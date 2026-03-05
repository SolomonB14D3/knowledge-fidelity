#!/usr/bin/env python3
"""Extract full SVD spectrum at representative layers for developmental sweep.

The standard scale_audit.py only saves top-5 singular values per behavior per
layer. This script extracts the FULL spectrum (all singular values) at the
first and last transformer block for each of the 4 subspace behaviors.

This enables:
- Spectral decay analysis across scales and conditions
- Precise effective dimensionality comparison (not limited to saved top-k)
- Entropy-based dimensionality metrics

Usage:
    # Single checkpoint
    python experiments/developmental_sweep/extract_svd_spectrum.py \
        --checkpoint results/scale_ladder/3M_seed42 --device mps

    # All checkpoints in a sweep
    python experiments/developmental_sweep/extract_svd_spectrum.py \
        --sweep results/scale_ladder --pattern "3M_seed42*" --device mps
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale_ladder.configs import SUBSPACE_BEHAVIORS


def extract_spectrum(model, tokenizer, layers, behaviors, device="cpu"):
    """Extract full SVD spectrum at given layers for each behavior.

    Returns dict:
        {behavior: {layer_idx: {
            singular_values: [...],
            explained_variance_cumulative: [...],
            effective_dim_90: int,
            effective_dim_95: int,
            spectral_entropy: float,
            n_pairs: int,
        }}}
    """
    from rho_eval.interpretability.activation import (
        LayerActivationCapture,
        build_contrast_pairs,
    )
    from rho_eval.interpretability.subspaces import _load_probes_for_behavior

    results = {}

    for behavior in behaviors:
        print(f"\n  [{behavior}] Loading probes...", flush=True)
        probes = _load_probes_for_behavior(behavior)
        pairs = build_contrast_pairs(behavior, probes)

        if len(pairs) < 3:
            print(f"    WARNING: Only {len(pairs)} pairs — skipping")
            continue

        print(f"    {len(pairs)} contrast pairs", flush=True)

        cap = LayerActivationCapture(model, layers)
        pos_by_layer = {l: [] for l in layers}
        neg_by_layer = {l: [] for l in layers}

        for i, pair in enumerate(pairs):
            # Positive
            inputs = tokenizer(
                pair["positive"], return_tensors="pt",
                truncation=True, max_length=512,
            ).to(device)
            model(**inputs)
            for layer_idx in layers:
                h = cap.get(layer_idx)
                pos_by_layer[layer_idx].append(h[0, -1, :].cpu())

            # Negative
            inputs = tokenizer(
                pair["negative"], return_tensors="pt",
                truncation=True, max_length=512,
            ).to(device)
            model(**inputs)
            for layer_idx in layers:
                h = cap.get(layer_idx)
                neg_by_layer[layer_idx].append(h[0, -1, :].cpu())

            cap.clear()

            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(pairs)} pairs", flush=True)

        cap.remove()

        behavior_results = {}
        for layer_idx in layers:
            pos_stack = torch.stack(pos_by_layer[layer_idx])
            neg_stack = torch.stack(neg_by_layer[layer_idx])

            diffs = pos_stack - neg_stack
            diffs_centered = diffs - diffs.mean(dim=0)
            diffs_cpu = diffs_centered.float()

            # Full SVD — no truncation
            U, S, Vh = torch.linalg.svd(diffs_cpu, full_matrices=False)

            svals = S.tolist()
            total_var = (S ** 2).sum().item()

            if total_var > 0:
                cumvar = ((S ** 2).cumsum(0) / total_var).tolist()
            else:
                cumvar = [1.0] * len(svals)

            # Effective dimensionality at 90% and 95% thresholds
            eff_dim_90 = len(svals)
            eff_dim_95 = len(svals)
            for j, cv in enumerate(cumvar):
                if cv >= 0.90 and eff_dim_90 == len(svals):
                    eff_dim_90 = j + 1
                if cv >= 0.95 and eff_dim_95 == len(svals):
                    eff_dim_95 = j + 1

            # Spectral entropy (normalized by log(rank))
            if total_var > 0:
                probs = (S ** 2 / total_var).numpy()
                probs = probs[probs > 1e-12]  # filter zeros
                entropy = -np.sum(probs * np.log(probs))
                max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0
                spectral_entropy = float(entropy / max_entropy)
            else:
                spectral_entropy = 0.0

            behavior_results[layer_idx] = {
                "singular_values": svals,
                "explained_variance_cumulative": cumvar,
                "effective_dim_90": eff_dim_90,
                "effective_dim_95": eff_dim_95,
                "spectral_entropy": round(spectral_entropy, 4),
                "n_pairs": len(pairs),
                "total_variance": round(total_var, 6),
            }

            print(
                f"    Layer {layer_idx}: eff_dim_90={eff_dim_90}, "
                f"eff_dim_95={eff_dim_95}, "
                f"entropy={spectral_entropy:.3f}, "
                f"top-1 var={cumvar[0]:.1%}",
                flush=True,
            )

        results[behavior] = behavior_results

    return results


def process_checkpoint(checkpoint_dir, device="cpu"):
    """Load model and extract full SVD spectrum."""
    from experiments.scale_ladder.scale_audit import load_checkpoint

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"  ERROR: Checkpoint not found: {checkpoint_path}")
        return None

    model, tokenizer = load_checkpoint(checkpoint_dir, device=device)

    # Representative layers: first and last transformer block
    n_layers = model.config.n_layer
    layers = [0, n_layers - 1]
    if n_layers > 2:
        # Also add middle layer for 4+ layer models
        layers = [0, n_layers // 2, n_layers - 1]
    layers = sorted(set(layers))

    print(f"  Extracting SVD spectrum at layers {layers}", flush=True)

    with torch.no_grad():
        spectrum = extract_spectrum(
            model, tokenizer, layers, SUBSPACE_BEHAVIORS, device=device
        )

    # Serialize: convert layer indices to strings for JSON
    output = {
        "checkpoint": str(checkpoint_path),
        "n_layers": n_layers,
        "d_model": model.config.n_embd,
        "n_params": sum(p.numel() for p in model.parameters()),
        "analyzed_layers": layers,
        "behaviors": SUBSPACE_BEHAVIORS,
        "spectra": {
            beh: {str(layer): data for layer, data in layers_dict.items()}
            for beh, layers_dict in spectrum.items()
        },
    }

    out_path = checkpoint_path / "svd_spectrum.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        prog="extract-svd-spectrum",
        description="Extract full SVD spectrum at representative layers",
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint directory")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Parent directory containing multiple checkpoints")
    parser.add_argument("--pattern", type=str, default="*_seed42*",
                        help="Glob pattern for checkpoint dirs within --sweep")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps", "cuda"])
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip checkpoints that already have svd_spectrum.json")
    args = parser.parse_args()

    if args.checkpoint:
        checkpoints = [Path(args.checkpoint)]
    elif args.sweep:
        sweep_dir = Path(args.sweep)
        checkpoints = sorted(
            d for d in sweep_dir.glob(args.pattern)
            if d.is_dir() and (d / "model").exists()
        )
        if not checkpoints:
            print(f"  ERROR: No matching checkpoints in {sweep_dir} with pattern {args.pattern}")
            return 1
    else:
        parser.error("Must specify --checkpoint or --sweep")

    print(f"\n  Processing {len(checkpoints)} checkpoint(s)...\n")

    for i, ckpt in enumerate(checkpoints):
        if args.skip_existing and (ckpt / "svd_spectrum.json").exists():
            print(f"  [{i+1}/{len(checkpoints)}] SKIP (exists): {ckpt.name}")
            continue

        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(checkpoints)}] {ckpt.name}")
        print(f"{'='*60}")
        t0 = time.time()

        process_checkpoint(str(ckpt), device=args.device)

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.0f}s")

    print(f"\n  All done!")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
