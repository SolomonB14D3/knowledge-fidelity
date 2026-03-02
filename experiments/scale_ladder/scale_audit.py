#!/usr/bin/env python3
"""Run rho-eval audit + subspace analysis on a trained scale ladder model.

Measures behavioral subspace structure at all layers: effective dimensionality,
Grassmann angles between behavior pairs, and bootstrap d-prime for each behavior.

Usage:
    python experiments/scale_ladder/scale_audit.py --checkpoint results/scale_ladder/7M_seed42
    python experiments/scale_ladder/scale_audit.py --checkpoint results/scale_ladder/64M_seed42 --device mps
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale_ladder.configs import SUBSPACE_BEHAVIORS


def load_checkpoint(checkpoint_dir, device="cpu"):
    """Load a trained model from HuggingFace save_pretrained format."""
    from transformers import AutoModelForCausalLM, GPT2TokenizerFast

    model_path = Path(checkpoint_dir) / "model"
    if not model_path.exists():
        model_path = Path(checkpoint_dir)

    print(f"  Loading model from {model_path}...", flush=True)
    import torch
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to(torch.device(device))
    model.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.n_layer
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params:,} params, {n_layers} layers, d={model.config.n_embd}")

    return model, tokenizer


def run_rho_audit(model, tokenizer, device="cpu"):
    """Run the full 8-behavior rho-eval audit."""
    from rho_eval.audit import audit

    print(f"\n  Running rho-eval audit (all 8 behaviors)...", flush=True)
    t0 = time.time()

    report = audit(
        model=model,
        tokenizer=tokenizer,
        behaviors="all",
        seed=42,
        device=device,
    )

    elapsed = time.time() - t0
    print(f"  Audit complete in {elapsed:.0f}s", flush=True)

    # Print summary
    print(f"\n  {'Behavior':<20s} {'rho':>8s} {'Status':>8s}")
    print(f"  {'-'*40}")
    for result in report.behaviors.values():
        print(f"  {result.behavior:<20s} {result.rho:>8.3f} {result.status:>8s}")

    return report


def run_subspace_extraction(model, tokenizer, device="cpu"):
    """Extract behavioral subspaces at ALL layers."""
    from rho_eval.interpretability.subspaces import extract_subspaces
    from rho_eval.interpretability.overlap import compute_overlap

    n_layers = model.config.n_layer
    all_layers = list(range(n_layers))

    print(f"\n  Extracting subspaces at all {n_layers} layers...", flush=True)
    print(f"  Behaviors: {SUBSPACE_BEHAVIORS}")
    t0 = time.time()

    subspaces = extract_subspaces(
        model,
        tokenizer,
        behaviors=SUBSPACE_BEHAVIORS,
        layers=all_layers,
        device=device,
        max_rank=50,
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f"  Subspace extraction complete in {elapsed:.0f}s", flush=True)

    # Compute overlap
    print(f"\n  Computing pairwise overlaps (Grassmann angles)...", flush=True)
    overlap = compute_overlap(subspaces, top_k=10)

    # Extract per-layer effective dimensionality
    eff_dim_data = {}
    for beh_name, layer_dict in subspaces.items():
        eff_dim_data[beh_name] = {}
        for layer_idx, result in layer_dict.items():
            eff_dim_data[beh_name][layer_idx] = {
                "effective_dim": result.effective_dim,
                "explained_variance_90": float(result.explained_variance[min(result.effective_dim, len(result.explained_variance) - 1)]) if result.effective_dim < len(result.explained_variance) else 1.0,
                "total_variance_captured": float(result.explained_variance[-1]) if len(result.explained_variance) > 0 else 0.0,
                "singular_values_top5": [float(v) for v in result.singular_values[:5]],
            }

    return subspaces, overlap, eff_dim_data


def bootstrap_dprime(model, tokenizer, n_resamples=100, seed=42, device="cpu"):
    """Compute bootstrap d-prime for each behavior.

    d-prime measures how well the model discriminates between positive
    and negative examples for each behavior. Higher d-prime = the model
    has learned behavioral structure at this scale.
    """
    import torch
    from rho_eval.behaviors import get_behavior

    print(f"\n  Computing bootstrap d-prime ({n_resamples} resamples)...", flush=True)
    rng = np.random.RandomState(seed)

    results = {}
    for beh_name in SUBSPACE_BEHAVIORS:
        beh = get_behavior(beh_name)
        probes = beh.load_probes(seed=seed)

        # Get confidence scores for positive and negative examples
        pos_confs = []
        neg_confs = []

        for probe in probes:
            if "positive_text" in probe and "negative_text" in probe:
                pos_text = probe["positive_text"]
                neg_text = probe["negative_text"]
            elif "text" in probe:
                # Some behaviors use text + truthful_answer / sycophantic_answer
                if "truthful_answer" in probe and "sycophantic_answer" in probe:
                    pos_text = f"{probe['text']}\n{probe['truthful_answer']}"
                    neg_text = f"{probe['text']}\n{probe['sycophantic_answer']}"
                else:
                    continue
            else:
                continue

            # Get model log-probs as confidence proxy
            for text, collector in [(pos_text, pos_confs), (neg_text, neg_confs)]:
                tokens = tokenizer.encode(text, return_tensors="pt").to(
                    next(model.parameters()).device
                )
                if tokens.shape[1] < 2:
                    continue
                if tokens.shape[1] > 256:
                    tokens = tokens[:, :256]

                with torch.no_grad():
                    outputs = model(tokens)
                    logits = outputs.logits[:, :-1, :]
                    targets = tokens[:, 1:]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
                    mean_lp = token_log_probs.mean().item()
                    collector.append(mean_lp)

        if len(pos_confs) < 5 or len(neg_confs) < 5:
            results[beh_name] = {
                "dprime_mean": 0.0, "dprime_std": 0.0,
                "n_pos": len(pos_confs), "n_neg": len(neg_confs),
                "note": "insufficient samples",
            }
            continue

        pos_arr = np.array(pos_confs)
        neg_arr = np.array(neg_confs)

        # Bootstrap d-prime
        dprimes = []
        for _ in range(n_resamples):
            idx_p = rng.choice(len(pos_arr), len(pos_arr), replace=True)
            idx_n = rng.choice(len(neg_arr), len(neg_arr), replace=True)
            p, n = pos_arr[idx_p], neg_arr[idx_n]
            pooled_var = 0.5 * (p.var() + n.var())
            if pooled_var < 1e-12:
                dprimes.append(0.0)
            else:
                d = (p.mean() - n.mean()) / np.sqrt(pooled_var)
                dprimes.append(d)

        results[beh_name] = {
            "dprime_mean": round(float(np.mean(dprimes)), 4),
            "dprime_std": round(float(np.std(dprimes)), 4),
            "dprime_ci95": [
                round(float(np.percentile(dprimes, 2.5)), 4),
                round(float(np.percentile(dprimes, 97.5)), 4),
            ],
            "n_pos": len(pos_confs),
            "n_neg": len(neg_confs),
            "pos_mean": round(float(pos_arr.mean()), 4),
            "neg_mean": round(float(neg_arr.mean()), 4),
        }

        print(f"    {beh_name:<15s}: d'={results[beh_name]['dprime_mean']:+.3f} "
              f"(±{results[beh_name]['dprime_std']:.3f}), "
              f"n_pos={len(pos_confs)}, n_neg={len(neg_confs)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        prog="scale-audit",
        description="Audit a scale ladder model with rho-eval + subspace analysis",
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained model checkpoint directory")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps", "cuda"])
    parser.add_argument("--skip-subspace", action="store_true",
                        help="Skip subspace extraction (just run audit + d-prime)")
    parser.add_argument("--skip-dprime", action="store_true",
                        help="Skip bootstrap d-prime")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"  ERROR: Checkpoint not found: {checkpoint_path}")
        return 1

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Scale Audit — {checkpoint_path.name}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_checkpoint(args.checkpoint, device=args.device)

    # 1. Rho-eval audit
    report = run_rho_audit(model, tokenizer, device=args.device)

    # Save audit report
    audit_data = {
        "checkpoint": str(checkpoint_path),
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_layers": model.config.n_layer,
        "d_model": model.config.n_embd,
        "results": [
            {
                "behavior": r.behavior,
                "rho": r.rho,
                "positive_count": r.positive_count,
                "total": r.total,
                "status": r.status,
            }
            for r in report.behaviors.values()
        ],
    }
    audit_path = checkpoint_path / "audit_report.json"
    audit_path.write_text(json.dumps(audit_data, indent=2))
    print(f"\n  Saved: {audit_path}")

    # 2. Subspace extraction
    subspace_data = None
    if not args.skip_subspace:
        subspaces, overlap, eff_dim_data = run_subspace_extraction(
            model, tokenizer, device=args.device
        )

        # Serialize overlap (convert numpy to native types)
        overlap_serializable = {}
        for key, matrix in overlap.items():
            if hasattr(matrix, "tolist"):
                overlap_serializable[key] = matrix.tolist()
            elif isinstance(matrix, dict):
                overlap_serializable[key] = {
                    str(k): v.tolist() if hasattr(v, "tolist") else v
                    for k, v in matrix.items()
                }
            else:
                overlap_serializable[key] = str(matrix)

        subspace_data = {
            "checkpoint": str(checkpoint_path),
            "behaviors": SUBSPACE_BEHAVIORS,
            "n_layers": model.config.n_layer,
            "effective_dimensionality": {
                beh: {str(k): v for k, v in layers.items()}
                for beh, layers in eff_dim_data.items()
            },
            "overlap": overlap_serializable,
        }

        sub_path = checkpoint_path / "subspace_report.json"
        sub_path.write_text(json.dumps(subspace_data, indent=2, default=str))
        print(f"  Saved: {sub_path}")

    # 3. Bootstrap d-prime
    dprime_data = None
    if not args.skip_dprime:
        dprime_data = bootstrap_dprime(
            model, tokenizer, n_resamples=100, seed=42, device=args.device
        )

        dp_path = checkpoint_path / "dprime_bootstrap.json"
        dp_path.write_text(json.dumps(dprime_data, indent=2))
        print(f"  Saved: {dp_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total audit time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
