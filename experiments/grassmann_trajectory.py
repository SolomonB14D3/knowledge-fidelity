#!/usr/bin/env python3
"""Grassmann angle trajectory extraction for behavioral subspace analysis.

Phase A: Extract baseline and post-SVD subspaces (run once before surgery).
Phase B: Extract subspaces at each LoRA checkpoint (run after surgery completes).

Produces Grassmann principal angle trajectories showing how behavioral subspace
geometry evolves during rho-guided SFT at different γ values.

Usage:
    # Phase A: baseline extraction (run once)
    python experiments/grassmann_trajectory.py --phase A \
        --model Qwen/Qwen2.5-7B-Instruct \
        -o results/grassmann_trajectory

    # Phase B: checkpoint extraction (run after surgery)
    python experiments/grassmann_trajectory.py --phase B \
        --model Qwen/Qwen2.5-7B-Instruct \
        --checkpoint-dir results/grassmann_trajectory/gamma_0.03 \
        -o results/grassmann_trajectory

    # Both phases
    python experiments/grassmann_trajectory.py --phase AB \
        --model Qwen/Qwen2.5-7B-Instruct \
        --checkpoint-dir results/grassmann_trajectory/gamma_0.03 \
        -o results/grassmann_trajectory
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SUBSPACE_BEHAVIORS = ["factual", "sycophancy", "toxicity", "bias"]


def _serialize_overlap(overlap_dict: dict) -> dict:
    """Convert OverlapMatrix objects to JSON-serializable dicts."""
    result = {}
    for layer_idx, om in overlap_dict.items():
        result[str(layer_idx)] = {
            "layer_idx": om.layer_idx,
            "behaviors": om.behaviors,
            "cosine_matrix": om.cosine_matrix,
            "shared_variance": om.shared_variance,
            "subspace_angles": om.subspace_angles,
            "rank_used": om.rank_used,
        }
    return result


def _serialize_subspaces_summary(subspaces: dict) -> dict:
    """Extract key metrics from subspaces (skip raw tensors for JSON)."""
    result = {}
    for beh_name, layer_dict in subspaces.items():
        result[beh_name] = {}
        for layer_idx, sr in layer_dict.items():
            result[beh_name][str(layer_idx)] = {
                "effective_dim": sr.effective_dim,
                "singular_values_top5": [float(v) for v in sr.singular_values[:5]],
                "explained_variance_90": float(
                    sr.explained_variance[min(sr.effective_dim, len(sr.explained_variance) - 1)]
                ) if sr.effective_dim < len(sr.explained_variance) else 1.0,
            }
    return result


def phase_a(model_name: str, output_dir: str, compress_ratio: float = 0.7):
    """Phase A: Extract baseline and post-SVD behavioral subspaces.

    Produces:
        - baseline_subspaces.json: subspace summary + overlap before any intervention
        - compressed_subspaces.json: subspace summary + overlap after SVD compression
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rho_eval.interpretability.subspaces import extract_subspaces
    from rho_eval.interpretability.overlap import compute_overlap

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if (out / "baseline_subspaces.json").exists() and (out / "compressed_subspaces.json").exists():
        print("  Phase A already complete, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"  PHASE A: Baseline Subspace Extraction")
    print(f"  Model: {model_name}")
    print(f"{'='*60}\n")

    # Load model in PyTorch (CPU — inference only)
    print("  Loading model (PyTorch CPU)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.0f}s")

    # ── Baseline subspaces (untouched model) ──
    print("\n  Extracting BASELINE subspaces...", flush=True)
    t0 = time.time()
    baseline_subspaces = extract_subspaces(
        model, tokenizer, behaviors=SUBSPACE_BEHAVIORS,
        device="cpu", max_rank=50, verbose=True,
    )
    baseline_overlap = compute_overlap(baseline_subspaces, top_k=10)
    elapsed = time.time() - t0
    print(f"  Baseline extraction: {elapsed:.0f}s")

    baseline_data = {
        "stage": "baseline",
        "model": model_name,
        "subspaces": _serialize_subspaces_summary(baseline_subspaces),
        "overlap": _serialize_overlap(baseline_overlap),
        "elapsed_sec": elapsed,
    }
    (out / "baseline_subspaces.json").write_text(json.dumps(baseline_data, indent=2))
    print(f"  Saved → {out / 'baseline_subspaces.json'}")

    # ── Post-SVD subspaces (compressed, before LoRA training) ──
    print(f"\n  Applying SVD compression (ratio={compress_ratio})...", flush=True)
    t0 = time.time()

    # Apply SVD compression to Q/K/O projections (same as surgery pipeline)
    n_compressed = 0
    for layer in model.model.layers:
        for proj_name in ["q_proj", "k_proj", "o_proj"]:
            proj = getattr(layer.self_attn, proj_name, None)
            if proj is None or not hasattr(proj, "weight"):
                continue
            W = proj.weight.data.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            k = max(1, int(S.shape[0] * compress_ratio))
            W_approx = (U[:, :k] * S[:k]) @ Vh[:k, :]
            proj.weight.data = W_approx.to(proj.weight.dtype)
            n_compressed += 1

    print(f"  Compressed {n_compressed} matrices in {time.time() - t0:.0f}s")

    print("\n  Extracting POST-SVD subspaces...", flush=True)
    t0 = time.time()
    compressed_subspaces = extract_subspaces(
        model, tokenizer, behaviors=SUBSPACE_BEHAVIORS,
        device="cpu", max_rank=50, verbose=True,
    )
    compressed_overlap = compute_overlap(compressed_subspaces, top_k=10)
    elapsed = time.time() - t0
    print(f"  Compressed extraction: {elapsed:.0f}s")

    compressed_data = {
        "stage": "compressed",
        "model": model_name,
        "compress_ratio": compress_ratio,
        "n_compressed_matrices": n_compressed,
        "subspaces": _serialize_subspaces_summary(compressed_subspaces),
        "overlap": _serialize_overlap(compressed_overlap),
        "elapsed_sec": elapsed,
    }
    (out / "compressed_subspaces.json").write_text(json.dumps(compressed_data, indent=2))
    print(f"  Saved → {out / 'compressed_subspaces.json'}")

    # ── Compare baseline vs compressed ──
    print("\n  === Baseline → Compressed angle shifts ===")
    for layer_idx in sorted(set(baseline_overlap.keys()) & set(compressed_overlap.keys())):
        b_angles = baseline_overlap[layer_idx].subspace_angles
        c_angles = compressed_overlap[layer_idx].subspace_angles
        behaviors = baseline_overlap[layer_idx].behaviors
        print(f"  Layer {layer_idx}:")
        for i in range(len(behaviors)):
            for j in range(i + 1, len(behaviors)):
                delta = c_angles[i][j] - b_angles[i][j]
                print(f"    {behaviors[i][:4]}↔{behaviors[j][:4]}: "
                      f"{b_angles[i][j]:.1f}° → {c_angles[i][j]:.1f}° (Δ={delta:+.1f}°)")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def phase_b(
    model_name: str,
    checkpoint_dir: str,
    output_dir: str,
    compress_ratio: float = 0.7,
    gamma_label: str = "unknown",
):
    """Phase B: Extract subspaces at each LoRA checkpoint.

    Loads the SVD-compressed base model once, then for each checkpoint:
    applies LoRA weights → extracts subspaces → computes Grassmann angles.

    Produces:
        - grassmann_trajectory_{gamma_label}.json: angles at each checkpoint step
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rho_eval.interpretability.subspaces import extract_subspaces
    from rho_eval.interpretability.overlap import compute_overlap

    out = Path(output_dir)
    ckpt_dir = Path(checkpoint_dir)

    # Find all LoRA checkpoints
    ckpt_files = sorted(ckpt_dir.glob("lora_step_*.npz"))
    if not ckpt_files:
        print(f"  No LoRA checkpoints found in {ckpt_dir}")
        return

    steps = [int(f.stem.split("_")[-1]) for f in ckpt_files]
    print(f"\n{'='*60}")
    print(f"  PHASE B: Grassmann Trajectory Extraction")
    print(f"  Model: {model_name}, γ={gamma_label}")
    print(f"  Checkpoints: {steps}")
    print(f"{'='*60}\n")

    # Load and compress base model once
    print("  Loading + compressing base model (PyTorch CPU)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Apply same SVD compression as surgery
    for layer in model.model.layers:
        for proj_name in ["q_proj", "k_proj", "o_proj"]:
            proj = getattr(layer.self_attn, proj_name, None)
            if proj is None or not hasattr(proj, "weight"):
                continue
            W = proj.weight.data.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            k = max(1, int(S.shape[0] * compress_ratio))
            W_approx = (U[:, :k] * S[:k]) @ Vh[:k, :]
            proj.weight.data = W_approx.to(proj.weight.dtype)

    print(f"  Base model ready in {time.time() - t0:.0f}s")

    # Save base weights for restoration between checkpoints
    base_state = {name: param.clone() for name, param in model.named_parameters()}

    trajectory = []

    for ckpt_file in ckpt_files:
        step = int(ckpt_file.stem.split("_")[-1])
        print(f"\n  ── Step {step} ──────────────────────────────────", flush=True)

        # Restore base weights
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(base_state[name])

        # Load and apply LoRA checkpoint
        if step > 0:  # step 0 is random init, apply it too
            lora_data = np.load(str(ckpt_file))
            n_applied = _apply_lora_checkpoint(model, lora_data)
            print(f"  Applied {n_applied} LoRA matrices from {ckpt_file.name}")
        else:
            # Step 0: apply random LoRA init
            lora_data = np.load(str(ckpt_file))
            n_applied = _apply_lora_checkpoint(model, lora_data)
            print(f"  Applied step-0 (random init) LoRA: {n_applied} matrices")

        # Extract subspaces
        t0 = time.time()
        subspaces = extract_subspaces(
            model, tokenizer, behaviors=SUBSPACE_BEHAVIORS,
            device="cpu", max_rank=50, verbose=False,
        )
        overlap = compute_overlap(subspaces, top_k=10, verbose=False)
        elapsed = time.time() - t0
        print(f"  Extraction: {elapsed:.0f}s")

        # Record
        entry = {
            "step": step,
            "gamma": gamma_label,
            "subspaces": _serialize_subspaces_summary(subspaces),
            "overlap": _serialize_overlap(overlap),
            "elapsed_sec": elapsed,
        }
        trajectory.append(entry)

        # Print key angle
        for layer_idx in sorted(overlap.keys()):
            angles = overlap[layer_idx].subspace_angles
            behaviors = overlap[layer_idx].behaviors
            if "sycophancy" in behaviors and "bias" in behaviors:
                si = behaviors.index("sycophancy")
                bi = behaviors.index("bias")
                print(f"  Layer {layer_idx}: syc↔bias = {angles[si][bi]:.1f}°")

    # Save trajectory
    traj_path = out / f"grassmann_trajectory_{gamma_label}.json"
    traj_path.write_text(json.dumps(trajectory, indent=2))
    print(f"\n  Trajectory saved → {traj_path}")

    del model, base_state
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return trajectory


def _apply_lora_checkpoint(model, lora_data: dict) -> int:
    """Apply saved LoRA adapter weights to a PyTorch model.

    The LoRA checkpoint contains {layer_name.lora_a: array, layer_name.lora_b: array}.
    We reconstruct the low-rank update: W_new = W_base + lora_b @ lora_a (with scaling).

    For simplicity, we fuse the LoRA directly into the weight matrices rather than
    maintaining separate adapter layers.
    """
    n_applied = 0

    # Build a map of module names to modules
    module_map = {name: mod for name, mod in model.named_modules()}

    # Group LoRA weights by module
    lora_modules = {}
    for key in lora_data.files:
        # Key format: "model.layers.0.self_attn.q_proj.lora_a"
        parts = key.rsplit(".", 1)
        if len(parts) == 2:
            module_name, weight_type = parts
            if module_name not in lora_modules:
                lora_modules[module_name] = {}
            lora_modules[module_name][weight_type] = torch.from_numpy(lora_data[key]).float()

    for module_name, weights in lora_modules.items():
        if "lora_a" not in weights or "lora_b" not in weights:
            continue

        # Find the corresponding model module
        # MLX names: model.layers.0.self_attn.q_proj
        # PyTorch names: model.model.layers.0.self_attn.q_proj
        # Try both naming conventions
        for candidate in [module_name, f"model.{module_name}"]:
            if candidate in module_map:
                module = module_map[candidate]
                break
        else:
            continue

        if not hasattr(module, "weight"):
            continue

        lora_a = weights["lora_a"]  # MLX shape: (in_features, rank)
        lora_b = weights["lora_b"]  # MLX shape: (rank, out_features)

        # LoRA in MLX: y = x @ W^T + scale * (x @ lora_a @ lora_b)
        # Effective PyTorch weight update: W += scale * lora_b^T @ lora_a^T
        # Default scale = alpha/rank = 16/8 = 2.0
        scale = 2.0  # lora_alpha / lora_rank
        with torch.no_grad():
            # Transpose both to convert from MLX convention to PyTorch
            # lora_b^T: (out_features, rank), lora_a^T: (rank, in_features)
            # Product: (out_features, in_features) = PyTorch W shape
            delta = scale * (lora_b.T @ lora_a.T)
            module.weight.data += delta.to(module.weight.dtype)
        n_applied += 1

    return n_applied


def main():
    parser = argparse.ArgumentParser(description="Grassmann angle trajectory extraction")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--phase", choices=["A", "B", "AB"], default="AB",
                        help="Which phase to run")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory with lora_step_*.npz files (Phase B)")
    parser.add_argument("--compress-ratio", type=float, default=0.7,
                        help="SVD compression ratio (default: 0.7)")
    parser.add_argument("--gamma-label", type=str, default="unknown",
                        help="Gamma value label for output naming")
    parser.add_argument("-o", "--output-dir", type=str,
                        default="results/grassmann_trajectory",
                        help="Output directory")
    args = parser.parse_args()

    if "A" in args.phase:
        phase_a(args.model, args.output_dir, args.compress_ratio)

    if "B" in args.phase:
        if not args.checkpoint_dir:
            parser.error("--checkpoint-dir required for Phase B")
        phase_b(
            args.model, args.checkpoint_dir, args.output_dir,
            args.compress_ratio, args.gamma_label,
        )


if __name__ == "__main__":
    main()
