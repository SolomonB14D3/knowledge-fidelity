"""Diagnostic: Re-run bias evaluation on 7B with new diversified probes (MLX backend).

MLX version of diagnose_bias_7b.py. Runs entirely on Apple Silicon GPU via MLX,
avoiding the MPS deadlock issues with PyTorch on 7B+ models.

Pipeline per config:
  1. Load model via mlx_lm.load()
  2. Baseline bias audit
  3. SVD compress Q/K/O projections (MLX numpy-like ops)
  4. LoRA SFT with rho-guided contrastive loss (mlx_rho_guided_sft)
  5. Final bias audit
  6. Per-category disaggregated comparison

Usage:
    python experiments/diagnose_bias_7b_mlx.py Qwen/Qwen2.5-7B-Instruct --configs star control
    python experiments/diagnose_bias_7b_mlx.py Qwen/Qwen2.5-7B-Instruct --configs star --bias-only
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np

# Ensure src/ on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ── Named configs matching the sweep ─────────────────────────────────────

NAMED_CONFIGS = {
    "star": {
        "compress_ratio": 0.7,
        "freeze_fraction": 0.75,
        "sae_layer": None,
        "rho_weight": 0.2,
        "tag": "cr0.7_saeNone_rho0.2",
    },
    "control": {
        "compress_ratio": 0.7,
        "freeze_fraction": 0.75,
        "sae_layer": 17,
        "rho_weight": 0.0,
        "tag": "cr0.7_sae17_rho0.0",
    },
    "sae_rho": {
        "compress_ratio": 0.7,
        "freeze_fraction": 0.75,
        "sae_layer": 17,
        "rho_weight": 0.2,
        "tag": "cr0.7_sae17_rho0.2",
    },
    "rho_only": {
        "compress_ratio": 1.0,
        "freeze_fraction": 0.75,
        "sae_layer": None,
        "rho_weight": 0.2,
        "tag": "cr1.0_saeNone_rho0.2",
    },
}


def mlx_svd_compress(model, ratio: float = 0.7) -> int:
    """SVD compress Q/K/O attention projections in an MLX model.

    Equivalent to rho_eval.svd.compress_qko() but operates on MLX weight
    matrices using numpy for SVD (mlx doesn't have linalg.svd yet).
    """
    import mlx.core as mx

    compressed = 0
    safe_projections = ["q_proj", "k_proj", "o_proj"]

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for proj_name in safe_projections:
            if not hasattr(attn, proj_name):
                continue

            proj = getattr(attn, proj_name)
            W_mx = proj.weight
            # Convert to float32 numpy for SVD (MLX bfloat16 can't go direct to numpy)
            W = np.array(W_mx.astype(mx.float32))

            if len(W.shape) != 2 or min(W.shape) <= 10:
                continue

            rank = max(1, int(min(W.shape) * ratio))

            try:
                U, S, Vh = np.linalg.svd(W, full_matrices=False)
                W_approx = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
                proj.weight = mx.array(W_approx).astype(W_mx.dtype)
                compressed += 1
            except Exception as e:
                print(f"    SVD failed for layer {i} {proj_name}: {e}", flush=True)
                continue

    mx.eval(model.parameters())
    return compressed


def mlx_freeze_layers(model, ratio: float = 0.75) -> dict:
    """Mark layers as frozen by disabling LoRA for bottom layers.

    In MLX, freezing is handled by only applying LoRA to unfrozen layers.
    We return the freeze info so the LoRA setup knows which layers to skip.
    """
    n_layers = len(model.model.layers)
    n_freeze = int(n_layers * ratio)

    return {
        "n_layers": n_layers,
        "n_frozen": n_freeze,
        "freeze_ratio": ratio,
        "frozen_layer_indices": list(range(n_freeze)),
        "trainable_layer_indices": list(range(n_freeze, n_layers)),
    }


def run_bias_audit(model, tokenizer, label: str) -> dict:
    """Run bias audit and return structured results."""
    from rho_eval.audit import audit

    t0 = time.time()
    report = audit(model=model, tokenizer=tokenizer, behaviors=["bias"], device="mlx")
    elapsed = time.time() - t0

    bias_result = report.behaviors["bias"]
    print(f"  [{label}] Bias audit: {elapsed:.1f}s", flush=True)
    print(f"    rho={bias_result.rho:.4f}, "
          f"{bias_result.positive_count}/{bias_result.total} "
          f"({bias_result.retention:.1%})", flush=True)

    # Category breakdown
    meta = bias_result.metadata or {}
    cat_metrics = meta.get("category_metrics", {})
    if cat_metrics:
        print(f"    {'Category':<30s} {'Accuracy':>8s} {'N':>4s}", flush=True)
        for cat, data in sorted(cat_metrics.items(), key=lambda x: -x[1]["accuracy"]):
            print(f"    {cat:<30s} {data['accuracy']:>7.1%} {data['n']:>4d}", flush=True)

    return {
        "rho": bias_result.rho,
        "positive_count": bias_result.positive_count,
        "total": bias_result.total,
        "retention": bias_result.retention,
        "category_metrics": cat_metrics,
        "elapsed": elapsed,
    }


def run_diagnostic_mlx(
    model_name: str,
    config_name: str,
    output_dir: Path,
):
    """Run a single diagnostic config entirely on MLX."""
    import mlx_lm
    import mlx.core as mx

    params = NAMED_CONFIGS[config_name]
    tag = params["tag"]
    run_dir = output_dir / f"{tag}_diag_mlx"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  DIAGNOSTIC (MLX): {config_name} ({tag})", flush=True)
    print(f"  compress={params['compress_ratio']}, freeze={params['freeze_fraction']}, "
          f"rho={params['rho_weight']}, sae={params['sae_layer']}", flush=True)
    print(f"  Output: {run_dir}", flush=True)
    print(f"{'='*70}", flush=True)

    t_start = time.time()

    # ── Load model ───────────────────────────────────────────────────
    print(f"\n  Loading {model_name} via MLX...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(model_name)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # ── Phase 1: Baseline bias audit ─────────────────────────────────
    print(f"\n  Phase 1: Baseline audit", flush=True)
    baseline = run_bias_audit(model, tokenizer, "baseline")

    # ── Phase 2: SVD compression ─────────────────────────────────────
    if params["compress_ratio"] < 1.0:
        print(f"\n  Phase 2: SVD compression (ratio={params['compress_ratio']})", flush=True)
        t0 = time.time()
        n_compressed = mlx_svd_compress(model, ratio=params["compress_ratio"])
        freeze_info = mlx_freeze_layers(model, ratio=params["freeze_fraction"])
        mx.eval(model.parameters())  # Ensure all weights are materialized
        print(f"  Compressed {n_compressed} matrices, "
              f"frozen {freeze_info['n_frozen']}/{freeze_info['n_layers']} layers "
              f"in {time.time()-t0:.1f}s", flush=True)
    else:
        freeze_info = {"n_frozen": 0, "n_layers": len(model.model.layers),
                       "frozen_layer_indices": [], "trainable_layer_indices": list(range(len(model.model.layers)))}

    # ── Phase 3: SAE (skip for now — SAE is PyTorch-only) ────────────
    if params["sae_layer"] is not None:
        print(f"\n  Phase 3: SAE at layer {params['sae_layer']} — SKIPPED (MLX mode)")
        print(f"    SAE features are PyTorch-only. MLX diagnostic runs without SAE.", flush=True)

    # ── Phase 4: LoRA SFT ────────────────────────────────────────────
    if params["rho_weight"] > 0:
        print(f"\n  Phase 4: LoRA SFT (rho_weight={params['rho_weight']})", flush=True)
        t0 = time.time()

        from rho_eval.alignment.dataset import (
            _build_trap_texts, _load_alpaca_texts, BehavioralContrastDataset,
        )
        from rho_eval.alignment.mlx_trainer import mlx_rho_guided_sft
        import random as _random

        # Build raw text list for MLX (bypasses PyTorch TextDataset)
        rng = _random.Random(42)
        sft_texts = []
        trap_texts = _build_trap_texts(["sycophancy"], seed=42)
        rng.shuffle(trap_texts)
        sft_texts.extend(trap_texts[:400])  # 20% of 2000
        try:
            alpaca_texts = _load_alpaca_texts(1600, seed=42)
            sft_texts.extend(alpaca_texts)
        except Exception as e:
            print(f"    Alpaca load failed ({e}), using traps only", flush=True)
        rng.shuffle(sft_texts)
        sft_texts = sft_texts[:2000]
        print(f"  {len(sft_texts)} SFT texts loaded", flush=True)

        contrast_dataset = BehavioralContrastDataset(behaviors=["sycophancy"], seed=42)

        sft_result = mlx_rho_guided_sft(
            model, tokenizer,
            sft_texts,
            contrast_dataset=contrast_dataset,
            rho_weight=params["rho_weight"],
            epochs=1,
            lr=2e-4,
            margin=0.1,
        )
        sft_elapsed = time.time() - t0
        print(f"  SFT complete in {sft_elapsed:.1f}s "
              f"({sft_result['steps']} steps)", flush=True)
    else:
        print(f"\n  Phase 4: SFT skipped (rho_weight=0)", flush=True)

    # ── Phase 5: Final bias audit ────────────────────────────────────
    print(f"\n  Phase 5: Final audit", flush=True)
    final = run_bias_audit(model, tokenizer, "final")

    # ── Results ──────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    print(f"\n  {'='*60}", flush=True)
    print(f"  RESULTS: {config_name} ({tag})", flush=True)
    print(f"  Total time: {total_elapsed/60:.1f} min", flush=True)
    print(f"  Bias: {baseline['rho']:.4f} → {final['rho']:.4f} "
          f"(Δ={final['rho']-baseline['rho']:+.4f})", flush=True)

    # Per-category comparison
    if baseline["category_metrics"] and final["category_metrics"]:
        print(f"\n  {'Category':<30s} {'Before':>8s} {'After':>8s} {'Delta':>8s}", flush=True)
        print(f"  {'-'*58}", flush=True)
        for cat in sorted(final["category_metrics"].keys()):
            before = baseline["category_metrics"].get(cat, {}).get("accuracy", 0)
            after = final["category_metrics"][cat]["accuracy"]
            delta = after - before
            marker = "▼" if delta < -0.05 else ("▲" if delta > 0.05 else " ")
            print(f"  {cat:<30s} {before:>7.1%} {after:>7.1%} {delta:>+7.1%} {marker}", flush=True)

    # Save results
    result_data = {
        "config": config_name,
        "tag": tag,
        "model": model_name,
        "backend": "mlx",
        "total_elapsed_sec": total_elapsed,
        "baseline_bias": baseline,
        "final_bias": final,
    }
    result_path = run_dir / "diagnostic_result.json"
    result_path.write_text(json.dumps(result_data, indent=2, default=str))
    print(f"\n  Saved to {result_path}", flush=True)

    return result_data


def main():
    parser = argparse.ArgumentParser(
        description="Bias diagnostic for 7B hybrid sweep configs (MLX backend)",
    )
    parser.add_argument("model", help="HuggingFace model ID")
    parser.add_argument(
        "--configs", nargs="+", default=["star", "control"],
        choices=list(NAMED_CONFIGS.keys()),
        help="Which configs to diagnose (default: star control)",
    )
    parser.add_argument(
        "--bias-only", action="store_true",
        help="Only evaluate bias behavior (always true in MLX mode)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/hybrid_sweep/diagnostics",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMLX Bias Diagnostic", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Configs: {args.configs}", flush=True)
    print(f"  Output: {output_dir}", flush=True)

    all_results = {}
    for config_name in args.configs:
        result = run_diagnostic_mlx(args.model, config_name, output_dir)
        all_results[config_name] = result

    # Summary comparison
    if len(all_results) >= 2:
        print(f"\n{'='*70}", flush=True)
        print(f"  COMPARISON SUMMARY", flush=True)
        print(f"{'='*70}", flush=True)
        for name, r in all_results.items():
            b = r["baseline_bias"]["rho"]
            f = r["final_bias"]["rho"]
            print(f"  {name:<15s}: bias {b:.4f} → {f:.4f} (Δ={f-b:+.4f})", flush=True)

    print(f"\nDone!", flush=True)


if __name__ == "__main__":
    main()
