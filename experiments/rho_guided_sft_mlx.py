#!/usr/bin/env python3
"""Rho-Guided SFT Experiment — MLX Backend (Apple Silicon).

MLX equivalent of rho_guided_sft.py. Runs on Apple Silicon unified
memory (~10× faster than CPU-only PyTorch for 7B models).

Which experiment script should I use?
  - rho_guided_sft.py      — PyTorch, any hardware (CPU/CUDA/MPS*)
  - rho_guided_sft_mlx.py  — MLX, Apple Silicon only (M1/M2/M3/M4)
  * MPS has known NaN gradient bugs with Qwen + frozen layers;
    the PyTorch version forces CPU training as a workaround.

Requirements:
  pip install mlx mlx-lm    (for this MLX script)
  pip install peft           (for the PyTorch script)

Compares standard SFT (CE only) vs rho-guided SFT (CE + auxiliary
contrastive loss) across multiple rho_weight values and seeds.

Usage:
    python experiments/rho_guided_sft_mlx.py
    python experiments/rho_guided_sft_mlx.py --model qwen2.5-7b
    python experiments/rho_guided_sft_mlx.py --validate
    python experiments/rho_guided_sft_mlx.py --analyze results/alignment/mlx_rho_sft_sweep.json
    python experiments/rho_guided_sft_mlx.py --full-audit  # PyTorch audit after each run
"""

import argparse
import gc
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results" / "alignment"

MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
}

DEFAULT_RHO_WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
DEFAULT_SEEDS = [42, 123, 456]

BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias"]
ALL_EVAL_BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias", "reasoning", "refusal"]


# ── Quick MLX Confidence Scorer ───────────────────────────────────────

def mlx_quick_rho(model, tokenizer, behaviors: list[str], seed: int = 42) -> dict:
    """Quick confidence-gap scoring using the MLX model.

    Computes mean(logprob_positive - logprob_negative) per behavior.
    Not the same as full Spearman ρ, but strongly correlated — sufficient
    for tracking training progress within a sweep.

    Args:
        model: MLX model (fused, eval mode).
        tokenizer: MLX tokenizer.
        behaviors: List of behavior names.
        seed: Seed for probe loading.

    Returns:
        Dict mapping behavior name → confidence gap (float).
    """
    import mlx.core as mx
    import mlx.nn as nn
    from rho_eval.behaviors import get_behavior
    from rho_eval.interpretability.activation import build_contrast_pairs

    scores = {}

    for bname in behaviors:
        try:
            beh = get_behavior(bname)
            probes = beh.load_probes(seed=seed)
            pairs = build_contrast_pairs(bname, probes)
        except Exception as e:
            print(f"    [quick-rho] skip {bname}: {e}")
            scores[bname] = 0.0
            continue

        pos_lps, neg_lps = [], []
        for pair in pairs:
            pos_lp = _mlx_mean_logprob(model, tokenizer, pair["positive"])
            neg_lp = _mlx_mean_logprob(model, tokenizer, pair["negative"])
            pos_lps.append(pos_lp)
            neg_lps.append(neg_lp)

        # Confidence gap: positive means model prefers positive text
        gap = float(np.mean(pos_lps) - np.mean(neg_lps))
        scores[bname] = gap

    return scores


def _mlx_mean_logprob(
    model, tokenizer, text: str, max_length: int = 256,
) -> float:
    """MLX equivalent of get_mean_logprob() from behaviors/metrics.py.

    Computes mean per-token log probability under the model
    (teacher-forced, no gradients).
    """
    import mlx.core as mx
    import mlx.nn as nn

    tokens = tokenizer.encode(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    if len(tokens) < 2:
        return 0.0

    input_ids = mx.array(tokens)[None, :]
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    logits = model(inputs)

    # Per-token cross-entropy (mean = mean negative log prob)
    ce = nn.losses.cross_entropy(logits, targets).mean()
    mx.eval(ce)

    val = -float(ce)
    return val if np.isfinite(val) else 0.0


# ── MLX ↔ PyTorch Conversion ─────────────────────────────────────────

def mlx_to_pytorch_audit(
    mlx_model, model_name: str, tokenizer_name: str = None,
) -> dict:
    """Convert MLX model to PyTorch, run full audit(), return scores.

    Saves MLX weights as safetensors, loads into a fresh PyTorch HF model,
    then runs the standard audit() pipeline.

    Args:
        mlx_model: Fused MLX model.
        model_name: HuggingFace model ID (for loading PyTorch architecture).
        tokenizer_name: Optional separate tokenizer name.

    Returns:
        Dict mapping behavior name → ρ score.
    """
    import tempfile
    import mlx.core as mx
    from mlx.utils import tree_flatten

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file as load_safetensors

    from rho_eval.audit import audit

    tmp_dir = Path(tempfile.mkdtemp())
    weights_path = tmp_dir / "model.safetensors"

    # Save MLX weights
    weights = dict(tree_flatten(mlx_model.parameters()))
    mx.save_safetensors(str(weights_path), weights)

    # Load PyTorch model
    torch_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
    if torch_tokenizer.pad_token is None:
        torch_tokenizer.pad_token = torch_tokenizer.eos_token

    torch_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    )

    # Load fused weights
    state_dict = load_safetensors(str(weights_path))
    torch_model.load_state_dict(state_dict, strict=False)
    torch_model.eval()

    # Run audit
    report = audit(
        model=torch_model, tokenizer=torch_tokenizer,
        behaviors="all", device="cpu",
    )

    scores = {
        bname: r.rho for bname, r in report.behaviors.items()
    }

    # Cleanup
    del torch_model, state_dict
    gc.collect()

    # Remove temp files
    weights_path.unlink(missing_ok=True)
    tmp_dir.rmdir()

    return scores


# ── Main Pipeline ─────────────────────────────────────────────────────

def run_sweep(
    model_name: str,
    rho_weights: list[float],
    seeds: list[int],
    sft_size: int = 2000,
    epochs: int = 1,
    lr: float = 2e-4,
    lora_rank: int = 8,
    margin: float = 0.1,
    full_audit: bool = False,
    results_path: Path | None = None,
    verbose: bool = True,
):
    """Run the full rho_weight × seeds sweep using MLX.

    For each (rho_weight, seed):
      1. Restore original model state
      2. Train with mlx_rho_guided_sft(rho_weight=...)
      3. Quick MLX eval (confidence gaps) — always
      4. Optionally: full PyTorch audit (--full-audit)
      5. Save checkpoint after each run
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load as mlx_load

    from rho_eval.alignment.dataset import (
        _load_alpaca_texts, _build_trap_texts,
        BehavioralContrastDataset, CONTRAST_BEHAVIORS,
    )
    from rho_eval.alignment.mlx_trainer import mlx_rho_guided_sft

    seed_tag = "_".join(str(s) for s in seeds)
    results_path = results_path or (
        RESULTS_DIR / f"mlx_rho_sft_sweep_{model_name.replace('/', '_')}_s{seed_tag}.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Loading (MLX): {model_name}")
    print(f"  Sweep:   rho_weight={rho_weights}, seeds={seeds}")
    print(f"{'='*70}\n")

    model, tokenizer = mlx_load(model_name)
    model.eval()

    # ── Baseline quick eval ───────────────────────────────────────────
    print("Evaluating baseline (quick MLX confidence gaps)...")
    baseline_quick = mlx_quick_rho(model, tokenizer, ALL_EVAL_BEHAVIORS)
    print("  Baseline confidence gaps:")
    for bname, gap in baseline_quick.items():
        print(f"    {bname:12s}: {gap:+.4f}")

    # Full PyTorch baseline if requested
    baseline_scores = {}
    if full_audit:
        print("\nRunning full PyTorch audit for baseline...")
        baseline_scores = mlx_to_pytorch_audit(model, model_name)
        print("  Baseline rho:")
        for bname, rho in baseline_scores.items():
            print(f"    {bname:12s}: {rho:.4f}")

    # ── Save initial weights ──────────────────────────────────────────
    print("\nSaving initial weights for restore between runs...")
    initial_path = results_path.parent / "mlx_initial_weights.safetensors"
    initial_weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(initial_path), initial_weights)
    del initial_weights
    gc.collect()
    print(f"  Saved to: {initial_path.name}")

    # ── Prepare SFT texts (raw strings) ───────────────────────────────
    print("\nPreparing SFT texts...")
    trap_ratio = 0.2
    n_traps = int(sft_size * trap_ratio)
    remaining = sft_size - n_traps

    trap_texts = _build_trap_texts(list(CONTRAST_BEHAVIORS), seed=42)
    random.Random(42).shuffle(trap_texts)
    trap_texts = trap_texts[:n_traps]
    print(f"  {len(trap_texts)} behavioral trap texts")

    alpaca_texts = _load_alpaca_texts(remaining, seed=42)
    print(f"  {len(alpaca_texts)} Alpaca instruction texts")

    sft_texts = trap_texts + alpaca_texts
    random.Random(42).shuffle(sft_texts)
    sft_texts = sft_texts[:sft_size]
    print(f"  Total: {len(sft_texts)} SFT texts")

    # ── Sweep ─────────────────────────────────────────────────────────
    all_results = {
        "model": model_name,
        "backend": "mlx",
        "baseline_quick": baseline_quick,
        "baseline_audit": baseline_scores if full_audit else None,
        "config": {
            "sft_size": sft_size,
            "epochs": epochs,
            "lr": lr,
            "lora_rank": lora_rank,
            "margin": margin,
            "rho_weights": rho_weights,
            "seeds": seeds,
            "behaviors": BEHAVIORS,
            "full_audit": full_audit,
        },
        "runs": [],
        "timestamp": datetime.now().isoformat(),
    }

    total_runs = len(rho_weights) * len(seeds)
    run_idx = 0

    for rho_weight in rho_weights:
        for seed in seeds:
            run_idx += 1
            label = f"w={rho_weight:.2f}/s={seed}"
            print(f"\n{'─'*60}")
            print(f"  Run {run_idx}/{total_runs}: rho_weight={rho_weight}, seed={seed}")
            print(f"{'─'*60}")

            # Restore original weights
            model.load_weights(str(initial_path), strict=False)
            mx.eval(model.parameters())

            # Rebuild contrast dataset with this seed
            contrast_seed = BehavioralContrastDataset(
                behaviors=BEHAVIORS, seed=seed,
            )

            t_start = time.time()

            # Train
            try:
                train_result = mlx_rho_guided_sft(
                    model, tokenizer,
                    sft_texts, contrast_seed,
                    rho_weight=rho_weight,
                    epochs=epochs,
                    lr=lr,
                    lora_rank=lora_rank,
                    margin=margin,
                    verbose=verbose,
                )
                model = train_result["merged_model"]
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                all_results["runs"].append({
                    "rho_weight": rho_weight,
                    "seed": seed,
                    "error": str(e),
                })
                _save_checkpoint(all_results, results_path)
                continue

            # Quick MLX eval
            print(f"\n  Quick eval {label}...")
            quick_scores = mlx_quick_rho(model, tokenizer, ALL_EVAL_BEHAVIORS)

            # Optional full PyTorch audit
            audit_scores = {}
            if full_audit:
                print(f"  Full PyTorch audit {label}...")
                try:
                    audit_scores = mlx_to_pytorch_audit(model, model_name)
                except Exception as e:
                    print(f"  AUDIT ERROR: {e}")

            elapsed = time.time() - t_start

            # Record
            run_record = {
                "rho_weight": rho_weight,
                "seed": seed,
                "quick_scores": quick_scores,
                "audit_scores": audit_scores if full_audit else None,
                "quick_deltas": {
                    bname: quick_scores.get(bname, 0) - baseline_quick.get(bname, 0)
                    for bname in ALL_EVAL_BEHAVIORS
                },
                "train_ce_loss": train_result.get("ce_loss", 0),
                "train_rho_loss": train_result.get("rho_loss", 0),
                "train_steps": train_result.get("steps", 0),
                "elapsed_seconds": elapsed,
            }

            if full_audit and audit_scores and baseline_scores:
                run_record["audit_deltas"] = {
                    bname: audit_scores.get(bname, 0) - baseline_scores.get(bname, 0)
                    for bname in ALL_EVAL_BEHAVIORS
                }

            all_results["runs"].append(run_record)

            # Print summary
            print(f"\n  Results for {label} ({elapsed:.1f}s):")
            for bname in ALL_EVAL_BEHAVIORS:
                q_score = quick_scores.get(bname, float("nan"))
                q_delta = run_record["quick_deltas"].get(bname, 0)
                marker = "+" if q_delta > 0.01 else ("-" if q_delta < -0.01 else "=")
                line = f"    {bname:12s}: gap={q_score:+.4f} ({q_delta:+.4f}) {marker}"
                if full_audit and audit_scores:
                    a_score = audit_scores.get(bname, float("nan"))
                    line += f"  | ρ={a_score:.4f}"
                print(line)

            # Save checkpoint
            _save_checkpoint(all_results, results_path)
            print(f"  Checkpoint saved: {results_path.name}")

    # ── Final Summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE (MLX): {total_runs} runs")
    print(f"  Results: {results_path}")
    print(f"{'='*70}")

    # Cleanup temp weights
    initial_path.unlink(missing_ok=True)

    return all_results


def _save_checkpoint(results: dict, path: Path):
    """Atomic save to JSON."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.rename(path)


# ── Analysis (reuses logic from PyTorch version) ─────────────────────

def analyze(results_path: Path):
    """Analyze MLX sweep results and print summary statistics."""
    with open(results_path) as f:
        data = json.load(f)

    runs = [r for r in data["runs"] if "error" not in r]
    if not runs:
        print("No successful runs found.")
        return

    # Use quick_scores for analysis (always available)
    score_key = "quick_scores"
    delta_key = "quick_deltas"
    metric_label = "confidence gap"

    # If full audit was done, prefer those
    if data.get("config", {}).get("full_audit") and runs[0].get("audit_scores"):
        score_key = "audit_scores"
        delta_key = "audit_deltas"
        metric_label = "ρ"

    baseline = data.get("baseline_quick", {})

    from collections import defaultdict
    by_weight = defaultdict(list)
    for r in runs:
        by_weight[r["rho_weight"]].append(r)

    print(f"\nModel: {data['model']} (MLX backend)")
    print(f"Metric: {metric_label}")
    print(f"Runs:  {len(runs)} successful / {len(data['runs'])} total\n")

    # Header
    behaviors = sorted(
        set().union(*(r.get(score_key, {}).keys() for r in runs))
    )
    header = f"{'rho_weight':>10s} {'n':>3s}"
    for beh in behaviors:
        header += f" | {beh:>12s}"
    print(header)
    print("-" * len(header))

    # Baseline
    row = f"{'baseline':>10s} {'':>3s}"
    for beh in behaviors:
        row += f" | {baseline.get(beh, 0):12.4f}"
    print(row)
    print("-" * len(header))

    # Per-weight aggregates
    for weight in sorted(by_weight.keys()):
        weight_runs = by_weight[weight]
        n = len(weight_runs)
        row = f"{weight:10.2f} {n:3d}"
        for beh in behaviors:
            deltas = [
                r.get(delta_key, {}).get(beh, 0) for r in weight_runs
            ]
            mean_delta = np.mean(deltas)
            if abs(mean_delta) > 0.01:
                row += f" | {mean_delta:+11.4f}*"
            else:
                row += f" | {mean_delta:+12.4f}"
        print(row)

    # Best weight per behavior
    print(f"\nBest rho_weight per behavior (by mean delta):")
    for beh in behaviors:
        best_weight = None
        best_delta = -float("inf")
        for weight, weight_runs in by_weight.items():
            if weight == 0.0:
                continue
            deltas = [
                r.get(delta_key, {}).get(beh, 0) for r in weight_runs
            ]
            mean_d = np.mean(deltas)
            if mean_d > best_delta:
                best_delta = mean_d
                best_weight = weight
        if best_weight is not None:
            std_deltas = [
                r.get(delta_key, {}).get(beh, 0) for r in by_weight.get(0.0, [])
            ]
            std_mean = np.mean(std_deltas) if std_deltas else 0
            print(f"  {beh:12s}: w={best_weight:.2f} → {best_delta:+.4f} "
                  f"(vs std SFT: {std_mean:+.4f})")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rho-guided SFT experiment (MLX backend for Apple Silicon)"
    )
    parser.add_argument(
        "--model", default="qwen2.5-0.5b",
        help="Model key or HF name (default: qwen2.5-0.5b)",
    )
    parser.add_argument(
        "--rho-weights", default=None,
        help="Comma-separated rho_weight values (default: 0,0.05,0.1,0.2,0.3,0.5)",
    )
    parser.add_argument(
        "--seeds", default=None,
        help="Comma-separated seeds (default: 42,123,456)",
    )
    parser.add_argument(
        "--sft-size", type=int, default=2000,
        help="Number of SFT examples (default: 2000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Training epochs (default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--margin", type=float, default=0.1,
        help="Contrastive margin (default: 0.1)",
    )
    parser.add_argument(
        "--full-audit", action="store_true",
        help="Run full PyTorch audit after each run (slower but gives true ρ)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Quick validation run (Qwen-0.5B, 1 seed, 2 weights, 200 SFT)",
    )
    parser.add_argument(
        "--analyze", default=None,
        help="Analyze existing results file instead of running",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: auto-generated)",
    )

    args = parser.parse_args()

    # Analysis mode
    if args.analyze:
        analyze(Path(args.analyze))
        return

    # Resolve model name
    model_name = MODELS.get(args.model, args.model)

    # Validation mode
    if args.validate:
        model_name = MODELS.get("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B")
        rho_weights = [0.0, 0.2]
        seeds = [42]
        sft_size = 200
        print("VALIDATION MODE: Qwen-0.5B, 2 weights, 1 seed, 200 SFT examples")
    else:
        rho_weights = (
            [float(w) for w in args.rho_weights.split(",")]
            if args.rho_weights
            else DEFAULT_RHO_WEIGHTS
        )
        seeds = (
            [int(s) for s in args.seeds.split(",")]
            if args.seeds
            else DEFAULT_SEEDS
        )
        sft_size = args.sft_size

    results_path = Path(args.output) if args.output else None

    run_sweep(
        model_name=model_name,
        rho_weights=rho_weights,
        seeds=seeds,
        sft_size=sft_size,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        margin=args.margin,
        full_audit=args.full_audit,
        results_path=results_path,
    )


if __name__ == "__main__":
    main()
