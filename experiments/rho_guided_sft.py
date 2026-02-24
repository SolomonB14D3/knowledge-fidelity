#!/usr/bin/env python3
"""Rho-Guided SFT Experiment.

Compares standard SFT (CE only) vs rho-guided SFT (CE + auxiliary
contrastive loss) across multiple rho_weight values and seeds.

Tests the hypothesis: adding a differentiable contrastive loss derived
from behavioral probe pairs improves rho scores compared to standard SFT.

Usage:
    python experiments/rho_guided_sft.py
    python experiments/rho_guided_sft.py --model Qwen/Qwen2.5-7B-Instruct
    python experiments/rho_guided_sft.py --validate
    python experiments/rho_guided_sft.py --analyze results/alignment/rho_sft_sweep.json
"""

import argparse
import copy
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Configuration ─────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results" / "alignment"

MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}

DEFAULT_RHO_WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
DEFAULT_SEEDS = [42, 123, 456]

BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias"]
ALL_EVAL_BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias", "reasoning"]


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
    device: str = "cpu",
    results_path: Path | None = None,
    verbose: bool = True,
):
    """Run the full rho_weight × seeds sweep for a single model.

    For each (rho_weight, seed):
      1. Restore original model state
      2. Train with rho_guided_sft(rho_weight=...)
      3. Evaluate all 5 behaviors via audit()
      4. Save checkpoint after each run

    Args:
        model_name: HuggingFace model ID.
        rho_weights: List of rho_weight values to sweep.
        seeds: List of random seeds.
        sft_size: Number of SFT examples.
        epochs: Training epochs per run.
        lr: Learning rate.
        lora_rank: LoRA rank.
        margin: Contrastive margin.
        device: Torch device.
        results_path: Path to save JSON results (auto-generated if None).
        verbose: Print progress.
    """
    from rho_eval.audit import audit
    from rho_eval.alignment.dataset import load_sft_dataset, BehavioralContrastDataset
    from rho_eval.alignment.trainer import rho_guided_sft

    results_path = results_path or (
        RESULTS_DIR / f"rho_sft_sweep_{model_name.replace('/', '_')}.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Loading: {model_name}")
    print(f"  Sweep:   rho_weight={rho_weights}, seeds={seeds}")
    print(f"{'='*70}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    # Save original state for restore between runs
    print("Saving original state_dict...")
    original_state = copy.deepcopy(model.state_dict())

    # ── Baseline evaluation (no SFT) ─────────────────────────────────
    print("\nEvaluating baseline (no SFT)...")
    baseline_report = audit(
        model=model, tokenizer=tokenizer,
        behaviors="all", device=device,
    )
    baseline_scores = {
        bname: r.rho for bname, r in baseline_report.behaviors.items()
    }
    print("  Baseline rho:")
    for bname, rho in baseline_scores.items():
        print(f"    {bname:12s}: {rho:.4f}")

    # ── Prepare datasets (once per model) ─────────────────────────────
    print("\nPreparing datasets...")
    sft_data = load_sft_dataset(
        tokenizer, n=sft_size,
        include_traps=True, seed=42,
    )
    contrast_data = BehavioralContrastDataset(
        behaviors=BEHAVIORS, seed=42,
    )

    # ── Sweep ─────────────────────────────────────────────────────────
    all_results = {
        "model": model_name,
        "baseline": baseline_scores,
        "config": {
            "sft_size": sft_size,
            "epochs": epochs,
            "lr": lr,
            "lora_rank": lora_rank,
            "margin": margin,
            "rho_weights": rho_weights,
            "seeds": seeds,
            "behaviors": BEHAVIORS,
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

            # Restore original state
            model.load_state_dict(original_state)
            model.to(device)

            # Rebuild datasets with this seed for reproducibility
            contrast_seed = BehavioralContrastDataset(
                behaviors=BEHAVIORS, seed=seed,
            )

            t_start = time.time()

            # Train
            try:
                train_result = rho_guided_sft(
                    model, tokenizer,
                    sft_data, contrast_seed,
                    rho_weight=rho_weight,
                    epochs=epochs,
                    lr=lr,
                    lora_rank=lora_rank,
                    margin=margin,
                    device=device,
                    verbose=verbose,
                )
                model = train_result["merged_model"]
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results["runs"].append({
                    "rho_weight": rho_weight,
                    "seed": seed,
                    "error": str(e),
                })
                _save_checkpoint(all_results, results_path)
                continue

            # Evaluate
            print(f"\n  Evaluating {label}...")
            try:
                report = audit(
                    model=model, tokenizer=tokenizer,
                    behaviors="all", device=device,
                )
                eval_scores = {
                    bname: r.rho for bname, r in report.behaviors.items()
                }
            except Exception as e:
                print(f"  EVAL ERROR: {e}")
                eval_scores = {}

            elapsed = time.time() - t_start

            # Record
            run_record = {
                "rho_weight": rho_weight,
                "seed": seed,
                "scores": eval_scores,
                "deltas": {
                    bname: eval_scores.get(bname, 0) - baseline_scores.get(bname, 0)
                    for bname in ALL_EVAL_BEHAVIORS
                },
                "train_ce_loss": train_result.get("ce_loss", 0),
                "train_rho_loss": train_result.get("rho_loss", 0),
                "train_steps": train_result.get("steps", 0),
                "elapsed_seconds": elapsed,
            }
            all_results["runs"].append(run_record)

            # Print summary
            print(f"\n  Results for {label}:")
            for bname in ALL_EVAL_BEHAVIORS:
                score = eval_scores.get(bname, float("nan"))
                delta = run_record["deltas"].get(bname, 0)
                marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else "=")
                print(f"    {bname:12s}: {score:.4f} ({delta:+.4f}) {marker}")

            # Save checkpoint after each run
            _save_checkpoint(all_results, results_path)
            print(f"  Checkpoint saved: {results_path.name}")

    # ── Final Summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE: {total_runs} runs")
    print(f"  Results: {results_path}")
    print(f"{'='*70}")

    return all_results


def _save_checkpoint(results: dict, path: Path):
    """Atomic save to JSON."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.rename(path)


# ── Analysis ──────────────────────────────────────────────────────────

def analyze(results_path: Path):
    """Analyze sweep results and print summary statistics."""
    with open(results_path) as f:
        data = json.load(f)

    baseline = data["baseline"]
    runs = [r for r in data["runs"] if "error" not in r]

    if not runs:
        print("No successful runs found.")
        return

    # Group by rho_weight
    from collections import defaultdict
    by_weight = defaultdict(list)
    for r in runs:
        by_weight[r["rho_weight"]].append(r)

    print(f"\nModel: {data['model']}")
    print(f"Runs:  {len(runs)} successful / {len(data['runs'])} total\n")

    # Header
    behaviors = sorted(set().union(*(r["scores"].keys() for r in runs)))
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
            deltas = [r["deltas"].get(beh, 0) for r in weight_runs]
            mean_delta = np.mean(deltas)
            row += f" | {mean_delta:+11.4f}*" if abs(mean_delta) > 0.01 else f" | {mean_delta:+12.4f}"
        print(row)

    # Best weight per behavior
    print(f"\nBest rho_weight per behavior (by mean delta):")
    for beh in behaviors:
        best_weight = None
        best_delta = -float("inf")
        for weight, weight_runs in by_weight.items():
            if weight == 0.0:
                continue  # skip standard SFT
            deltas = [r["deltas"].get(beh, 0) for r in weight_runs]
            mean_d = np.mean(deltas)
            if mean_d > best_delta:
                best_delta = mean_d
                best_weight = weight
        if best_weight is not None:
            std_deltas = [r["deltas"].get(beh, 0) for r in by_weight.get(0.0, [])]
            std_mean = np.mean(std_deltas) if std_deltas else 0
            print(f"  {beh:12s}: w={best_weight:.2f} → {best_delta:+.4f} "
                  f"(vs std SFT: {std_mean:+.4f})")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rho-guided SFT experiment: sweep rho_weight × seeds"
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
        "--device", default=None,
        help="Torch device (default: auto-detect)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Quick validation run (Qwen-0.5B, 1 seed, 2 weights)",
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

    # Resolve device
    if args.device:
        device = args.device
    else:
        from rho_eval.utils import get_device
        device = str(get_device())

    # Validation mode: minimal config for fast testing
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
        device=device,
        results_path=results_path,
    )


if __name__ == "__main__":
    main()
