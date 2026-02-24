#!/usr/bin/env python3
"""Phase 3: Calibration evaluation for rho-guided SFT models (MLX backend).

Computes Expected Calibration Error (ECE) and Brier score per behavior
on baseline vs rho-guided models. Answers: "Does rho-guidance improve
not just accuracy but also calibration?"

A model is well-calibrated when its confidence matches its accuracy:
  - When it's 80% sure positive > negative, it should be right 80% of the time.
  - Low ECE = confidence tracks accuracy (trustworthy predictions).
  - Low Brier = both accurate and well-calibrated (proper scoring rule).

Modes:
  1. Standalone:  Score a model directly:
       python experiments/calibration_eval_mlx.py --model qwen2.5-7b

  2. Sweep-integrated: Score all rho_weight conditions from a sweep,
     re-training each one on-the-fly for calibration eval:
       python experiments/calibration_eval_mlx.py --sweep results/alignment/mlx_rho_sft_sweep_*.json

  3. Quick:  Use --behaviors factual,toxicity for fast subset.

Usage:
    # Score baseline model
    python experiments/calibration_eval_mlx.py --model qwen2.5-7b

    # Score after rho-guided SFT with specific rho_weight
    python experiments/calibration_eval_mlx.py --model qwen2.5-7b \\
        --rho-weights 0.0,0.2 --seeds 42,123

    # Analyze saved sweep results + run calibration
    python experiments/calibration_eval_mlx.py --analyze results/alignment/calibration_*.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results" / "alignment"

MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
}

BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias"]


# ── Calibration Evaluation ───────────────────────────────────────────

def evaluate_calibration(
    model,
    tokenizer,
    behaviors: list[str] | None = None,
    seed: int = 42,
    n_bins: int = 10,
    verbose: bool = True,
) -> dict:
    """Run calibration metrics on a loaded model.

    Returns per-behavior ECE, Brier, accuracy, and confidence stats.
    """
    from rho_eval.behaviors.metrics import calibration_metrics

    if verbose:
        print(f"  Computing calibration metrics (seed={seed})...")

    t0 = time.time()
    results = calibration_metrics(
        model, tokenizer,
        behaviors=behaviors or BEHAVIORS,
        seed=seed,
        n_bins=n_bins,
    )
    elapsed = time.time() - t0

    if verbose:
        print(f"  Calibration eval done in {elapsed:.1f}s")
        print(f"  {'Behavior':12s}  {'ECE':>8s}  {'Brier':>8s}  {'Acc':>6s}  {'Conf':>6s}  {'Gap':>8s}")
        print(f"  {'─'*58}")
        for bname, bdata in results.items():
            if "error" in bdata:
                print(f"  {bname:12s}  ERROR: {bdata['error']}")
            else:
                print(f"  {bname:12s}  {bdata['ece']:8.4f}  {bdata['brier']:8.4f}  "
                      f"{bdata['accuracy']:5.1%}  {bdata['mean_confidence']:5.1%}  "
                      f"{bdata['mean_gap']:+8.4f}")

    return results


# ── Full Sweep with Calibration ──────────────────────────────────────

def run_calibration_sweep(
    model_name: str,
    rho_weights: list[float],
    seeds: list[int],
    sft_size: int = 1000,
    epochs: int = 1,
    lr: float = 2e-4,
    lora_rank: int = 8,
    margin: float = 0.1,
    behaviors: list[str] | None = None,
    n_bins: int = 10,
    results_path: Path | None = None,
    verbose: bool = True,
):
    """Train with each rho_weight × seed, then measure calibration.

    This is the full Phase 3 pipeline:
      1. Load model
      2. Score baseline calibration
      3. For each (rho_weight, seed):
         a. Restore weights
         b. Train with mlx_rho_guided_sft
         c. Measure ECE + Brier per behavior
      4. Save results JSON
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load as mlx_load

    from rho_eval.alignment.dataset import (
        _load_alpaca_texts, _build_trap_texts,
        BehavioralContrastDataset, CONTRAST_BEHAVIORS,
    )
    from rho_eval.alignment.mlx_trainer import mlx_rho_guided_sft

    behaviors = behaviors or BEHAVIORS
    results_path = results_path or (
        RESULTS_DIR / f"calibration_{model_name.replace('/', '_')}.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Phase 3: Calibration Eval (MLX): {model_name}")
    print(f"  rho_weights={rho_weights}, seeds={seeds}")
    print(f"  behaviors={behaviors}")
    print(f"{'='*70}\n")

    model, tokenizer = mlx_load(model_name)
    model.eval()

    # ── Baseline calibration ──────────────────────────────────────
    print("Evaluating baseline calibration...")
    baseline_calib = evaluate_calibration(
        model, tokenizer, behaviors=behaviors,
        seed=42, n_bins=n_bins, verbose=verbose,
    )

    # Strip details for JSON (too large)
    baseline_summary = _strip_details(baseline_calib)

    # Save initial weights
    print("\nSaving initial weights...")
    initial_path = results_path.parent / "calib_initial_weights.safetensors"
    initial_weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(initial_path), initial_weights)
    del initial_weights
    gc.collect()

    # Prepare SFT texts
    print("\nPreparing SFT texts...")
    trap_ratio = 0.2
    n_traps = int(sft_size * trap_ratio)
    remaining = sft_size - n_traps

    trap_texts = _build_trap_texts(list(CONTRAST_BEHAVIORS), seed=42)
    random.Random(42).shuffle(trap_texts)
    trap_texts = trap_texts[:n_traps]

    alpaca_texts = _load_alpaca_texts(remaining, seed=42)
    sft_texts = trap_texts + alpaca_texts
    random.Random(42).shuffle(sft_texts)
    sft_texts = sft_texts[:sft_size]
    print(f"  {len(sft_texts)} SFT texts")

    # Results container
    all_results = {
        "model": model_name,
        "backend": "mlx",
        "experiment": "calibration_eval",
        "baseline_calibration": baseline_summary,
        "config": {
            "rho_weights": rho_weights,
            "sft_size": sft_size,
            "epochs": epochs,
            "lr": lr,
            "lora_rank": lora_rank,
            "margin": margin,
            "seeds": seeds,
            "behaviors": behaviors,
            "n_bins": n_bins,
        },
        "runs": [],
        "timestamp": datetime.now().isoformat(),
    }

    total_runs = len(rho_weights) * len(seeds)
    run_idx = 0

    for rho_weight in rho_weights:
        for seed in seeds:
            run_idx += 1
            label = f"λ={rho_weight}/s={seed}"
            print(f"\n{'─'*60}")
            print(f"  Run {run_idx}/{total_runs}: rho_weight={rho_weight}, seed={seed}")
            print(f"{'─'*60}")

            # Restore original weights
            model.load_weights(str(initial_path), strict=False)
            mx.eval(model.parameters())

            # Build contrast dataset
            contrast_ds = BehavioralContrastDataset(
                behaviors=BEHAVIORS, seed=seed,
            )

            t_start = time.time()

            try:
                train_result = mlx_rho_guided_sft(
                    model, tokenizer,
                    sft_texts, contrast_ds,
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

            # Calibration eval
            print(f"\n  Calibration eval {label}...")
            calib_result = evaluate_calibration(
                model, tokenizer, behaviors=behaviors,
                seed=seed, n_bins=n_bins, verbose=verbose,
            )

            elapsed = time.time() - t_start

            # Record
            run_record = {
                "rho_weight": rho_weight,
                "seed": seed,
                "calibration": _strip_details(calib_result),
                "train_ce_loss": train_result.get("ce_loss", 0),
                "train_rho_loss": train_result.get("rho_loss", 0),
                "train_steps": train_result.get("steps", 0),
                "elapsed_seconds": elapsed,
            }

            # Compute deltas from baseline
            for bname in behaviors:
                if bname in calib_result and bname in baseline_calib:
                    base = baseline_calib[bname]
                    post = calib_result[bname]
                    if "error" not in base and "error" not in post:
                        run_record.setdefault("deltas", {})[bname] = {
                            "ece_delta": post["ece"] - base["ece"],
                            "brier_delta": post["brier"] - base["brier"],
                            "acc_delta": post["accuracy"] - base["accuracy"],
                        }

            all_results["runs"].append(run_record)

            # Print summary
            print(f"\n  Calibration change for {label}:")
            for bname in behaviors:
                if "deltas" in run_record and bname in run_record["deltas"]:
                    d = run_record["deltas"][bname]
                    ece_dir = "↓" if d["ece_delta"] < -0.001 else ("↑" if d["ece_delta"] > 0.001 else "=")
                    brier_dir = "↓" if d["brier_delta"] < -0.001 else ("↑" if d["brier_delta"] > 0.001 else "=")
                    acc_dir = "↑" if d["acc_delta"] > 0.01 else ("↓" if d["acc_delta"] < -0.01 else "=")
                    print(f"    {bname:12s}: ECE {d['ece_delta']:+.4f}{ece_dir}  "
                          f"Brier {d['brier_delta']:+.4f}{brier_dir}  "
                          f"Acc {d['acc_delta']:+.1%}{acc_dir}")

            _save_checkpoint(all_results, results_path)
            print(f"  Checkpoint saved: {results_path.name}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CALIBRATION EVAL COMPLETE: {total_runs} runs")
    print(f"  Results: {results_path}")
    print(f"{'='*70}")

    # Cleanup
    initial_path.unlink(missing_ok=True)

    return all_results


# ── Analysis ─────────────────────────────────────────────────────────

def analyze_calibration(results_path: str | Path):
    """Print analysis tables from saved calibration results JSON."""
    path = Path(results_path)
    with open(path) as f:
        data = json.load(f)

    model = data["model"]
    print(f"\n{'='*70}")
    print(f"  Calibration Analysis: {model}")
    print(f"{'='*70}\n")

    # Baseline
    print("Baseline calibration:")
    print(f"  {'Behavior':12s}  {'ECE':>8s}  {'Brier':>8s}  {'Acc':>6s}  {'Conf':>6s}")
    print(f"  {'─'*46}")
    baseline = data.get("baseline_calibration", {})
    for bname, bdata in baseline.items():
        if "error" not in bdata:
            print(f"  {bname:12s}  {bdata['ece']:8.4f}  {bdata['brier']:8.4f}  "
                  f"{bdata['accuracy']:5.1%}  {bdata['mean_confidence']:5.1%}")

    # Per-condition tables
    runs = data.get("runs", [])
    if not runs:
        print("\nNo runs found.")
        return

    # Group by rho_weight
    from collections import defaultdict
    by_rho = defaultdict(list)
    for run in runs:
        if "error" not in run:
            by_rho[run["rho_weight"]].append(run)

    behaviors = data.get("config", {}).get("behaviors", BEHAVIORS)

    for bname in behaviors:
        print(f"\n{'─'*60}")
        print(f"  {bname.upper()} — Calibration vs rho_weight")
        print(f"{'─'*60}")
        print(f"  {'λ_ρ':>6s}  {'ECE':>10s}  {'Brier':>10s}  {'Accuracy':>10s}  {'Conf':>10s}")
        print(f"  {'─'*52}")

        # Baseline row
        if bname in baseline and "error" not in baseline[bname]:
            b = baseline[bname]
            print(f"  {'base':>6s}  {b['ece']:10.4f}  {b['brier']:10.4f}  "
                  f"{b['accuracy']:9.1%}  {b['mean_confidence']:9.1%}")

        for rho_w in sorted(by_rho.keys()):
            runs_at_rho = by_rho[rho_w]
            eces, briers, accs, confs = [], [], [], []
            for run in runs_at_rho:
                calib = run.get("calibration", {}).get(bname, {})
                if "error" not in calib and calib:
                    eces.append(calib["ece"])
                    briers.append(calib["brier"])
                    accs.append(calib["accuracy"])
                    confs.append(calib["mean_confidence"])

            if eces:
                ece_mean, ece_std = np.mean(eces), np.std(eces)
                brier_mean, brier_std = np.mean(briers), np.std(briers)
                acc_mean = np.mean(accs)
                conf_mean = np.mean(confs)

                if len(eces) > 1:
                    print(f"  {rho_w:6.2f}  {ece_mean:6.4f}±{ece_std:.3f}  "
                          f"{brier_mean:6.4f}±{brier_std:.3f}  "
                          f"{acc_mean:9.1%}  {conf_mean:9.1%}")
                else:
                    print(f"  {rho_w:6.2f}  {ece_mean:10.4f}  {brier_mean:10.4f}  "
                          f"{acc_mean:9.1%}  {conf_mean:9.1%}")

    # Significance test: ECE reduction from baseline
    print(f"\n{'─'*60}")
    print(f"  ECE Improvement Tests (vs baseline)")
    print(f"{'─'*60}")

    for bname in behaviors:
        if bname not in baseline or "error" in baseline.get(bname, {}):
            continue
        base_ece = baseline[bname]["ece"]

        for rho_w in sorted(by_rho.keys()):
            if rho_w == 0.0:
                continue
            runs_at_rho = by_rho[rho_w]
            eces = []
            for run in runs_at_rho:
                calib = run.get("calibration", {}).get(bname, {})
                if "error" not in calib and calib:
                    eces.append(calib["ece"])

            if len(eces) >= 2:
                from scipy import stats
                t_stat, p_val = stats.ttest_1samp(eces, base_ece)
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
                direction = "better" if np.mean(eces) < base_ece else "worse"
                print(f"  {bname:12s} λ={rho_w}: ECE {np.mean(eces):.4f} vs {base_ece:.4f} "
                      f"({direction}) p={p_val:.4f} {sig}")


def _strip_details(calib_result: dict) -> dict:
    """Strip per-probe details from calibration results for JSON storage."""
    stripped = {}
    for bname, bdata in calib_result.items():
        if isinstance(bdata, dict):
            stripped[bname] = {k: v for k, v in bdata.items()
                               if k not in ("details", "ece_bins")}
        else:
            stripped[bname] = bdata
    return stripped


def _save_checkpoint(results: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.rename(path)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Calibration evaluation")
    parser.add_argument("--model", default="qwen2.5-7b",
                        help="Model key or HF name")
    parser.add_argument("--rho-weights", default="0.0,0.1,0.2,0.5",
                        help="Comma-separated rho weights to evaluate")
    parser.add_argument("--seeds", default="42,123",
                        help="Comma-separated seeds")
    parser.add_argument("--sft-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--behaviors", default=",".join(BEHAVIORS),
                        help="Comma-separated behaviors to evaluate")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Number of ECE bins")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only evaluate baseline model (no training)")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Path to results JSON for analysis (skip training)")

    args = parser.parse_args()

    if args.analyze:
        analyze_calibration(args.analyze)
        return

    model_name = MODELS.get(args.model, args.model)
    rho_weights = [float(w) for w in args.rho_weights.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    behaviors = [b.strip() for b in args.behaviors.split(",")]

    if args.baseline_only:
        from mlx_lm import load as mlx_load
        model, tokenizer = mlx_load(model_name)
        model.eval()
        evaluate_calibration(model, tokenizer, behaviors=behaviors, verbose=True)
    else:
        run_calibration_sweep(
            model_name=model_name,
            rho_weights=rho_weights,
            seeds=seeds,
            sft_size=args.sft_size,
            epochs=args.epochs,
            lr=args.lr,
            lora_rank=args.lora_rank,
            margin=args.margin,
            behaviors=behaviors,
            n_bins=args.n_bins,
        )


if __name__ == "__main__":
    main()
