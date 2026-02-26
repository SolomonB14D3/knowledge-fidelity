#!/usr/bin/env python3
"""External Benchmark Evaluation: Full behavioral audit + TruthfulQA MC2.

Trains rho-guided SFT models at different lambda values, then evaluates
each checkpoint on ALL 8 behavioral dimensions plus TruthfulQA MC2.
Reports both internal rho metrics AND external accuracy metrics to show
that internal improvements translate to real task performance.

Key metrics extracted:
  - reasoning: adversarial_accuracy, clean_accuracy, accuracy_drop
  - bias: accuracy on BBQ (= rho), bias_rate
  - factual: rho (Spearman correlation), retention
  - toxicity: AUC, confidence_gap
  - sycophancy: truthful_rate (= rho), sycophancy_rate
  - refusal: AUC, confidence_gap
  - deception: AUC, confidence_gap
  - overrefusal: answer_rate (= rho), refusal_rate
  - TruthfulQA: MC2 score, MC1 accuracy

Usage:
    python experiments/external_eval_mlx.py --model qwen2.5-7b \\
        --rho-weights 0.0,0.2,0.5 --seeds 42,123,456

    # Quick test with subset
    python experiments/external_eval_mlx.py --model qwen2.5-7b \\
        --rho-weights 0.0,0.5 --seeds 42 --n-tqa 100
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "alignment"

MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
}


# ── Full Audit with Accuracy Extraction ──────────────────────────────

def mlx_to_pytorch_full_audit(
    mlx_model, model_name: str, tokenizer_name: str = None,
) -> dict:
    """Convert MLX model to PyTorch, run full audit(), return detailed results.

    Unlike mlx_to_pytorch_audit() which only returns {behavior: rho},
    this returns the full BehaviorResult including metadata (accuracy,
    confidence gaps, category breakdowns, etc).

    Returns:
        Dict with:
          - "scores": {behavior: rho}
          - "accuracy": {behavior: {metric: value}} -- external accuracy metrics
          - "retention": {behavior: retention}
          - "metadata": {behavior: metadata_dict}
          - "elapsed": total seconds
    """
    import tempfile
    import mlx.core as mx
    from mlx.utils import tree_flatten

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file as load_safetensors

    from rho_eval.audit import audit

    t0 = time.time()
    tmp_dir = Path(tempfile.mkdtemp())
    weights_path = tmp_dir / "model.safetensors"

    # Save MLX weights
    weights = dict(tree_flatten(mlx_model.parameters()))
    mx.save_safetensors(str(weights_path), weights)
    del weights
    gc.collect()

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

    # Run full audit on all 8 behaviors
    report = audit(
        model=torch_model, tokenizer=torch_tokenizer,
        behaviors="all", device="cpu",
    )

    # Extract detailed results
    scores = {}
    accuracy = {}
    retention = {}
    metadata = {}

    for bname, result in report.behaviors.items():
        scores[bname] = result.rho
        retention[bname] = result.retention

        # Make metadata JSON-serializable
        meta = {}
        for k, v in result.metadata.items():
            if isinstance(v, (float, int, str, bool)):
                meta[k] = v
            elif isinstance(v, dict):
                meta[k] = {
                    kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                    for kk, vv in v.items()
                }
            elif isinstance(v, np.floating):
                meta[k] = float(v)
        metadata[bname] = meta

        # Extract key accuracy metrics per behavior
        acc = {}
        if bname == "reasoning":
            acc["adversarial_accuracy"] = result.metadata.get("adversarial_accuracy", 0.0)
            acc["clean_accuracy"] = result.metadata.get("clean_accuracy", 0.0)
            acc["accuracy_drop"] = result.metadata.get("accuracy_drop", 0.0)
        elif bname == "bias":
            acc["bbq_accuracy"] = result.rho  # rho = accuracy for bias
            acc["bias_rate"] = result.metadata.get("bias_rate", 0.0)
        elif bname == "sycophancy":
            acc["truthful_rate"] = result.rho  # rho = non-sycophantic rate
            acc["sycophancy_rate"] = result.metadata.get("sycophancy_rate", 0.0)
        elif bname == "overrefusal":
            acc["answer_rate"] = result.rho  # rho = answer rate
            acc["refusal_rate"] = result.metadata.get("refusal_rate", 0.0)
        elif bname == "toxicity":
            acc["auc"] = result.rho
            acc["confidence_gap"] = result.metadata.get("confidence_gap", 0.0)
        elif bname == "factual":
            acc["retention"] = result.retention
        elif bname == "refusal":
            acc["auc"] = result.rho
            acc["confidence_gap"] = result.metadata.get("confidence_gap", 0.0)
        elif bname == "deception":
            acc["auc"] = result.rho
            acc["confidence_gap"] = result.metadata.get("confidence_gap", 0.0)
        accuracy[bname] = acc

    # Cleanup
    del torch_model, state_dict
    gc.collect()
    weights_path.unlink(missing_ok=True)
    tmp_dir.rmdir()

    elapsed = time.time() - t0

    return {
        "scores": scores,
        "accuracy": accuracy,
        "retention": retention,
        "metadata": metadata,
        "elapsed": elapsed,
    }


# ── Main Sweep Pipeline ──────────────────────────────────────────────

def run_external_eval_sweep(
    model_name: str,
    rho_weights: list[float] = [0.0, 0.2, 0.5],
    seeds: list[int] = [42, 123, 456],
    n_tqa: int | None = None,
    sft_size: int = 1000,
    epochs: int = 1,
    lr: float = 2e-4,
    lora_rank: int = 8,
    margin: float = 0.1,
    results_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Train with each rho_weight x seed, then evaluate on all benchmarks.

    For each (rho_weight, seed):
      1. Train with mlx_rho_guided_sft()
      2. Run full audit (8 behaviors) via PyTorch conversion
      3. Run TruthfulQA MC2
      4. Extract accuracy metrics and rho scores
      5. Save checkpoint results incrementally

    Returns:
        Dict with baseline, per-run results, and summary statistics.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load as mlx_load

    from rho_eval.alignment.dataset import (
        _load_alpaca_texts, _build_trap_texts,
        BehavioralContrastDataset, CONTRAST_BEHAVIORS,
    )
    from rho_eval.alignment.mlx_trainer import mlx_rho_guided_sft

    # Import TruthfulQA scorer from sibling experiment
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from truthfulqa_mc2_mlx import load_truthfulqa_mc2, score_mc2

    results_path = results_path or (
        RESULTS_DIR / f"external_eval_{model_name.replace('/', '_')}.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  External Benchmark Evaluation (MLX): {model_name}")
    print(f"  rho_weights={rho_weights}, seeds={seeds}")
    print(f"  Behaviors: all 8 + TruthfulQA MC2")
    print(f"{'='*70}\n")

    # Load model
    model, tokenizer = mlx_load(model_name)
    model.eval()

    # Load TruthfulQA
    print("Loading TruthfulQA MC2...")
    tqa_questions = load_truthfulqa_mc2(n=n_tqa, seed=42)

    # ── Baseline evaluation ───────────────────────────────────────────

    print("\n  Evaluating baseline (all 8 behaviors)...")
    baseline_audit = mlx_to_pytorch_full_audit(model, model_name)

    print("  Baseline rho scores:")
    for bname in sorted(baseline_audit["scores"].keys()):
        rho = baseline_audit["scores"][bname]
        print(f"    {bname:12s}: {rho:.4f}")

    print("\n  Evaluating baseline TruthfulQA MC2...")
    baseline_tqa = score_mc2(model, tokenizer, tqa_questions, verbose=verbose)
    baseline_tqa_summary = {k: v for k, v in baseline_tqa.items() if k != "details"}

    # ── Save initial weights ──────────────────────────────────────────

    print("\nSaving initial weights...")
    initial_path = results_path.parent / "exteval_initial_weights.safetensors"
    initial_weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(initial_path), initial_weights)
    del initial_weights
    gc.collect()

    # ── Prepare SFT data ──────────────────────────────────────────────

    print("\nPreparing SFT data...")
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

    # ── Training behaviors (same as standard sweep) ───────────────────
    TRAIN_BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias", "refusal", "deception"]

    # ── Results container ─────────────────────────────────────────────

    all_results = {
        "model": model_name,
        "experiment": "external_eval",
        "baseline": {
            "audit": baseline_audit,
            "truthfulqa": baseline_tqa_summary,
        },
        "config": {
            "rho_weights": rho_weights,
            "seeds": seeds,
            "sft_size": sft_size,
            "epochs": epochs,
            "lr": lr,
            "lora_rank": lora_rank,
            "margin": margin,
            "n_tqa_questions": n_tqa or len(tqa_questions),
        },
        "runs": [],
        "timestamp": datetime.now().isoformat(),
    }

    # ── Main sweep loop ───────────────────────────────────────────────

    total_runs = len(rho_weights) * len(seeds)
    run_idx = 0

    for rho_weight in rho_weights:
        for seed in seeds:
            run_idx += 1
            label = f"lambda={rho_weight}/seed={seed}"
            print(f"\n{'='*60}")
            print(f"  Run {run_idx}/{total_runs}: {label}")
            print(f"{'='*60}")

            # Restore initial weights
            model.load_weights(str(initial_path), strict=False)
            mx.eval(model.parameters())

            # Build contrast dataset with this seed
            contrast_ds = BehavioralContrastDataset(
                behaviors=TRAIN_BEHAVIORS, seed=seed,
            )

            t_start = time.time()

            # ── Train ─────────────────────────────────────────────────
            try:
                train_result = mlx_rho_guided_sft(
                    model, tokenizer,
                    sft_texts, contrast_ds,
                    rho_weight=rho_weight,
                    epochs=epochs, lr=lr,
                    lora_rank=lora_rank, margin=margin,
                    verbose=verbose,
                )
                model = train_result["merged_model"]
                train_ce = train_result.get("ce_loss", 0.0)
                train_rho = train_result.get("rho_loss", 0.0)
                train_steps = train_result.get("steps", 0)
            except Exception as e:
                import traceback
                print(f"  TRAIN ERROR: {e}")
                traceback.print_exc()
                all_results["runs"].append({
                    "rho_weight": rho_weight,
                    "seed": seed,
                    "error": str(e),
                })
                _save_checkpoint(all_results, results_path)
                continue

            # ── Full audit ────────────────────────────────────────────
            print(f"\n  Running full audit (8 behaviors)...")
            try:
                audit_result = mlx_to_pytorch_full_audit(model, model_name)
            except Exception as e:
                print(f"  AUDIT ERROR: {e}")
                audit_result = {"scores": {}, "accuracy": {}, "retention": {}, "metadata": {}, "elapsed": 0}

            # ── TruthfulQA MC2 ────────────────────────────────────────
            print(f"\n  Running TruthfulQA MC2...")
            try:
                tqa_result = score_mc2(model, tokenizer, tqa_questions, verbose=verbose)
                tqa_summary = {k: v for k, v in tqa_result.items() if k != "details"}
            except Exception as e:
                print(f"  TQA ERROR: {e}")
                tqa_summary = {"mc2_score": 0.0, "mc1_accuracy": 0.0, "error": str(e)}

            elapsed = time.time() - t_start

            # ── Compute deltas ────────────────────────────────────────
            score_deltas = {}
            for bname, rho in audit_result["scores"].items():
                baseline_rho = baseline_audit["scores"].get(bname, 0.0)
                score_deltas[bname] = rho - baseline_rho

            tqa_delta = tqa_summary.get("mc2_score", 0.0) - baseline_tqa["mc2_score"]

            # ── Record ────────────────────────────────────────────────
            run_record = {
                "rho_weight": rho_weight,
                "seed": seed,
                "scores": audit_result["scores"],
                "score_deltas": score_deltas,
                "accuracy": audit_result["accuracy"],
                "retention": audit_result["retention"],
                "truthfulqa": tqa_summary,
                "truthfulqa_delta": tqa_delta,
                "train_ce_loss": float(train_ce) if train_ce else 0.0,
                "train_rho_loss": float(train_rho) if train_rho else 0.0,
                "train_steps": train_steps,
                "elapsed_seconds": elapsed,
            }
            all_results["runs"].append(run_record)

            # Print summary
            print(f"\n  Results for {label}:")
            print(f"    {'Behavior':12s} {'rho':>8s} {'delta':>8s}  Key Accuracy Metric")
            print(f"    {'─'*50}")
            for bname in sorted(audit_result["scores"].keys()):
                rho = audit_result["scores"][bname]
                delta = score_deltas.get(bname, 0.0)
                acc_str = ""
                acc = audit_result["accuracy"].get(bname, {})
                if bname == "reasoning":
                    acc_str = f"adv_acc={acc.get('adversarial_accuracy', 0):.1%}"
                elif bname == "bias":
                    acc_str = f"bbq_acc={acc.get('bbq_accuracy', 0):.1%}"
                elif bname == "overrefusal":
                    acc_str = f"answer_rate={acc.get('answer_rate', 0):.1%}"
                elif bname == "sycophancy":
                    acc_str = f"truthful={acc.get('truthful_rate', 0):.1%}"
                print(f"    {bname:12s} {rho:8.4f} {delta:+8.4f}  {acc_str}")

            print(f"    {'─'*50}")
            print(f"    TruthfulQA MC2: {tqa_summary.get('mc2_score', 0):.4f} "
                  f"(delta={tqa_delta:+.4f})")
            print(f"    Elapsed: {elapsed:.0f}s")

            _save_checkpoint(all_results, results_path)

    # ── Summary Table ─────────────────────────────────────────────────

    _print_summary(all_results, baseline_audit, baseline_tqa)

    # Cleanup
    initial_path.unlink(missing_ok=True)

    print(f"\n  Results saved: {results_path}")
    return all_results


def _print_summary(all_results: dict, baseline_audit: dict, baseline_tqa: dict):
    """Print a summary table of all runs grouped by rho_weight."""
    print(f"\n{'='*70}")
    print(f"  EXTERNAL EVALUATION SUMMARY")
    print(f"{'='*70}")

    # Group by rho_weight
    by_rho = defaultdict(list)
    for run in all_results["runs"]:
        if "error" not in run:
            by_rho[run["rho_weight"]].append(run)

    # Key behaviors to summarize
    key_behaviors = ["factual", "toxicity", "bias", "reasoning", "deception", "overrefusal"]

    # Rho scores table
    print(f"\n  Rho Scores (mean +/- std across seeds):")
    print(f"  {'lambda':>6s}", end="")
    for bname in key_behaviors:
        print(f"  {bname:>10s}", end="")
    print(f"  {'TQA_MC2':>8s}")
    print(f"  {'─'*80}")

    # Baseline row
    print(f"  {'base':>6s}", end="")
    for bname in key_behaviors:
        rho = baseline_audit["scores"].get(bname, 0)
        print(f"  {rho:10.4f}", end="")
    print(f"  {baseline_tqa['mc2_score']:8.4f}")

    # Per-lambda rows
    for rho_w in sorted(by_rho.keys()):
        runs = by_rho[rho_w]
        print(f"  {rho_w:6.2f}", end="")
        for bname in key_behaviors:
            vals = [r["scores"].get(bname, 0) for r in runs]
            if len(vals) > 1:
                print(f"  {np.mean(vals):5.3f}+{np.std(vals):.3f}", end="")
            else:
                print(f"  {vals[0]:10.4f}", end="")
        tqa_vals = [r["truthfulqa"].get("mc2_score", 0) for r in runs]
        if len(tqa_vals) > 1:
            print(f"  {np.mean(tqa_vals):.3f}+{np.std(tqa_vals):.3f}")
        else:
            print(f"  {tqa_vals[0]:8.4f}")

    # Delta table
    print(f"\n  Delta from Baseline:")
    print(f"  {'lambda':>6s}", end="")
    for bname in key_behaviors:
        print(f"  {bname:>10s}", end="")
    print(f"  {'TQA_MC2':>8s}")
    print(f"  {'─'*80}")

    for rho_w in sorted(by_rho.keys()):
        runs = by_rho[rho_w]
        print(f"  {rho_w:6.2f}", end="")
        for bname in key_behaviors:
            deltas = [r["score_deltas"].get(bname, 0) for r in runs]
            mean_d = np.mean(deltas)
            sign = "+" if mean_d >= 0 else ""
            print(f"  {sign}{mean_d:9.4f}", end="")
        tqa_deltas = [r.get("truthfulqa_delta", 0) for r in runs]
        mean_tqa = np.mean(tqa_deltas)
        sign = "+" if mean_tqa >= 0 else ""
        print(f"  {sign}{mean_tqa:7.4f}")

    # Accuracy metrics table
    print(f"\n  Key Accuracy Metrics:")
    print(f"  {'lambda':>6s}  {'GSM8K_adv':>9s}  {'GSM8K_cln':>9s}  "
          f"{'BBQ_acc':>7s}  {'OverRef':>7s}  {'TQA_MC2':>8s}")
    print(f"  {'─'*60}")

    # Baseline
    base_acc = baseline_audit["accuracy"]
    print(f"  {'base':>6s}  "
          f"{base_acc.get('reasoning', {}).get('adversarial_accuracy', 0):9.1%}  "
          f"{base_acc.get('reasoning', {}).get('clean_accuracy', 0):9.1%}  "
          f"{base_acc.get('bias', {}).get('bbq_accuracy', 0):7.1%}  "
          f"{base_acc.get('overrefusal', {}).get('answer_rate', 0):7.1%}  "
          f"{baseline_tqa['mc2_score']:8.4f}")

    for rho_w in sorted(by_rho.keys()):
        runs = by_rho[rho_w]
        adv_accs = [r["accuracy"].get("reasoning", {}).get("adversarial_accuracy", 0) for r in runs]
        cln_accs = [r["accuracy"].get("reasoning", {}).get("clean_accuracy", 0) for r in runs]
        bbq_accs = [r["accuracy"].get("bias", {}).get("bbq_accuracy", 0) for r in runs]
        ovr_accs = [r["accuracy"].get("overrefusal", {}).get("answer_rate", 0) for r in runs]
        tqa_mc2s = [r["truthfulqa"].get("mc2_score", 0) for r in runs]

        print(f"  {rho_w:6.2f}  "
              f"{np.mean(adv_accs):9.1%}  "
              f"{np.mean(cln_accs):9.1%}  "
              f"{np.mean(bbq_accs):7.1%}  "
              f"{np.mean(ovr_accs):7.1%}  "
              f"{np.mean(tqa_mc2s):8.4f}")


def _save_checkpoint(results: dict, path: Path):
    """Save results incrementally with atomic write."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.rename(path)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="External Benchmark Evaluation: Full audit + TruthfulQA MC2"
    )
    parser.add_argument("--model", default="qwen2.5-7b",
                        help="Model key or HuggingFace ID")
    parser.add_argument("--rho-weights", default="0.0,0.2,0.5",
                        help="Comma-separated rho weights")
    parser.add_argument("--seeds", default="42,123,456",
                        help="Comma-separated random seeds")
    parser.add_argument("--n-tqa", type=int, default=None,
                        help="Subsample N TruthfulQA questions (None = all 817)")
    parser.add_argument("--sft-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--results-path", type=str, default=None,
                        help="Custom output path for results JSON")

    args = parser.parse_args()

    model_name = MODELS.get(args.model, args.model)
    rho_weights = [float(w) for w in args.rho_weights.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    results_path = Path(args.results_path) if args.results_path else None

    run_external_eval_sweep(
        model_name=model_name,
        rho_weights=rho_weights,
        seeds=seeds,
        n_tqa=args.n_tqa,
        sft_size=args.sft_size,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        margin=args.margin,
        results_path=results_path,
    )


if __name__ == "__main__":
    main()
