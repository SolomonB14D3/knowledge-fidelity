#!/usr/bin/env python3
"""Ablation experiments for rho-guided SFT (MLX backend).

Isolates the contribution of each component in the rho-guided SFT pipeline:

Conditions:
  A. "rho-guided"    : CE(SFT) + λ_ρ * contrastive(behavioral_pairs) [the real method]
  B. "contrastive-only": contrastive(behavioral_pairs) only, no SFT CE loss
  C. "shuffled-pairs" : CE(SFT) + λ_ρ * contrastive(SHUFFLED pairs)
  D. "sft-only"       : CE(SFT) only, no contrastive (= rho_weight=0 baseline)

If the ρ-loss is doing real work:
  - A should beat D (already proven by the main sweep)
  - B should fix toxicity but may hurt factual (no general SFT signal)
  - C should NOT fix toxicity (random pairs have no signal)
  - A should beat B on factual stability (the combination is the key)

Usage:
    python experiments/ablation_sft_mlx.py --model qwen2.5-7b --seeds 42,123
    python experiments/ablation_sft_mlx.py --model llama3.1-8b --seeds 42,123
"""

from __future__ import annotations

import argparse
import copy
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
ALL_EVAL_BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias", "reasoning"]

# Ablation conditions
CONDITIONS = ["sft-only", "rho-guided", "contrastive-only", "shuffled-pairs"]


# ── Reuse scoring from main experiment ─────────────────────────────────

def _mlx_mean_logprob(model, tokenizer, text: str, max_length: int = 256) -> float:
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
    ce = nn.losses.cross_entropy(logits, targets).mean()
    mx.eval(ce)
    val = -float(ce)
    return val if np.isfinite(val) else 0.0


def mlx_quick_rho(model, tokenizer, behaviors, seed=42):
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
            pos_lps.append(_mlx_mean_logprob(model, tokenizer, pair["positive"]))
            neg_lps.append(_mlx_mean_logprob(model, tokenizer, pair["negative"]))
        scores[bname] = float(np.mean(pos_lps) - np.mean(neg_lps))

    return scores


# ── Shuffled Dataset Wrapper ───────────────────────────────────────────

class ShuffledContrastDataset:
    """Wraps a BehavioralContrastDataset but randomly swaps pos/neg labels."""

    def __init__(self, real_dataset, shuffle_seed=999):
        self.real_dataset = real_dataset
        self.shuffle_seed = shuffle_seed

    def __len__(self):
        return len(self.real_dataset)

    def sample(self, k, rng=None):
        """Sample k pairs with randomly shuffled positive/negative labels."""
        real_pairs = self.real_dataset.sample(k, rng)
        shuffle_rng = random.Random(self.shuffle_seed)

        shuffled = []
        for pair in real_pairs:
            if shuffle_rng.random() < 0.5:
                # Swap labels
                shuffled.append({
                    "positive": pair["negative"],
                    "negative": pair["positive"],
                })
            else:
                shuffled.append(pair)

        return shuffled


# ── Empty Dataset (for contrastive-only: no SFT texts) ────────────────

class EmptyContrastDataset:
    """Produces no pairs — used when we want CE only."""
    def __len__(self):
        return 0
    def sample(self, k, rng=None):
        return []


# ── Main Ablation Pipeline ────────────────────────────────────────────

def run_ablation(
    model_name: str,
    conditions: list[str],
    seeds: list[int],
    rho_weight: float = 0.2,
    sft_size: int = 1000,
    epochs: int = 1,
    lr: float = 2e-4,
    lora_rank: int = 8,
    margin: float = 0.1,
    results_path: Path | None = None,
    verbose: bool = True,
):
    """Run ablation conditions × seeds."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load as mlx_load

    from rho_eval.alignment.dataset import (
        _load_alpaca_texts, _build_trap_texts,
        BehavioralContrastDataset, CONTRAST_BEHAVIORS,
    )
    from rho_eval.alignment.mlx_trainer import mlx_rho_guided_sft

    results_path = results_path or (
        RESULTS_DIR / f"ablation_{model_name.replace('/', '_')}.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Ablation Study (MLX): {model_name}")
    print(f"  Conditions: {conditions}")
    print(f"  Seeds: {seeds}, rho_weight={rho_weight}")
    print(f"{'='*70}\n")

    model, tokenizer = mlx_load(model_name)
    model.eval()

    # Baseline eval
    print("Evaluating baseline (quick MLX confidence gaps)...")
    baseline_quick = mlx_quick_rho(model, tokenizer, ALL_EVAL_BEHAVIORS)
    for bname, gap in baseline_quick.items():
        print(f"    {bname:12s}: {gap:+.4f}")

    # Baseline calibration
    print("\nEvaluating baseline calibration (ECE + Brier)...")
    try:
        from rho_eval.behaviors.metrics import calibration_metrics
        baseline_calib = calibration_metrics(
            model, tokenizer, behaviors=BEHAVIORS, seed=42,
        )
        baseline_calib_summary = {
            bname: {k: v for k, v in bdata.items()
                    if k not in ("details", "ece_bins")}
            for bname, bdata in baseline_calib.items()
            if isinstance(bdata, dict)
        }
        for bname, bdata in baseline_calib_summary.items():
            if "error" not in bdata:
                print(f"    {bname:12s}: ECE={bdata['ece']:.4f}  Brier={bdata['brier']:.4f}  "
                      f"Acc={bdata['accuracy']:.1%}")
    except Exception as e:
        print(f"    [calib] Baseline error: {e}")
        baseline_calib_summary = {}

    # Save initial weights
    print("\nSaving initial weights...")
    initial_path = results_path.parent / "ablation_initial_weights.safetensors"
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
        "experiment": "ablation",
        "baseline_quick": baseline_quick,
        "baseline_calibration": baseline_calib_summary,
        "config": {
            "rho_weight": rho_weight,
            "sft_size": sft_size,
            "epochs": epochs,
            "lr": lr,
            "lora_rank": lora_rank,
            "margin": margin,
            "conditions": conditions,
            "seeds": seeds,
        },
        "runs": [],
        "timestamp": datetime.now().isoformat(),
    }

    total_runs = len(conditions) * len(seeds)
    run_idx = 0

    for condition in conditions:
        for seed in seeds:
            run_idx += 1
            label = f"{condition}/s={seed}"
            print(f"\n{'─'*60}")
            print(f"  Run {run_idx}/{total_runs}: condition={condition}, seed={seed}")
            print(f"{'─'*60}")

            # Restore original weights
            model.load_weights(str(initial_path), strict=False)
            mx.eval(model.parameters())

            # Build contrast dataset for this seed
            real_contrast = BehavioralContrastDataset(
                behaviors=BEHAVIORS, seed=seed,
            )

            # Configure condition
            if condition == "sft-only":
                # CE only — no contrastive signal
                this_rho_weight = 0.0
                this_contrast = real_contrast  # unused since weight=0
                this_sft_texts = sft_texts

            elif condition == "rho-guided":
                # The real method: CE + λ_ρ * contrastive
                this_rho_weight = rho_weight
                this_contrast = real_contrast
                this_sft_texts = sft_texts

            elif condition == "contrastive-only":
                # Contrastive loss only — use behavioral pairs as SFT texts too
                # but with high rho_weight so contrastive dominates
                this_rho_weight = rho_weight
                this_contrast = real_contrast
                # Use the positive texts from contrast pairs as SFT data
                # (minimal CE signal, contrastive is the main driver)
                contrast_texts = []
                for pair in real_contrast.sample(len(real_contrast), random.Random(seed)):
                    contrast_texts.append(pair["positive"])
                this_sft_texts = contrast_texts[:sft_size] if len(contrast_texts) >= sft_size else contrast_texts

            elif condition == "shuffled-pairs":
                # Same setup as rho-guided but with shuffled pos/neg labels
                this_rho_weight = rho_weight
                this_contrast = ShuffledContrastDataset(real_contrast, shuffle_seed=seed)
                this_sft_texts = sft_texts

            else:
                raise ValueError(f"Unknown condition: {condition}")

            t_start = time.time()

            try:
                train_result = mlx_rho_guided_sft(
                    model, tokenizer,
                    this_sft_texts, this_contrast,
                    rho_weight=this_rho_weight,
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
                    "condition": condition,
                    "seed": seed,
                    "error": str(e),
                })
                _save_checkpoint(all_results, results_path)
                continue

            # Quick eval
            print(f"\n  Quick eval {label}...")
            quick_scores = mlx_quick_rho(model, tokenizer, ALL_EVAL_BEHAVIORS)

            # Calibration metrics (ECE + Brier)
            print(f"  Calibration eval {label}...")
            try:
                from rho_eval.behaviors.metrics import calibration_metrics
                calib = calibration_metrics(
                    model, tokenizer,
                    behaviors=BEHAVIORS,
                    seed=seed,
                )
                calib_summary = {
                    bname: {k: v for k, v in bdata.items()
                            if k not in ("details", "ece_bins")}
                    for bname, bdata in calib.items()
                    if isinstance(bdata, dict)
                }
            except Exception as e:
                print(f"    [calib] Error: {e}")
                calib_summary = {}

            elapsed = time.time() - t_start

            # Record
            run_record = {
                "condition": condition,
                "seed": seed,
                "quick_scores": quick_scores,
                "quick_deltas": {
                    bname: quick_scores.get(bname, 0) - baseline_quick.get(bname, 0)
                    for bname in ALL_EVAL_BEHAVIORS
                },
                "calibration": calib_summary,
                "train_ce_loss": train_result.get("ce_loss", 0),
                "train_rho_loss": train_result.get("rho_loss", 0),
                "train_steps": train_result.get("steps", 0),
                "elapsed_seconds": elapsed,
            }
            all_results["runs"].append(run_record)

            # Print summary
            print(f"\n  Results for {label} ({elapsed:.1f}s):")
            for bname in ALL_EVAL_BEHAVIORS:
                q_score = quick_scores.get(bname, float("nan"))
                q_delta = run_record["quick_deltas"].get(bname, 0)
                marker = "+" if q_delta > 0.01 else ("-" if q_delta < -0.01 else "=")
                print(f"    {bname:12s}: gap={q_score:+.4f} ({q_delta:+.4f}) {marker}")
            if calib_summary:
                print(f"  Calibration:")
                for bname in BEHAVIORS:
                    if bname in calib_summary and "error" not in calib_summary[bname]:
                        c = calib_summary[bname]
                        print(f"    {bname:12s}: ECE={c['ece']:.4f}  Brier={c['brier']:.4f}  "
                              f"Acc={c['accuracy']:.1%}")

            _save_checkpoint(all_results, results_path)
            print(f"  Checkpoint saved: {results_path.name}")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ABLATION COMPLETE: {total_runs} runs")
    print(f"  Results: {results_path}")
    print(f"{'='*70}")

    # Cleanup
    initial_path.unlink(missing_ok=True)

    return all_results


def _save_checkpoint(results: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.rename(path)


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablation study for rho-guided SFT")
    parser.add_argument("--model", default="qwen2.5-7b",
                        help="Model key or HF name")
    parser.add_argument("--conditions", default=",".join(CONDITIONS),
                        help="Comma-separated conditions to run")
    parser.add_argument("--seeds", default="42,123",
                        help="Comma-separated seeds")
    parser.add_argument("--rho-weight", type=float, default=0.2,
                        help="Rho weight for conditions that use it")
    parser.add_argument("--sft-size", type=int, default=1000,
                        help="Number of SFT texts")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.1)

    args = parser.parse_args()

    model_name = MODELS.get(args.model, args.model)
    conditions = [c.strip() for c in args.conditions.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    run_ablation(
        model_name=model_name,
        conditions=conditions,
        seeds=seeds,
        rho_weight=args.rho_weight,
        sft_size=args.sft_size,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
