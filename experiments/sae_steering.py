#!/usr/bin/env python3
"""SAE Steering Experiment: Disentangled vs Linear Behavioral Control.

Compares SAE-based feature steering against SVD-based linear steering,
measuring both target behavior improvement and collateral damage to
other behaviors. The key hypothesis: SAE steering can improve sycophancy
at Layer 17 without degrading bias (which shares the same SVD subspace).

Sweeps:
  - Expansion factors: [4, 8, 16]
  - Sparsity lambdas: [1e-4, 1e-3, 1e-2]
  - Scale factors: [0.0, 1.5, 2.0, 3.0, 4.0]
  - SVD alphas: [2.0, 4.0, 6.0] (for comparison)

Usage:
    python experiments/sae_steering.py
    python experiments/sae_steering.py --model Qwen/Qwen2.5-7B-Instruct
    python experiments/sae_steering.py --validate
    python experiments/sae_steering.py --analyze results/steering/sae_sweep.json
"""

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np


# ── Configuration ─────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results" / "steering"

MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}

DEFAULT_TARGET_LAYER = 17
BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias"]
EVAL_BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias", "reasoning"]

DEFAULT_EXPANSION_FACTORS = [4, 8, 16]
DEFAULT_SPARSITY_LAMBDAS = [1e-4, 1e-3, 1e-2]
DEFAULT_SCALES = [0.0, 1.5, 2.0, 3.0, 4.0]
SVD_ALPHAS = [2.0, 4.0, 6.0]


# ── Main Pipeline ─────────────────────────────────────────────────────

def run_experiment(
    model_name: str,
    layer_idx: int = DEFAULT_TARGET_LAYER,
    expansion_factors: list[int] | None = None,
    sparsity_lambdas: list[float] | None = None,
    scales: list[float] | None = None,
    target_behavior: str = "sycophancy",
    n_epochs: int = 5,
    max_probes: int | None = None,
    device: str = "cpu",
    results_path: Path | None = None,
    verbose: bool = True,
):
    """Run the full SAE steering experiment.

    Pipeline:
    1. Load model, evaluate baseline
    2. Collect activations at target layer
    3. For each (expansion, sparsity) config:
       a. Train Gated SAE
       b. Identify behavioral features
       c. Evaluate SAE steering at each scale
    4. Run SVD steering for comparison
    5. Compute and report key metrics

    Args:
        model_name: HuggingFace model ID.
        layer_idx: Target layer (default: 17 for sycophancy-bias overlap).
        expansion_factors: SAE expansion factors to test.
        sparsity_lambdas: L1 weights to test.
        scales: SAE feature scale factors to test.
        target_behavior: Primary behavior to steer (default: sycophancy).
        n_epochs: SAE training epochs.
        max_probes: Cap on probes per behavior.
        device: Torch device.
        results_path: Output JSON path.
        verbose: Print progress.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rho_eval.steering import (
        SAEConfig, GatedSAE, collect_activations,
        train_sae, identify_behavioral_features,
        evaluate_sae_steering, feature_overlap_matrix,
    )
    from rho_eval.interpretability import extract_subspaces
    from rho_eval.interpretability.surgical import evaluate_surgical, evaluate_baseline

    expansion_factors = expansion_factors or DEFAULT_EXPANSION_FACTORS
    sparsity_lambdas = sparsity_lambdas or DEFAULT_SPARSITY_LAMBDAS
    scales = scales or DEFAULT_SCALES

    results_path = results_path or (
        RESULTS_DIR / f"sae_sweep_{model_name.replace('/', '_')}.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SAE Steering Experiment")
    print(f"  Model:  {model_name}")
    print(f"  Layer:  {layer_idx}")
    print(f"  Target: {target_behavior}")
    print(f"{'='*70}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    # ── Baseline evaluation ──────────────────────────────────────
    print("\n[1/5] Evaluating baseline...")
    baseline_scores = evaluate_baseline(model, tokenizer, EVAL_BEHAVIORS, device, verbose)

    all_results = {
        "model": model_name,
        "layer_idx": layer_idx,
        "target_behavior": target_behavior,
        "baseline": baseline_scores,
        "config": {
            "expansion_factors": expansion_factors,
            "sparsity_lambdas": sparsity_lambdas,
            "scales": scales,
            "n_epochs": n_epochs,
            "max_probes": max_probes,
        },
        "sae_runs": [],
        "svd_runs": [],
        "timestamp": datetime.now().isoformat(),
    }

    # ── Collect activations (once) ───────────────────────────────
    print("\n[2/5] Collecting activations...")
    act_data = collect_activations(
        model, tokenizer, BEHAVIORS, layer_idx,
        device=device, max_probes=max_probes, verbose=verbose,
    )

    # ── SAE sweep ────────────────────────────────────────────────
    print(f"\n[3/5] SAE sweep: "
          f"{len(expansion_factors)} expansions x {len(sparsity_lambdas)} lambdas")

    total_configs = len(expansion_factors) * len(sparsity_lambdas)
    config_idx = 0

    for expansion in expansion_factors:
        for sparsity_lambda in sparsity_lambdas:
            config_idx += 1
            label = f"exp={expansion}, lambda={sparsity_lambda}"
            print(f"\n{'─'*60}")
            print(f"  Config {config_idx}/{total_configs}: {label}")
            print(f"{'─'*60}")

            config = SAEConfig(
                hidden_dim=act_data.hidden_dim,
                expansion_factor=expansion,
                sparsity_lambda=sparsity_lambda,
                n_epochs=n_epochs,
                device=device,
            )

            # Train SAE
            try:
                sae = GatedSAE(config.hidden_dim, config.expansion_factor)
                stats = train_sae(sae, act_data, config, verbose=verbose)
                sae = stats["sae"]
            except Exception as e:
                print(f"  ERROR training SAE: {e}")
                all_results["sae_runs"].append({
                    "expansion": expansion,
                    "sparsity_lambda": sparsity_lambda,
                    "error": str(e),
                })
                _save_checkpoint(all_results, results_path)
                continue

            # Identify features
            reports, behavioral_features = identify_behavioral_features(
                sae, act_data, threshold=2.0,
            )
            overlap = feature_overlap_matrix(reports, BEHAVIORS)

            n_features_per = {b: len(v) for b, v in behavioral_features.items()}
            print(f"  Features per behavior: {n_features_per}")

            # Evaluate steering
            if target_behavior not in behavioral_features or not behavioral_features[target_behavior]:
                print(f"  WARNING: No features for {target_behavior} — skipping steering")
                all_results["sae_runs"].append({
                    "expansion": expansion,
                    "sparsity_lambda": sparsity_lambda,
                    "n_features": n_features_per,
                    "overlap": overlap,
                    "train_stats": {
                        k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in stats.items() if k != "sae"
                    },
                    "steering": [],
                    "warning": f"No features for {target_behavior}",
                })
                _save_checkpoint(all_results, results_path)
                continue

            try:
                steering_results = evaluate_sae_steering(
                    model, tokenizer, sae, behavioral_features,
                    target_behavior=target_behavior,
                    layer_idx=layer_idx,
                    eval_behaviors=EVAL_BEHAVIORS,
                    scales=scales,
                    device=device,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"  ERROR evaluating: {e}")
                steering_results = []

            run_record = {
                "expansion": expansion,
                "sparsity_lambda": sparsity_lambda,
                "n_features": n_features_per,
                "overlap": overlap,
                "train_stats": {
                    k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in stats.items() if k != "sae"
                },
                "steering": steering_results,
            }
            all_results["sae_runs"].append(run_record)
            _save_checkpoint(all_results, results_path)

    # ── SVD comparison ───────────────────────────────────────────
    print(f"\n[4/5] SVD steering comparison...")
    try:
        subspaces = extract_subspaces(
            model, tokenizer, BEHAVIORS,
            layers=[layer_idx], device=device, verbose=verbose,
        )

        for alpha in SVD_ALPHAS:
            print(f"\n  SVD alpha={alpha}:")
            result = evaluate_surgical(
                model, tokenizer, subspaces,
                target_behavior=target_behavior,
                layer_idx=layer_idx,
                eval_behaviors=EVAL_BEHAVIORS,
                alpha=alpha, device=device, verbose=verbose,
            )
            svd_record = result.to_dict()
            svd_record["baseline_rho_scores"] = baseline_scores
            # Compute collateral
            collateral = {}
            for beh in EVAL_BEHAVIORS:
                if beh != target_behavior:
                    collateral[beh] = (
                        result.rho_scores.get(beh, 0) - baseline_scores.get(beh, 0)
                    )
            svd_record["collateral"] = collateral
            all_results["svd_runs"].append(svd_record)

        _save_checkpoint(all_results, results_path)
    except Exception as e:
        print(f"  SVD comparison failed: {e}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n[5/5] Summary")
    _print_summary(all_results)

    print(f"\n  Results saved to: {results_path}")
    return all_results


def _save_checkpoint(results: dict, path: Path):
    """Atomic save to JSON."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.rename(path)


def _print_summary(results: dict):
    """Print experiment summary."""
    baseline = results["baseline"]
    target = results["target_behavior"]

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT SUMMARY: Steering {target}")
    print(f"{'='*70}")

    print(f"\n  Baseline rho scores:")
    for beh, rho in baseline.items():
        print(f"    {beh:12s}: {rho:.4f}")

    # Best SAE config
    best_sae = None
    best_sae_delta = -float("inf")
    best_sae_collateral = float("inf")

    for run in results.get("sae_runs", []):
        for sr in run.get("steering", []):
            target_rho = sr.get("rho_scores", {}).get(target, 0)
            delta = target_rho - baseline.get(target, 0)
            collateral_sum = sum(abs(v) for v in sr.get("collateral", {}).values())

            # Best = highest target improvement with lowest collateral
            score = delta - 0.5 * collateral_sum
            if score > best_sae_delta:
                best_sae_delta = score
                best_sae_collateral = collateral_sum
                best_sae = {
                    "expansion": run["expansion"],
                    "lambda": run["sparsity_lambda"],
                    "scale": sr["scale"],
                    "rho_scores": sr.get("rho_scores", {}),
                    "collateral": sr.get("collateral", {}),
                }

    if best_sae:
        print(f"\n  Best SAE config:")
        print(f"    Expansion: {best_sae['expansion']}, "
              f"Lambda: {best_sae['lambda']}, Scale: {best_sae['scale']}")
        for beh, rho in best_sae["rho_scores"].items():
            delta = rho - baseline.get(beh, 0)
            marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else "=")
            print(f"    {beh:12s}: {rho:.4f} ({delta:+.4f}) {marker}")

    # Best SVD
    best_svd = None
    best_svd_delta = -float("inf")

    for run in results.get("svd_runs", []):
        target_rho = run.get("rho_scores", {}).get(target, 0)
        delta = target_rho - baseline.get(target, 0)
        collateral_sum = sum(abs(v) for v in run.get("collateral", {}).values())
        score = delta - 0.5 * collateral_sum
        if score > best_svd_delta:
            best_svd_delta = score
            best_svd = run

    if best_svd:
        print(f"\n  Best SVD config (alpha={best_svd.get('config', {}).get('alpha', '?')}):")
        for beh, rho in best_svd.get("rho_scores", {}).items():
            delta = rho - baseline.get(beh, 0)
            marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else "=")
            print(f"    {beh:12s}: {rho:.4f} ({delta:+.4f}) {marker}")

    # Comparison
    if best_sae and best_svd:
        print(f"\n  KEY COMPARISON ({target}):")
        sae_target_delta = best_sae["rho_scores"].get(target, 0) - baseline.get(target, 0)
        svd_target_delta = best_svd.get("rho_scores", {}).get(target, 0) - baseline.get(target, 0)
        sae_collateral = sum(abs(v) for v in best_sae.get("collateral", {}).values())
        svd_collateral = sum(abs(v) for v in best_svd.get("collateral", {}).values())
        print(f"    SAE: target delta={sae_target_delta:+.4f}, "
              f"collateral={sae_collateral:.4f}")
        print(f"    SVD: target delta={svd_target_delta:+.4f}, "
              f"collateral={svd_collateral:.4f}")
        if sae_collateral < svd_collateral and sae_target_delta > 0:
            print(f"    --> SAE achieves {'better' if sae_target_delta >= svd_target_delta else 'comparable'} "
                  f"steering with {(1 - sae_collateral/max(svd_collateral, 1e-8))*100:.0f}% less collateral")


# ── Analysis ──────────────────────────────────────────────────────────

def analyze(results_path: Path):
    """Analyze existing experiment results."""
    with open(results_path) as f:
        data = json.load(f)
    _print_summary(data)


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAE steering experiment: disentangled vs linear behavioral control"
    )
    parser.add_argument("--model", default="qwen2.5-0.5b",
                        help="Model key or HF name (default: qwen2.5-0.5b)")
    parser.add_argument("--layer", type=int, default=DEFAULT_TARGET_LAYER,
                        help=f"Target layer (default: {DEFAULT_TARGET_LAYER})")
    parser.add_argument("--target", default="sycophancy",
                        help="Target behavior to steer (default: sycophancy)")
    parser.add_argument("--expansions", default=None,
                        help="Comma-separated expansion factors (default: 4,8,16)")
    parser.add_argument("--lambdas", default=None,
                        help="Comma-separated sparsity lambdas (default: 1e-4,1e-3,1e-2)")
    parser.add_argument("--scales", default=None,
                        help="Comma-separated scale factors")
    parser.add_argument("--epochs", type=int, default=5,
                        help="SAE training epochs (default: 5)")
    parser.add_argument("--max-probes", type=int, default=None,
                        help="Max probes per behavior")
    parser.add_argument("--device", default=None,
                        help="Torch device (default: auto-detect)")
    parser.add_argument("--validate", action="store_true",
                        help="Quick validation run")
    parser.add_argument("--analyze", default=None,
                        help="Analyze existing results file")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")

    args = parser.parse_args()

    if args.analyze:
        analyze(Path(args.analyze))
        return

    model_name = MODELS.get(args.model, args.model)

    if args.device:
        device = args.device
    else:
        from rho_eval.utils import get_device
        device = str(get_device())

    if args.validate:
        model_name = MODELS.get("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B")
        expansion_factors = [4]
        sparsity_lambdas = [1e-3]
        scales = [0.0, 2.0]
        n_epochs = 2
        max_probes = 30
        print("VALIDATION MODE: Qwen-0.5B, 1 expansion, 1 lambda, 2 scales, 30 probes")
    else:
        expansion_factors = (
            [int(x) for x in args.expansions.split(",")]
            if args.expansions else None
        )
        sparsity_lambdas = (
            [float(x) for x in args.lambdas.split(",")]
            if args.lambdas else None
        )
        scales = (
            [float(x) for x in args.scales.split(",")]
            if args.scales else None
        )
        n_epochs = args.epochs
        max_probes = args.max_probes

    results_path = Path(args.output) if args.output else None

    run_experiment(
        model_name=model_name,
        layer_idx=args.layer,
        expansion_factors=expansion_factors,
        sparsity_lambdas=sparsity_lambdas,
        scales=scales,
        target_behavior=args.target,
        n_epochs=n_epochs,
        max_probes=max_probes,
        device=device,
        results_path=results_path,
    )


if __name__ == "__main__":
    main()
