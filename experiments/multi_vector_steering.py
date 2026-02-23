#!/usr/bin/env python3
"""Phase 5.1: Multi-vector steering — resolving the Layer 17 trade-off.

The Phase 3 steering experiments discovered that Layer 17 is a behavioral
bottleneck: sycophancy steering (ρ 0.120→0.413 at α=+4.0) simultaneously
collapses bias detection (ρ 0.773→0.337). Each behavior's best steering
config lives at a different layer:

    Factual:     Layer 24 (86%), α=+4.0  → ρ 0.474→0.626
    Sycophancy:  Layer 17 (61%), α=+4.0  → ρ 0.120→0.413
    Bias:        Layer 14 (50%), α=−4.0  → ρ 0.773→0.810

This script tests "steering cocktails" — multiple vectors applied
simultaneously at different layers — to find a configuration where:

    Sycophancy ρ ≥ 0.35  (substantial improvement from 0.120 baseline)
    Bias ρ ≥ 0.70        (close to 0.773 baseline retention)

If a null point exists, we also test adding a third factual vector
at Layer 24 to create a full 3-behavior cocktail.

Usage:
    python experiments/multi_vector_steering.py
    python experiments/multi_vector_steering.py --quick
    python experiments/multi_vector_steering.py --cross-model mistralai/Mistral-7B-Instruct-v0.3
"""

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Reuse everything from Phase 3 — no duplication
from steering_vectors import (
    ActivationCapture,
    SteeringHook,
    build_contrast_pairs,
    extract_steering_vectors,
    load_model,
    _save,
    DEVICE,
)

from knowledge_fidelity.behavioral import (
    load_behavioral_probes,
    evaluate_behavior,
)
from knowledge_fidelity.probes import get_all_probes
from knowledge_fidelity.utils import get_layers, free_memory

RESULTS_DIR = Path(__file__).parent.parent / "results" / "steering"


# ── Multi-Layer Steering ─────────────────────────────────────────────────

class MultiSteeringHook:
    """Apply steering vectors at multiple layers simultaneously.

    Each entry in vectors_config is a (layer_idx, vector, alpha) tuple.
    Registers independent hooks on each layer — no interaction between them,
    which is exactly what we want: each vector steers its own layer.

    Usage:
        configs = [
            (17, syc_vector, 4.0),   # sycophancy at Layer 17
            (14, bias_vector, -2.0),  # bias at Layer 14
        ]
        hook = MultiSteeringHook(model, configs)
        model.generate(...)  # steered with both vectors
        hook.remove()
    """

    def __init__(self, model, vectors_config: list[tuple[int, torch.Tensor, float]]):
        """
        Args:
            model: HuggingFace causal LM
            vectors_config: list of (layer_idx, vector, alpha) tuples
        """
        self._hooks = []
        layers = get_layers(model)

        for layer_idx, vector, alpha in vectors_config:
            # Closure factory to avoid late-binding bug
            def make_steer_fn(vec, a):
                def _steer(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        h = h + a * vec.to(h.device, h.dtype)
                        return (h,) + output[1:]
                    else:
                        return output + a * vec.to(output.device, output.dtype)
                return _steer

            handle = layers[layer_idx].register_forward_hook(
                make_steer_fn(vector, alpha)
            )
            self._hooks.append(handle)

    def remove(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def evaluate_multi_steered(
    model,
    tokenizer,
    behavior: str,
    probes: list[dict],
    vectors_config: list[tuple[int, torch.Tensor, float]],
    device: str = "cpu",
) -> dict:
    """Evaluate a model with multiple steering vectors applied simultaneously.

    Args:
        vectors_config: list of (layer_idx, vector, alpha) tuples
    """
    hook = MultiSteeringHook(model, vectors_config)
    try:
        result = evaluate_behavior(behavior, model, tokenizer, probes, device)
    finally:
        hook.remove()
    return result


# ── Vector Extraction ────────────────────────────────────────────────────

def extract_all_vectors(model, tokenizer, behaviors, candidate_layers, device):
    """Extract steering vectors for all behaviors at all candidate layers.

    Returns:
        Dict: {behavior: {layer_idx: vector_tensor}}
        Dict: {behavior: {layer_idx: norm_float}}
    """
    all_vectors = {}
    all_norms = {}

    for behavior in behaviors:
        print(f"\n  Extracting vectors for: {behavior}")

        if behavior == "factual":
            probes = get_all_probes()
        else:
            probes = load_behavioral_probes(behavior, seed=42)

        pairs = build_contrast_pairs(behavior, probes)
        print(f"    {len(pairs)} contrast pairs")

        vectors = extract_steering_vectors(
            model, tokenizer, pairs, candidate_layers, device, method="mean_diff"
        )

        all_vectors[behavior] = vectors
        all_norms[behavior] = {
            str(k): round(float(v.norm()), 4) for k, v in vectors.items()
        }

    return all_vectors, all_norms


# ── Grid Search ──────────────────────────────────────────────────────────

def run_cocktail_grid(
    model, tokenizer, all_vectors, all_probes,
    syc_layer, bias_layer,
    syc_alphas, bias_alphas,
    eval_behaviors, device,
):
    """Run the 2D cocktail grid: sycophancy alpha × bias alpha.

    For each combination, evaluates ALL behaviors to measure cross-talk.

    Returns:
        List of grid result dicts
    """
    grid_results = []
    total = len(syc_alphas) * len(bias_alphas)
    idx = 0

    for syc_alpha in syc_alphas:
        for bias_alpha in bias_alphas:
            idx += 1
            print(f"\n  [{idx}/{total}] syc@L{syc_layer} α={syc_alpha:+.1f}, "
                  f"bias@L{bias_layer} α={bias_alpha:+.1f}")

            config = [
                (syc_layer, all_vectors["sycophancy"][syc_layer], syc_alpha),
                (bias_layer, all_vectors["bias"][bias_layer], bias_alpha),
            ]

            point = {
                "config": {
                    "sycophancy": {"layer": syc_layer, "alpha": syc_alpha},
                    "bias": {"layer": bias_layer, "alpha": bias_alpha},
                },
                "results": {},
            }

            t0 = time.time()
            for beh in eval_behaviors:
                probes = all_probes[beh]
                result = evaluate_multi_steered(
                    model, tokenizer, beh, probes, config, device
                )
                rho = result["rho"]
                point["results"][beh] = {
                    k: v for k, v in result.items() if k != "details"
                }
                print(f"    {beh}: ρ={rho:.4f}" if rho is not None else f"    {beh}: ρ=N/A",
                      end="")
            point["elapsed_s"] = round(time.time() - t0, 1)
            print(f"  [{point['elapsed_s']:.0f}s]")

            grid_results.append(point)

    return grid_results


def find_null_point(grid_results, baselines,
                    syc_target=0.35, bias_target=0.70):
    """Find the best Pareto-optimal cocktail meeting both targets.

    Strategy: among configs where sycophancy ρ ≥ target AND bias ρ ≥ target,
    pick the one maximizing (syc_rho + bias_rho) — the joint objective.

    Returns:
        Best config dict, or None if no config meets both targets.
    """
    candidates = []

    for point in grid_results:
        syc_rho = point["results"].get("sycophancy", {}).get("rho")
        bias_rho = point["results"].get("bias", {}).get("rho")

        if syc_rho is None or bias_rho is None:
            continue
        if isinstance(syc_rho, float) and math.isnan(syc_rho):
            continue
        if isinstance(bias_rho, float) and math.isnan(bias_rho):
            continue

        if syc_rho >= syc_target and bias_rho >= bias_target:
            candidates.append(point)

    if not candidates:
        # Fallback: find closest to meeting both targets
        print("  No config meets both targets. Finding closest...")
        best_score = -float("inf")
        best = None
        for point in grid_results:
            syc_rho = point["results"].get("sycophancy", {}).get("rho", 0)
            bias_rho = point["results"].get("bias", {}).get("rho", 0)
            if isinstance(syc_rho, float) and math.isnan(syc_rho):
                continue
            if isinstance(bias_rho, float) and math.isnan(bias_rho):
                continue
            # Weighted score: both targets matter equally
            score = min(syc_rho / syc_target, 1.0) + min(bias_rho / bias_target, 1.0)
            if score > best_score:
                best_score = score
                best = point
        return best, False

    # Among candidates, maximize joint objective
    best = max(candidates, key=lambda p: (
        p["results"]["sycophancy"]["rho"] + p["results"]["bias"]["rho"]
    ))
    return best, True


# ── Triple Cocktail ──────────────────────────────────────────────────────

def run_triple_cocktail(
    model, tokenizer, all_vectors, all_probes,
    null_point_config, factual_layer, factual_alphas,
    eval_behaviors, device,
):
    """Add a factual vector on top of the null-point cocktail.

    Tests whether we can improve factual ρ without disrupting the
    sycophancy-bias balance.
    """
    base_config = null_point_config["config"]
    syc_layer = base_config["sycophancy"]["layer"]
    syc_alpha = base_config["sycophancy"]["alpha"]
    bias_layer = base_config["bias"]["layer"]
    bias_alpha = base_config["bias"]["alpha"]

    results = []

    for fact_alpha in factual_alphas:
        print(f"\n  Triple: syc@L{syc_layer}={syc_alpha:+.1f}, "
              f"bias@L{bias_layer}={bias_alpha:+.1f}, "
              f"fact@L{factual_layer}={fact_alpha:+.1f}")

        config = [
            (syc_layer, all_vectors["sycophancy"][syc_layer], syc_alpha),
            (bias_layer, all_vectors["bias"][bias_layer], bias_alpha),
            (factual_layer, all_vectors["factual"][factual_layer], fact_alpha),
        ]

        point = {
            "config": {
                "sycophancy": {"layer": syc_layer, "alpha": syc_alpha},
                "bias": {"layer": bias_layer, "alpha": bias_alpha},
                "factual": {"layer": factual_layer, "alpha": fact_alpha},
            },
            "results": {},
        }

        t0 = time.time()
        for beh in eval_behaviors:
            probes = all_probes[beh]
            result = evaluate_multi_steered(
                model, tokenizer, beh, probes, config, device
            )
            point["results"][beh] = {
                k: v for k, v in result.items() if k != "details"
            }
            rho = result["rho"]
            print(f"    {beh}: ρ={rho:.4f}" if rho is not None else f"    {beh}: ρ=N/A",
                  end="")
        point["elapsed_s"] = round(time.time() - t0, 1)
        print(f"  [{point['elapsed_s']:.0f}s]")

        results.append(point)

    return results


# ── Cross-Model Validation ───────────────────────────────────────────────

def run_cross_model(
    model_id, null_point_config, eval_behaviors, device,
):
    """Test the winning cocktail on a different model.

    Re-extracts vectors for the new model (vectors are model-specific),
    then applies the same layer percentages and alphas.
    """
    print(f"\n{'='*60}")
    print(f"  CROSS-MODEL: {model_id}")
    print(f"{'='*60}")

    model, tokenizer = load_model(model_id, device)
    n_layers = len(get_layers(model))

    # Map layer indices from original model to this model via percentages
    base_config = null_point_config["config"]
    orig_n_layers = 28  # Qwen2.5-7B

    layer_map = {}
    for beh, cfg in base_config.items():
        orig_layer = cfg["layer"]
        pct = orig_layer / orig_n_layers
        new_layer = max(0, min(n_layers - 1, int(pct * n_layers)))
        layer_map[beh] = new_layer
        print(f"  {beh}: Layer {orig_layer} ({pct:.0%}) → Layer {new_layer}")

    # All unique layers needed
    unique_layers = sorted(set(layer_map.values()))

    # Load probes and extract vectors
    all_probes = {}
    all_vectors = {}

    for beh in eval_behaviors:
        if beh == "factual":
            all_probes[beh] = get_all_probes()
        else:
            all_probes[beh] = load_behavioral_probes(beh, seed=42)

    behaviors_to_steer = [b for b in base_config.keys()]
    for beh in behaviors_to_steer:
        if beh == "factual":
            probes = get_all_probes()
        else:
            probes = load_behavioral_probes(beh, seed=42)

        pairs = build_contrast_pairs(beh, probes)
        vectors = extract_steering_vectors(
            model, tokenizer, pairs, unique_layers, device
        )
        all_vectors[beh] = vectors

    # Compute baselines
    baselines = {}
    for beh in eval_behaviors:
        result = evaluate_behavior(beh, model, tokenizer, all_probes[beh], device)
        baselines[beh] = {k: v for k, v in result.items() if k != "details"}
        print(f"  Baseline {beh}: ρ={result['rho']:.4f}")

    # Apply cocktail
    config_tuples = []
    for beh, cfg in base_config.items():
        new_layer = layer_map[beh]
        alpha = cfg["alpha"]
        config_tuples.append((new_layer, all_vectors[beh][new_layer], alpha))

    steered_results = {}
    for beh in eval_behaviors:
        result = evaluate_multi_steered(
            model, tokenizer, beh, all_probes[beh], config_tuples, device
        )
        steered_results[beh] = {k: v for k, v in result.items() if k != "details"}
        print(f"  Steered {beh}: ρ={result['rho']:.4f}")

    # Cleanup
    del model, tokenizer
    free_memory()

    return {
        "model_id": model_id,
        "n_layers": n_layers,
        "layer_map": {beh: {"original": base_config[beh]["layer"],
                            "mapped": layer_map[beh],
                            "alpha": base_config[beh]["alpha"]}
                      for beh in base_config},
        "baselines": baselines,
        "steered": steered_results,
    }


# ── Main Experiment ──────────────────────────────────────────────────────

def run_experiment(
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    cross_model_id: str = None,
    device: str = DEVICE,
    quick: bool = False,
):
    """Full multi-vector steering cocktail experiment."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_short = model_id.split("/")[-1].lower().replace("-", "_")
    output_path = RESULTS_DIR / f"cocktail_{model_short}.json"

    # ── Grid parameters ──
    if quick:
        syc_alphas = [1.0, 2.0, 4.0]
        bias_alphas = [-1.0, -2.0, -4.0]
        factual_alphas = [1.0, 2.0, 4.0]
    else:
        syc_alphas = [0.5, 1.0, 2.0, 3.0, 4.0]
        bias_alphas = [-0.5, -1.0, -2.0, -3.0, -4.0]
        factual_alphas = [0.5, 1.0, 2.0, 3.0, 4.0]

    eval_behaviors = ["factual", "sycophancy", "bias"]

    # Fixed layer assignments from Phase 3 best configs
    syc_layer = 17   # 61% depth — sycophancy sweet spot
    bias_layer = 14   # 50% depth — bias sweet spot
    factual_layer = 24  # 86% depth — factual sweet spot

    print(f"\n{'='*60}")
    print(f"  MULTI-VECTOR STEERING COCKTAILS")
    print(f"  Model: {model_id}")
    print(f"  Grid: {len(syc_alphas)} × {len(bias_alphas)} = "
          f"{len(syc_alphas) * len(bias_alphas)} combos")
    print(f"  Layers: syc@{syc_layer}, bias@{bias_layer}, fact@{factual_layer}")
    print(f"{'='*60}")

    # ── Load model ──
    model, tokenizer = load_model(model_id, device)
    n_layers = len(get_layers(model))

    # All candidate layers for vector extraction
    candidate_layers = sorted({syc_layer, bias_layer, factual_layer})

    # ── Load probes ──
    all_probes = {}
    for beh in eval_behaviors:
        if beh == "factual":
            all_probes[beh] = get_all_probes()
        else:
            all_probes[beh] = load_behavioral_probes(beh, seed=42)
        print(f"  Loaded {len(all_probes[beh])} {beh} probes")

    # ── Compute baselines ──
    print("\n  Computing baselines...")
    baselines = {}
    for beh in eval_behaviors:
        result = evaluate_behavior(beh, model, tokenizer, all_probes[beh], device)
        baselines[beh] = {k: v for k, v in result.items() if k != "details"}
        print(f"    {beh}: ρ={result['rho']:.4f}")

    # ── Extract steering vectors ──
    print("\n  Extracting steering vectors...")
    all_vectors, all_norms = extract_all_vectors(
        model, tokenizer, eval_behaviors, candidate_layers, device
    )

    # ── Initialize results ──
    results = {
        "model_id": model_id,
        "experiment": "multi_vector_steering",
        "n_layers": n_layers,
        "device": str(device),
        "timestamp_start": datetime.now().isoformat(),
        "layer_assignments": {
            "sycophancy": syc_layer,
            "bias": bias_layer,
            "factual": factual_layer,
        },
        "syc_alphas": syc_alphas,
        "bias_alphas": bias_alphas,
        "baselines": baselines,
        "vectors_extracted": {
            beh: {"layers": candidate_layers, "norms": all_norms[beh]}
            for beh in eval_behaviors
        },
    }
    _save(results, output_path)

    # ── 2D Cocktail Grid ──
    print(f"\n{'='*60}")
    print(f"  COCKTAIL GRID: {len(syc_alphas)}×{len(bias_alphas)}")
    print(f"{'='*60}")

    t0 = time.time()
    grid_results = run_cocktail_grid(
        model, tokenizer, all_vectors, all_probes,
        syc_layer, bias_layer,
        syc_alphas, bias_alphas,
        eval_behaviors, device,
    )
    results["cocktail_grid"] = grid_results
    results["grid_elapsed_s"] = round(time.time() - t0, 1)
    _save(results, output_path)

    # ── Find null point ──
    null_point, meets_targets = find_null_point(grid_results, baselines)

    if null_point:
        results["null_point"] = {
            "config": null_point["config"],
            "factual_rho": null_point["results"].get("factual", {}).get("rho"),
            "sycophancy_rho": null_point["results"].get("sycophancy", {}).get("rho"),
            "bias_rho": null_point["results"].get("bias", {}).get("rho"),
            "meets_targets": meets_targets,
        }
        print(f"\n  NULL POINT {'FOUND' if meets_targets else '(closest, targets not met)'}:")
        print(f"    Config: {null_point['config']}")
        for beh in eval_behaviors:
            rho = null_point["results"].get(beh, {}).get("rho", "N/A")
            baseline = baselines[beh]["rho"]
            delta = rho - baseline if isinstance(rho, (int, float)) and not math.isnan(rho) else None
            delta_str = f" (Δ={delta:+.4f})" if delta is not None else ""
            print(f"    {beh}: ρ={rho:.4f}{delta_str} (baseline={baseline:.4f})")
    else:
        results["null_point"] = None
        print("\n  No null point found.")

    _save(results, output_path)

    # ── Triple Cocktail (if null point found) ──
    if null_point:
        print(f"\n{'='*60}")
        print(f"  TRIPLE COCKTAIL: adding factual @ Layer {factual_layer}")
        print(f"{'='*60}")

        triple_results = run_triple_cocktail(
            model, tokenizer, all_vectors, all_probes,
            null_point, factual_layer, factual_alphas,
            eval_behaviors, device,
        )
        results["triple_cocktail"] = triple_results

        # Find best triple
        best_triple = None
        best_score = -float("inf")
        for pt in triple_results:
            f_rho = pt["results"].get("factual", {}).get("rho", 0)
            s_rho = pt["results"].get("sycophancy", {}).get("rho", 0)
            b_rho = pt["results"].get("bias", {}).get("rho", 0)
            if any(isinstance(r, float) and math.isnan(r) for r in [f_rho, s_rho, b_rho]):
                continue
            score = f_rho + s_rho + b_rho
            if score > best_score:
                best_score = score
                best_triple = pt

        if best_triple:
            results["best_triple"] = {
                "config": best_triple["config"],
                "factual_rho": best_triple["results"].get("factual", {}).get("rho"),
                "sycophancy_rho": best_triple["results"].get("sycophancy", {}).get("rho"),
                "bias_rho": best_triple["results"].get("bias", {}).get("rho"),
            }
            print(f"\n  BEST TRIPLE:")
            print(f"    Config: {best_triple['config']}")
            for beh in eval_behaviors:
                rho = best_triple["results"].get(beh, {}).get("rho", "N/A")
                print(f"    {beh}: ρ={rho:.4f}")

        _save(results, output_path)

    # ── Cleanup primary model ──
    del model, tokenizer
    free_memory()

    # ── Cross-model validation ──
    if cross_model_id and null_point:
        cross_results = run_cross_model(
            cross_model_id, null_point, eval_behaviors, device
        )
        results["cross_model"] = cross_results
        _save(results, output_path)

    # ── Final summary ──
    results["timestamp_end"] = datetime.now().isoformat()
    _save(results, output_path)

    print(f"\n{'='*60}")
    print(f"  SUMMARY: Multi-Vector Steering")
    print(f"{'='*60}")
    print(f"\n  Baselines:")
    for beh, data in baselines.items():
        print(f"    {beh:>12}: ρ={data['rho']:.4f}")

    if null_point:
        print(f"\n  Null Point ({'targets met' if meets_targets else 'best available'}):")
        for beh in eval_behaviors:
            rho = null_point["results"].get(beh, {}).get("rho", "N/A")
            print(f"    {beh:>12}: ρ={rho:.4f}")

    if results.get("best_triple"):
        bt = results["best_triple"]
        print(f"\n  Best Triple Cocktail:")
        print(f"    factual:     ρ={bt['factual_rho']:.4f}")
        print(f"    sycophancy:  ρ={bt['sycophancy_rho']:.4f}")
        print(f"    bias:        ρ={bt['bias_rho']:.4f}")

    print(f"\n  Results: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-vector steering cocktails — resolving the Layer 17 trade-off"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct",
        help="Primary model (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--cross-model", default=None,
        help="Secondary model for cross-validation (e.g. mistralai/Mistral-7B-Instruct-v0.3)",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3×3 grid instead of 5×5")
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    run_experiment(
        model_id=args.model,
        cross_model_id=args.cross_model,
        device=args.device,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
