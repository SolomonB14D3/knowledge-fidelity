#!/usr/bin/env python3
"""Mistral layer heatmap: find where sycophancy lives without bias collapse.

On Qwen, Layer 17 is the only sycophancy-responsive layer — but steering there
destroys bias. Does Mistral have a different sweet spot where sycophancy
resistance improves WITHOUT bias collapse?

Sweep: apply sycophancy vector at every 2nd layer (L10–L30), α=+4.0,
measure all 3 behaviors at each point.

Usage:
    python experiments/mistral_layer_heatmap.py
    python experiments/mistral_layer_heatmap.py --alpha 2.0
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from steering_vectors import (
    build_contrast_pairs,
    extract_steering_vectors,
    load_model,
    _save,
    DEVICE,
)
from multi_vector_steering import MultiSteeringHook, evaluate_multi_steered

from knowledge_fidelity.behavioral import (
    load_behavioral_probes,
    evaluate_behavior,
)
from knowledge_fidelity.probes import get_all_probes
from knowledge_fidelity.utils import get_layers, free_memory

RESULTS_DIR = Path(__file__).parent.parent / "results" / "steering"


def run_heatmap(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    alpha: float = 4.0,
    device: str = DEVICE,
):
    """Sweep sycophancy vector across layers, measure all behaviors."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_short = model_id.split("/")[-1].lower().replace("-", "_")
    output_path = RESULTS_DIR / f"heatmap_{model_short}.json"

    # Target layers: every 2nd layer from 10 to 30 (for 32-layer Mistral)
    sweep_layers = list(range(10, 31, 2))  # [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

    eval_behaviors = ["factual", "sycophancy", "bias"]

    print(f"\n{'='*60}")
    print(f"  MISTRAL LAYER HEATMAP")
    print(f"  Model: {model_id}")
    print(f"  Sweep: sycophancy vector at {len(sweep_layers)} layers")
    print(f"  Alpha: +{alpha}")
    print(f"  Layers: {sweep_layers}")
    print(f"{'='*60}")

    # ── Load model ──
    model, tokenizer = load_model(model_id, device)
    n_layers = len(get_layers(model))
    print(f"  Total layers: {n_layers}")

    # Clamp sweep layers to actual model size
    sweep_layers = [l for l in sweep_layers if l < n_layers]
    print(f"  Valid sweep layers: {sweep_layers}")

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

    # ── Extract sycophancy vectors at ALL sweep layers ──
    print(f"\n  Extracting sycophancy vectors at {len(sweep_layers)} layers...")
    syc_probes = load_behavioral_probes("sycophancy", seed=42)
    pairs = build_contrast_pairs("sycophancy", syc_probes)
    vectors = extract_steering_vectors(
        model, tokenizer, pairs, sweep_layers, device
    )
    norms = {str(k): round(float(v.norm()), 4) for k, v in vectors.items()}
    print(f"  Norms: {norms}")

    # ── Initialize results ──
    results = {
        "model_id": model_id,
        "experiment": "layer_heatmap",
        "n_layers": n_layers,
        "device": str(device),
        "alpha": alpha,
        "sweep_layers": sweep_layers,
        "timestamp_start": datetime.now().isoformat(),
        "baselines": baselines,
        "vector_norms": norms,
    }
    _save(results, output_path)

    # ── Sweep ──
    print(f"\n{'='*60}")
    print(f"  LAYER SWEEP: α=+{alpha}")
    print(f"{'='*60}")

    sweep_results = []
    for i, layer in enumerate(sweep_layers):
        print(f"\n  [{i+1}/{len(sweep_layers)}] Layer {layer} ({layer/n_layers:.0%} depth)")

        config = [(layer, vectors[layer], alpha)]

        point = {
            "layer": layer,
            "depth_pct": round(layer / n_layers, 3),
            "alpha": alpha,
            "results": {},
        }

        t0 = time.time()
        for beh in eval_behaviors:
            result = evaluate_multi_steered(
                model, tokenizer, beh, all_probes[beh], config, device
            )
            point["results"][beh] = {
                k: v for k, v in result.items() if k != "details"
            }
            rho = result["rho"]
            baseline = baselines[beh]["rho"]
            delta = rho - baseline if rho is not None else None
            delta_str = f" (Δ={delta:+.4f})" if delta is not None else ""
            print(f"    {beh}: ρ={rho:.4f}{delta_str}", end="")

        point["elapsed_s"] = round(time.time() - t0, 1)
        print(f"  [{point['elapsed_s']:.0f}s]")

        sweep_results.append(point)

    results["sweep"] = sweep_results
    results["timestamp_end"] = datetime.now().isoformat()

    # ── Find best layer (max sycophancy delta while bias stays above threshold) ──
    bias_threshold = baselines["bias"]["rho"] * 0.85  # Allow 15% drop
    candidates = []
    for pt in sweep_results:
        syc_rho = pt["results"]["sycophancy"]["rho"]
        bias_rho = pt["results"]["bias"]["rho"]
        if bias_rho >= bias_threshold:
            candidates.append(pt)

    if candidates:
        best = max(candidates, key=lambda p: p["results"]["sycophancy"]["rho"])
        results["best_layer"] = {
            "layer": best["layer"],
            "depth_pct": best["depth_pct"],
            "sycophancy_rho": best["results"]["sycophancy"]["rho"],
            "bias_rho": best["results"]["bias"]["rho"],
            "factual_rho": best["results"]["factual"]["rho"],
            "bias_retained": True,
        }
        print(f"\n  BEST LAYER (bias ≥ {bias_threshold:.3f}):")
        print(f"    Layer {best['layer']} ({best['depth_pct']:.0%})")
        for beh in eval_behaviors:
            r = best["results"][beh]["rho"]
            b = baselines[beh]["rho"]
            print(f"    {beh}: ρ={r:.4f} (Δ={r-b:+.4f})")
    else:
        results["best_layer"] = None
        print("\n  No layer meets bias retention threshold.")

    _save(results, output_path)

    # ── Summary table ──
    print(f"\n{'='*60}")
    print(f"  HEATMAP SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Layer':>6} {'Depth':>6} {'Fact ρ':>8} {'Syc ρ':>8} {'Bias ρ':>8} {'ΔSyc':>8} {'ΔBias':>8}")
    print(f"  {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for pt in sweep_results:
        l = pt["layer"]
        d = pt["depth_pct"]
        fr = pt["results"]["factual"]["rho"]
        sr = pt["results"]["sycophancy"]["rho"]
        br = pt["results"]["bias"]["rho"]
        ds = sr - baselines["sycophancy"]["rho"]
        db = br - baselines["bias"]["rho"]
        marker = " ★" if results.get("best_layer") and l == results["best_layer"]["layer"] else ""
        print(f"  {l:>6} {d:>5.0%} {fr:>8.4f} {sr:>8.4f} {br:>8.4f} {ds:>+8.4f} {db:>+8.4f}{marker}")

    print(f"\n  Baselines: fact={baselines['factual']['rho']:.4f}, "
          f"syc={baselines['sycophancy']['rho']:.4f}, "
          f"bias={baselines['bias']['rho']:.4f}")
    print(f"  Results: {output_path}")

    # Cleanup
    del model, tokenizer
    free_memory()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Mistral layer heatmap: sycophancy vector sweep"
    )
    parser.add_argument(
        "--model", default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Model to sweep",
    )
    parser.add_argument("--alpha", type=float, default=4.0,
                        help="Steering alpha (default: 4.0)")
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    run_heatmap(
        model_id=args.model,
        alpha=args.alpha,
        device=args.device,
    )


if __name__ == "__main__":
    main()
