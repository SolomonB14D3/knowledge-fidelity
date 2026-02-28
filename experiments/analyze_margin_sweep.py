#!/usr/bin/env python3
"""Analyze margin sweep results and test the critical γ* prediction.

Theory predicts:
  γ* ≈ 0.024 is the minimum margin for bias preservation at λ_ρ=0.2.

  The interference ratio I(γ)/I(0) = γ/s_∞, where s_∞ is the asymptotic
  CE separation without margin. The margin γ is the sole controller of
  cross-dimensional interference, independent of λ_ρ, η, and T.

Usage:
  python experiments/analyze_margin_sweep.py
"""
import json
import sys
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("results/alignment")

# Known data points (from ablation_5seed_analysis)
KNOWN_POINTS = {
    0.00: {"d_bias": -0.011, "d_factual": +0.136, "d_toxicity": +0.560, "source": "ablation"},
    0.10: {"d_bias": +0.034, "d_factual": +0.163, "d_toxicity": +0.621, "source": "ablation"},
}

# Predicted critical margin
GAMMA_STAR = 0.0244

def load_sweep_results():
    """Load margin sweep JSON files."""
    points = dict(KNOWN_POINTS)

    for f in sorted(RESULTS_DIR.glob("margin_sweep_gamma*.json")):
        gamma_str = f.stem.split("gamma")[1]
        gamma = float(gamma_str)

        with open(f) as fh:
            data = json.load(fh)

        # Handle both schema variants
        if "runs" in data:
            runs = data["runs"]
        elif "merged_deltas" in data:
            md = data["merged_deltas"]
            # Find the rho-guided condition (or the only condition)
            for cond, beh_dict in md.items():
                n_seeds = len(next(iter(beh_dict.values())))
                d_bias_vals = beh_dict.get("bias", [0.0] * n_seeds)
                d_fact_vals = beh_dict.get("factual", [0.0] * n_seeds)
                d_tox_vals = beh_dict.get("toxicity", [0.0] * n_seeds)
                points[gamma] = {
                    "d_bias": float(np.mean(d_bias_vals)),
                    "d_bias_std": float(np.std(d_bias_vals, ddof=1)) if len(d_bias_vals) >= 2 else 0,
                    "d_factual": float(np.mean(d_fact_vals)),
                    "d_toxicity": float(np.mean(d_tox_vals)),
                    "n_seeds": n_seeds,
                    "source": f.name,
                }
                break
        else:
            # Try direct format: list of run dicts
            if isinstance(data, list):
                runs = data
            else:
                # Try to extract from whatever format this is
                runs = []
                for key, val in data.items():
                    if isinstance(val, dict) and "quick_deltas" in val:
                        runs.append(val)

            if runs:
                d_bias_vals = [r.get("quick_deltas", {}).get("bias", 0) for r in runs]
                d_fact_vals = [r.get("quick_deltas", {}).get("factual", 0) for r in runs]
                d_tox_vals = [r.get("quick_deltas", {}).get("toxicity", 0) for r in runs]
                points[gamma] = {
                    "d_bias": float(np.mean(d_bias_vals)),
                    "d_bias_std": float(np.std(d_bias_vals, ddof=1)) if len(d_bias_vals) >= 2 else 0,
                    "d_factual": float(np.mean(d_fact_vals)),
                    "d_toxicity": float(np.mean(d_tox_vals)),
                    "n_seeds": len(runs),
                    "source": f.name,
                }

    return points


def analyze(points):
    """Test the critical margin prediction."""
    print("=" * 70)
    print("MARGIN SWEEP ANALYSIS: Testing γ* ≈ 0.024 prediction")
    print("=" * 70)

    gammas = sorted(points.keys())
    print(f"\nData points: {len(gammas)} margin values")
    print(f"\n{'γ':>6s}  {'Δρ_bias':>8s}  {'Δρ_fact':>8s}  {'Δρ_tox':>8s}  {'Source'}")
    print("-" * 60)

    for g in gammas:
        p = points[g]
        std_str = f" ±{p.get('d_bias_std', 0):.4f}" if p.get('d_bias_std', 0) > 0 else ""
        print(f"{g:>6.3f}  {p['d_bias']:>+8.4f}{std_str}  {p['d_factual']:>+8.4f}  {p['d_toxicity']:>+8.4f}  {p.get('source', 'known')}")

    # Fit linear model to all bias data points
    gs = np.array(gammas)
    biases = np.array([points[g]["d_bias"] for g in gammas])

    if len(gs) >= 2:
        # Linear regression
        coeffs = np.polyfit(gs, biases, 1)
        slope, intercept = coeffs
        gamma_star_fit = -intercept / slope if slope != 0 else float('inf')

        print(f"\n--- LINEAR MODEL ---")
        print(f"Δρ_bias(γ) = {intercept:+.4f} + {slope:.3f}·γ")
        print(f"Fitted γ* (zero crossing): {gamma_star_fit:.4f}")
        print(f"Predicted γ* (from theory): {GAMMA_STAR:.4f}")
        print(f"Ratio (fitted/predicted): {gamma_star_fit/GAMMA_STAR:.2f}×")

        # How close is the prediction?
        error = abs(gamma_star_fit - GAMMA_STAR) / GAMMA_STAR * 100
        print(f"Prediction error: {error:.1f}%")

        if error < 50:
            print(f"\n✓ PREDICTION CONFIRMED (within 50%)")
        elif error < 100:
            print(f"\n~ PREDICTION PARTIALLY CONFIRMED (within 100%, order-of-magnitude correct)")
        else:
            print(f"\n✗ PREDICTION NOT CONFIRMED (off by >{error:.0f}%)")

    # Check the key qualitative prediction: γ=0.02 inverts, γ=0.03 preserves
    print(f"\n--- QUALITATIVE PREDICTION TEST ---")
    if 0.02 in points and 0.03 in points:
        bias_02 = points[0.02]["d_bias"]
        bias_03 = points[0.03]["d_bias"]
        print(f"γ=0.02: Δρ_bias = {bias_02:+.4f} ({'INVERTED ✓' if bias_02 < 0 else 'PRESERVED ✗'} — predicted: inverted)")
        print(f"γ=0.03: Δρ_bias = {bias_03:+.4f} ({'PRESERVED ✓' if bias_03 > 0 else 'INVERTED ✗'} — predicted: preserved)")

        if bias_02 < 0 and bias_03 > 0:
            print(f"\n✓ SHARP TRANSITION CONFIRMED between γ=0.02 and γ=0.03")
            print(f"  The critical margin lies in [0.02, 0.03], consistent with γ*=0.024")
        elif bias_02 < 0 and bias_03 < 0:
            print(f"\n~ γ* is higher than predicted (both still inverted)")
        elif bias_02 > 0 and bias_03 > 0:
            print(f"\n~ γ* is lower than predicted (both already preserved)")
    else:
        print("  (waiting for sweep results at γ=0.02, 0.03)")

    return gammas, biases


if __name__ == "__main__":
    points = load_sweep_results()
    analyze(points)
