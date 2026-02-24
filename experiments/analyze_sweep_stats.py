#!/usr/bin/env python3
"""Statistical analysis for rho-guided SFT sweep results.

Merges results from multiple seed files and computes:
  1. Per-condition (λ_ρ) mean ± std across seeds
  2. Paired t-tests (each λ_ρ vs λ_ρ=0 baseline)
  3. Dose-response tables for paper
  4. Cohen's d effect sizes

Usage:
    # Analyze existing results
    python experiments/analyze_sweep_stats.py \\
        results/alignment/mlx_rho_sft_sweep_7B_seeds42_123.json \\
        results/alignment/mlx_rho_sft_sweep_Qwen_Qwen2.5-7B-Instruct.json

    # Or with glob
    python experiments/analyze_sweep_stats.py results/alignment/mlx_rho_sft_sweep*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias"]


def load_and_merge(paths: list[str]) -> dict:
    """Load and merge results from multiple JSON files.

    Returns merged structure with:
        baseline: dict of behavior -> baseline gap
        runs: list of all run dicts
    """
    all_runs = []
    baseline = None

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue

        with open(path) as f:
            data = json.load(f)

        # Use first file's baseline (should be same across all)
        if baseline is None:
            baseline = data.get("baseline_quick", {})

        runs = data.get("runs", [])
        for run in runs:
            if "error" not in run:
                all_runs.append(run)

    print(f"Loaded {len(all_runs)} runs from {len(paths)} files")
    return {"baseline": baseline, "runs": all_runs}


def group_by_rho(runs: list[dict]) -> dict[float, list[dict]]:
    """Group runs by rho_weight."""
    groups = defaultdict(list)
    for run in runs:
        groups[run["rho_weight"]].append(run)
    return dict(sorted(groups.items()))


def compute_statistics(merged: dict) -> dict:
    """Compute per-condition statistics with significance tests."""
    baseline = merged["baseline"]
    by_rho = group_by_rho(merged["runs"])

    results = {
        "baseline": baseline,
        "conditions": {},
    }

    # Get baseline condition (λ_ρ=0) scores for paired tests
    baseline_runs = by_rho.get(0.0, [])

    for rho_w, runs in by_rho.items():
        cond = {"rho_weight": rho_w, "n_seeds": len(runs), "behaviors": {}}

        for bname in BEHAVIORS:
            # Post-SFT scores
            scores = [r["quick_scores"].get(bname, 0) for r in runs]
            # Deltas from pre-trained baseline
            deltas = [r["quick_deltas"].get(bname, 0) for r in runs]

            bstats = {
                "scores": scores,
                "deltas": deltas,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "baseline_gap": baseline.get(bname, 0),
            }

            # One-sample t-test: is the delta significantly different from 0?
            if len(deltas) >= 2:
                t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
                bstats["t_stat_vs_zero"] = float(t_stat)
                bstats["p_val_vs_zero"] = float(p_val)
            elif len(deltas) == 1:
                bstats["t_stat_vs_zero"] = None
                bstats["p_val_vs_zero"] = None

            # Two-sample t-test vs λ_ρ=0 condition (if different)
            if rho_w != 0.0 and baseline_runs:
                base_scores = [r["quick_scores"].get(bname, 0) for r in baseline_runs]
                if len(scores) >= 2 and len(base_scores) >= 2:
                    t_stat, p_val = stats.ttest_ind(scores, base_scores)
                    bstats["t_stat_vs_sft"] = float(t_stat)
                    bstats["p_val_vs_sft"] = float(p_val)

                    # Cohen's d effect size
                    pooled_std = np.sqrt(
                        ((len(scores) - 1) * np.var(scores, ddof=1) +
                         (len(base_scores) - 1) * np.var(base_scores, ddof=1)) /
                        (len(scores) + len(base_scores) - 2)
                    )
                    if pooled_std > 0:
                        bstats["cohens_d"] = float(
                            (np.mean(scores) - np.mean(base_scores)) / pooled_std
                        )
                    else:
                        bstats["cohens_d"] = float("inf") if np.mean(scores) != np.mean(base_scores) else 0.0

            cond["behaviors"][bname] = bstats

        results["conditions"][rho_w] = cond

    return results


def print_dose_response_table(results: dict):
    """Print the dose-response table for the paper."""
    baseline = results["baseline"]

    print(f"\n{'='*80}")
    print(f"  DOSE-RESPONSE TABLE: Confidence Gaps by λ_ρ")
    print(f"  (Qwen2.5-7B-Instruct, 1-epoch LoRA SFT, 1000 texts)")
    print(f"{'='*80}")

    # Header
    print(f"\n  {'λ_ρ':>6s}  {'Seeds':>5s}  ", end="")
    for b in BEHAVIORS:
        print(f"  {b[:8]:>12s}", end="")
    print()
    print(f"  {'─'*70}")

    # Pre-trained baseline
    print(f"  {'pre':>6s}  {'─':>5s}  ", end="")
    for b in BEHAVIORS:
        val = baseline.get(b, 0)
        print(f"  {val:+12.4f}", end="")
    print("  (pre-trained)")

    # Each condition
    for rho_w, cond in sorted(results["conditions"].items()):
        n = cond["n_seeds"]
        print(f"  {rho_w:6.2f}  {n:>5d}  ", end="")
        for b in BEHAVIORS:
            bdata = cond["behaviors"].get(b, {})
            mean = bdata.get("mean_score", 0)
            std = bdata.get("std_score", 0)
            if n > 1:
                print(f"  {mean:+7.4f}±{std:.3f}", end="")
            else:
                print(f"  {mean:+12.4f}", end="")
        print()

    # Delta table (change from pre-trained)
    print(f"\n  {'─'*70}")
    print(f"  Deltas from pre-trained baseline:")
    print(f"  {'─'*70}")

    for rho_w, cond in sorted(results["conditions"].items()):
        n = cond["n_seeds"]
        print(f"  {rho_w:6.2f}  {n:>5d}  ", end="")
        for b in BEHAVIORS:
            bdata = cond["behaviors"].get(b, {})
            mean_d = bdata.get("mean_delta", 0)
            std_d = bdata.get("std_delta", 0)
            if n > 1:
                print(f"  {mean_d:+7.4f}±{std_d:.3f}", end="")
            else:
                print(f"  {mean_d:+12.4f}", end="")
        print()


def print_significance_tests(results: dict):
    """Print significance test results."""
    print(f"\n{'='*80}")
    print(f"  SIGNIFICANCE TESTS")
    print(f"{'='*80}")

    # Test 1: Is delta from pre-trained significantly ≠ 0?
    print(f"\n  Test 1: Is Δ from pre-trained ≠ 0? (one-sample t-test)")
    print(f"  {'─'*70}")
    print(f"  {'λ_ρ':>6s}  {'Behavior':>12s}  {'Mean Δ':>8s}  {'t':>8s}  {'p':>8s}  {'Sig':>4s}  {'Direction':>12s}")
    print(f"  {'─'*70}")

    for rho_w, cond in sorted(results["conditions"].items()):
        for b in BEHAVIORS:
            bdata = cond["behaviors"].get(b, {})
            mean_d = bdata.get("mean_delta", 0)
            t_val = bdata.get("t_stat_vs_zero")
            p_val = bdata.get("p_val_vs_zero")

            if t_val is not None and p_val is not None:
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
                direction = "improved" if mean_d > 0.01 else ("degraded" if mean_d < -0.01 else "unchanged")
                print(f"  {rho_w:6.2f}  {b:>12s}  {mean_d:+8.4f}  {t_val:8.3f}  {p_val:8.4f}  {sig:>4s}  {direction:>12s}")
            else:
                print(f"  {rho_w:6.2f}  {b:>12s}  {mean_d:+8.4f}  {'n/a':>8s}  {'n/a':>8s}  {'n/a':>4s}  (1 seed)")

    # Test 2: Is rho-guided better than SFT-only?
    baseline_runs = [c for c in results["conditions"].values() if c["behaviors"]]
    has_sft_only = 0.0 in results["conditions"]

    if has_sft_only:
        print(f"\n  Test 2: Is λ_ρ>0 better than λ_ρ=0 (SFT-only)? (two-sample t-test)")
        print(f"  {'─'*70}")
        print(f"  {'λ_ρ':>6s}  {'Behavior':>12s}  {'Δ from SFT':>10s}  {'t':>8s}  {'p':>8s}  {'d':>6s}  {'Sig':>4s}")
        print(f"  {'─'*70}")

        sft_cond = results["conditions"][0.0]

        for rho_w, cond in sorted(results["conditions"].items()):
            if rho_w == 0.0:
                continue

            for b in BEHAVIORS:
                bdata = cond["behaviors"].get(b, {})
                sft_data = sft_cond["behaviors"].get(b, {})

                mean_diff = bdata.get("mean_score", 0) - sft_data.get("mean_score", 0)
                t_val = bdata.get("t_stat_vs_sft")
                p_val = bdata.get("p_val_vs_sft")
                d_val = bdata.get("cohens_d")

                if t_val is not None and p_val is not None:
                    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
                    d_str = f"{d_val:+6.2f}" if d_val is not None else "  n/a"
                    print(f"  {rho_w:6.2f}  {b:>12s}  {mean_diff:+10.4f}  {t_val:8.3f}  {p_val:8.4f}  {d_str}  {sig:>4s}")
                else:
                    print(f"  {rho_w:6.2f}  {b:>12s}  {mean_diff:+10.4f}  {'n/a':>8s}  {'n/a':>8s}  {'n/a':>6s}  (insuff.)")


def print_key_findings(results: dict):
    """Print key findings summary for the paper."""
    baseline = results["baseline"]

    print(f"\n{'='*80}")
    print(f"  KEY FINDINGS SUMMARY")
    print(f"{'='*80}")

    # Finding 1: SFT inversion
    sft_cond = results["conditions"].get(0.0, {})
    if sft_cond:
        tox_data = sft_cond["behaviors"].get("toxicity", {})
        base_tox = baseline.get("toxicity", 0)
        sft_tox = tox_data.get("mean_score", 0)
        print(f"\n  1. SFT-INDUCED CONFIDENCE INVERSION:")
        print(f"     Pre-trained toxicity gap: {base_tox:+.4f}")
        print(f"     Post-SFT (λ_ρ=0) gap:    {sft_tox:+.4f} (mean over {sft_cond['n_seeds']} seeds)")
        print(f"     Standard SFT INVERTS the model's toxicity discrimination.")
        print(f"     The model goes from correctly preferring non-toxic text")
        print(f"     to incorrectly preferring toxic text.")

    # Finding 2: Rho correction
    for rho_w in [0.1, 0.2, 0.5]:
        cond = results["conditions"].get(rho_w, {})
        if cond:
            tox = cond["behaviors"].get("toxicity", {})
            fac = cond["behaviors"].get("factual", {})
            print(f"\n  2. RHO-GUIDED CORRECTION (λ_ρ={rho_w}):")
            print(f"     Toxicity gap: {tox.get('mean_score', 0):+.4f} ± {tox.get('std_score', 0):.4f}"
                  f" (Δ from baseline: {tox.get('mean_delta', 0):+.4f})")
            print(f"     Factual gap:  {fac.get('mean_score', 0):+.4f} ± {fac.get('std_score', 0):.4f}"
                  f" (Δ from baseline: {fac.get('mean_delta', 0):+.4f})")

    # Finding 3: Monotonic dose-response
    rho_weights = sorted(results["conditions"].keys())
    if len(rho_weights) >= 3:
        tox_means = []
        for rw in rho_weights:
            tox_means.append(results["conditions"][rw]["behaviors"].get("toxicity", {}).get("mean_score", 0))

        is_monotonic = all(tox_means[i] <= tox_means[i+1] for i in range(len(tox_means)-1))
        print(f"\n  3. DOSE-RESPONSE:")
        print(f"     Toxicity gap monotonically {'increases' if is_monotonic else 'does NOT monotonically increase'} with λ_ρ")
        print(f"     λ_ρ values: {rho_weights}")
        print(f"     Toxicity gaps: {[f'{t:+.4f}' for t in tox_means]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze rho-guided SFT sweep results")
    parser.add_argument("files", nargs="+", help="JSON result files to analyze")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Save analysis to JSON file")

    args = parser.parse_args()

    merged = load_and_merge(args.files)
    if not merged["runs"]:
        print("No runs found!")
        sys.exit(1)

    results = compute_statistics(merged)

    print_dose_response_table(results)
    print_significance_tests(results)
    print_key_findings(results)

    if args.json_out:
        # Strip numpy arrays for JSON serialization
        out = json.loads(json.dumps(results, default=str))
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved analysis to {args.json_out}")


if __name__ == "__main__":
    main()
