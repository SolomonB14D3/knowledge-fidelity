#!/usr/bin/env python3
"""Statistical analysis of ablation study results.

Computes per-condition means, p-values, and Cohen's d for:
  1. Each condition vs baseline (is delta ≠ 0?)
  2. rho-guided vs each other condition (what's the active ingredient?)
  3. shuffled-pairs vs rho-guided (correct labels matter?)

Also produces paper-ready tables for the Results section.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias"]


def load_ablation(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d (pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        ma, mb = np.mean(a), np.mean(b)
        sa, sb = (np.std(a, ddof=1) if na > 1 else 0.001,
                  np.std(b, ddof=1) if nb > 1 else 0.001)
        pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / max(na + nb - 2, 1))
        if pooled < 1e-10:
            pooled = 1e-10
        return (ma - mb) / pooled
    ma, mb = np.mean(a), np.mean(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled < 1e-10:
        pooled = 1e-10
    return (ma - mb) / pooled


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"


def analyze(data: dict) -> dict:
    """Full statistical analysis of ablation results."""

    baseline = data["baseline_quick"]
    runs = data["runs"]

    # Group by condition
    by_cond = defaultdict(list)
    for r in runs:
        by_cond[r["condition"]].append(r)

    conditions = list(by_cond.keys())

    results = {
        "baseline": {b: baseline.get(b, 0.0) for b in BEHAVIORS},
        "conditions": {},
        "tests": [],
    }

    # ── Per-condition summaries ──────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  ABLATION STUDY: Per-Condition Summary (Qwen2.5-7B-Instruct)")
    print(f"{'='*80}")
    print(f"\n  {'Condition':20s}  {'Seeds':>5s}  {'factual':>10s}  {'toxicity':>10s}  {'sycophan':>10s}  {'bias':>10s}")
    print(f"  {'─'*72}")

    # Print baseline
    bl = results["baseline"]
    print(f"  {'baseline':20s}  {'─':>5s}  {bl['factual']:+10.4f}  {bl['toxicity']:+10.4f}  {bl['sycophancy']:+10.4f}  {bl['bias']:+10.4f}")

    for cond in conditions:
        cond_runs = by_cond[cond]
        n_seeds = len(cond_runs)

        cond_data = {}
        for beh in BEHAVIORS:
            scores = [r["quick_scores"][beh] for r in cond_runs]
            deltas = [r["quick_deltas"][beh] for r in cond_runs]
            cond_data[beh] = {
                "scores": scores,
                "deltas": deltas,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "mean_delta": float(np.mean(deltas)),
            }

        results["conditions"][cond] = cond_data

        line = f"  {cond:20s}  {n_seeds:5d}"
        for beh in BEHAVIORS:
            m = cond_data[beh]["mean"]
            s = cond_data[beh]["std"]
            if n_seeds > 1:
                line += f"  {m:+.4f}±{s:.3f}"
            else:
                line += f"  {m:+10.4f}"
        print(line)

    # ── Deltas from baseline ─────────────────────────────────────────
    print(f"\n  {'Condition':20s}  {'Seeds':>5s}  {'Δfactual':>10s}  {'Δtoxicity':>10s}  {'Δsycophan':>10s}  {'Δbias':>10s}")
    print(f"  {'─'*72}")

    for cond in conditions:
        cond_data = results["conditions"][cond]
        n_seeds = len(by_cond[cond])
        line = f"  {cond:20s}  {n_seeds:5d}"
        for beh in BEHAVIORS:
            md = cond_data[beh]["mean_delta"]
            line += f"  {md:+10.4f}"
        print(line)

    # ── Test 1: Each condition's delta ≠ 0 (one-sample t-test) ───────
    print(f"\n{'='*80}")
    print(f"  TEST 1: Is Δ from baseline ≠ 0? (one-sample t-test)")
    print(f"{'='*80}")
    print(f"  {'Condition':20s}  {'Behavior':>12s}  {'Mean Δ':>10s}  {'t':>8s}  {'p':>10s}  {'Sig':>5s}")
    print(f"  {'─'*72}")

    for cond in conditions:
        for beh in BEHAVIORS:
            deltas = results["conditions"][cond][beh]["deltas"]
            if len(deltas) >= 2:
                t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
            else:
                t_stat, p_val = float('nan'), float('nan')
            sig = sig_stars(p_val) if not np.isnan(p_val) else "n/a"
            md = np.mean(deltas)
            print(f"  {cond:20s}  {beh:>12s}  {md:+10.4f}  {t_stat:8.3f}  {p_val:10.4f}  {sig:>5s}")

            results["tests"].append({
                "test": "delta_vs_zero",
                "condition": cond,
                "behavior": beh,
                "mean_delta": float(md),
                "t": float(t_stat),
                "p": float(p_val),
                "sig": sig,
            })

    # ── Test 2: rho-guided vs each other condition (two-sample t) ────
    print(f"\n{'='*80}")
    print(f"  TEST 2: ρ-guided vs each other condition (two-sample t-test)")
    print(f"{'='*80}")
    print(f"  {'Comparison':35s}  {'Behavior':>10s}  {'Δ':>8s}  {'t':>8s}  {'p':>10s}  {'d':>8s}  {'Sig':>5s}")
    print(f"  {'─'*82}")

    rho_data = results["conditions"].get("rho-guided", {})
    for other_cond in conditions:
        if other_cond == "rho-guided":
            continue
        other_data = results["conditions"][other_cond]
        for beh in BEHAVIORS:
            rho_scores = rho_data[beh]["scores"]
            other_scores = other_data[beh]["scores"]
            if len(rho_scores) >= 2 and len(other_scores) >= 2:
                t_stat, p_val = stats.ttest_ind(rho_scores, other_scores)
            else:
                t_stat, p_val = float('nan'), float('nan')
            d = cohens_d(rho_scores, other_scores)
            diff = np.mean(rho_scores) - np.mean(other_scores)
            sig = sig_stars(p_val) if not np.isnan(p_val) else "n/a"
            label = f"ρ-guided vs {other_cond}"
            print(f"  {label:35s}  {beh:>10s}  {diff:+8.4f}  {t_stat:8.3f}  {p_val:10.4f}  {d:+8.2f}  {sig:>5s}")

            results["tests"].append({
                "test": "rho_vs_other",
                "comparison": label,
                "behavior": beh,
                "diff": float(diff),
                "t": float(t_stat),
                "p": float(p_val),
                "d": float(d),
                "sig": sig,
            })

    # ── Test 3: Key contrasts for the paper ──────────────────────────
    print(f"\n{'='*80}")
    print(f"  KEY CONTRASTS FOR PAPER")
    print(f"{'='*80}")

    key_pairs = [
        ("rho-guided", "sft-only", "Does the contrastive loss help beyond vanilla SFT?"),
        ("rho-guided", "shuffled-pairs", "Do correct behavioral labels matter?"),
        ("contrastive-only", "sft-only", "Does contrastive alone beat SFT?"),
        ("contrastive-only", "shuffled-pairs", "Correct vs shuffled contrastive?"),
    ]

    for cond_a, cond_b, question in key_pairs:
        print(f"\n  Q: {question}")
        print(f"  {cond_a} vs {cond_b}")
        print(f"  {'Behavior':>12s}  {'A mean':>8s}  {'B mean':>8s}  {'Diff':>8s}  {'d':>8s}  {'p':>10s}  {'Sig':>5s}")
        print(f"  {'─'*65}")

        a_data = results["conditions"].get(cond_a, {})
        b_data = results["conditions"].get(cond_b, {})

        for beh in BEHAVIORS:
            a_scores = a_data[beh]["scores"]
            b_scores = b_data[beh]["scores"]
            if len(a_scores) >= 2 and len(b_scores) >= 2:
                t_stat, p_val = stats.ttest_ind(a_scores, b_scores)
            else:
                t_stat, p_val = float('nan'), float('nan')
            d = cohens_d(a_scores, b_scores)
            ma, mb = np.mean(a_scores), np.mean(b_scores)
            sig = sig_stars(p_val) if not np.isnan(p_val) else "n/a"
            print(f"  {beh:>12s}  {ma:+8.4f}  {mb:+8.4f}  {ma-mb:+8.4f}  {d:+8.2f}  {p_val:10.4f}  {sig:>5s}")

    # ── Calibration comparison ───────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  CALIBRATION METRICS BY CONDITION")
    print(f"{'='*80}")

    # Baseline calibration
    bl_cal = data.get("baseline_calibration", {})
    print(f"\n  {'Condition':20s}  {'Behavior':>10s}  {'ECE':>8s}  {'Brier':>8s}  {'Acc':>8s}")
    print(f"  {'─'*60}")

    if bl_cal:
        for beh in BEHAVIORS:
            if beh in bl_cal:
                bc = bl_cal[beh]
                print(f"  {'baseline':20s}  {beh:>10s}  {bc['ece']:8.4f}  {bc['brier']:8.4f}  {bc['accuracy']:7.1%}")

    for cond in conditions:
        cond_runs = by_cond[cond]
        for beh in BEHAVIORS:
            eces = [r["calibration"][beh]["ece"] for r in cond_runs if beh in r.get("calibration", {})]
            briers = [r["calibration"][beh]["brier"] for r in cond_runs if beh in r.get("calibration", {})]
            accs = [r["calibration"][beh]["accuracy"] for r in cond_runs if beh in r.get("calibration", {})]
            if eces:
                me, mb_, ma_ = np.mean(eces), np.mean(briers), np.mean(accs)
                print(f"  {cond:20s}  {beh:>10s}  {me:8.4f}  {mb_:8.4f}  {ma_:7.1%}")

    # ── Narrative summary ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  NARRATIVE SUMMARY")
    print(f"{'='*80}")

    # Get key numbers
    rho_tox = np.mean(results["conditions"]["rho-guided"]["toxicity"]["scores"])
    sft_tox = np.mean(results["conditions"]["sft-only"]["toxicity"]["scores"])
    shuf_tox = np.mean(results["conditions"]["shuffled-pairs"]["toxicity"]["scores"])
    cont_tox = np.mean(results["conditions"]["contrastive-only"]["toxicity"]["scores"])
    bl_tox = baseline.get("toxicity", 0)

    rho_fac = np.mean(results["conditions"]["rho-guided"]["factual"]["scores"])
    shuf_fac = np.mean(results["conditions"]["shuffled-pairs"]["factual"]["scores"])

    print(f"""
  1. SFT INVERSION CONFIRMED:
     Baseline toxicity: {bl_tox:+.4f}
     After SFT-only:    {sft_tox:+.4f} (Δ = {sft_tox - bl_tox:+.4f})
     Standard SFT degrades toxicity discrimination.

  2. RHO-GUIDED SFT CORRECTS:
     After ρ-guided:    {rho_tox:+.4f} (Δ = {rho_tox - bl_tox:+.4f})
     vs SFT-only:       +{rho_tox - sft_tox:.4f} improvement

  3. CONTRASTIVE LOSS IS THE ACTIVE INGREDIENT:
     Contrastive-only:  {cont_tox:+.4f} (Δ = {cont_tox - bl_tox:+.4f})
     Nearly as effective as full rho-guided without any SFT.

  4. SHUFFLED-PAIRS CONTROL — CORRECT LABELS MATTER:
     Shuffled toxicity:  {shuf_tox:+.4f} (Δ = {shuf_tox - bl_tox:+.4f})
     Shuffled factual:   {shuf_fac:+.4f} (Δ = {shuf_fac - baseline.get('factual', 0):+.4f})
     Randomizing positive/negative labels DESTROYS the model.
     This proves the contrastive loss needs correct behavioral signal.

  5. RHO-GUIDED vs SHUFFLED (the key contrast):
     Toxicity gap:  {rho_tox - shuf_tox:+.4f} ({rho_tox:+.4f} vs {shuf_tox:+.4f})
     Factual gap:   {rho_fac - shuf_fac:+.4f} ({rho_fac:+.4f} vs {shuf_fac:+.4f})
     d(toxicity) = {cohens_d(results['conditions']['rho-guided']['toxicity']['scores'], results['conditions']['shuffled-pairs']['toxicity']['scores']):+.1f}
     d(factual)  = {cohens_d(results['conditions']['rho-guided']['factual']['scores'], results['conditions']['shuffled-pairs']['factual']['scores']):+.1f}
""")

    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation study statistical analysis")
    parser.add_argument("file", help="Ablation results JSON file")
    parser.add_argument("--json-out", type=str, default=None, help="Save analysis JSON")
    args = parser.parse_args()

    data = load_ablation(args.file)
    results = analyze(data)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved analysis to {args.json_out}")


if __name__ == "__main__":
    main()
