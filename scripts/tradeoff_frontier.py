#!/usr/bin/env python3
"""Trade-off frontier analysis at high dose (λ=0.5).

Plots factual vs sycophancy across all seeds and γ values at λ=0.5,
and extends to multi-dose fronts. Also does the decoupler validation:
checks factual↔toxicity correlation as f(γ) with existing data.

Usage:
  python scripts/tradeoff_frontier.py [--refresh-db]
"""

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "results" / "master.db"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

CORE_BEHAVIORS = ["factual", "toxicity", "bias", "sycophancy"]


def safe_float(val, default=np.nan):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def savefig(fig, name):
    path = DOCS_DIR / name
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Trade-Off Frontier at λ=0.5
# ═══════════════════════════════════════════════════════════════════════════

def trade_off_frontier(conn):
    """Plot factual vs sycophancy across seeds at λ=0.5, colored by γ."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Trade-Off Frontier at λ=0.5")
    print("=" * 70)

    # Get all runs at λ=0.5 (both margins)
    rows_margin = conn.execute("""
        SELECT seed, d_factual, d_sycophancy, d_toxicity, d_bias, margin
        FROM alignment_runs
        WHERE model LIKE '%Qwen2.5-7B%'
        AND d_factual IS NOT NULL
        AND condition = '' AND rho_weight = 0.5
        ORDER BY seed
    """).fetchall()

    # Also get λ=0.2 for comparison
    rows_02 = conn.execute("""
        SELECT seed, d_factual, d_sycophancy, d_toxicity, d_bias, margin
        FROM alignment_runs
        WHERE model LIKE '%Qwen2.5-7B%'
        AND d_factual IS NOT NULL
        AND condition = '' AND rho_weight = 0.2
        ORDER BY seed
    """).fetchall()

    # Also get no-margin runs (γ=0, λ=0.2)
    rows_no_margin = conn.execute("""
        SELECT seed, d_factual, d_sycophancy, d_toxicity, d_bias
        FROM alignment_runs
        WHERE experiment LIKE 'mlx_no_margin%'
        AND d_factual IS NOT NULL
        ORDER BY seed
    """).fetchall()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── Panel 1: Factual vs Sycophancy at λ=0.5 (per seed) ──
    ax = axes[0, 0]
    for r in rows_margin:
        margin = safe_float(r["margin"])
        fact = safe_float(r["d_factual"])
        syc = safe_float(r["d_sycophancy"])
        if np.isnan(margin) or np.isnan(fact) or np.isnan(syc):
            continue

        color = "green" if margin > 0.05 else "red"
        marker = "o" if margin > 0.05 else "s"
        label_prefix = "γ=0.1" if margin > 0.05 else "γ=0"
        ax.scatter(fact, syc, c=color, marker=marker, s=80,
                  edgecolors="black", linewidths=0.5, alpha=0.8,
                  label=f"{label_prefix}, s{r['seed']}" if r == rows_margin[0] or margin < 0.05 else "")
        ax.annotate(f"s{r['seed']}", (fact, syc), fontsize=7,
                   textcoords="offset points", xytext=(5, 5))

    # Fit line through γ=0.1 points
    facts_g01 = [safe_float(r["d_factual"]) for r in rows_margin if safe_float(r["margin"]) > 0.05]
    sycs_g01 = [safe_float(r["d_sycophancy"]) for r in rows_margin if safe_float(r["margin"]) > 0.05]
    valid = [(f, s) for f, s in zip(facts_g01, sycs_g01) if not np.isnan(f) and not np.isnan(s)]
    if len(valid) >= 3:
        fs, ss = zip(*valid)
        rho, p = stats.spearmanr(fs, ss)
        # Linear fit for visualization
        coeff = np.polyfit(fs, ss, 1)
        x_fit = np.linspace(min(fs), max(fs), 50)
        ax.plot(x_fit, np.polyval(coeff, x_fit), "g--", alpha=0.5,
               label=f"γ=0.1 fit: ρ={rho:+.3f}")

    ax.set_xlabel("Δρ_factual")
    ax.set_ylabel("Δρ_sycophancy")
    ax.set_title(f"Trade-Off Frontier at λ=0.5\nfactual↔sycophancy ρ={rho:+.3f} (p={p:.3f})")
    ax.legend(fontsize=7, loc="best")

    # ── Panel 2: All behaviors at λ=0.5, each seed is a point in 4D ──
    ax = axes[0, 1]

    seed_data_05 = defaultdict(dict)
    for r in rows_margin:
        if safe_float(r["margin"]) > 0.05:  # γ=0.1 only
            seed = r["seed"]
            for b in CORE_BEHAVIORS:
                seed_data_05[seed][b] = safe_float(r[f"d_{b}"])

    # Parallel coordinates style
    seeds = sorted(seed_data_05.keys())
    seed_colors = {42: "C0", 123: "C1", 456: "C2", 789: "C3", 1337: "C4"}

    for seed in seeds:
        vals = [seed_data_05[seed].get(b, np.nan) for b in CORE_BEHAVIORS]
        color = seed_colors.get(seed, "gray")
        ax.plot(range(len(CORE_BEHAVIORS)), vals, "o-", color=color,
               label=f"s{seed}", markersize=6, alpha=0.8)

    ax.set_xticks(range(len(CORE_BEHAVIORS)))
    ax.set_xticklabels(CORE_BEHAVIORS, fontsize=9)
    ax.set_ylabel("Δρ")
    ax.set_title("Per-Seed Behavioral Profile at λ=0.5, γ=0.1")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # ── Panel 3: Multi-dose factual-sycophancy frontier ──
    ax = axes[1, 0]

    for rw_val, rows_rw, color, marker in [
        (0.2, rows_02, "blue", "^"),
        (0.5, rows_margin, "green", "o"),
    ]:
        for r in rows_rw:
            if safe_float(r["margin"]) > 0.05:
                fact = safe_float(r["d_factual"])
                syc = safe_float(r["d_sycophancy"])
                if not np.isnan(fact) and not np.isnan(syc):
                    ax.scatter(fact, syc, c=color, marker=marker, s=60,
                              edgecolors="black", linewidths=0.5, alpha=0.7)

    # No-margin runs
    for r in rows_no_margin:
        fact = safe_float(r["d_factual"])
        syc = safe_float(r["d_sycophancy"])
        if not np.isnan(fact) and not np.isnan(syc):
            ax.scatter(fact, syc, c="red", marker="s", s=60,
                      edgecolors="black", linewidths=0.5, alpha=0.7)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=8, label="γ=0, λ=0.2"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="blue",
               markeredgecolor="black", markersize=8, label="γ=0.1, λ=0.2"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
               markeredgecolor="black", markersize=8, label="γ=0.1, λ=0.5"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    ax.set_xlabel("Δρ_factual")
    ax.set_ylabel("Δρ_sycophancy")
    ax.set_title("Multi-Dose Trade-Off Frontier\nFactual vs Sycophancy")

    # ── Panel 4: Factual vs Toxicity colored by (γ, λ) ──
    ax = axes[1, 1]

    for r in rows_margin:
        margin = safe_float(r["margin"])
        fact = safe_float(r["d_factual"])
        tox = safe_float(r["d_toxicity"])
        if np.isnan(fact) or np.isnan(tox):
            continue
        color = "green" if margin > 0.05 else "red"
        marker = "o" if margin > 0.05 else "s"
        ax.scatter(fact, tox, c=color, marker=marker, s=60,
                  edgecolors="black", linewidths=0.5, alpha=0.7)

    for r in rows_02:
        if safe_float(r["margin"]) > 0.05:
            fact = safe_float(r["d_factual"])
            tox = safe_float(r["d_toxicity"])
            if not np.isnan(fact) and not np.isnan(tox):
                ax.scatter(fact, tox, c="blue", marker="^", s=60,
                          edgecolors="black", linewidths=0.5, alpha=0.7)

    for r in rows_no_margin:
        fact = safe_float(r["d_factual"])
        tox = safe_float(r["d_toxicity"])
        if not np.isnan(fact) and not np.isnan(tox):
            ax.scatter(fact, tox, c="red", marker="s", s=60,
                      edgecolors="black", linewidths=0.5, alpha=0.7)
            ax.annotate(f"s{r['seed']}", (fact, tox), fontsize=6,
                       textcoords="offset points", xytext=(4, 4), color="red")

    # Compute correlations for each group
    for group_name, group_rows, margin_filter in [
        ("γ=0", rows_no_margin, None),
        ("γ=0.1, λ=0.2", rows_02, 0.1),
        ("γ=0.1, λ=0.5", rows_margin, 0.1),
    ]:
        if margin_filter is not None:
            fs = [safe_float(r["d_factual"]) for r in group_rows if safe_float(r["margin"]) > 0.05]
            ts = [safe_float(r["d_toxicity"]) for r in group_rows if safe_float(r["margin"]) > 0.05]
        else:
            fs = [safe_float(r["d_factual"]) for r in group_rows]
            ts = [safe_float(r["d_toxicity"]) for r in group_rows]

        valid = [(f, t) for f, t in zip(fs, ts) if not np.isnan(f) and not np.isnan(t)]
        if len(valid) >= 4:
            fv, tv = zip(*valid)
            r, p = stats.spearmanr(fv, tv)
            print(f"  {group_name}: factual↔toxicity ρ={r:+.3f} (p={p:.3f}, n={len(valid)})")

    ax.set_xlabel("Δρ_factual")
    ax.set_ylabel("Δρ_toxicity")
    ax.set_title("Factual vs Toxicity\n(γ decoupling effect)")
    ax.legend(handles=legend_elements, fontsize=8)

    fig.suptitle("Trade-Off Frontier Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "tradeoff_frontier.png")

    return {
        "factual_syc_rho_05": float(rho) if 'rho' in dir() else None,
        "factual_syc_p_05": float(p) if 'p' in dir() else None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Margin Decoupler Validation
# ═══════════════════════════════════════════════════════════════════════════

def decoupler_validation(conn):
    """Validate the margin as dimensional decoupler with existing data."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Margin Decoupler Validation")
    print("=" * 70)

    # We have two γ values: 0.0 and 0.1, both at λ=0.2
    # Compute within-group correlations for all behavior pairs

    # γ=0 data (no-margin experiment)
    rows_g0 = conn.execute("""
        SELECT seed, d_factual, d_toxicity, d_bias, d_sycophancy, d_refusal
        FROM alignment_runs
        WHERE experiment LIKE 'mlx_no_margin%'
        AND d_factual IS NOT NULL
        ORDER BY seed
    """).fetchall()

    # γ=0.1, λ=0.2 data
    rows_g1 = conn.execute("""
        SELECT seed, d_factual, d_toxicity, d_bias, d_sycophancy, d_refusal
        FROM alignment_runs
        WHERE model LIKE '%Qwen2.5-7B%'
        AND d_factual IS NOT NULL
        AND condition = '' AND rho_weight = 0.2 AND margin = 0.1
        ORDER BY seed
    """).fetchall()

    behs = ["factual", "toxicity", "bias", "sycophancy"]
    n_behs = len(behs)
    pairs = [(i, j) for i in range(n_behs) for j in range(i + 1, n_behs)]

    results = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for gamma_idx, (gamma_label, rows_g) in enumerate([("γ=0.0", rows_g0), ("γ=0.1", rows_g1)]):
        ax = axes[gamma_idx]

        data = {}
        for b in behs:
            vals = [safe_float(r[f"d_{b}"]) for r in rows_g]
            data[b] = np.array(vals)

        corr_matrix = np.full((n_behs, n_behs), np.nan)
        p_matrix = np.full((n_behs, n_behs), np.nan)

        for i in range(n_behs):
            for j in range(n_behs):
                x = data[behs[i]]
                y = data[behs[j]]
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() >= 4:
                    r, p = stats.spearmanr(x[mask], y[mask])
                    corr_matrix[i, j] = r
                    p_matrix[i, j] = p

        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(n_behs))
        ax.set_xticklabels(behs, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_behs))
        ax.set_yticklabels(behs, fontsize=9)
        ax.set_title(f"{gamma_label} (n={len(rows_g)})")

        for i in range(n_behs):
            for j in range(n_behs):
                if not np.isnan(corr_matrix[i, j]):
                    sig = "*" if p_matrix[i, j] < 0.05 else ""
                    ax.text(j, i, f"{corr_matrix[i,j]:+.2f}{sig}", ha="center", va="center",
                           fontsize=9, color="white" if abs(corr_matrix[i,j]) > 0.5 else "black")

        results[gamma_label] = corr_matrix.copy()
        print(f"\n  {gamma_label}:")
        for i, j in pairs:
            val = corr_matrix[i, j]
            p_val = p_matrix[i, j]
            sig = "*" if p_val < 0.05 else ""
            print(f"    {behs[i]}↔{behs[j]}: ρ={val:+.3f} (p={p_val:.3f}){sig}")

    # Panel 3: Difference matrix (decoupling magnitude)
    ax = axes[2]
    if "γ=0.0" in results and "γ=0.1" in results:
        diff = results["γ=0.1"] - results["γ=0.0"]
        im = ax.imshow(diff, cmap="PiYG", aspect="equal",
                       vmin=-np.nanmax(np.abs(diff)), vmax=np.nanmax(np.abs(diff)))
        ax.set_xticks(range(n_behs))
        ax.set_xticklabels(behs, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_behs))
        ax.set_yticklabels(behs, fontsize=9)
        ax.set_title("Δcorrelation (γ=0.1 − γ=0.0)")
        plt.colorbar(im, ax=ax, shrink=0.8)

        for i in range(n_behs):
            for j in range(n_behs):
                if not np.isnan(diff[i, j]):
                    ax.text(j, i, f"{diff[i,j]:+.2f}", ha="center", va="center",
                           fontsize=9, color="white" if abs(diff[i,j]) > 0.5 * np.nanmax(np.abs(diff)) else "black")

        # Report the biggest decoupling
        print("\n  Biggest decoupling effects (γ=0.1 vs γ=0.0):")
        for i, j in pairs:
            d = diff[i, j]
            print(f"    {behs[i]}↔{behs[j]}: Δρ = {d:+.3f} "
                  f"({results['γ=0.0'][i,j]:+.3f} → {results['γ=0.1'][i,j]:+.3f})")

    fig.suptitle("Margin as Dimensional Decoupler\n"
                 "Left: γ=0 correlations | Center: γ=0.1 correlations | Right: Difference",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "decoupler_validation.png")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Dose-Dependent Correlation Shift
# ═══════════════════════════════════════════════════════════════════════════

def dose_correlation_shift(conn):
    """How do within-dose correlations change as λ_ρ increases?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Dose-Dependent Correlation Shift")
    print("=" * 70)

    behs = ["factual", "toxicity", "bias", "sycophancy"]
    lambdas = [0.0, 0.1, 0.2, 0.5]

    # Track correlations across doses
    pair_trends = {}  # (b1, b2) -> [(lambda, rho, p)]

    for rw in lambdas:
        rows = conn.execute("""
            SELECT seed, d_factual, d_toxicity, d_bias, d_sycophancy
            FROM alignment_runs
            WHERE model LIKE '%Qwen2.5-7B%'
            AND d_factual IS NOT NULL AND condition = '' AND margin = 0.1
            AND rho_weight = ?
            ORDER BY seed
        """, (rw,)).fetchall()

        if len(rows) < 4:
            continue

        data = {b: np.array([safe_float(r[f"d_{b}"]) for r in rows]) for b in behs}

        for i in range(len(behs)):
            for j in range(i + 1, len(behs)):
                b1, b2 = behs[i], behs[j]
                x, y = data[b1], data[b2]
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() >= 4:
                    r, p = stats.spearmanr(x[mask], y[mask])
                    key = (b1, b2)
                    if key not in pair_trends:
                        pair_trends[key] = []
                    pair_trends[key].append((rw, r, p))

    # Plot correlation trends
    n_pairs = len(pair_trends)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for p_idx, ((b1, b2), trend) in enumerate(sorted(pair_trends.items())):
        if p_idx >= 6:
            break
        ax = axes[p_idx]
        lams = [t[0] for t in trend]
        rhos = [t[1] for t in trend]
        ps = [t[2] for t in trend]

        # Color by significance
        colors = ["green" if p < 0.05 else "gray" for p in ps]
        sizes = [100 if p < 0.05 else 40 for p in ps]

        ax.scatter(lams, rhos, c=colors, s=sizes, edgecolors="black", linewidths=0.5, zorder=3)
        ax.plot(lams, rhos, "k--", alpha=0.3, zorder=2)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("λ_ρ")
        ax.set_ylabel("Spearman ρ")
        ax.set_title(f"{b1} ↔ {b2}")
        ax.set_ylim(-1.1, 1.1)

        # Annotate significant points
        for lam, rho, p in trend:
            if p < 0.05:
                ax.annotate(f"p={p:.3f}", (lam, rho), fontsize=7,
                           textcoords="offset points", xytext=(5, 8), color="green")

        # Report
        for lam, rho, p in trend:
            sig = "*" if p < 0.05 else ""
            print(f"  {b1}↔{b2} at λ={lam}: ρ={rho:+.3f} (p={p:.3f}){sig}")

    fig.suptitle("Within-Dose Correlation Shift Across λ_ρ\n"
                 "Green = significant (p<0.05) | Gray = non-significant",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "dose_correlation_shift.png")

    return pair_trends


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-db", action="store_true")
    args = parser.parse_args()

    if args.refresh_db:
        print("Rebuilding master.db...")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_master_db.py")], check=True)

    conn = get_conn()

    results = {}
    results["frontier"] = trade_off_frontier(conn)
    results["decoupler"] = decoupler_validation(conn)
    results["dose_shift"] = dose_correlation_shift(conn)

    conn.close()

    # Save results (convert tuple keys and numpy types)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = DOCS_DIR / "tradeoff_frontier.json"
    with open(output, "w") as f:
        json.dump(make_serializable(results), f, indent=2, default=str)
    print(f"\n→ Results saved to {output}")


if __name__ == "__main__":
    main()
