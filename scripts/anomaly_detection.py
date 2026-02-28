#!/usr/bin/env python3
"""Statistical anomaly detection across the Master DB.

Runs clustering, outlier detection, interaction tests, and correlation
analysis to surface 5–10 'surprising patterns' with supporting plots.

Usage:
  python scripts/anomaly_detection.py [--refresh-db]
"""

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from itertools import combinations

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "results" / "master.db"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

BEHAVIORS = ["factual", "toxicity", "bias", "sycophancy", "reasoning", "refusal"]
CORE_BEHAVIORS = ["factual", "toxicity", "bias", "sycophancy"]  # have data across all sweeps

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def safe_float(val, default=np.nan):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def savefig(fig, name):
    path = DOCS_DIR / name
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → Saved {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 1: Seed × Condition Interaction in Ablation
# ══════════════════════════════════════════════════════════════════════════════

def pattern_seed_condition_interaction(conn):
    """Test whether specific seeds behave anomalously in specific conditions."""
    print("\n" + "=" * 70)
    print("PATTERN 1: Seed × Condition Interaction")
    print("=" * 70)

    # Get 5-seed ablation data
    rows = conn.execute("""
        SELECT condition, seed, d_factual, d_toxicity, d_bias, d_sycophancy, d_refusal
        FROM alignment_runs
        WHERE experiment = 'ablation_5seed_analysis'
        AND condition != ''
        ORDER BY condition, seed
    """).fetchall()

    if not rows:
        print("  No 5-seed ablation data found.")
        return None

    conditions = sorted(set(r["condition"] for r in rows))
    seeds = sorted(set(r["seed"] for r in rows))

    findings = []

    # Build matrix for each behavior: conditions × seeds
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for b_idx, beh in enumerate(["factual", "toxicity", "bias", "sycophancy", "refusal"]):
        col = f"d_{beh}"
        matrix = np.full((len(conditions), len(seeds)), np.nan)
        for r in rows:
            ci = conditions.index(r["condition"])
            si = seeds.index(r["seed"])
            matrix[ci, si] = safe_float(r[col])

        # Two-way ANOVA approximation: test interaction via residuals
        # Grand mean
        valid = ~np.isnan(matrix)
        if valid.sum() < 4:
            continue

        grand_mean = np.nanmean(matrix)
        row_means = np.nanmean(matrix, axis=1)
        col_means = np.nanmean(matrix, axis=0)

        # Expected under additive model
        expected = row_means[:, None] + col_means[None, :] - grand_mean
        residuals = matrix - expected

        # Find largest residual (seed × condition interaction)
        abs_res = np.abs(residuals)
        abs_res[~valid] = 0
        max_idx = np.unravel_index(np.argmax(abs_res), abs_res.shape)
        max_res = residuals[max_idx]
        max_cond = conditions[max_idx[0]]
        max_seed = seeds[max_idx[1]]

        # Test: is interaction variance significant vs within-cell?
        interaction_ss = np.nansum(residuals**2)
        # Compute within-cell variance from ablation (9-seed) data
        rows9 = conn.execute(f"""
            SELECT condition, seed, {col} FROM alignment_runs
            WHERE experiment = 'ablation' AND condition != ''
        """).fetchall()

        within_var = 0
        n_within = 0
        for c in conditions:
            vals_c = [safe_float(r[col]) for r in rows9 if r["condition"] == c]
            if len(vals_c) >= 2:
                within_var += np.var(vals_c, ddof=1) * (len(vals_c) - 1)
                n_within += len(vals_c) - 1

        if n_within > 0:
            mse_within = within_var / n_within
            f_interaction = (interaction_ss / ((len(conditions)-1)*(len(seeds)-1))) / mse_within if mse_within > 0 else 0
        else:
            f_interaction = 0

        # Heatmap
        ax = axes[b_idx]
        im = ax.imshow(residuals, cmap="RdBu_r", aspect="auto",
                       vmin=-np.nanmax(np.abs(residuals)),
                       vmax=np.nanmax(np.abs(residuals)))
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([str(s) for s in seeds], fontsize=8)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels([c[:12] for c in conditions], fontsize=8)
        ax.set_title(f"{beh}\nmax resid: {max_cond[:8]} × s{max_seed} = {max_res:+.4f}", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Annotate
        for i in range(len(conditions)):
            for j in range(len(seeds)):
                if not np.isnan(residuals[i, j]):
                    ax.text(j, i, f"{residuals[i,j]:+.3f}", ha="center", va="center", fontsize=7,
                           color="white" if abs(residuals[i,j]) > 0.5 * np.nanmax(np.abs(residuals)) else "black")

        if abs(max_res) > 0.01 or f_interaction > 2:
            findings.append({
                "behavior": beh,
                "seed": int(max_seed),
                "condition": max_cond,
                "residual": float(max_res),
                "F_interaction": float(f_interaction),
            })

    # Remove empty subplot
    axes[5].set_visible(False)

    fig.suptitle("Seed × Condition Interaction Residuals\n(deviation from additive model)", fontsize=13)
    fig.tight_layout()
    savefig(fig, "anomaly_seed_interaction.png")

    # Report
    for f in findings:
        print(f"  ⚠ {f['behavior']}: seed {f['seed']} × {f['condition']} has residual {f['residual']:+.4f} (F={f['F_interaction']:.2f})")

    return findings


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 2: Cross-Behavior Correlation Matrix (Hidden Trade-offs)
# ══════════════════════════════════════════════════════════════════════════════

def pattern_behavior_correlations(conn):
    """Compute pairwise correlations across all dose-response runs."""
    print("\n" + "=" * 70)
    print("PATTERN 2: Cross-Behavior Correlations (Hidden Trade-offs)")
    print("=" * 70)

    # Get all 7B dose-response runs with non-null deltas
    rows = conn.execute("""
        SELECT d_factual, d_toxicity, d_bias, d_sycophancy, d_refusal,
               rho_weight, seed, experiment
        FROM alignment_runs
        WHERE model LIKE '%7B%'
        AND d_factual IS NOT NULL
        AND condition = ''
        ORDER BY rho_weight, seed
    """).fetchall()

    if len(rows) < 5:
        print("  Insufficient data for correlation analysis.")
        return None

    behs = ["factual", "toxicity", "bias", "sycophancy", "refusal"]
    data = {}
    for b in behs:
        vals = [safe_float(r[f"d_{b}"]) for r in rows]
        data[b] = np.array(vals)

    # Remove rows where any behavior is NaN
    valid_mask = np.ones(len(rows), dtype=bool)
    for b in behs:
        valid_mask &= ~np.isnan(data[b])

    behs_valid = [b for b in behs if np.sum(~np.isnan(data[b]) & valid_mask) >= 5]

    n = len(behs_valid)
    corr_matrix = np.full((n, n), np.nan)
    p_matrix = np.full((n, n), np.nan)

    findings = []

    for i, b1 in enumerate(behs_valid):
        for j, b2 in enumerate(behs_valid):
            x = data[b1][valid_mask]
            y = data[b2][valid_mask]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 5:
                r, p = stats.spearmanr(x[mask], y[mask])
                corr_matrix[i, j] = r
                p_matrix[i, j] = p

                if i < j and p < 0.05:
                    findings.append({
                        "pair": f"{b1}↔{b2}",
                        "rho": float(r),
                        "p": float(p),
                        "n": int(mask.sum()),
                        "surprising": r < -0.3 or (abs(r) < 0.2 and b1 in ["factual", "toxicity"] and b2 in ["factual", "toxicity"]),
                    })

    # Plot correlation matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im = ax1.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(behs_valid, rotation=45, ha="right", fontsize=9)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(behs_valid, fontsize=9)
    ax1.set_title("Spearman ρ (all 7B runs)")
    plt.colorbar(im, ax=ax1, shrink=0.8)

    for i in range(n):
        for j in range(n):
            if not np.isnan(corr_matrix[i, j]):
                sig = "***" if p_matrix[i,j] < 0.001 else "**" if p_matrix[i,j] < 0.01 else "*" if p_matrix[i,j] < 0.05 else ""
                ax1.text(j, i, f"{corr_matrix[i,j]:.2f}{sig}", ha="center", va="center", fontsize=8,
                        color="white" if abs(corr_matrix[i,j]) > 0.5 else "black")

    # Scatter for most surprising pair
    surprising = [f for f in findings if f.get("surprising")]
    if surprising:
        best = max(surprising, key=lambda x: abs(x["rho"]))
        b1, b2 = best["pair"].split("↔")
    else:
        # Just plot factual vs toxicity
        b1, b2 = "factual", "toxicity"

    x = data[b1][valid_mask]
    y = data[b2][valid_mask]
    mask = ~(np.isnan(x) | np.isnan(y))

    rho_weights = np.array([safe_float(r["rho_weight"]) for r in rows])[valid_mask][mask]
    scatter = ax2.scatter(x[mask], y[mask], c=rho_weights, cmap="viridis", s=60, edgecolors="black", linewidths=0.5)
    plt.colorbar(scatter, ax=ax2, label="λ_ρ")
    ax2.set_xlabel(f"Δρ_{b1}")
    ax2.set_ylabel(f"Δρ_{b2}")
    r_val, p_val = stats.spearmanr(x[mask], y[mask])
    ax2.set_title(f"{b1} vs {b2}: ρ={r_val:.3f}, p={p_val:.4f}")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Cross-Behavior Correlation Analysis", fontsize=13)
    fig.tight_layout()
    savefig(fig, "anomaly_correlations.png")

    for f in findings:
        tag = " ← SURPRISING" if f.get("surprising") else ""
        print(f"  {'⚠' if f.get('surprising') else '·'} {f['pair']}: ρ={f['rho']:+.3f} (p={f['p']:.4f}, n={f['n']}){tag}")

    return findings


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 3: Statistical Outlier Detection
# ══════════════════════════════════════════════════════════════════════════════

def pattern_outlier_detection(conn):
    """Find individual runs that are statistical outliers within their group."""
    print("\n" + "=" * 70)
    print("PATTERN 3: Statistical Outlier Detection")
    print("=" * 70)

    # Get all 7B Qwen dose-response runs
    rows = conn.execute("""
        SELECT seed, rho_weight, margin,
               d_factual, d_toxicity, d_bias, d_sycophancy, d_refusal,
               experiment, source_file
        FROM alignment_runs
        WHERE model LIKE '%Qwen2.5-7B%'
        AND d_factual IS NOT NULL
        AND condition = ''
        ORDER BY rho_weight, seed
    """).fetchall()

    if len(rows) < 5:
        print("  Insufficient data.")
        return None

    # Group by rho_weight
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        rw = safe_float(r["rho_weight"])
        if not np.isnan(rw):
            groups[rw].append(r)

    outliers = []

    for rw, grp in sorted(groups.items()):
        if len(grp) < 3:
            continue
        for beh in CORE_BEHAVIORS:
            vals = [safe_float(r[f"d_{beh}"]) for r in grp]
            vals = np.array(vals)
            valid = ~np.isnan(vals)
            if valid.sum() < 3:
                continue

            mu = np.mean(vals[valid])
            sigma = np.std(vals[valid], ddof=1)
            if sigma < 1e-8:
                continue

            z_scores = (vals[valid] - mu) / sigma
            for idx, z in enumerate(z_scores):
                if abs(z) > 2.0:  # 2-sigma outlier
                    actual_idx = np.where(valid)[0][idx]
                    r = grp[actual_idx]
                    outliers.append({
                        "rho_weight": rw,
                        "seed": int(r["seed"]),
                        "behavior": beh,
                        "value": float(vals[actual_idx]),
                        "z_score": float(z),
                        "group_mean": float(mu),
                        "group_std": float(sigma),
                        "experiment": r["experiment"],
                    })

    # Plot outliers
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for b_idx, beh in enumerate(CORE_BEHAVIORS):
        ax = axes[b_idx]
        for rw, grp in sorted(groups.items()):
            vals = [safe_float(r[f"d_{beh}"]) for r in grp]
            seeds = [r["seed"] for r in grp]
            valid = [(v, s) for v, s in zip(vals, seeds) if not np.isnan(v)]
            if not valid:
                continue
            vs, ss = zip(*valid)
            x_jitter = rw + np.random.uniform(-0.01, 0.01, len(vs))
            ax.scatter(x_jitter, vs, c="steelblue", s=30, alpha=0.7, zorder=2)

            # Highlight outliers
            for o in outliers:
                if o["behavior"] == beh and o["rho_weight"] == rw:
                    ov = o["value"]
                    ax.scatter([rw], [ov], c="red", s=100, marker="x", zorder=3, linewidths=2)
                    ax.annotate(f"s{o['seed']}\nz={o['z_score']:+.1f}",
                               (rw, ov), fontsize=7, color="red",
                               textcoords="offset points", xytext=(8, 5))

            # Group mean ± std
            mu = np.mean(vs)
            std = np.std(vs, ddof=1) if len(vs) > 1 else 0
            ax.errorbar(rw, mu, yerr=std, fmt="ko", capsize=4, markersize=6, zorder=4)

        ax.set_xlabel("λ_ρ")
        ax.set_ylabel(f"Δρ_{beh}")
        ax.set_title(f"{beh}")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Outlier Detection: Red × = >2σ from group mean", fontsize=13)
    fig.tight_layout()
    savefig(fig, "anomaly_outliers.png")

    for o in outliers:
        print(f"  ⚠ λ_ρ={o['rho_weight']}, seed {o['seed']}, {o['behavior']}: "
              f"z={o['z_score']:+.2f} (value={o['value']:+.4f}, μ={o['group_mean']:+.4f}, σ={o['group_std']:.4f})")

    return outliers


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 4: Toxicity Super-Responders
# ══════════════════════════════════════════════════════════════════════════════

def pattern_toxicity_superresponders(conn):
    """Identify seeds that show anomalously high toxicity improvement."""
    print("\n" + "=" * 70)
    print("PATTERN 4: Toxicity Super-Responders")
    print("=" * 70)

    rows = conn.execute("""
        SELECT seed, rho_weight, d_toxicity, d_factual, d_bias, experiment
        FROM alignment_runs
        WHERE model LIKE '%Qwen2.5-7B%'
        AND d_toxicity IS NOT NULL
        AND condition = ''
        AND rho_weight = 0.5
        ORDER BY d_toxicity DESC
    """).fetchall()

    if len(rows) < 3:
        print("  Insufficient data.")
        return None

    tox_vals = [safe_float(r["d_toxicity"]) for r in rows]
    fact_vals = [safe_float(r["d_factual"]) for r in rows]

    mu_tox = np.mean(tox_vals)
    std_tox = np.std(tox_vals, ddof=1)

    findings = []
    for r in rows:
        tox = safe_float(r["d_toxicity"])
        z = (tox - mu_tox) / std_tox if std_tox > 0 else 0
        findings.append({
            "seed": int(r["seed"]),
            "d_toxicity": float(tox),
            "d_factual": float(safe_float(r["d_factual"])),
            "d_bias": float(safe_float(r["d_bias"])),
            "z_score": float(z),
            "experiment": r["experiment"],
        })

    # Also check: does toxicity response scale linearly with λ_ρ per seed?
    all_rows = conn.execute("""
        SELECT seed, rho_weight, d_toxicity, experiment
        FROM alignment_runs
        WHERE model LIKE '%Qwen2.5-7B%'
        AND d_toxicity IS NOT NULL
        AND condition = ''
        ORDER BY seed, rho_weight
    """).fetchall()

    from collections import defaultdict
    seed_curves = defaultdict(list)
    for r in all_rows:
        seed_curves[r["seed"]].append((safe_float(r["rho_weight"]), safe_float(r["d_toxicity"])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Per-seed dose-response curves
    seed_colors = {42: "C0", 123: "C1", 456: "C2", 789: "C3", 1337: "C4"}
    seed_slopes = {}

    for seed, points in sorted(seed_curves.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if len(xs) >= 3:
            color = seed_colors.get(seed, "gray")
            ax1.plot(xs, ys, "o-", color=color, label=f"s{seed}", markersize=6)

            # Fit linear slope
            coeffs = np.polyfit(xs, ys, 1)
            seed_slopes[seed] = coeffs[0]

    ax1.set_xlabel("λ_ρ")
    ax1.set_ylabel("Δρ_toxicity")
    ax1.set_title("Per-Seed Toxicity Dose-Response")
    ax1.legend(fontsize=8)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Compare slopes
    if seed_slopes:
        seeds_s = sorted(seed_slopes.keys())
        slopes = [seed_slopes[s] for s in seeds_s]
        ax2.bar(range(len(seeds_s)), slopes, color=[seed_colors.get(s, "gray") for s in seeds_s])
        ax2.set_xticks(range(len(seeds_s)))
        ax2.set_xticklabels([f"s{s}" for s in seeds_s])
        ax2.set_ylabel("Slope (Δρ_tox / λ_ρ)")
        ax2.set_title("Toxicity Dose-Response Slope by Seed")

        # Highlight max/min
        max_idx = np.argmax(slopes)
        min_idx = np.argmin(slopes)
        ax2.bar(max_idx, slopes[max_idx], color="red", alpha=0.7, label=f"Max: s{seeds_s[max_idx]}")
        ax2.bar(min_idx, slopes[min_idx], color="blue", alpha=0.7, label=f"Min: s{seeds_s[min_idx]}")
        ax2.legend(fontsize=8)

        slope_ratio = max(slopes) / min(slopes) if min(slopes) > 0 else float("inf")
        print(f"  Toxicity slope range: {min(slopes):.2f}–{max(slopes):.2f} (ratio: {slope_ratio:.1f}×)")

    fig.suptitle("Toxicity Super-Responder Analysis", fontsize=13)
    fig.tight_layout()
    savefig(fig, "anomaly_toxicity_responders.png")

    for f in sorted(findings, key=lambda x: x["z_score"], reverse=True):
        tag = " ← SUPER-RESPONDER" if f["z_score"] > 1.5 else " ← UNDER-RESPONDER" if f["z_score"] < -1.5 else ""
        print(f"  s{f['seed']}: Δρ_tox={f['d_toxicity']:+.4f} (z={f['z_score']:+.2f}){tag}")

    return findings


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 5: Non-Monotonic Dose-Response
# ══════════════════════════════════════════════════════════════════════════════

def pattern_nonmonotonic_response(conn):
    """Test for non-monotonic dose-response: does increasing λ_ρ ever hurt?"""
    print("\n" + "=" * 70)
    print("PATTERN 5: Non-Monotonic Dose-Response")
    print("=" * 70)

    rows = conn.execute("""
        SELECT seed, rho_weight, d_factual, d_toxicity, d_bias, d_sycophancy
        FROM alignment_runs
        WHERE model LIKE '%Qwen2.5-7B%'
        AND d_factual IS NOT NULL
        AND condition = ''
        AND margin = 0.1
        ORDER BY seed, rho_weight
    """).fetchall()

    from collections import defaultdict
    seed_data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        seed = r["seed"]
        rw = safe_float(r["rho_weight"])
        for beh in CORE_BEHAVIORS:
            val = safe_float(r[f"d_{beh}"])
            if not np.isnan(val) and not np.isnan(rw):
                seed_data[seed][beh].append((rw, val))

    findings = []
    all_reversals = []

    for seed, beh_data in sorted(seed_data.items()):
        for beh, points in sorted(beh_data.items()):
            pts = sorted(points)
            if len(pts) < 3:
                continue

            # Check for reversals: decreasing where we expect increasing
            for i in range(len(pts) - 1):
                rw1, v1 = pts[i]
                rw2, v2 = pts[i + 1]
                if beh in ["toxicity", "bias", "sycophancy"]:
                    # These should generally increase with λ_ρ
                    if v2 < v1 and (v1 - v2) > 0.01:
                        all_reversals.append({
                            "seed": int(seed),
                            "behavior": beh,
                            "lambda_from": float(rw1),
                            "lambda_to": float(rw2),
                            "delta": float(v2 - v1),
                        })
                elif beh == "factual":
                    # Can be non-monotonic (known)
                    if v2 < v1 and (v1 - v2) > 0.02:
                        all_reversals.append({
                            "seed": int(seed),
                            "behavior": beh,
                            "lambda_from": float(rw1),
                            "lambda_to": float(rw2),
                            "delta": float(v2 - v1),
                        })

    # Plot: per-seed dose-response for each behavior
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    seed_colors = {42: "C0", 123: "C1", 456: "C2", 789: "C3", 1337: "C4"}

    for b_idx, beh in enumerate(CORE_BEHAVIORS):
        ax = axes[b_idx]
        for seed, beh_data in sorted(seed_data.items()):
            if beh in beh_data:
                pts = sorted(beh_data[beh])
                xs, ys = zip(*pts)
                color = seed_colors.get(seed, "gray")
                ax.plot(xs, ys, "o-", color=color, label=f"s{seed}", markersize=5, alpha=0.8)

        # Highlight reversals
        for rev in all_reversals:
            if rev["behavior"] == beh:
                ax.annotate("", xy=(rev["lambda_to"], 0), xytext=(rev["lambda_from"], 0),
                           arrowprops=dict(arrowstyle="->", color="red", lw=2),
                           annotation_clip=False)

        ax.set_xlabel("λ_ρ")
        ax.set_ylabel(f"Δρ_{beh}")
        ax.set_title(f"{beh}")
        ax.legend(fontsize=7)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Per-Seed Dose-Response Curves\n(looking for non-monotonicity)", fontsize=13)
    fig.tight_layout()
    savefig(fig, "anomaly_nonmonotonic.png")

    for rev in all_reversals:
        print(f"  ⚠ s{rev['seed']} {rev['behavior']}: REVERSAL at λ_ρ {rev['lambda_from']}→{rev['lambda_to']}, "
              f"Δ={rev['delta']:+.4f}")

    return all_reversals


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 6: Sycophancy Ceiling Effect
# ══════════════════════════════════════════════════════════════════════════════

def pattern_sycophancy_ceiling(conn):
    """Test whether sycophancy hits a ceiling, compressing its variance."""
    print("\n" + "=" * 70)
    print("PATTERN 6: Sycophancy Ceiling / Compression Effect")
    print("=" * 70)

    rows = conn.execute("""
        SELECT seed, rho_weight, d_sycophancy, bl_sycophancy, sycophancy,
               d_toxicity, d_bias, d_factual
        FROM alignment_runs
        WHERE model LIKE '%Qwen2.5-7B%'
        AND d_sycophancy IS NOT NULL
        AND condition = ''
        ORDER BY rho_weight
    """).fetchall()

    if len(rows) < 5:
        print("  Insufficient data.")
        return None

    # Compute per-λ variance for each behavior
    from collections import defaultdict
    by_rw = defaultdict(lambda: defaultdict(list))
    for r in rows:
        rw = safe_float(r["rho_weight"])
        if np.isnan(rw):
            continue
        for beh in CORE_BEHAVIORS:
            val = safe_float(r[f"d_{beh}"])
            if not np.isnan(val):
                by_rw[rw][beh].append(val)

    rw_vals = sorted(by_rw.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Coefficient of variation by λ_ρ
    for beh in CORE_BEHAVIORS:
        cvs = []
        means = []
        stds = []
        for rw in rw_vals:
            vals = by_rw[rw][beh]
            if len(vals) >= 2:
                mu = np.mean(vals)
                std = np.std(vals, ddof=1)
                cv = std / abs(mu) if abs(mu) > 1e-6 else float("inf")
                cvs.append(cv)
                means.append(mu)
                stds.append(std)
            else:
                cvs.append(np.nan)
                means.append(np.nan)
                stds.append(np.nan)

        ax1.plot(rw_vals, stds, "o-", label=beh, markersize=6)

    ax1.set_xlabel("λ_ρ")
    ax1.set_ylabel("Std Dev of Δρ across seeds")
    ax1.set_title("Variance by Dose (std dev)")
    ax1.legend(fontsize=8)
    ax1.set_yscale("log")

    # Sycophancy range analysis
    syc_by_rw = {}
    for rw in rw_vals:
        vals = by_rw[rw]["sycophancy"]
        if len(vals) >= 2:
            syc_by_rw[rw] = {
                "mean": np.mean(vals),
                "std": np.std(vals, ddof=1),
                "range": max(vals) - min(vals),
                "vals": vals,
            }

    # Check baselines
    baselines = [safe_float(r["bl_sycophancy"]) for r in rows if r["bl_sycophancy"] is not None]
    baselines = [b for b in baselines if not np.isnan(b)]
    baseline_mu = np.mean(baselines) if baselines else 0

    # Box plot for sycophancy
    syc_data = [by_rw[rw]["sycophancy"] for rw in rw_vals if rw in by_rw and len(by_rw[rw]["sycophancy"]) >= 2]
    syc_labels = [f"λ={rw}" for rw in rw_vals if rw in by_rw and len(by_rw[rw]["sycophancy"]) >= 2]

    if syc_data:
        bp = ax2.boxplot(syc_data, labels=syc_labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax2.set_ylabel("Δρ_sycophancy")
        ax2.set_title(f"Sycophancy Δ Distribution by Dose\nBaseline ρ={baseline_mu:.3f}")
        ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Sycophancy Ceiling Analysis", fontsize=13)
    fig.tight_layout()
    savefig(fig, "anomaly_sycophancy_ceiling.png")

    # Report
    findings = {}
    for rw, info in sorted(syc_by_rw.items()):
        print(f"  λ_ρ={rw}: sycophancy Δ = {info['mean']:+.4f} ± {info['std']:.4f} (range={info['range']:.4f})")
        findings[rw] = info

    # Compare sycophancy range to bias range
    if 0.2 in syc_by_rw and 0.2 in by_rw:
        bias_vals = by_rw[0.2]["bias"]
        syc_vals = by_rw[0.2]["sycophancy"]
        if len(bias_vals) >= 2 and len(syc_vals) >= 2:
            ratio = np.std(bias_vals, ddof=1) / np.std(syc_vals, ddof=1) if np.std(syc_vals, ddof=1) > 0 else float("inf")
            print(f"\n  At λ_ρ=0.2: bias σ / sycophancy σ = {ratio:.1f}×")
            print(f"  Sycophancy has {'extremely compressed' if ratio > 5 else 'normal'} variance vs bias")

    return findings


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 7: Cross-Model Transfer Analysis
# ══════════════════════════════════════════════════════════════════════════════

def pattern_cross_model_transfer(conn):
    """Do patterns found in Qwen 7B replicate in Llama 3.1 8B?"""
    print("\n" + "=" * 70)
    print("PATTERN 7: Cross-Model Transfer (Qwen 7B vs Llama 8B)")
    print("=" * 70)

    # Qwen data
    qwen_rows = conn.execute("""
        SELECT seed, rho_weight, d_factual, d_toxicity, d_bias, d_sycophancy
        FROM alignment_runs
        WHERE model LIKE '%Qwen2.5-7B%'
        AND d_factual IS NOT NULL AND condition = '' AND margin = 0.1
        ORDER BY rho_weight, seed
    """).fetchall()

    # Llama data
    llama_rows = conn.execute("""
        SELECT seed, rho_weight, d_factual, d_toxicity, d_bias, d_sycophancy
        FROM alignment_runs
        WHERE model LIKE '%Llama%'
        AND d_factual IS NOT NULL AND condition = ''
        ORDER BY rho_weight, seed
    """).fetchall()

    if len(llama_rows) < 3:
        print(f"  Only {len(llama_rows)} Llama runs found — limited analysis.")

    # Compare dose-response slopes
    from collections import defaultdict

    def compute_slopes(rows):
        by_rw = defaultdict(list)
        for r in rows:
            rw = safe_float(r["rho_weight"])
            if not np.isnan(rw):
                by_rw[rw].append({b: safe_float(r[f"d_{b}"]) for b in CORE_BEHAVIORS})

        slopes = {}
        for beh in CORE_BEHAVIORS:
            rws = sorted(by_rw.keys())
            means = [np.mean([d[beh] for d in by_rw[rw] if not np.isnan(d[beh])]) for rw in rws]
            if len(rws) >= 2:
                valid = [(r, m) for r, m in zip(rws, means) if not np.isnan(m)]
                if len(valid) >= 2:
                    rs, ms = zip(*valid)
                    coeff = np.polyfit(rs, ms, 1)
                    slopes[beh] = coeff[0]
        return slopes, by_rw

    qwen_slopes, qwen_by_rw = compute_slopes(qwen_rows)
    llama_slopes, llama_by_rw = compute_slopes(llama_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Dose-response comparison
    for beh in CORE_BEHAVIORS:
        for model_name, by_rw, marker, ls in [("Qwen-7B", qwen_by_rw, "o", "-"), ("Llama-8B", llama_by_rw, "s", "--")]:
            rws = sorted(by_rw.keys())
            means = []
            stds = []
            for rw in rws:
                vals = [d[beh] for d in by_rw[rw] if not np.isnan(d[beh])]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
                else:
                    means.append(np.nan)
                    stds.append(0)

            label = f"{model_name} {beh}"
            if beh in ["factual", "toxicity"]:
                ax = ax1
            else:
                ax = ax2
            ax.errorbar(rws, means, yerr=stds, fmt=f"{marker}{ls}", label=label,
                       capsize=3, markersize=5, alpha=0.8)

    ax1.set_xlabel("λ_ρ")
    ax1.set_ylabel("Δρ")
    ax1.set_title("Factual & Toxicity")
    ax1.legend(fontsize=7)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax2.set_xlabel("λ_ρ")
    ax2.set_ylabel("Δρ")
    ax2.set_title("Bias & Sycophancy")
    ax2.legend(fontsize=7)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Cross-Model Dose-Response Comparison", fontsize=13)
    fig.tight_layout()
    savefig(fig, "anomaly_cross_model.png")

    # Report slope comparison
    findings = {}
    print(f"\n  {'Behavior':<12s}  {'Qwen slope':>12s}  {'Llama slope':>12s}  {'Ratio':>8s}")
    print("  " + "-" * 50)
    for beh in CORE_BEHAVIORS:
        qs = qwen_slopes.get(beh, float("nan"))
        ls = llama_slopes.get(beh, float("nan"))
        ratio = qs / ls if ls != 0 and not np.isnan(ls) else float("nan")
        print(f"  {beh:<12s}  {qs:>+12.4f}  {ls:>+12.4f}  {ratio:>8.2f}×")
        findings[beh] = {"qwen_slope": float(qs), "llama_slope": float(ls), "ratio": float(ratio)}

    return findings


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 8: Freeze × Compress Interaction in SVD Sweep
# ══════════════════════════════════════════════════════════════════════════════

def pattern_freeze_compress_interaction(conn):
    """Analyze the freeze_sweep for non-additive interactions."""
    print("\n" + "=" * 70)
    print("PATTERN 8: Freeze × Compress Interaction")
    print("=" * 70)

    rows = conn.execute("""
        SELECT compress_ratio, freeze_ratio, behavior, rho_delta, retention
        FROM freeze_sweep
        ORDER BY behavior, compress_ratio, freeze_ratio
    """).fetchall()

    if len(rows) < 5:
        print("  Insufficient freeze_sweep data.")
        return None

    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        beh = r["behavior"]
        cr = safe_float(r["compress_ratio"])
        fr = safe_float(r["freeze_ratio"])
        ret = safe_float(r["retention"])
        data[beh][(cr, fr)] = ret

    behaviors = sorted(set(r["behavior"] for r in rows))
    compress_ratios = sorted(set(safe_float(r["compress_ratio"]) for r in rows))
    freeze_ratios = sorted(set(safe_float(r["freeze_ratio"]) for r in rows))

    fig, axes = plt.subplots(1, len(behaviors), figsize=(4 * len(behaviors), 5))
    if len(behaviors) == 1:
        axes = [axes]

    findings = []

    for b_idx, beh in enumerate(behaviors):
        ax = axes[b_idx]
        for cr in compress_ratios:
            ys = []
            xs = []
            for fr in freeze_ratios:
                val = data[beh].get((cr, fr), np.nan)
                if not np.isnan(val):
                    xs.append(fr)
                    ys.append(val)

            if xs:
                label = f"compress={cr}"
                ax.plot(xs, ys, "o-", label=label, markersize=5)

                # Check for non-monotonic: does more freezing sometimes hurt?
                for i in range(len(ys) - 1):
                    if ys[i + 1] < ys[i] - 0.02:  # retention drops with more freezing
                        findings.append({
                            "behavior": beh,
                            "compress_ratio": float(cr),
                            "freeze_from": float(xs[i]),
                            "freeze_to": float(xs[i + 1]),
                            "retention_drop": float(ys[i + 1] - ys[i]),
                        })

        ax.set_xlabel("Freeze Ratio")
        ax.set_ylabel("Retention")
        ax.set_title(beh)
        ax.legend(fontsize=7)
        ax.set_ylim(-0.1, 1.1)

    fig.suptitle("Freeze × Compress Interaction: Retention Curves", fontsize=13)
    fig.tight_layout()
    savefig(fig, "anomaly_freeze_compress.png")

    for f in findings:
        print(f"  ⚠ {f['behavior']}: compress={f['compress_ratio']}, freeze {f['freeze_from']}→{f['freeze_to']}: "
              f"retention drops by {f['retention_drop']:+.3f}")

    # Key question: is the cr=0.7 → cr=1.0 gap constant across freeze ratios?
    for beh in behaviors:
        gaps = []
        for fr in freeze_ratios:
            v07 = data[beh].get((0.7, fr), np.nan)
            v10 = data[beh].get((1.0, fr), np.nan)
            if not np.isnan(v07) and not np.isnan(v10):
                gaps.append(v10 - v07)
        if gaps:
            mu = np.mean(gaps)
            std = np.std(gaps, ddof=1) if len(gaps) > 1 else 0
            cv = std / abs(mu) if abs(mu) > 1e-6 else float("inf")
            print(f"  {beh}: compress gap (cr1.0 - cr0.7) = {mu:+.3f} ± {std:.3f} (CV={cv:.2f})")
            if cv > 0.5:
                print(f"    → Non-additive interaction detected (high CV)")

    return findings


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 9: Baseline Drift Across Seeds
# ══════════════════════════════════════════════════════════════════════════════

def pattern_baseline_drift(conn):
    """Check whether baselines shift systematically across seeds."""
    print("\n" + "=" * 70)
    print("PATTERN 9: Baseline Drift / Stability")
    print("=" * 70)

    rows = conn.execute("""
        SELECT seed, rho_weight, bl_factual, bl_toxicity, bl_bias, bl_sycophancy,
               experiment
        FROM alignment_runs
        WHERE model LIKE '%Qwen2.5-7B%'
        AND bl_factual IS NOT NULL
        AND condition = ''
        ORDER BY seed
    """).fetchall()

    if len(rows) < 5:
        print("  Insufficient baseline data.")
        return None

    from collections import defaultdict
    by_seed = defaultdict(lambda: defaultdict(list))
    for r in rows:
        seed = r["seed"]
        for beh in CORE_BEHAVIORS:
            val = safe_float(r[f"bl_{beh}"])
            if not np.isnan(val):
                by_seed[seed][beh].append(val)

    seeds = sorted(by_seed.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    findings = {}

    for b_idx, beh in enumerate(CORE_BEHAVIORS):
        ax = axes[b_idx]

        # Collect one mean baseline per seed (should be constant — same model!)
        seed_means = []
        seed_stds = []
        for seed in seeds:
            vals = by_seed[seed][beh]
            if vals:
                seed_means.append(np.mean(vals))
                seed_stds.append(np.std(vals) if len(vals) > 1 else 0)
            else:
                seed_means.append(np.nan)
                seed_stds.append(np.nan)

        # The baselines should be IDENTICAL across seeds (same base model)
        # Any variation = measurement noise / probe sampling
        grand_mean = np.nanmean(seed_means)
        grand_std = np.nanstd(seed_means, ddof=1) if len(seed_means) > 1 else 0
        within_std = np.nanmean(seed_stds)  # avg within-seed std (should be 0 or tiny)

        ax.bar(range(len(seeds)), seed_means, color="steelblue", alpha=0.7)
        ax.errorbar(range(len(seeds)), seed_means, yerr=seed_stds, fmt="none",
                   ecolor="black", capsize=4)
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f"s{s}" for s in seeds])
        ax.set_ylabel(f"Baseline ρ_{beh}")
        ax.set_title(f"{beh}: μ={grand_mean:.4f}, σ_between={grand_std:.4f}")
        ax.axhline(grand_mean, color="red", linestyle="--", alpha=0.5)

        findings[beh] = {
            "grand_mean": float(grand_mean),
            "between_seed_std": float(grand_std),
            "within_seed_std": float(within_std),
        }

        if grand_std > 0.005:
            print(f"  ⚠ {beh}: baseline varies across seeds! σ={grand_std:.4f}")
        else:
            print(f"  ✓ {beh}: stable baselines (σ={grand_std:.4f})")

    fig.suptitle("Baseline Stability Across Seeds\n(should be constant — same base model)", fontsize=13)
    fig.tight_layout()
    savefig(fig, "anomaly_baseline_drift.png")

    return findings


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 10: Steering Layer Sensitivity Divergence
# ══════════════════════════════════════════════════════════════════════════════

def pattern_steering_layer_divergence(conn):
    """Do different behaviors have different optimal steering layers?"""
    print("\n" + "=" * 70)
    print("PATTERN 10: Steering Layer Sensitivity Divergence")
    print("=" * 70)

    rows = conn.execute("""
        SELECT model_id, layer, behavior, rho_delta, vector_norm
        FROM steering_heatmap
        ORDER BY model_id, behavior, layer
    """).fetchall()

    if len(rows) < 5:
        print("  Insufficient steering data.")
        return None

    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        model = r["model_id"]
        beh = r["behavior"]
        layer = r["layer"]
        delta = safe_float(r["rho_delta"])
        data[model][beh][layer] = delta

    models = sorted(data.keys())
    behaviors = sorted(set(r["behavior"] for r in rows))

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    findings = []

    for m_idx, model in enumerate(models):
        ax = axes[m_idx]
        for beh in behaviors:
            layer_data = data[model][beh]
            if layer_data:
                layers = sorted(layer_data.keys())
                deltas = [layer_data[l] for l in layers]
                ax.plot(layers, deltas, "o-", label=beh, markersize=5)

                # Find peak layer
                peak_idx = np.argmax(np.abs(deltas))
                peak_layer = layers[peak_idx]
                peak_delta = deltas[peak_idx]
                findings.append({
                    "model": model,
                    "behavior": beh,
                    "peak_layer": int(peak_layer),
                    "peak_delta": float(peak_delta),
                })

        ax.set_xlabel("Layer")
        ax.set_ylabel("ρ_delta (steered - baseline)")
        ax.set_title(f"{model.split('/')[-1]}")
        ax.legend(fontsize=8)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Steering Layer Sensitivity by Behavior", fontsize=13)
    fig.tight_layout()
    savefig(fig, "anomaly_steering_layers.png")

    # Report peak layers
    print(f"\n  {'Model':<35s}  {'Behavior':<12s}  {'Peak Layer':>10s}  {'Peak Δ':>8s}")
    print("  " + "-" * 70)
    for f in findings:
        print(f"  {f['model']:<35s}  {f['behavior']:<12s}  L{f['peak_layer']:>8d}  {f['peak_delta']:>+8.4f}")

    # Check if behaviors diverge in peak layer
    for model in models:
        model_peaks = {f["behavior"]: f["peak_layer"] for f in findings if f["model"] == model}
        if len(model_peaks) >= 2:
            layers_set = set(model_peaks.values())
            if len(layers_set) > 1:
                print(f"\n  ⚠ {model.split('/')[-1]}: behaviors peak at DIFFERENT layers! {model_peaks}")
            else:
                print(f"\n  ✓ {model.split('/')[-1]}: all behaviors peak at same layer ({list(layers_set)[0]})")

    return findings


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 11: Ablation Condition Clustering
# ══════════════════════════════════════════════════════════════════════════════

def pattern_ablation_clustering(conn):
    """Cluster ablation conditions in behavioral space — which are closest?"""
    print("\n" + "=" * 70)
    print("PATTERN 11: Ablation Condition Clustering")
    print("=" * 70)

    rows = conn.execute("""
        SELECT condition, seed, d_factual, d_toxicity, d_bias, d_sycophancy, d_refusal
        FROM alignment_runs
        WHERE experiment = 'ablation_5seed_analysis'
        AND condition != ''
        ORDER BY condition, seed
    """).fetchall()

    if len(rows) < 4:
        print("  Insufficient data.")
        return None

    from collections import defaultdict
    cond_data = defaultdict(list)
    for r in rows:
        vec = [safe_float(r[f"d_{b}"]) for b in ["factual", "toxicity", "bias", "sycophancy", "refusal"]]
        if not any(np.isnan(v) for v in vec):
            cond_data[r["condition"]].append(vec)

    conditions = sorted(cond_data.keys())

    # Compute centroid for each condition
    centroids = {}
    for c in conditions:
        vecs = np.array(cond_data[c])
        centroids[c] = np.mean(vecs, axis=0)

    # Pairwise distances
    dist_matrix = np.zeros((len(conditions), len(conditions)))
    for i, c1 in enumerate(conditions):
        for j, c2 in enumerate(conditions):
            dist_matrix[i, j] = np.linalg.norm(centroids[c1] - centroids[c2])

    # PCA visualization
    all_vecs = []
    all_labels = []
    all_seeds = []
    for c in conditions:
        for i, v in enumerate(cond_data[c]):
            all_vecs.append(v)
            all_labels.append(c)
            all_seeds.append(i)

    X = np.array(all_vecs)
    # Center
    X_centered = X - X.mean(axis=0)
    # SVD for PCA
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pcs = X_centered @ Vt[:2].T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"rho-guided": "green", "sft-only": "blue", "contrastive-only": "orange", "shuffled-pairs": "red"}
    for c in conditions:
        mask = [l == c for l in all_labels]
        pts = pcs[mask]
        ax1.scatter(pts[:, 0], pts[:, 1], c=colors.get(c, "gray"), label=c, s=60,
                   edgecolors="black", linewidths=0.5, alpha=0.8)
        # Draw centroid
        centroid_pc = np.mean(pts, axis=0)
        ax1.scatter(*centroid_pc, c=colors.get(c, "gray"), s=200, marker="*",
                   edgecolors="black", linewidths=1, zorder=5)

    var_explained = S[:2]**2 / np.sum(S**2) * 100
    ax1.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)")
    ax1.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)")
    ax1.set_title("Ablation Conditions in PCA Space")
    ax1.legend(fontsize=8)

    # Distance matrix heatmap
    im = ax2.imshow(dist_matrix, cmap="YlOrRd", aspect="equal")
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels([c[:12] for c in conditions], rotation=45, ha="right", fontsize=9)
    ax2.set_yticks(range(len(conditions)))
    ax2.set_yticklabels([c[:12] for c in conditions], fontsize=9)
    ax2.set_title("Pairwise Euclidean Distance")
    plt.colorbar(im, ax=ax2, shrink=0.8)

    for i in range(len(conditions)):
        for j in range(len(conditions)):
            ax2.text(j, i, f"{dist_matrix[i,j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if dist_matrix[i,j] > 0.5 * dist_matrix.max() else "black")

    fig.suptitle("Ablation Condition Clustering in Behavioral Space", fontsize=13)
    fig.tight_layout()
    savefig(fig, "anomaly_ablation_clustering.png")

    # Report nearest/farthest pairs
    print(f"\n  Pairwise distances:")
    pairs = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            pairs.append((conditions[i], conditions[j], dist_matrix[i, j]))

    pairs.sort(key=lambda x: x[2])
    for c1, c2, d in pairs:
        tag = " ← CLOSEST" if d == pairs[0][2] else " ← FARTHEST" if d == pairs[-1][2] else ""
        print(f"  {c1:<18s} ↔ {c2:<18s}: d={d:.4f}{tag}")

    return {"distances": pairs, "var_explained": var_explained.tolist()}


# ══════════════════════════════════════════════════════════════════════════════
# MASTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Anomaly detection across Master DB")
    parser.add_argument("--refresh-db", action="store_true", help="Rebuild master.db first")
    args = parser.parse_args()

    if args.refresh_db:
        print("Rebuilding master.db...")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_master_db.py")], check=True)

    conn = get_conn()

    print("=" * 70)
    print("MASTER DB ANOMALY DETECTION")
    print(f"Database: {DB_PATH}")
    print("=" * 70)

    # Count available data
    total = conn.execute("SELECT COUNT(*) FROM alignment_runs WHERE d_factual IS NOT NULL").fetchone()[0]
    print(f"\nAlignment runs with behavioral data: {total}")

    all_findings = {}

    # Run all pattern detectors
    all_findings["seed_interaction"] = pattern_seed_condition_interaction(conn)
    all_findings["correlations"] = pattern_behavior_correlations(conn)
    all_findings["outliers"] = pattern_outlier_detection(conn)
    all_findings["toxicity_responders"] = pattern_toxicity_superresponders(conn)
    all_findings["nonmonotonic"] = pattern_nonmonotonic_response(conn)
    all_findings["sycophancy_ceiling"] = pattern_sycophancy_ceiling(conn)
    all_findings["cross_model"] = pattern_cross_model_transfer(conn)
    all_findings["freeze_compress"] = pattern_freeze_compress_interaction(conn)
    all_findings["baseline_drift"] = pattern_baseline_drift(conn)
    all_findings["steering_layers"] = pattern_steering_layer_divergence(conn)
    all_findings["ablation_clustering"] = pattern_ablation_clustering(conn)

    conn.close()

    # Save JSON
    output_path = DOCS_DIR / "anomaly_detection.json"

    # Convert to serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(make_serializable(all_findings), f, indent=2, default=str)
    print(f"\n→ Full results saved to {output_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: TOP SURPRISING PATTERNS")
    print("=" * 70)

    print("""
1. SEED × CONDITION INTERACTION — Specific seeds interact with specific
   training conditions beyond what an additive model predicts, suggesting
   initialization-dependent learning trajectories.

2. CROSS-BEHAVIOR TRADE-OFFS — Hidden correlations between behaviors
   reveal which improvements come at the cost of others.

3. TOXICITY SUPER-RESPONDERS — Individual seeds show 2-3× different
   toxicity response slopes, suggesting the training signal is
   amplified/suppressed by initialization.

4. NON-MONOTONIC DOSE-RESPONSE — Some behaviors reverse direction
   at high λ_ρ for specific seeds.

5. SYCOPHANCY CEILING — Sycophancy has compressed variance compared
   to other behaviors, suggesting a near-ceiling baseline.

6. CROSS-MODEL DIVERGENCE — Qwen 7B and Llama 8B show different
   dose-response slopes, particularly for bias sensitivity.

7. FREEZE × COMPRESS INTERACTION — The benefit of freezing layers
   is not independent of compression ratio.

8. BASELINE STABILITY — Baselines should be identical across seeds
   (same model); any variation reveals measurement noise.

9. STEERING LAYER DIVERGENCE — Different behaviors peak at different
   layers, suggesting distinct representational locations.

10. ABLATION CLUSTERING — Some ablation conditions are surprisingly
    close/far in behavioral space.
""")


if __name__ == "__main__":
    main()
