#!/usr/bin/env python3
"""Auto-generate paper-ready tables, correlation matrices, and variance plots
from the master SQLite database.

Outputs:
  docs/auto_report.md   — Markdown tables + stats for README/paper
  docs/dashboard.html   — Interactive Plotly dashboard (offline)

Usage:
  python scripts/generate_report.py
  python scripts/generate_report.py --db results/master.db --out docs/

Requires: pandas, numpy, scipy, matplotlib, plotly
"""
import argparse
import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# ── Constants ──────────────────────────────────────────────────────────────
BEHAVIORS = ["factual", "toxicity", "bias", "sycophancy", "reasoning",
             "refusal", "deception", "overrefusal"]

CONDITIONS = ["sft-only", "rho-guided", "contrastive-only", "shuffled-pairs"]


def connect(db_path: str) -> sqlite3.Connection:
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    return db


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Paper-ready tables
# ═══════════════════════════════════════════════════════════════════════════

def table_dose_response(db) -> str:
    """Table 1: Dose-response by λ_ρ (Qwen 7B, all seeds combined)."""
    rows = db.execute("""
        SELECT rho_weight,
               AVG(d_factual) as f_mu, AVG(d_toxicity) as t_mu,
               AVG(d_bias) as b_mu, AVG(d_sycophancy) as s_mu,
               COUNT(*) as n
        FROM alignment_runs
        WHERE source_file LIKE '%mlx_rho_sft_sweep%7B%'
          AND d_factual IS NOT NULL
        GROUP BY rho_weight
        ORDER BY rho_weight
    """).fetchall()

    lines = ["### Table: Dose-Response by λ_ρ (Qwen2.5-7B-Instruct, all seeds)",
             "",
             "| λ_ρ | N | Factual Δρ | Toxicity Δρ | Bias Δρ | Sycophancy Δρ |",
             "|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for r in rows:
        rw = f"{r['rho_weight']:.1f}" if r['rho_weight'] is not None else "?"
        lines.append(
            f"| {rw} | {r['n']} | {r['f_mu']:+.4f} | {r['t_mu']:+.4f} "
            f"| {r['b_mu']:+.4f} | {r['s_mu']:+.4f} |"
        )
    return "\n".join(lines)


def table_ablation(db) -> str:
    """Table 2: Ablation conditions (5-seed means ± std)."""
    rows = db.execute("""
        SELECT condition,
               AVG(d_factual) as f_mu, AVG(d_toxicity) as t_mu,
               AVG(d_bias) as b_mu, AVG(d_sycophancy) as s_mu,
               COUNT(*) as n
        FROM alignment_runs
        WHERE condition IS NOT NULL AND condition != ''
          AND d_factual IS NOT NULL
        GROUP BY condition
        ORDER BY f_mu DESC
    """).fetchall()

    # Also get stdevs
    all_rows = db.execute("""
        SELECT condition, d_factual, d_toxicity, d_bias, d_sycophancy
        FROM alignment_runs
        WHERE condition IS NOT NULL AND condition != ''
          AND d_factual IS NOT NULL
    """).fetchall()

    by_cond = defaultdict(list)
    for r in all_rows:
        by_cond[r['condition']].append(r)

    lines = ["### Table: Ablation Conditions (all seeds)",
             "",
             "| Condition | N | Factual Δρ | Toxicity Δρ | Bias Δρ | Sycophancy Δρ |",
             "|:---|:---:|:---:|:---:|:---:|:---:|"]
    for r in rows:
        cond = r['condition']
        runs = by_cond[cond]
        n = len(runs)
        fds = [x['d_factual'] for x in runs if x['d_factual'] is not None]
        tds = [x['d_toxicity'] for x in runs if x['d_toxicity'] is not None]
        bds = [x['d_bias'] for x in runs if x['d_bias'] is not None]
        sds = [x['d_sycophancy'] for x in runs if x['d_sycophancy'] is not None]

        def fmt(vals):
            if len(vals) < 2:
                return f"{np.mean(vals):+.3f}" if vals else "—"
            return f"{np.mean(vals):+.3f} ± {np.std(vals, ddof=1):.3f}"

        lines.append(f"| {cond} | {n} | {fmt(fds)} | {fmt(tds)} | {fmt(bds)} | {fmt(sds)} |")
    return "\n".join(lines)


def table_hybrid_sweep(db) -> str:
    """Table 3: Hybrid control sweep (0.5B)."""
    rows = db.execute("""
        SELECT config_tag, mean_before, mean_after, mean_delta,
               d_factual, d_toxicity, d_bias, d_sycophancy
        FROM hybrid_sweep
        ORDER BY mean_after DESC
    """).fetchall()

    lines = ["### Table: Hybrid Control Sweep (Qwen2.5-0.5B, 7 configs)",
             "",
             "| Config | Mean ρ (after) | Mean Δ | Factual | Toxicity | Bias | Sycophancy |",
             "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for r in rows:
        lines.append(
            f"| {r['config_tag']} | {r['mean_after']:.4f} | {r['mean_delta']:+.4f} "
            f"| {r['d_factual']:+.3f} | {r['d_toxicity']:+.3f} "
            f"| {r['d_bias']:+.3f} | {r['d_sycophancy']:+.3f} |"
        )
    return "\n".join(lines)


def table_variance_by_lambda(db) -> str:
    """Table 4: Variance (σ) by λ_ρ — shows U-shape."""
    rows = db.execute("""
        SELECT rho_weight, d_factual, d_toxicity, d_bias
        FROM alignment_runs
        WHERE source_file LIKE '%mlx_rho_sft_sweep%7B%'
          AND d_factual IS NOT NULL
        ORDER BY rho_weight
    """).fetchall()

    by_rw = defaultdict(list)
    for r in rows:
        by_rw[r['rho_weight']].append(r)

    lines = ["### Table: Variance by λ_ρ (Qwen2.5-7B, 5 seeds)",
             "",
             "| λ_ρ | N | Factual σ | Toxicity σ | Bias σ | Factual μ |",
             "|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for rw in sorted(by_rw.keys()):
        runs = by_rw[rw]
        n = len(runs)
        fds = [r['d_factual'] for r in runs]
        tds = [r['d_toxicity'] for r in runs if r['d_toxicity'] is not None]
        bds = [r['d_bias'] for r in runs if r['d_bias'] is not None]
        f_sig = np.std(fds, ddof=1) if n >= 2 else 0
        t_sig = np.std(tds, ddof=1) if len(tds) >= 2 else 0
        b_sig = np.std(bds, ddof=1) if len(bds) >= 2 else 0
        f_mu = np.mean(fds)
        lines.append(f"| {rw:.1f} | {n} | {f_sig:.4f} | {t_sig:.4f} | {b_sig:.4f} | {f_mu:+.4f} |")
    return "\n".join(lines)


def table_summary_stats(db) -> str:
    """Table 5: Summary statistics per behavior per condition."""
    rows = db.execute("""
        SELECT condition, d_factual, d_toxicity, d_bias, d_sycophancy,
               d_reasoning, d_refusal, d_deception, d_overrefusal
        FROM alignment_runs
        WHERE condition IS NOT NULL AND condition != ''
    """).fetchall()

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r['condition']].append(dict(r))

    lines = ["### Table: Summary Statistics (min/max/mean/std per behavior per condition)",
             ""]

    for cond in CONDITIONS:
        if cond not in by_cond:
            continue
        runs = by_cond[cond]
        n = len(runs)
        lines.append(f"\n**{cond}** (n={n})")
        lines.append("")
        lines.append("| Behavior | Min | Max | Mean | Std |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")

        for beh in BEHAVIORS:
            key = f"d_{beh}"
            vals = [r[key] for r in runs if r.get(key) is not None]
            if not vals or all(v == 0 for v in vals):
                continue
            lines.append(
                f"| {beh} | {min(vals):+.4f} | {max(vals):+.4f} "
                f"| {np.mean(vals):+.4f} | {np.std(vals, ddof=1):.4f} |"
            )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Correlation matrix
# ═══════════════════════════════════════════════════════════════════════════

def correlation_matrix(db) -> tuple[str, pd.DataFrame]:
    """Cross-behavior Spearman correlation matrix across all alignment runs."""
    rows = db.execute("""
        SELECT d_factual, d_toxicity, d_bias, d_sycophancy,
               d_reasoning, d_refusal, d_deception, d_overrefusal
        FROM alignment_runs
        WHERE d_factual IS NOT NULL
    """).fetchall()

    data = {beh: [] for beh in BEHAVIORS}
    for r in rows:
        for beh in BEHAVIORS:
            val = r[f"d_{beh}"]
            data[beh].append(val if val is not None else np.nan)

    df = pd.DataFrame(data)
    # Drop columns that are all NaN or have zero variance (→ NaN corr)
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, df.fillna(0).std() > 1e-12]  # drop zero-variance columns
    available = [b for b in BEHAVIORS if b in df.columns]

    # Spearman correlation
    corr = df[available].corr(method='spearman')

    lines = ["### Correlation Matrix (Spearman ρ across all alignment runs)",
             "",
             f"N = {len(df)} runs. Shows whether gains in one behavior predict gains/losses in another.",
             ""]

    # Build markdown table
    header = "| | " + " | ".join(available) + " |"
    sep = "|:---|" + "|".join([":---:" for _ in available]) + "|"
    lines.extend([header, sep])
    for b1 in available:
        vals = []
        for b2 in available:
            v = corr.loc[b1, b2]
            if b1 == b2:
                vals.append("1.000")
            elif abs(v) >= 0.3:
                vals.append(f"**{v:+.3f}**")
            else:
                vals.append(f"{v:+.3f}")
        lines.append(f"| {b1} | " + " | ".join(vals) + " |")

    return "\n".join(lines), corr


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Plots (matplotlib for static, plotly for dashboard)
# ═══════════════════════════════════════════════════════════════════════════

def plot_variance_ushape(db, out_dir: Path):
    """Plot factual σ vs λ_ρ showing U-shape."""
    import matplotlib.pyplot as plt

    rows = db.execute("""
        SELECT rho_weight, d_factual
        FROM alignment_runs
        WHERE source_file LIKE '%mlx_rho_sft_sweep%7B%'
          AND d_factual IS NOT NULL
        ORDER BY rho_weight
    """).fetchall()

    by_rw = defaultdict(list)
    for r in rows:
        by_rw[r['rho_weight']].append(r['d_factual'])

    rws = sorted(by_rw.keys())
    sigmas = [np.std(by_rw[rw], ddof=1) for rw in rws]
    means = [np.mean(by_rw[rw]) for rw in rws]
    ns = [len(by_rw[rw]) for rw in rws]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Variance plot
    ax1.bar(range(len(rws)), sigmas, color=['#e74c3c' if s == max(sigmas) else
            '#27ae60' if s == min(sigmas) else '#3498db' for s in sigmas],
            alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(len(rws)))
    ax1.set_xticklabels([f"λ={rw:.1f}" for rw in rws])
    ax1.set_ylabel("Factual Δρ Standard Deviation (σ)")
    ax1.set_title("Variance U-Shape: Sweet Spot at λ_ρ ≈ 0.2")
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    for i, (s, n) in enumerate(zip(sigmas, ns)):
        ax1.text(i, s + 0.003, f"σ={s:.3f}\nn={n}", ha='center', fontsize=8)

    # Jitter plot of individual seeds
    for i, rw in enumerate(rws):
        vals = by_rw[rw]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax2.scatter([i + j for j in jitter], vals, alpha=0.6, s=40,
                   color='#3498db', edgecolor='black', linewidth=0.5)
        ax2.plot([i - 0.2, i + 0.2], [np.mean(vals)] * 2, 'r-', linewidth=2)

    ax2.set_xticks(range(len(rws)))
    ax2.set_xticklabels([f"λ={rw:.1f}" for rw in rws])
    ax2.set_ylabel("Factual Δρ (per seed)")
    ax2.set_title("Individual Seeds: Variance Collapse at λ_ρ = 0.2")
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "variance_ushape.png", dpi=150, bbox_inches='tight')
    plt.close()
    return out_dir / "variance_ushape.png"


def plot_ablation_bars(db, out_dir: Path):
    """Bar chart of ablation conditions across behaviors."""
    import matplotlib.pyplot as plt

    rows = db.execute("""
        SELECT condition, d_factual, d_toxicity, d_bias, d_sycophancy
        FROM alignment_runs
        WHERE condition IS NOT NULL AND condition != ''
          AND d_factual IS NOT NULL
    """).fetchall()

    by_cond = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = r['condition']
        for beh in ['factual', 'toxicity', 'bias', 'sycophancy']:
            v = r[f'd_{beh}']
            if v is not None:
                by_cond[c][beh].append(v)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(['factual', 'toxicity', 'bias', 'sycophancy']))
    width = 0.2
    colors = {'rho-guided': '#27ae60', 'sft-only': '#e74c3c',
              'contrastive-only': '#3498db', 'shuffled-pairs': '#95a5a6'}

    for i, cond in enumerate(CONDITIONS):
        if cond not in by_cond:
            continue
        means = [np.mean(by_cond[cond].get(b, [0])) for b in ['factual', 'toxicity', 'bias', 'sycophancy']]
        stds = [np.std(by_cond[cond].get(b, [0]), ddof=1) if len(by_cond[cond].get(b, [])) >= 2 else 0
                for b in ['factual', 'toxicity', 'bias', 'sycophancy']]
        ax.bar(x + i * width, means, width, yerr=stds, label=cond,
               color=colors.get(cond, '#999'), alpha=0.85, capsize=3,
               edgecolor='black', linewidth=0.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['Factual', 'Toxicity', 'Bias', 'Sycophancy'])
    ax.set_ylabel('Δρ from Baseline')
    ax.set_title('Ablation: Behavioral Changes by Condition')
    ax.legend(loc='upper left', fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "ablation_conditions.png", dpi=150, bbox_inches='tight')
    plt.close()
    return out_dir / "ablation_conditions.png"


def plot_hybrid_heatmap(db, out_dir: Path):
    """Heatmap of hybrid sweep deltas by behavior."""
    import matplotlib.pyplot as plt

    rows = db.execute("""
        SELECT config_tag, d_factual, d_toxicity, d_bias, d_sycophancy,
               d_reasoning, d_refusal, d_deception, d_overrefusal
        FROM hybrid_sweep
        ORDER BY mean_after DESC
    """).fetchall()

    tags = [r['config_tag'] for r in rows]
    behs = BEHAVIORS
    data = np.array([[r[f'd_{b}'] if r[f'd_{b}'] is not None else 0 for b in behs] for r in rows])

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-0.07, vmax=0.1)
    ax.set_xticks(range(len(behs)))
    ax.set_xticklabels([b[:8] for b in behs], rotation=45, ha='right')
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags, fontsize=8)
    ax.set_title("Hybrid Sweep: Per-Behavior Δρ (green = gain, red = loss)")
    plt.colorbar(im, ax=ax, label='Δρ')

    for i in range(len(tags)):
        for j in range(len(behs)):
            ax.text(j, i, f"{data[i,j]:+.3f}", ha='center', va='center', fontsize=7,
                   color='white' if abs(data[i, j]) > 0.04 else 'black')

    plt.tight_layout()
    fig.savefig(out_dir / "hybrid_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    return out_dir / "hybrid_heatmap.png"


def plot_correlation_heatmap(corr: pd.DataFrame, out_dir: Path):
    """Heatmap of cross-behavior correlations."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    behs = list(corr.columns)
    ax.set_xticks(range(len(behs)))
    ax.set_xticklabels([b[:8] for b in behs], rotation=45, ha='right')
    ax.set_yticks(range(len(behs)))
    ax.set_yticklabels([b[:8] for b in behs])
    ax.set_title("Cross-Behavior Spearman Correlation")
    plt.colorbar(im, ax=ax, label='Spearman ρ')

    for i in range(len(behs)):
        for j in range(len(behs)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha='center', va='center', fontsize=8,
                   color='white' if abs(corr.values[i, j]) > 0.5 else 'black')

    plt.tight_layout()
    fig.savefig(out_dir / "correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    return out_dir / "correlation_matrix.png"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: HTML Dashboard (Plotly)
# ═══════════════════════════════════════════════════════════════════════════

def generate_dashboard(db, out_dir: Path):
    """Generate interactive HTML dashboard with Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ── Data extraction ──
    # 1. Dose-response
    dose_rows = db.execute("""
        SELECT rho_weight, d_factual, d_toxicity, d_bias, d_sycophancy, seed
        FROM alignment_runs
        WHERE source_file LIKE '%mlx_rho_sft_sweep%7B%'
          AND d_factual IS NOT NULL
        ORDER BY rho_weight
    """).fetchall()

    # 2. Ablation
    abl_rows = db.execute("""
        SELECT condition, d_factual, d_toxicity, d_bias, d_sycophancy
        FROM alignment_runs
        WHERE condition IS NOT NULL AND condition != '' AND d_factual IS NOT NULL
    """).fetchall()

    # 3. Hybrid sweep
    hybrid_rows = db.execute("""
        SELECT config_tag, mean_before, mean_after, mean_delta,
               d_factual, d_toxicity, d_bias, d_sycophancy,
               d_reasoning, d_refusal, d_deception, d_overrefusal
        FROM hybrid_sweep ORDER BY mean_after DESC
    """).fetchall()

    # 4. Attack/defense
    atk_rows = db.execute("""
        SELECT behavior, scale, rho, delta, n_regressed, mean_collateral
        FROM attack_defense ORDER BY behavior, scale
    """).fetchall()

    # ── Build dashboard ──
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Dose-Response: Factual Δρ by λ_ρ",
            "Variance U-Shape (Factual σ)",
            "Ablation Conditions",
            "Hybrid Sweep: Mean Δρ",
            "Attack/Defense: Refusal vs Deception",
            "Cross-Behavior Correlation"
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
    )

    # Panel 1: Dose-response scatter
    by_rw = defaultdict(list)
    for r in dose_rows:
        by_rw[r['rho_weight']].append(r['d_factual'])

    rws = sorted(by_rw.keys())
    means = [np.mean(by_rw[rw]) for rw in rws]
    stds = [np.std(by_rw[rw], ddof=1) if len(by_rw[rw]) >= 2 else 0 for rw in rws]
    fig.add_trace(go.Scatter(
        x=rws, y=means, mode='lines+markers',
        error_y=dict(type='data', array=stds, visible=True),
        name='Factual Δρ', line=dict(color='#27ae60', width=2),
        marker=dict(size=8)
    ), row=1, col=1)

    # Panel 2: Variance U-shape
    sigmas = [np.std(by_rw[rw], ddof=1) for rw in rws]
    colors = ['#27ae60' if s == min(sigmas) else '#e74c3c' if s == max(sigmas) else '#3498db'
              for s in sigmas]
    fig.add_trace(go.Bar(
        x=[f"λ={rw:.1f}" for rw in rws], y=sigmas,
        marker_color=colors, name='Factual σ',
        text=[f"σ={s:.3f}" for s in sigmas], textposition='outside'
    ), row=1, col=2)

    # Panel 3: Ablation bars
    by_cond = defaultdict(lambda: defaultdict(list))
    for r in abl_rows:
        for b in ['factual', 'toxicity', 'bias', 'sycophancy']:
            v = r[f'd_{b}']
            if v is not None:
                by_cond[r['condition']][b].append(v)

    abl_colors = {'rho-guided': '#27ae60', 'sft-only': '#e74c3c',
                  'contrastive-only': '#3498db', 'shuffled-pairs': '#95a5a6'}
    for cond in CONDITIONS:
        if cond not in by_cond:
            continue
        behs_short = ['factual', 'toxicity', 'bias', 'sycophancy']
        vals = [np.mean(by_cond[cond].get(b, [0])) for b in behs_short]
        fig.add_trace(go.Bar(
            x=behs_short, y=vals, name=cond,
            marker_color=abl_colors.get(cond, '#999')
        ), row=2, col=1)

    # Panel 4: Hybrid sweep
    tags = [r['config_tag'] for r in hybrid_rows]
    deltas = [r['mean_delta'] for r in hybrid_rows]
    h_colors = ['#27ae60' if d > 0 else '#e74c3c' for d in deltas]
    fig.add_trace(go.Bar(
        x=tags, y=deltas, marker_color=h_colors, name='Mean Δρ',
        text=[f"{d:+.4f}" for d in deltas], textposition='outside'
    ), row=2, col=2)

    # Panel 5: Attack/defense curves
    for beh in ['refusal', 'deception']:
        beh_rows = [r for r in atk_rows if r['behavior'] == beh]
        if beh_rows:
            scales = [r['scale'] for r in beh_rows]
            rhos = [r['rho'] for r in beh_rows]
            fig.add_trace(go.Scatter(
                x=scales, y=rhos, mode='lines+markers', name=beh.title(),
                line=dict(width=2)
            ), row=3, col=1)

    # Panel 6: Correlation heatmap
    corr_rows = db.execute("""
        SELECT d_factual, d_toxicity, d_bias, d_sycophancy
        FROM alignment_runs WHERE d_factual IS NOT NULL
    """).fetchall()
    corr_data = {b: [r[f'd_{b}'] for r in corr_rows]
                 for b in ['factual', 'toxicity', 'bias', 'sycophancy']}
    corr_df = pd.DataFrame(corr_data).corr(method='spearman')
    behs_short = list(corr_df.columns)
    fig.add_trace(go.Heatmap(
        z=corr_df.values, x=behs_short, y=behs_short,
        colorscale='RdBu_r', zmin=-1, zmax=1,
        text=[[f"{corr_df.values[i][j]:.2f}" for j in range(len(behs_short))]
              for i in range(len(behs_short))],
        texttemplate="%{text}", textfont={"size": 12},
    ), row=3, col=2)

    fig.update_layout(
        height=1200, width=1100,
        title_text=f"rho-eval Research Dashboard — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        showlegend=True,
        template='plotly_white',
    )

    html_path = out_dir / "dashboard.html"
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    return html_path


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate auto-report from master DB")
    parser.add_argument("--db", default="results/master.db", help="Path to master.db")
    parser.add_argument("--out", default="docs", help="Output directory")
    args = parser.parse_args()

    db_path = args.db
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating report from: {db_path}")
    print(f"Output directory: {out_dir}")

    db = connect(db_path)

    # ── Markdown report ──
    sections = []
    sections.append(f"# rho-eval Auto-Generated Report")
    sections.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    sections.append(f"*Source: {db_path}*\n")

    print("  [1/6] Dose-response table...")
    sections.append(table_dose_response(db))

    print("  [2/6] Ablation table...")
    sections.append(table_ablation(db))

    print("  [3/6] Hybrid sweep table...")
    sections.append(table_hybrid_sweep(db))

    print("  [4/6] Variance table...")
    sections.append(table_variance_by_lambda(db))

    print("  [5/6] Summary statistics...")
    sections.append(table_summary_stats(db))

    print("  [6/6] Correlation matrix...")
    corr_md, corr_df = correlation_matrix(db)
    sections.append(corr_md)

    md_path = out_dir / "auto_report.md"
    md_path.write_text("\n\n".join(sections) + "\n")
    print(f"\n  Markdown: {md_path}")

    # ── Plots ──
    print("\n  Generating plots...")
    p1 = plot_variance_ushape(db, out_dir)
    print(f"    {p1}")
    p2 = plot_ablation_bars(db, out_dir)
    print(f"    {p2}")
    p3 = plot_hybrid_heatmap(db, out_dir)
    print(f"    {p3}")
    p4 = plot_correlation_heatmap(corr_df, out_dir)
    print(f"    {p4}")

    # ── Dashboard ──
    print("\n  Generating HTML dashboard...")
    dash = generate_dashboard(db, out_dir)
    print(f"    {dash}")

    db.close()

    print(f"\n{'='*60}")
    print(f"Done! Outputs:")
    print(f"  {md_path}")
    print(f"  {out_dir / 'variance_ushape.png'}")
    print(f"  {out_dir / 'ablation_conditions.png'}")
    print(f"  {out_dir / 'hybrid_heatmap.png'}")
    print(f"  {out_dir / 'correlation_matrix.png'}")
    print(f"  {dash}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
