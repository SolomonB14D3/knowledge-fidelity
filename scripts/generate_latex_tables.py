#!/usr/bin/env python3
r"""Generate publication-ready LaTeX tables from the master SQLite database.

Each table is written as a standalone .tex file that can be included in a
paper via \input{tables/ablation.tex}.

Modes:
  --paper    Rounded numbers, significance stars, bold best values (default)
  --detailed Full precision, no formatting

Outputs (to paper/tables/):
  dose_response.tex   — Table 1: Dose-response by lambda_rho
  ablation.tex        — Table 2: 5-seed ablation (4 conditions)
  variance.tex        — Table 3: Variance (sigma) by lambda_rho
  hybrid_sweep.tex    — Table 4: Hybrid control sweep (0.5B)
  correlation.tex     — Table 5: Cross-behavior Spearman correlation

Usage:
  python scripts/generate_latex_tables.py
  python scripts/generate_latex_tables.py --detailed
  python scripts/generate_latex_tables.py --refresh-db
"""
import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEHAVIORS = ["factual", "toxicity", "bias", "sycophancy"]
BEHAVIORS_FULL = ["factual", "toxicity", "bias", "sycophancy", "refusal"]


def connect(db_path: str) -> sqlite3.Connection:
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    return db


# ── Formatting helpers ─────────────────────────────────────────────────────

def _fmt(val, decimals=3, sign=True, bold=False):
    """Format a float for LaTeX."""
    if val is None or np.isnan(val):
        return "---"
    s = f"{val:+.{decimals}f}" if sign else f"{val:.{decimals}f}"
    if bold:
        return r"\textbf{" + s + "}"
    return s


def _sig_stars(p):
    """Return significance stars (raw LaTeX, no $ delimiters)."""
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return r"^{***}"
    if p < 0.01:
        return r"^{**}"
    if p < 0.05:
        return r"^{*}"
    return ""


def _pm(mu, std, decimals=3, sign=True, stars=""):
    """Format mean +/- std for LaTeX.  Stars go inside the math env."""
    m = f"{mu:+.{decimals}f}" if sign else f"{mu:.{decimals}f}"
    s = f"{std:.{decimals}f}"
    return f"${m} \\pm {s}{stars}$"


# ── Table 1: Dose-Response ─────────────────────────────────────────────────

def table_dose_response(db, paper_mode: bool) -> str:
    rows = db.execute("""
        SELECT rho_weight, d_factual, d_toxicity, d_bias, d_sycophancy, seed
        FROM alignment_runs
        WHERE source_file LIKE '%mlx_rho_sft_sweep%7B%'
          AND d_factual IS NOT NULL
        ORDER BY rho_weight, seed
    """).fetchall()

    by_rw = defaultdict(list)
    for r in rows:
        by_rw[r["rho_weight"]].append(r)

    rws = sorted(by_rw.keys())
    dec = 3 if paper_mode else 4

    # Find best per column for bolding
    means = {}
    for rw in rws:
        runs = by_rw[rw]
        means[rw] = {b: np.mean([float(r[f"d_{b}"]) for r in runs]) for b in BEHAVIORS}

    best = {b: max(rws, key=lambda rw: means[rw][b]) for b in BEHAVIORS}

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Dose-response by $\lambda_\rho$ (Qwen2.5-7B-Instruct, 5 seeds).}")
    lines.append(r"\label{tab:dose_response}")
    lines.append(r"\begin{tabular}{c c r r r r}")
    lines.append(r"\toprule")
    lines.append(r"$\lambda_\rho$ & $N$ & Factual $\Delta\rho$ & Toxicity $\Delta\rho$ & Bias $\Delta\rho$ & Sycophancy $\Delta\rho$ \\")
    lines.append(r"\midrule")

    for rw in rws:
        runs = by_rw[rw]
        n = len(runs)
        cells = [f"{rw:.1f}", str(n)]
        for b in BEHAVIORS:
            vals = [float(r[f"d_{b}"]) for r in runs]
            mu = np.mean(vals)
            is_best = paper_mode and (rw == best[b])
            cells.append(_fmt(mu, dec, sign=True, bold=is_best))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Table 2: Ablation ─────────────────────────────────────────────────────

def table_ablation(db, paper_mode: bool) -> str:
    rows = db.execute("""
        SELECT condition, d_factual, d_toxicity, d_bias, d_sycophancy,
               d_refusal, seed
        FROM alignment_runs
        WHERE condition IS NOT NULL AND condition != ''
          AND d_factual IS NOT NULL
    """).fetchall()

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    cond_order = ["rho-guided", "sft-only", "contrastive-only", "shuffled-pairs"]
    dec = 3 if paper_mode else 4

    # Compute p-values vs sft-only for each behavior
    sft_vals = {b: [float(r[f"d_{b}"]) for r in by_cond["sft-only"]
                     if r[f"d_{b}"] is not None]
                for b in BEHAVIORS_FULL}

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study (5 seeds, Qwen2.5-7B-Instruct). " +
                 r"Significance vs.\ SFT-only: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{l c r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Condition & $N$ & Factual & Toxicity & Bias & Sycophancy & Refusal \\")
    lines.append(r"\midrule")

    # Find best mean per column
    cond_means = {}
    for cond in cond_order:
        if cond not in by_cond:
            continue
        runs = by_cond[cond]
        cond_means[cond] = {b: np.mean([float(r[f"d_{b}"]) for r in runs
                                         if r[f"d_{b}"] is not None])
                            for b in BEHAVIORS_FULL}

    best = {}
    for b in BEHAVIORS_FULL:
        valid = {c: cond_means[c][b] for c in cond_order if c in cond_means}
        best[b] = max(valid, key=valid.get) if valid else None

    for cond in cond_order:
        if cond not in by_cond:
            continue
        runs = by_cond[cond]
        n = len(runs)
        cond_label = cond.replace("-", " ").title()
        if cond == "rho-guided":
            cond_label = r"Rho-Guided"
        elif cond == "sft-only":
            cond_label = "SFT-Only"
        elif cond == "contrastive-only":
            cond_label = "Contrastive-Only"
        elif cond == "shuffled-pairs":
            cond_label = "Shuffled Pairs"

        cells = [cond_label, str(n)]
        for b in BEHAVIORS_FULL:
            vals = [float(r[f"d_{b}"]) for r in runs if r[f"d_{b}"] is not None]
            if not vals:
                cells.append("---")
                continue
            mu = np.mean(vals)
            std = np.std(vals, ddof=1) if len(vals) >= 2 else 0.0

            is_best = paper_mode and (cond == best.get(b))

            if paper_mode and cond != "sft-only" and len(vals) >= 2:
                # t-test vs sft-only
                sft = sft_vals.get(b, [])
                if len(sft) >= 2:
                    _, p = sp_stats.ttest_ind(vals, sft, equal_var=False)
                    stars = _sig_stars(p)
                else:
                    stars = ""
                cell = _pm(mu, std, dec, sign=True, stars=stars)
                if is_best:
                    cell = r"\textbf{" + cell + "}"
            else:
                cell = _pm(mu, std, dec, sign=True)
                if is_best:
                    cell = r"\textbf{" + cell + "}"
            cells.append(cell)

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Table 3: Variance by lambda_rho ───────────────────────────────────────

def table_variance(db, paper_mode: bool) -> str:
    rows = db.execute("""
        SELECT rho_weight, d_factual, d_toxicity, d_bias
        FROM alignment_runs
        WHERE source_file LIKE '%mlx_rho_sft_sweep%7B%'
          AND d_factual IS NOT NULL
        ORDER BY rho_weight
    """).fetchall()

    by_rw = defaultdict(list)
    for r in rows:
        by_rw[r["rho_weight"]].append(r)

    rws = sorted(by_rw.keys())
    dec = 3 if paper_mode else 4

    # Find min sigma for bolding
    sigmas = {}
    for rw in rws:
        fds = [float(r["d_factual"]) for r in by_rw[rw]]
        sigmas[rw] = np.std(fds, ddof=1) if len(fds) >= 2 else 999
    min_sig_rw = min(rws, key=lambda rw: sigmas[rw])

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Seed variance ($\sigma$) by $\lambda_\rho$ (Qwen2.5-7B-Instruct, 5 seeds). " +
                 r"Minimum factual $\sigma$ at $\lambda_\rho = 0.2$ demonstrates variance collapse.}")
    lines.append(r"\label{tab:variance}")
    lines.append(r"\begin{tabular}{c c r r r r}")
    lines.append(r"\toprule")
    lines.append(r"$\lambda_\rho$ & $N$ & Factual $\sigma$ & Toxicity $\sigma$ & Bias $\sigma$ & Factual $\mu$ \\")
    lines.append(r"\midrule")

    for rw in rws:
        runs = by_rw[rw]
        n = len(runs)
        fds = [float(r["d_factual"]) for r in runs]
        tds = [float(r["d_toxicity"]) for r in runs if r["d_toxicity"] is not None]
        bds = [float(r["d_bias"]) for r in runs if r["d_bias"] is not None]

        f_sig = np.std(fds, ddof=1) if len(fds) >= 2 else 0
        t_sig = np.std(tds, ddof=1) if len(tds) >= 2 else 0
        b_sig = np.std(bds, ddof=1) if len(bds) >= 2 else 0
        f_mu = np.mean(fds)

        is_min = paper_mode and (rw == min_sig_rw)
        f_sig_str = _fmt(f_sig, dec, sign=False, bold=is_min)

        cells = [
            f"{rw:.1f}", str(n),
            f_sig_str,
            _fmt(t_sig, dec, sign=False),
            _fmt(b_sig, dec, sign=False),
            _fmt(f_mu, dec, sign=True),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Table 4: Hybrid Control Sweep ─────────────────────────────────────────

def table_hybrid_sweep(db, paper_mode: bool) -> str:
    rows = db.execute("""
        SELECT config_tag, mean_before, mean_after, mean_delta,
               d_factual, d_toxicity, d_bias, d_sycophancy
        FROM hybrid_sweep
        ORDER BY mean_after DESC
    """).fetchall()

    dec = 3 if paper_mode else 4

    # Find best mean_delta for bolding
    best_delta = max(r["mean_delta"] for r in rows) if rows else 0

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Hybrid control sweep (Qwen2.5-0.5B, 7 configurations). " +
                 r"SVD $\times$ SAE $\times$ Rho-SFT grid.}")
    lines.append(r"\label{tab:hybrid_sweep}")
    lines.append(r"\begin{tabular}{l r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Config & Mean $\Delta\rho$ & Factual & Toxicity & Bias & Sycophancy \\")
    lines.append(r"\midrule")

    for r in rows:
        tag = r["config_tag"]
        # Make config tag readable
        tag_tex = tag.replace("_", r"\_")
        is_best = paper_mode and abs(r["mean_delta"] - best_delta) < 1e-6
        delta_str = _fmt(r["mean_delta"], dec, sign=True, bold=is_best)

        cells = [
            tag_tex,
            delta_str,
            _fmt(r["d_factual"], dec, sign=True),
            _fmt(r["d_toxicity"], dec, sign=True),
            _fmt(r["d_bias"], dec, sign=True),
            _fmt(r["d_sycophancy"], dec, sign=True),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Table 5: Correlation Matrix ───────────────────────────────────────────

def table_correlation(db, paper_mode: bool) -> str:
    import pandas as pd

    rows = db.execute("""
        SELECT d_factual, d_toxicity, d_bias, d_sycophancy, d_refusal
        FROM alignment_runs
        WHERE d_factual IS NOT NULL
    """).fetchall()

    data = {b: [] for b in BEHAVIORS_FULL}
    for r in rows:
        for b in BEHAVIORS_FULL:
            val = r[f"d_{b}"]
            data[b].append(float(val) if val is not None else np.nan)

    df = pd.DataFrame(data)
    # Drop zero-variance columns
    df = df.loc[:, df.fillna(0).std() > 1e-12]
    available = [b for b in BEHAVIORS_FULL if b in df.columns]

    corr = df[available].corr(method="spearman")
    n = len(df)
    dec = 2 if paper_mode else 3

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-behavior Spearman correlation ($N = " + str(n) +
                 r"$ runs). Bold: $|\rho| \geq 0.3$.}")
    lines.append(r"\label{tab:correlation}")

    col_spec = "l" + " r" * len(available)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header
    header_cells = [""] + [b.capitalize() for b in available]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for b1 in available:
        cells = [b1.capitalize()]
        for b2 in available:
            v = corr.loc[b1, b2]
            if b1 == b2:
                cells.append("1.00" if dec == 2 else "1.000")
            else:
                bold = paper_mode and abs(v) >= 0.3
                cells.append(_fmt(v, dec, sign=True, bold=bold))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready LaTeX tables from master DB"
    )
    parser.add_argument("--db", default="results/master.db", help="Path to master.db")
    parser.add_argument("--out", default="paper/tables", help="Output directory")
    parser.add_argument(
        "--detailed", action="store_true",
        help="Full precision, no bolding or significance stars",
    )
    parser.add_argument(
        "--refresh-db", action="store_true",
        help="Re-run build_master_db.py before generating tables",
    )
    args = parser.parse_args()

    paper_mode = not args.detailed

    if args.refresh_db:
        import subprocess
        etl_script = Path(__file__).parent / "build_master_db.py"
        print(f"Rebuilding master DB via {etl_script}...")
        subprocess.run([sys.executable, str(etl_script)], check=True)
        print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = connect(args.db)
    mode_str = "paper" if paper_mode else "detailed"
    print(f"Generating LaTeX tables ({mode_str} mode)")
    print(f"  DB: {args.db}")
    print(f"  Output: {out_dir}/")
    print()

    tables = [
        ("dose_response.tex", "Dose-Response", table_dose_response),
        ("ablation.tex", "Ablation", table_ablation),
        ("variance.tex", "Variance", table_variance),
        ("hybrid_sweep.tex", "Hybrid Sweep", table_hybrid_sweep),
        ("correlation.tex", "Correlation Matrix", table_correlation),
    ]

    for filename, label, func in tables:
        tex = func(db, paper_mode)
        path = out_dir / filename
        path.write_text(tex + "\n")
        print(f"  [{label}] -> {path}")

    db.close()

    print(f"\nDone! Include in your paper:")
    for filename, _, _ in tables:
        print(f"  \\input{{tables/{filename.replace('.tex', '')}}}")


if __name__ == "__main__":
    main()
