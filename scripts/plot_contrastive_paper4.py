#!/usr/bin/env python3
"""Generate Paper 4 multi-panel figure: contrastive data injection results.

Panel A: Vanilla vs contrastive ρ bar chart (7M, all 8 behaviors)
Panel B: Dose-response curve (0%, 3%, 5%, 10%)
Panel C: Scale-ladder overlay — contrastive 7M/12M vs vanilla scaling curve
Panel D: eff_dim comparison (bias across injection rates, per layer)

Usage:
    python scripts/plot_contrastive_paper4.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "scale_ladder"
FIGURES = ROOT / "figures" / "paper4"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Hardcoded clean results (all verified 0% overlap) ──

VANILLA_7M = {
    "bias": 0.000, "sycophancy": 0.000, "deception": 0.393, "factual": 0.334,
    "overrefusal": 1.000, "refusal": 0.689, "toxicity": 0.553, "reasoning": 0.010,
}
CLEAN_7M_5PCT = {
    "bias": 0.433, "sycophancy": 0.513, "deception": 0.396, "factual": 0.305,
    "overrefusal": 1.000, "refusal": 0.691, "toxicity": 0.552, "reasoning": 0.000,
}

# Dose-response (7M, bias+sycophancy injection, all clean)
DOSE_RESPONSE = {
    0:  {"bias": 0.000, "sycophancy": 0.000, "factual": 0.334},
    3:  {"bias": 0.359, "sycophancy": 0.513, "factual": 0.304},
    5:  {"bias": 0.433, "sycophancy": 0.513, "factual": 0.305},
    10: {"bias": 0.397, "sycophancy": 0.500, "factual": 0.235},
}

# Scale ladder vanilla ρ
SCALE_VANILLA = {
    "7M":  {"bias": 0.000, "sycophancy": 0.000, "params": 7.3},
    "12M": {"bias": 0.000, "sycophancy": 0.000, "params": 12.4},
    "18M": {"bias": 0.133, "sycophancy": 0.000, "params": 17.7},
    "34M": {"bias": 0.238, "sycophancy": 0.300, "params": 33.9},
    "64M": {"bias": 0.087, "sycophancy": 0.300, "params": 64.1},
}

# Contrastive overlay points
CONTR_7M  = {"bias": 0.433, "sycophancy": 0.513, "params": 7.3}
CONTR_12M = {"bias": 0.380, "sycophancy": 0.520, "params": 12.4}

# Bias eff_dim by injection rate (7M, 4 layers)
BIAS_EFFDIM = {
    "0% (vanilla)": [1, 1, 1, 1],
    "3%":           [1, 1, 1, 1],
    "5%":           [1, 1, 2, 2],
    "10%":          [2, 2, 2, 3],
}


def panel_a(ax):
    """Panel A: Side-by-side ρ bar chart."""
    behaviors = ["bias", "sycophancy", "factual", "toxicity", "deception", "refusal"]
    x = np.arange(len(behaviors))
    w = 0.35

    v = [VANILLA_7M[b] for b in behaviors]
    c = [CLEAN_7M_5PCT[b] for b in behaviors]

    ax.bar(x - w/2, v, w, label="Vanilla 7M", color="#4A90D9", alpha=0.85)
    bars_c = ax.bar(x + w/2, c, w, label="Contrastive 5%", color="#E85D4A", alpha=0.85)

    for i, b in enumerate(behaviors):
        if b in ("bias", "sycophancy"):
            bars_c[i].set_edgecolor("black")
            bars_c[i].set_linewidth(2)
            delta = c[i] - v[i]
            ax.annotate(f"+{delta:.2f}", (x[i] + w/2, c[i] + 0.02),
                       ha="center", fontsize=8, fontweight="bold", color="#C0392B")

    ax.set_ylabel("Spearman ρ")
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("(a) Behavioral ρ: Vanilla vs Contrastive 5%", fontsize=10, fontweight="bold")


def panel_b(ax):
    """Panel B: Dose-response curve."""
    pcts = sorted(DOSE_RESPONSE.keys())
    for beh, color, marker in [
        ("bias", "#4A90D9", "o"),
        ("sycophancy", "#E85D4A", "s"),
        ("factual", "#50B050", "^"),
    ]:
        vals = [DOSE_RESPONSE[p][beh] for p in pcts]
        ax.plot(pcts, vals, f"{marker}-", color=color, label=beh,
                markersize=8, linewidth=2)

    # Mark optimum
    ax.axvline(x=5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.annotate("optimum", (5, 0.55), fontsize=8, color="gray", ha="center")

    ax.set_xlabel("Injection rate (%)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("(b) Dose-Response (7M, clean probes)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.05, 0.65)


def panel_c(ax):
    """Panel C: Scale ladder with contrastive overlay."""
    scales = ["7M", "12M", "18M", "34M", "64M"]
    params = [SCALE_VANILLA[s]["params"] for s in scales]

    for beh, color, marker in [
        ("bias", "#4A90D9", "o"),
        ("sycophancy", "#E85D4A", "s"),
    ]:
        vanilla_rhos = [SCALE_VANILLA[s][beh] for s in scales]
        ax.plot(params, vanilla_rhos, f"{marker}--", color=color,
                label=f"{beh} (vanilla)", alpha=0.5, markersize=6)

        # Contrastive overlays
        for pt, lbl in [(CONTR_7M, "7M"), (CONTR_12M, "12M")]:
            ax.plot(pt["params"], pt[beh], marker, color=color, markersize=13,
                    markeredgecolor="black", markeredgewidth=2, zorder=10)

    # Annotations
    ax.annotate("7M+contr\nρ=0.43/0.51", (7.3, 0.53), fontsize=7,
               fontweight="bold", color="#333", ha="center",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
    ax.annotate("12M+contr\nρ=0.38/0.52", (12.4, 0.54), fontsize=7,
               fontweight="bold", color="#333", ha="center",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("(c) Scale Ladder + Contrastive Overlay", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(-0.05, 0.65)
    ax.set_xlim(0, 70)


def panel_d(ax):
    """Panel D: Bias eff_dim by layer across injection rates."""
    layers = [0, 1, 2, 3]
    colors = {"0% (vanilla)": "#999999", "3%": "#7BB3E0", "5%": "#E85D4A", "10%": "#C0392B"}
    markers = {"0% (vanilla)": "x", "3%": "^", "5%": "o", "10%": "s"}

    for rate, dims in BIAS_EFFDIM.items():
        ax.plot(layers, dims, f"{markers[rate]}-", color=colors[rate],
                label=rate, markersize=8, linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Dimensionality")
    ax.set_title("(d) Bias eff_dim by Layer (7M)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, title="Injection rate")
    ax.set_xticks(layers)
    ax.set_ylim(0.5, 3.5)
    ax.set_yticks([1, 2, 3])


def main():
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[1, 0]))
    panel_d(fig.add_subplot(gs[1, 1]))

    fig.suptitle("Small Models Can Learn Complex Behaviors — They Just Need the Right Examples",
                 fontsize=13, fontweight="bold", y=0.98)

    for ext in ["png", "pdf"]:
        out = FIGURES / f"contrastive_injection_results.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")

    plt.close()


if __name__ == "__main__":
    main()
