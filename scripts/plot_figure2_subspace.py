#!/usr/bin/env python3
"""Generate Figure 2: Cross-model subspace alignment across developmental scales.

3-panel figure showing that the shared activation direction between bias-only
and sycophancy-only injection models exists at all scales (strongest at 3M),
but cross-dimensional behavioral transfer only appears at 5M+.

This is the "readout threshold, not representation threshold" figure.

Data source: results/developmental_sweep/cross_model_subspace_{3M,5M,7M}.json
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "developmental_sweep"
FIGURES = ROOT / "figures" / "paper4"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────
def load_result(scale):
    path = RESULTS / f"cross_model_subspace_{scale}.json"
    with open(path) as f:
        return json.load(f)

data_3m = load_result("3M")
data_5m = load_result("5M")
data_7m = load_result("7M")

# ── Extract figure data ──────────────────────────────────────────────
# For each scale: first layer and last layer values
# 3M: 2 layers (0, 1) → first=0, last=1
# 5M: 2 layers (0, 1) → first=0, last=1
# 7M: 4 layers (0, 1, 2, 3) → first=0, last=3

scales = ["3M", "5M", "7M"]
d_models = [64, 96, 128]
n_probes = [60, 312, 186]

# Top-1 SVD cosine (bia↔syc) — first and last layer
svd_cos_first = [
    data_3m["part3_top_singular"]["0"]["bia_vs_syc_cosine"],  # 0.759
    data_5m["part3_top_singular"]["0"]["bia_vs_syc_cosine"],  # 0.782
    data_7m["part3_top_singular"]["0"]["bia_vs_syc_cosine"],  # 0.642
]
svd_cos_last = [
    data_3m["part3_top_singular"]["1"]["bia_vs_syc_cosine"],  # 0.778
    data_5m["part3_top_singular"]["1"]["bia_vs_syc_cosine"],  # 0.631
    data_7m["part3_top_singular"]["3"]["bia_vs_syc_cosine"],  # 0.581
]

# Mean activation-difference vector cosine (bia↔syc) — first and last layer
mean_cos_first = [
    data_3m["part2_mean_vector"]["0"]["bia_vs_syc"],  # 0.711
    data_5m["part2_mean_vector"]["0"]["bia_vs_syc"],  # 0.481
    data_7m["part2_mean_vector"]["0"]["bia_vs_syc"],  # 0.486
]
mean_cos_last = [
    data_3m["part2_mean_vector"]["1"]["bia_vs_syc"],  # 0.718
    data_5m["part2_mean_vector"]["1"]["bia_vs_syc"],  # 0.539
    data_7m["part2_mean_vector"]["3"]["bia_vs_syc"],  # 0.494
]

# Cross-transfer rho (syco-only → bias ρ)
cross_rho = [0.000, 0.290, 0.215]

# ── Colors & style ───────────────────────────────────────────────────
BLUE = "#4A90D9"
RED = "#E85D4A"
GRAY = "#888888"
DARK = "#333333"
outline = [pe.withStroke(linewidth=3, foreground="white")]

x = np.arange(len(scales))
bar_width = 0.32

# ── Figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

# ── Panel A: Top-1 SVD cosine ────────────────────────────────────────
ax = axes[0]
bars1 = ax.bar(x - bar_width/2, svd_cos_first, bar_width, color=BLUE,
               label="First layer", edgecolor="white", linewidth=0.5, zorder=3)
bars2 = ax.bar(x + bar_width/2, svd_cos_last, bar_width, color=RED,
               label="Last layer", edgecolor="white", linewidth=0.5, zorder=3)

# Value labels
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=DARK, path_effects=outline)

ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n(d={d})" for s, d in zip(scales, d_models)], fontsize=10)
ax.set_ylabel("Cosine similarity", fontsize=11)
ax.set_title("Top-1 SVD direction\n(bias↔sycophancy)", fontsize=11.5, fontweight="bold", pad=8)
ax.set_ylim(0, 0.95)
ax.legend(fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10)
ax.axhline(y=0, color="#ccc", linewidth=0.5, zorder=1)

# ── Panel B: Mean-vector cosine ──────────────────────────────────────
ax = axes[1]
bars1 = ax.bar(x - bar_width/2, mean_cos_first, bar_width, color=BLUE,
               label="First layer", edgecolor="white", linewidth=0.5, zorder=3)
bars2 = ax.bar(x + bar_width/2, mean_cos_last, bar_width, color=RED,
               label="Last layer", edgecolor="white", linewidth=0.5, zorder=3)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=DARK, path_effects=outline)

ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n(d={d})" for s, d in zip(scales, d_models)], fontsize=10)
ax.set_ylabel("Cosine similarity", fontsize=11)
ax.set_title("Mean activation-diff vector\n(bias↔sycophancy)", fontsize=11.5, fontweight="bold", pad=8)
ax.set_ylim(0, 0.85)
ax.legend(fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10)
ax.axhline(y=0, color="#ccc", linewidth=0.5, zorder=1)

# ── Panel C: Cross-transfer ρ ────────────────────────────────────────
ax = axes[2]
colors_rho = ["#cccccc", BLUE, BLUE]  # Gray for 3M (null), blue for 5M/7M
bars = ax.bar(x, cross_rho, bar_width * 1.8, color=colors_rho,
              edgecolor=["#aaa", "black", "black"],
              linewidth=[1, 2, 2], zorder=3)

for i, bar in enumerate(bars):
    h = bar.get_height()
    label = f"{cross_rho[i]:.3f}" if cross_rho[i] == 0 else f"{cross_rho[i]:.3f}"
    y_pos = max(h, 0.01) + 0.012
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=DARK, path_effects=outline)

# Threshold annotation
ax.annotate("readout\nthreshold", xy=(0.5, 0.145), xycoords="data",
            xytext=(0.7, 0.22), fontsize=9, fontstyle="italic", color="#555",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n(d={d})" for s, d in zip(scales, d_models)], fontsize=10)
ax.set_ylabel("Spearman ρ", fontsize=11)
ax.set_title("Cross-transfer\n(syco-only → bias ρ)", fontsize=11.5, fontweight="bold", pad=8)
ax.set_ylim(-0.02, 0.38)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10)
ax.axhline(y=0, color="#ccc", linewidth=0.5, zorder=1)

# ── Suptitle & layout ────────────────────────────────────────────────
fig.suptitle(
    "Shared activation direction exists before behavioral transfer emerges",
    fontsize=14, fontweight="bold", y=1.02,
)
fig.text(
    0.5, -0.04,
    "The bias↔sycophancy activation direction is strongest at 3M (d=64) "
    "but cross-transfer requires d≥96 (5M) — a readout threshold, not a representation threshold.",
    ha="center", fontsize=9.5, color="#555", fontstyle="italic",
    wrap=True,
)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────
for fmt, dpi in [("pdf", 300), ("png", 200)]:
    out = FIGURES / f"cross_model_subspace_alignment.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")

plt.close()
print("\nDone.")
