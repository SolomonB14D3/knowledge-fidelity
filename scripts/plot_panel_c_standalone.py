#!/usr/bin/env python3
"""Generate standalone Panel C for social media: scale ladder + contrastive overlay."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

FIGURES = Path(__file__).resolve().parents[1] / "figures" / "paper4"
FIGURES.mkdir(parents=True, exist_ok=True)

SCALE_VANILLA = {
    "7M":  {"bias": 0.000, "sycophancy": 0.000, "params": 7.3},
    "12M": {"bias": 0.000, "sycophancy": 0.000, "params": 12.4},
    "18M": {"bias": 0.133, "sycophancy": 0.000, "params": 17.7},
    "34M": {"bias": 0.238, "sycophancy": 0.300, "params": 33.9},
    "64M": {"bias": 0.087, "sycophancy": 0.300, "params": 64.1},
}

CONTR_7M  = {"bias": 0.433, "sycophancy": 0.513, "params": 7.3}
CONTR_12M = {"bias": 0.380, "sycophancy": 0.520, "params": 12.4}

fig, ax = plt.subplots(figsize=(8, 5.5))

scales = ["7M", "12M", "18M", "34M", "64M"]
params = [SCALE_VANILLA[s]["params"] for s in scales]

for beh, color, marker, label in [
    ("bias", "#4A90D9", "o", "Bias"),
    ("sycophancy", "#E85D4A", "s", "Sycophancy"),
]:
    vanilla_rhos = [SCALE_VANILLA[s][beh] for s in scales]
    ax.plot(params, vanilla_rhos, f"{marker}--", color=color,
            label=f"{label} (vanilla)", alpha=0.5, markersize=8, linewidth=2)

    # Contrastive overlay points
    for pt in [CONTR_7M, CONTR_12M]:
        ax.plot(pt["params"], pt[beh], marker, color=color, markersize=16,
                markeredgecolor="black", markeredgewidth=2.5, zorder=10)

# White outline effect for annotation text
outline = [pe.withStroke(linewidth=3, foreground="white")]

# Annotations — stacked on right side, 12M on top, 7M below
ax.annotate("12M + contrastive\nρ = 0.38 / 0.52", (12.4, 0.52), fontsize=10,
           fontweight="bold", color="#222", ha="left",
           xytext=(30, 0.52), arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#555", lw=1.5, alpha=0.95),
           path_effects=outline, zorder=20)
ax.annotate("7M + contrastive\nρ = 0.43 / 0.51", (7.3, 0.43), fontsize=10,
           fontweight="bold", color="#222", ha="left",
           xytext=(30, 0.38), arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#555", lw=1.5, alpha=0.95),
           path_effects=outline, zorder=20)

# Scale labels on vanilla curve
for s in scales:
    v = SCALE_VANILLA[s]
    y_offset = -0.04 if s != "64M" else 0.03
    ax.annotate(s, (v["params"], max(v["bias"], v["sycophancy"]) + y_offset),
               fontsize=8, color="#888", ha="center")

ax.set_xlabel("Parameters (millions)", fontsize=12)
ax.set_ylabel("Spearman ρ", fontsize=12)

# Title + subtitle
fig.suptitle("Contrastive injection lets 7M models outperform vanilla 34M–64M",
             fontsize=13, fontweight="bold", y=0.98)
ax.set_title("(5% injection rate)", fontsize=11, color="#555", fontstyle="italic", pad=8)

ax.legend(fontsize=10, loc="lower right")
ax.set_ylim(-0.12, 0.65)
ax.set_xlim(0, 70)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10)

# Footer note
fig.text(0.5, 0.01, "5% contrastive behavioral pairs during pretraining · 0.11% of total tokens",
         ha="center", fontsize=9, color="#777", fontstyle="italic")

out = FIGURES / "panel_c_standalone.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close()
