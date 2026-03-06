#!/usr/bin/env python3
"""Generate Figure 3: Deconcentration score vs behavioral ρ scatter plot.

The clearest plot in the paper — a single metric separates every productive
injection from every null condition. Direct injection shows perfect correlation;
cross-transfer uses a different geometric mechanism (shared direction readout).

Data source: results/developmental_sweep/decon_scores.json
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
DATA = ROOT / "results" / "developmental_sweep" / "decon_scores.json"
FIGURES = ROOT / "figures" / "paper4"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────
with open(DATA) as f:
    data = json.load(f)

conditions = data["conditions"]

# ── Categorize points ─────────────────────────────────────────────────
direct = [c for c in conditions if c["type"] == "direct"]
cross = [c for c in conditions if c["type"] == "cross"]
null = [c for c in conditions if c["type"] == "null"]

# ── Colors & style ───────────────────────────────────────────────────
BLUE = "#4A90D9"
RED = "#E85D4A"
GRAY = "#999999"
DARK = "#333333"
outline = [pe.withStroke(linewidth=3, foreground="white")]

fig, ax = plt.subplots(figsize=(9, 6.5))

# ── Plot points ───────────────────────────────────────────────────────
# Null conditions (gray triangles)
ax.scatter(
    [c["decon_score"] for c in null],
    [c["bias_rho"] for c in null],
    marker="^", s=120, c=GRAY, edgecolors="#666", linewidths=1.2,
    label="Null (NLI, Cal, Sub, Pri, Tox)", zorder=5, alpha=0.85,
)

# Cross-transfer (red squares)
ax.scatter(
    [c["decon_score"] for c in cross],
    [c["bias_rho"] for c in cross],
    marker="s", s=140, c=RED, edgecolors="black", linewidths=1.5,
    label="Cross-transfer (syco→bias)", zorder=6,
)

# Direct injection (blue circles)
ax.scatter(
    [c["decon_score"] for c in direct],
    [c["bias_rho"] for c in direct],
    marker="o", s=160, c=BLUE, edgecolors="black", linewidths=1.5,
    label="Direct injection (bias-only)", zorder=7,
)

# ── Annotations ───────────────────────────────────────────────────────
# Direct injection labels — offset to right
offsets_direct = {
    "Bias-only 3M": (0.08, -0.02),
    "Bias-only 5M": (0.08, -0.025),
    "Bias-only 7M": (0.08, -0.02),
}
for c in direct:
    dx, dy = offsets_direct.get(c["label"], (0.08, 0))
    ax.annotate(
        f"{c['scale']}", (c["decon_score"], c["bias_rho"]),
        xytext=(c["decon_score"] + dx, c["bias_rho"] + dy),
        fontsize=9.5, fontweight="bold", color=BLUE,
        path_effects=outline, zorder=15,
    )

# Cross-transfer labels
offsets_cross = {
    "Syco-only 3M": (0.06, 0.02),
    "Syco-only 5M": (-0.02, 0.03),
    "Syco-only 7M": (-0.02, 0.025),
}
for c in cross:
    dx, dy = offsets_cross.get(c["label"], (0.06, 0.02))
    ax.annotate(
        f"{c['scale']}", (c["decon_score"], c["bias_rho"]),
        xytext=(c["decon_score"] + dx, c["bias_rho"] + dy),
        fontsize=9.5, fontweight="bold", color=RED,
        path_effects=outline, zorder=15,
    )

# Null condition labels — compact
null_labels = {
    "NLI 3M": "NLI",
    "Calculator 3M": "Cal",
    "Subitizing 3M": "Sub",
    "Primitive 3M": "Pri",
    "NLI 7M": "NLI₇",
    "Toxicity 7M": "Tox₇",
}
# Spread null labels vertically to avoid overlap
null_y_offsets = {
    "NLI 3M": -0.04,
    "Calculator 3M": 0.025,
    "Subitizing 3M": -0.04,
    "Primitive 3M": 0.025,
    "NLI 7M": 0.025,
    "Toxicity 7M": -0.04,
}
for c in null:
    lbl = null_labels.get(c["label"], c["label"])
    dy = null_y_offsets.get(c["label"], 0.02)
    ax.annotate(
        lbl, (c["decon_score"], c["bias_rho"]),
        xytext=(c["decon_score"] + 0.01, c["bias_rho"] + dy),
        fontsize=8, color=GRAY, fontstyle="italic",
        path_effects=outline, zorder=14,
    )

# ── Vertical separator at decon ≈ 0.1 ────────────────────────────────
ax.axvline(x=0.1, color="#ccc", linewidth=1.5, linestyle="--", zorder=1)
ax.text(0.1, -0.06, "decon = 0.1", fontsize=8, color="#aaa", ha="center",
        fontstyle="italic")

# ── Trend line for direct injection ───────────────────────────────────
# Simple visual: connect direct points with a faint line
direct_sorted = sorted(direct, key=lambda c: c["decon_score"])
ax.plot(
    [c["decon_score"] for c in direct_sorted],
    [c["bias_rho"] for c in direct_sorted],
    "--", color=BLUE, alpha=0.3, linewidth=1.5, zorder=2,
)

# ── Zone labels ───────────────────────────────────────────────────────
ax.text(-0.15, 0.25, "inflationary\n(noise)", fontsize=9, color="#bbb",
        ha="center", fontstyle="italic", rotation=90, va="center")
ax.text(0.65, 0.46, "deconcentrating\n(restructuring)", fontsize=9,
        color=BLUE, alpha=0.4, ha="center", fontstyle="italic", va="center")

# ── Cross-transfer annotation box ────────────────────────────────────
ax.annotate(
    "Cross-transfer:\nshared direction readout\n(not target deconcentration)",
    xy=(0.11, 0.29), xytext=(0.35, 0.32),
    fontsize=8.5, fontstyle="italic", color="#555", ha="left",
    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ddd", lw=1, alpha=0.9),
    zorder=15,
)

# ── Axes & style ─────────────────────────────────────────────────────
ax.set_xlabel("Deconcentration score (bias subspace, last layer)", fontsize=12)
ax.set_ylabel("Bias ρ (behavioral output)", fontsize=12)
ax.set_xlim(-0.25, 1.95)
ax.set_ylim(-0.08, 0.50)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10)
ax.axhline(y=0, color="#eee", linewidth=0.8, zorder=0)
ax.axvline(x=0, color="#eee", linewidth=0.8, zorder=0)
ax.legend(fontsize=10, loc="upper left", framealpha=0.9)

# ── Title ─────────────────────────────────────────────────────────────
fig.suptitle(
    "Deconcentration score separates productive from null injection",
    fontsize=13, fontweight="bold", y=0.97,
)
ax.set_title(
    "decon = (1 − SV₁ᵖᵒˢᵗ / SV₁ᵛᵃⁿ) × (eff_dimᵖᵒˢᵗ / eff_dimᵛᵃⁿ)",
    fontsize=10, color="#555", fontstyle="italic", pad=10,
)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────
for fmt, dpi in [("pdf", 300), ("png", 200)]:
    out = FIGURES / f"decon_score_scatter.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")

plt.close()
print("\nDone.")
