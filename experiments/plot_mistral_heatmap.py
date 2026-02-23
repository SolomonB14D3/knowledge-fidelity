#!/usr/bin/env python3
"""Plot Mistral layer heatmap: Sensitivity Map with Alignment Kill Zone.

Reads results/steering/heatmap_mistral_7b_instruct_v0.3.json
and generates:
  - figures/mistral_sensitivity_map.png  (the "Heatmap of Resistance")
  - figures/mistral_layer_heatmap.png    (line chart with zones)
  - figures/mistral_layer_heatmap_delta.png (compact delta grid)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / "steering" / "heatmap_mistral_7b_instruct_v0.3.json"
FIGURES = ROOT / "figures"

# ── Project colors (consistent with cocktail_tradeoff.py) ──
BEHAVIOR_COLORS = {
    "factual": "#2196F3",     # Blue
    "sycophancy": "#FF5722",  # Red-orange
    "bias": "#9C27B0",        # Purple
}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    with open(RESULTS) as f:
        data = json.load(f)

    baselines = data["baselines"]
    sweep = data["sweep"]
    best = data.get("best_layer")

    layers = [pt["layer"] for pt in sweep]
    depths = [pt["depth_pct"] for pt in sweep]

    fact_rho = [pt["results"]["factual"]["rho"] for pt in sweep]
    syc_rho = [pt["results"]["sycophancy"]["rho"] for pt in sweep]
    bias_rho = [pt["results"]["bias"]["rho"] for pt in sweep]

    fact_base = baselines["factual"]["rho"]
    syc_base = baselines["sycophancy"]["rho"]
    bias_base = baselines["bias"]["rho"]

    # ══════════════════════════════════════════════════════════════════
    # Figure 1: THE SENSITIVITY MAP ("Heatmap of Resistance")
    # ══════════════════════════════════════════════════════════════════
    fig_map, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                                  gridspec_kw={"hspace": 0.05})

    behaviors = [
        ("Factual", fact_rho, fact_base, BEHAVIOR_COLORS["factual"]),
        ("Sycophancy", syc_rho, syc_base, BEHAVIOR_COLORS["sycophancy"]),
        ("Bias", bias_rho, bias_base, BEHAVIOR_COLORS["bias"]),
    ]

    for ax, (name, rhos, base, color) in zip(axes, behaviors):
        deltas = [r - base for r in rhos]

        # Color bars: green for improvement, red for degradation
        bar_colors = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]
        bars = ax.bar(range(len(layers)), deltas, color=bar_colors, alpha=0.8,
                      edgecolor="white", linewidth=0.5)

        # Zero line
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

        # Kill zone shading (L14-L18 = indices 2-4)
        for idx in range(len(layers)):
            if layers[idx] in [14, 16, 18]:
                ax.axvspan(idx - 0.4, idx + 0.4, alpha=0.12, color="red",
                           zorder=0)

        # Annotate deltas on bars
        for idx, (d, r) in enumerate(zip(deltas, rhos)):
            va = "bottom" if d >= 0 else "top"
            offset = 0.008 if d >= 0 else -0.008
            ax.text(idx, d + offset, f"{d:+.3f}", ha="center", va=va,
                    fontsize=7.5, fontweight="bold",
                    color="darkgreen" if d >= 0 else "darkred")

        ax.set_ylabel(f"{name}\n$\\Delta\\rho$", fontsize=11, fontweight="bold",
                      rotation=0, labelpad=50, va="center")
        ax.set_ylim(min(deltas) - 0.08, max(deltas) + 0.08)
        ax.grid(True, alpha=0.15, axis="y")
        ax.tick_params(axis="y", labelsize=9)

        # Add baseline annotation on right
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([0])
        ax2.set_yticklabels([f"base={base:.3f}"], fontsize=8, color="gray")
        ax2.tick_params(axis="y", length=0)

    # Kill Zone label
    kill_center = (layers.index(14) + layers.index(18)) / 2
    axes[0].annotate("ALIGNMENT\nKILL ZONE",
                     xy=(kill_center, axes[0].get_ylim()[1] * 0.7),
                     fontsize=11, fontweight="bold", color="#B71C1C",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFCDD2",
                               edgecolor="#B71C1C", alpha=0.9))

    # Factual sweet spot label
    l24_idx = layers.index(24)
    axes[0].annotate("FACTUAL\nSWEET SPOT",
                     xy=(l24_idx, fact_rho[l24_idx] - fact_base),
                     xytext=(l24_idx + 1.5, 0.10),
                     fontsize=9, fontweight="bold", color="#1565C0",
                     ha="center",
                     arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#BBDEFB",
                               edgecolor="#1565C0", alpha=0.9))

    # X-axis labels (bottom panel only)
    axes[-1].set_xticks(range(len(layers)))
    axes[-1].set_xticklabels(
        [f"L{l}\n({d:.0%})" for l, d in zip(layers, depths)],
        fontsize=9
    )
    axes[-1].set_xlabel("Layer (sycophancy vector applied @ $\\alpha$=+4.0)", fontsize=12)

    # Title
    fig_map.suptitle(
        "Mistral-7B Sensitivity Map: Behavioral Impact of Sycophancy Steering at Each Layer",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Subtitle
    fig_map.text(0.5, 0.94,
                 "Sycophancy suppression via activation steering is architecture-contingent. "
                 "No layer achieves meaningful sycophancy improvement.",
                 ha="center", fontsize=10, fontstyle="italic", color="#555")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_map = FIGURES / "mistral_sensitivity_map.png"
    fig_map.savefig(out_map, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_map}")

    # ══════════════════════════════════════════════════════════════════
    # Figure 2: Line chart with zones (existing style, updated)
    # ══════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Top panel: rho values across layers
    ax1.plot(layers, fact_rho, "o-", color=BEHAVIOR_COLORS["factual"],
             label=f"Factual (baseline={fact_base:.3f})", linewidth=2, markersize=8)
    ax1.plot(layers, syc_rho, "s-", color=BEHAVIOR_COLORS["sycophancy"],
             label=f"Sycophancy (baseline={syc_base:.3f})", linewidth=2, markersize=8)
    ax1.plot(layers, bias_rho, "D-", color=BEHAVIOR_COLORS["bias"],
             label=f"Bias (baseline={bias_base:.3f})", linewidth=2, markersize=8)

    # Baseline reference lines
    ax1.axhline(fact_base, color=BEHAVIOR_COLORS["factual"], linestyle="--", alpha=0.3)
    ax1.axhline(syc_base, color=BEHAVIOR_COLORS["sycophancy"], linestyle="--", alpha=0.3)
    ax1.axhline(bias_base, color=BEHAVIOR_COLORS["bias"], linestyle="--", alpha=0.3)

    # Bias safety threshold (85%)
    bias_thresh = bias_base * 0.85
    ax1.axhline(bias_thresh, color=BEHAVIOR_COLORS["bias"], linestyle=":",
                alpha=0.5, label=f"Bias safety threshold (85% = {bias_thresh:.3f})")

    # Kill zone shading (L14-L18)
    ax1.axvspan(13, 19, alpha=0.08, color="red")
    ax1.text(16, 0.92, "ALIGNMENT\nKILL ZONE", ha="center", va="center",
             fontsize=11, fontweight="bold", color="#B71C1C",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFCDD2",
                       edgecolor="#B71C1C", alpha=0.85))

    # Safe zone shading (L20-L28)
    ax1.axvspan(19, 29, alpha=0.04, color="green")
    ax1.text(24, 0.05, "Safe zone\n(bias retained)", ha="center", va="center",
             fontsize=9, color="green", alpha=0.8)

    # Mark L24 factual peak
    l24_idx = layers.index(24)
    ax1.annotate(f"L24: Factual peak\n($\\rho$={fact_rho[l24_idx]:.3f}, $\\Delta$=+0.117)",
                 xy=(24, fact_rho[l24_idx]), xytext=(24 - 5, 0.85),
                 fontsize=9, fontweight="bold", color=BEHAVIOR_COLORS["factual"],
                 arrowprops=dict(arrowstyle="->", color=BEHAVIOR_COLORS["factual"], lw=1.5))

    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("$\\rho$ (Spearman correlation)", fontsize=12)
    ax1.set_title("Mistral-7B Layer Heatmap: Sycophancy Vector @ $\\alpha$=+4.0\n"
                  "Sycophancy suppression is architecture-contingent: no layer works on Mistral",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=9)
    ax1.set_xticks(layers)
    ax1.set_xticklabels([f"L{l}\n({d:.0%})" for l, d in zip(layers, depths)], fontsize=8)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.2)

    # Bottom panel: Vector norms
    norms = data["vector_norms"]
    norm_values = [norms[str(l)] for l in layers]

    # Color bars by zone
    bar_colors_norm = []
    for l in layers:
        if l in [14, 16, 18]:
            bar_colors_norm.append("#F44336")  # Red for kill zone
        elif 20 <= l <= 28:
            bar_colors_norm.append("#4CAF50")  # Green for safe zone
        else:
            bar_colors_norm.append("gray")

    ax2.bar(layers, norm_values, color=bar_colors_norm, alpha=0.6, width=1.5)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Vector norm ||v||", fontsize=12)
    ax2.set_title("Steering vector magnitude grows with depth (norm $\\neq$ effectiveness)",
                  fontsize=11)
    ax2.set_xticks(layers)
    ax2.set_xticklabels([f"L{l}" for l in layers], fontsize=8)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out = FIGURES / "mistral_layer_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # ══════════════════════════════════════════════════════════════════
    # Figure 3: Compact delta grid
    # ══════════════════════════════════════════════════════════════════
    fig2, ax = plt.subplots(figsize=(14, 4.5))

    beh_list = ["factual", "sycophancy", "bias"]
    base_vals = [fact_base, syc_base, bias_base]
    rho_arrays = [fact_rho, syc_rho, bias_rho]

    # Build delta matrix: rows = behaviors, cols = layers
    delta_matrix = np.zeros((3, len(layers)))
    for i, (rhos, base) in enumerate(zip(rho_arrays, base_vals)):
        for j, rho in enumerate(rhos):
            delta_matrix[i, j] = rho - base

    # Custom diverging colormap: red for negative, green for positive
    im = ax.imshow(delta_matrix, cmap="RdYlGn", aspect="auto",
                   vmin=-0.55, vmax=0.15)

    # Labels
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Factual $\\Delta\\rho$", "Sycophancy $\\Delta\\rho$",
                         "Bias $\\Delta\\rho$"], fontsize=11)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}\n({d:.0%})" for l, d in zip(layers, depths)], fontsize=9)
    ax.set_xlabel("Layer (sycophancy vector applied @ $\\alpha$=+4.0)", fontsize=11)
    ax.set_title("Mistral-7B: $\\Delta\\rho$ from Baseline "
                 "--- Alignment Kill Zone at L14-L18",
                 fontsize=13, fontweight="bold")

    # Annotate cells with values
    for i in range(3):
        for j in range(len(layers)):
            val = delta_matrix[i, j]
            color = "white" if abs(val) > 0.25 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)

    # Kill zone border
    from matplotlib.patches import Rectangle
    kill_start = layers.index(14)
    kill_end = layers.index(18)
    rect = Rectangle((kill_start - 0.5, -0.5), kill_end - kill_start + 1, 3,
                      linewidth=2.5, edgecolor="#B71C1C", facecolor="none",
                      linestyle="--")
    ax.add_patch(rect)
    ax.text(kill_start + (kill_end - kill_start) / 2, -0.85,
            "ALIGNMENT KILL ZONE", ha="center", fontsize=9,
            fontweight="bold", color="#B71C1C")

    plt.colorbar(im, ax=ax, label="$\\Delta\\rho$ from baseline", shrink=0.8)
    plt.tight_layout()

    out2 = FIGURES / "mistral_layer_heatmap_delta.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")

    # ── Print key finding ──
    print("\n" + "="*60)
    print("KEY FINDING: Sycophancy steering does NOT work on Mistral.")
    print("="*60)
    print(f"  Alignment Kill Zone: L14-L18 (bias drops 34-46%, zero syc benefit)")
    print(f"  Best sycophancy gain: +0.013 (L14) — noise-level, with bias collapse")
    print(f"  Factual sweet spot: L24 (+0.117) — transfers across architectures")
    print(f"\n  NEGATIVE CAPABILITY CLAIM:")
    print(f"  Sycophancy suppression via activation steering is")
    print(f"  architecture-contingent. Do not apply Qwen recipes to Mistral.")


if __name__ == "__main__":
    main()
