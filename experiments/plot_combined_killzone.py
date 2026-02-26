#!/usr/bin/env python3
"""Combined Kill Zone Heatmap across all tested models.

Generates a single figure showing behavioral sensitivity (Δρ from baseline)
at each layer depth for Qwen-2.5-7B, Mistral-7B, and Llama-3.1-8B.
Normalizes layers to depth percentage (0-100%) for cross-architecture comparison.

Outputs:
  figures/combined_killzone_heatmap.png   — 9-panel grid (3 models × 3 behaviors)
  figures/combined_killzone_summary.png   — 3-panel overlay (one per behavior, all models)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / "steering"
FIGURES = ROOT / "figures"

# ── Color scheme ──
MODEL_COLORS = {
    "Qwen-2.5-7B":  "#2196F3",   # Blue
    "Mistral-7B":   "#FF5722",   # Red-orange
    "Llama-3.1-8B": "#4CAF50",   # Green
}
BEHAVIOR_COLORS = {
    "factual":    "#2196F3",
    "sycophancy": "#FF5722",
    "bias":       "#9C27B0",
}


def load_qwen():
    """Load Qwen sweep, extract α=+4.0 points."""
    with open(RESULTS / "steering_qwen2.5_7b_instruct.json") as f:
        data = json.load(f)

    model_name = "Qwen-2.5-7B"
    n_layers = data["n_layers"]  # 28
    baselines = {
        beh: data["behaviors"][beh]["baseline"]["rho"]
        for beh in ["factual", "sycophancy", "bias"]
    }

    # Extract α=+4.0 sweep points
    sweep = {}
    for beh in ["factual", "sycophancy", "bias"]:
        points = []
        for pt in data["behaviors"][beh]["sweep"]:
            if pt["alpha"] == 4.0:
                points.append({
                    "layer": pt["layer"],
                    "depth_pct": pt["layer_pct"],
                    "rho": pt["rho"],
                    "delta": pt["delta_rho"],
                })
        sweep[beh] = points

    return model_name, n_layers, baselines, sweep


def load_heatmap_model(filename, model_name):
    """Load Mistral/Llama heatmap format."""
    with open(RESULTS / filename) as f:
        data = json.load(f)

    n_layers = data["n_layers"]
    baselines = {
        beh: data["baselines"][beh]["rho"]
        for beh in ["factual", "sycophancy", "bias"]
    }

    sweep = {}
    for beh in ["factual", "sycophancy", "bias"]:
        points = []
        for pt in data["sweep"]:
            rho = pt["results"][beh]["rho"]
            points.append({
                "layer": pt["layer"],
                "depth_pct": pt["depth_pct"],
                "rho": rho,
                "delta": rho - baselines[beh],
            })
        sweep[beh] = points

    return model_name, n_layers, baselines, sweep


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Load all models
    models = [
        load_qwen(),
        load_heatmap_model("heatmap_mistral_7b_instruct_v0.3.json", "Mistral-7B"),
        load_heatmap_model("heatmap_llama_3.1_8b_instruct.json", "Llama-3.1-8B"),
    ]

    behaviors = ["factual", "sycophancy", "bias"]
    beh_labels = {"factual": "Factual", "sycophancy": "Sycophancy", "bias": "Bias"}

    # ══════════════════════════════════════════════════════════════════
    # Figure 1: 9-panel grid (3 models × 3 behaviors)
    # ══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 3, figsize=(18, 10), sharey="row")

    for col, (model_name, n_layers, baselines, sweep) in enumerate(models):
        color = MODEL_COLORS[model_name]

        for row, beh in enumerate(behaviors):
            ax = axes[row, col]
            points = sweep[beh]
            depths = [p["depth_pct"] * 100 for p in points]
            deltas = [p["delta"] for p in points]

            # Color bars by sign
            bar_colors = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]
            bar_width = 3.5 if len(points) <= 6 else 2.0
            bars = ax.bar(depths, deltas, width=bar_width, color=bar_colors,
                          alpha=0.8, edgecolor="white", linewidth=0.5)

            # Zero line
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

            # Annotate values on bars
            for d_pct, delta in zip(depths, deltas):
                if abs(delta) > 0.01:
                    va = "bottom" if delta >= 0 else "top"
                    offset = 0.012 if delta >= 0 else -0.012
                    ax.text(d_pct, delta + offset, f"{delta:+.3f}",
                            ha="center", va=va, fontsize=6.5, fontweight="bold",
                            color="darkgreen" if delta >= 0 else "darkred")

            # Kill zone shading
            if model_name == "Qwen-2.5-7B":
                ax.axvspan(58, 64, alpha=0.10, color="red", zorder=0)
            elif model_name == "Mistral-7B":
                ax.axvspan(41, 58, alpha=0.10, color="red", zorder=0)
            elif model_name == "Llama-3.1-8B":
                ax.axvspan(41, 52, alpha=0.10, color="red", zorder=0)

            # Labels
            if col == 0:
                ax.set_ylabel(f"{beh_labels[beh]}\n$\\Delta\\rho$",
                              fontsize=11, fontweight="bold",
                              rotation=0, labelpad=45, va="center")
            if row == 0:
                ax.set_title(f"{model_name}\n({n_layers} layers)",
                             fontsize=12, fontweight="bold", color=color)
            if row == 2:
                ax.set_xlabel("Depth (%)", fontsize=10)

            ax.set_xlim(15, 100)
            ax.grid(True, alpha=0.15, axis="y")
            ax.tick_params(labelsize=8)

            # Baseline annotation
            base = baselines[beh]
            ax.text(0.97, 0.95, f"base={base:.3f}",
                    transform=ax.transAxes, fontsize=7, color="gray",
                    ha="right", va="top")

    # Kill zone legend
    kill_patch = mpatches.Patch(facecolor="red", alpha=0.10,
                                 edgecolor="red", linewidth=1,
                                 label="Kill Zone")
    fig.legend(handles=[kill_patch], loc="lower center", ncol=1,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Cross-Architecture Kill Zone Map: Behavioral Impact of Sycophancy Steering ($\\alpha$=+4.0)",
        fontsize=15, fontweight="bold", y=0.99
    )
    fig.text(0.5, 0.955,
             "Kill zones cluster at 40-60% depth across architectures. "
             "Factual sweet spot at ~75-86% depth is universal.",
             ha="center", fontsize=10, fontstyle="italic", color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    out1 = FIGURES / "combined_killzone_heatmap.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════
    # Figure 2: 3-panel overlay (one per behavior, all models overlaid)
    # ══════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax, beh in zip(axes2, behaviors):
        for model_name, n_layers, baselines, sweep in models:
            points = sweep[beh]
            depths = [p["depth_pct"] * 100 for p in points]
            deltas = [p["delta"] for p in points]
            color = MODEL_COLORS[model_name]

            ax.plot(depths, deltas, "o-", color=color, label=model_name,
                    linewidth=2.2, markersize=7, alpha=0.85)

        # Universal kill zone band (40-60%)
        ax.axvspan(40, 60, alpha=0.06, color="red", zorder=0)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

        ax.set_title(beh_labels[beh], fontsize=14, fontweight="bold",
                     color=BEHAVIOR_COLORS[beh])
        ax.set_xlabel("Depth (%)", fontsize=11)
        ax.set_xlim(20, 100)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=9)

    axes2[0].set_ylabel("$\\Delta\\rho$ from baseline", fontsize=12)
    axes2[0].legend(fontsize=10, loc="upper left")

    # Kill zone annotation on sycophancy panel
    axes2[1].text(50, axes2[1].get_ylim()[1] * 0.75,
                  "KILL ZONE\n(40-60% depth)",
                  ha="center", fontsize=10, fontweight="bold", color="#B71C1C",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFCDD2",
                            edgecolor="#B71C1C", alpha=0.85))

    # Factual sweet spot annotation
    axes2[0].annotate("Universal factual\nsweet spot (~80%)",
                      xy=(82, 0.08), fontsize=9, fontweight="bold",
                      color="#1565C0", ha="center",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#BBDEFB",
                                edgecolor="#1565C0", alpha=0.85))

    fig2.suptitle(
        "Behavioral Sensitivity by Depth: Cross-Architecture Comparison",
        fontsize=14, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    out2 = FIGURES / "combined_killzone_summary.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig2)

    # ══════════════════════════════════════════════════════════════════
    # Figure 3: Compact delta grid (all models stacked)
    # ══════════════════════════════════════════════════════════════════
    # Interpolate all models to common depth grid (25% to 95%, step 5%)
    common_depths = np.arange(25, 96, 5)  # 15 points

    fig3, axes3 = plt.subplots(3, 1, figsize=(16, 8), sharex=True,
                                gridspec_kw={"hspace": 0.15})

    for ax, (model_name, n_layers, baselines, sweep) in zip(axes3, models):
        # Build interpolated delta matrix: 3 behaviors × common_depths
        delta_matrix = np.full((3, len(common_depths)), np.nan)

        for i, beh in enumerate(behaviors):
            points = sweep[beh]
            depths_raw = np.array([p["depth_pct"] * 100 for p in points])
            deltas_raw = np.array([p["delta"] for p in points])

            # Interpolate to common grid
            delta_interp = np.interp(common_depths, depths_raw, deltas_raw,
                                     left=np.nan, right=np.nan)
            delta_matrix[i, :] = delta_interp

        # Mask NaN for display
        masked = np.ma.masked_invalid(delta_matrix)

        im = ax.imshow(masked, cmap="RdYlGn", aspect="auto",
                       vmin=-0.50, vmax=0.30,
                       extent=[common_depths[0] - 2.5, common_depths[-1] + 2.5,
                               2.5, -0.5])

        # Cell value annotations
        for i in range(3):
            for j in range(len(common_depths)):
                val = delta_matrix[i, j]
                if np.isfinite(val):
                    color = "white" if abs(val) > 0.15 else "black"
                    ax.text(common_depths[j], i, f"{val:+.2f}",
                            ha="center", va="center", fontsize=7,
                            fontweight="bold", color=color)

        # Kill zone border
        if model_name == "Qwen-2.5-7B":
            kz_start, kz_end = 57.5, 62.5
        elif model_name == "Mistral-7B":
            kz_start, kz_end = 37.5, 57.5
        else:
            kz_start, kz_end = 37.5, 52.5

        rect = Rectangle((kz_start, -0.5), kz_end - kz_start, 3,
                          linewidth=2.5, edgecolor="#B71C1C", facecolor="none",
                          linestyle="--")
        ax.add_patch(rect)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Factual", "Sycophancy", "Bias"], fontsize=10)
        ax.set_ylabel(f"{model_name}", fontsize=11, fontweight="bold",
                      color=MODEL_COLORS[model_name], rotation=0,
                      labelpad=85, va="center")

    axes3[-1].set_xticks(common_depths)
    axes3[-1].set_xticklabels([f"{d:.0f}%" for d in common_depths], fontsize=9)
    axes3[-1].set_xlabel("Layer Depth (%)", fontsize=12)

    fig3.suptitle(
        "Kill Zone Atlas: $\\Delta\\rho$ from Baseline at Each Depth\n"
        "(Sycophancy steering vector, $\\alpha$=+4.0, dashed border = kill zone)",
        fontsize=13, fontweight="bold", y=0.98
    )

    # Single colorbar
    cbar_ax = fig3.add_axes([0.92, 0.15, 0.015, 0.7])
    fig3.colorbar(im, cax=cbar_ax, label="$\\Delta\\rho$")

    out3 = FIGURES / "combined_killzone_atlas.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved: {out3}")
    plt.close(fig3)

    # ── Summary stats ──
    print("\n" + "=" * 70)
    print("CROSS-ARCHITECTURE KILL ZONE SUMMARY")
    print("=" * 70)
    for model_name, n_layers, baselines, sweep in models:
        # Find worst bias delta
        bias_pts = sweep["bias"]
        worst_bias = min(bias_pts, key=lambda p: p["delta"])
        # Find best factual delta
        best_fact = max(sweep["factual"], key=lambda p: p["delta"])
        # Find best sycophancy delta
        best_syc = max(sweep["sycophancy"], key=lambda p: p["delta"])

        print(f"\n  {model_name} ({n_layers} layers):")
        print(f"    Worst bias damage:   L{worst_bias['layer']} "
              f"({worst_bias['depth_pct']:.0%}) Δ={worst_bias['delta']:+.3f}")
        print(f"    Best factual gain:   L{best_fact['layer']} "
              f"({best_fact['depth_pct']:.0%}) Δ={best_fact['delta']:+.3f}")
        print(f"    Best sycophancy:     L{best_syc['layer']} "
              f"({best_syc['depth_pct']:.0%}) Δ={best_syc['delta']:+.3f}")


if __name__ == "__main__":
    main()
