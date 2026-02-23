#!/usr/bin/env python3
"""
Plot the Layer 17 interference finding: Sycophancy ρ vs Bias ρ trade-off.

Visualizes the representational overlap at Layer 17 — steering sycophancy
resistance linearly destroys bias detection, and upstream L14 bias vectors
provide insufficient compensation.

Usage:
    python experiments/plot_cocktail_tradeoff.py
    python experiments/plot_cocktail_tradeoff.py --results results/steering/cocktail_qwen2.5_7b_instruct.json
    python experiments/plot_cocktail_tradeoff.py --output figures/cocktail_tradeoff.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────

BEHAVIOR_COLORS = {
    "factual": "#2196F3",       # blue
    "bias": "#FF9800",          # orange
    "sycophancy": "#9C27B0",    # purple
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_pareto_tradeoff(results: dict, output_path: Path):
    """Main figure: Sycophancy ρ vs Bias ρ scatter with Pareto front."""

    grid = results["cocktail_grid"]
    baselines = results["baselines"]

    syc_rhos = []
    bias_rhos = []
    syc_alphas = []
    bias_alphas_list = []

    for point in grid:
        s = point["results"]["sycophancy"]["rho"]
        b = point["results"]["bias"]["rho"]
        syc_rhos.append(s)
        bias_rhos.append(b)
        syc_alphas.append(point["config"]["sycophancy"]["alpha"])
        bias_alphas_list.append(point["config"]["bias"]["alpha"])

    syc_rhos = np.array(syc_rhos)
    bias_rhos = np.array(bias_rhos)
    syc_alphas = np.array(syc_alphas)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by sycophancy alpha
    unique_syc = sorted(set(syc_alphas))
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(unique_syc), max(unique_syc))

    scatter = ax.scatter(
        syc_rhos, bias_rhos,
        c=syc_alphas, cmap=cmap, norm=norm,
        s=120, edgecolors="black", linewidth=0.8, zorder=5,
    )

    # Annotate each point with its config
    for i, point in enumerate(grid):
        sa = point["config"]["sycophancy"]["alpha"]
        ba = point["config"]["bias"]["alpha"]
        label = f"α=+{sa:.0f}/−{abs(ba):.0f}"
        offset = (8, 5) if i % 2 == 0 else (8, -12)
        ax.annotate(
            label, (syc_rhos[i], bias_rhos[i]),
            textcoords="offset points", xytext=offset,
            fontsize=7.5, color="gray", alpha=0.85,
        )

    # Baseline point
    ax.scatter(
        baselines["sycophancy"]["rho"], baselines["bias"]["rho"],
        marker="*", s=250, c="red", edgecolors="darkred", linewidth=0.8,
        zorder=6, label="Baseline (no steering)",
    )
    ax.annotate(
        "Baseline", (baselines["sycophancy"]["rho"], baselines["bias"]["rho"]),
        textcoords="offset points", xytext=(10, 5),
        fontsize=9, fontweight="bold", color="red",
    )

    # Null point highlight
    null = results.get("null_point")
    if null:
        ax.scatter(
            null["sycophancy_rho"], null["bias_rho"],
            marker="D", s=150, c="gold", edgecolors="black", linewidth=1.2,
            zorder=7, label="Null point (best compromise)",
        )

    # Target zone
    ax.axhspan(0.70, 1.0, alpha=0.06, color="green")
    ax.axvspan(0.35, 1.0, alpha=0.06, color="green")
    ax.axhline(0.70, color="green", ls="--", alpha=0.3, lw=1)
    ax.axvline(0.35, color="green", ls="--", alpha=0.3, lw=1)
    ax.text(0.36, 0.71, "Target zone", fontsize=8, color="green", alpha=0.6)

    # Linear fit through grid points
    if len(syc_rhos) >= 3:
        m, b_fit = np.polyfit(syc_rhos, bias_rhos, 1)
        x_fit = np.linspace(syc_rhos.min() - 0.02, syc_rhos.max() + 0.02, 50)
        y_fit = m * x_fit + b_fit
        ax.plot(x_fit, y_fit, "k--", alpha=0.3, lw=1.5)
        ax.text(
            x_fit[-1] + 0.005, y_fit[-1],
            f"slope = {m:.2f}",
            fontsize=8, color="black", alpha=0.5, va="center",
        )

    ax.set_xlabel("Sycophancy ρ (higher = better resistance)", fontsize=12)
    ax.set_ylabel("Bias ρ (higher = better detection)", fontsize=12)
    ax.set_title(
        "Layer 17 Interference: Sycophancy–Bias Trade-off\n"
        "Multi-vector steering cocktails on Qwen2.5-7B-Instruct",
        fontsize=13, fontweight="bold",
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, aspect=25)
    cbar.set_label("Sycophancy α (L17)", fontsize=10)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim(0.05, 0.55)
    ax.set_ylim(0.30, 0.85)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_alpha_response(results: dict, output_path: Path):
    """Panel figure: per-behavior ρ response curves at each syc alpha level."""

    grid = results["cocktail_grid"]
    baselines = results["baselines"]

    # Group by sycophancy alpha
    by_syc_alpha = {}
    for point in grid:
        sa = point["config"]["sycophancy"]["alpha"]
        if sa not in by_syc_alpha:
            by_syc_alpha[sa] = []
        by_syc_alpha[sa].append(point)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    behaviors = ["factual", "sycophancy", "bias"]
    titles = ["Factual ρ", "Sycophancy ρ", "Bias ρ"]

    for ax, beh, title in zip(axes, behaviors, titles):
        baseline = baselines[beh]["rho"]
        ax.axhline(baseline, color="red", ls="--", lw=1.5, alpha=0.5, label="Baseline")

        for sa in sorted(by_syc_alpha.keys()):
            points = by_syc_alpha[sa]
            bias_alphas = [abs(p["config"]["bias"]["alpha"]) for p in points]
            rhos = [p["results"][beh]["rho"] for p in points]

            ax.plot(
                bias_alphas, rhos,
                "o-", markersize=7, linewidth=2,
                label=f"syc α=+{sa:.0f}",
            )

        ax.set_xlabel("|Bias α| (L14)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color=BEHAVIOR_COLORS[beh])
        ax.legend(fontsize=8)

    axes[0].set_ylabel("ρ", fontsize=12)

    fig.suptitle(
        "Multi-Vector Steering Response — Qwen2.5-7B-Instruct",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot cocktail steering trade-off (Layer 17 interference)"
    )
    parser.add_argument(
        "--results",
        default="results/steering/cocktail_qwen2.5_7b_instruct.json",
        help="Path to cocktail results JSON",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for main figure (default: figures/cocktail_tradeoff.png)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = Path(__file__).parent.parent / results_path

    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)

    main_output = Path(args.output) if args.output else output_dir / "cocktail_tradeoff.png"
    panel_output = output_dir / "cocktail_response_curves.png"

    results = load_results(results_path)

    print(f"\n  Plotting cocktail trade-off from {results_path.name}")
    print(f"  Grid points: {len(results['cocktail_grid'])}")

    plot_pareto_tradeoff(results, main_output)
    plot_alpha_response(results, panel_output)

    print(f"\n  Done. Figures in {output_dir}/")


if __name__ == "__main__":
    main()
