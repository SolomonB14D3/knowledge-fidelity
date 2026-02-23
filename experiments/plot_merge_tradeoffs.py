#!/usr/bin/env python3
"""
Plot merge method tradeoffs: behavioral rho across merge strategies.

Visualizes the factual/bias/sycophancy trade-off gradient from conservative
(baseline) to aggressive (DARE-TIES) merging.

Usage:
    python experiments/plot_merge_tradeoffs.py
    python experiments/plot_merge_tradeoffs.py --results results/leaderboard/merged_audit.json
    python experiments/plot_merge_tradeoffs.py --output figures/merge_tradeoffs.png
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

BEHAVIOR_MARKERS = {
    "factual": "o",
    "bias": "s",
    "sycophancy": "D",
}

BEHAVIOR_LABELS = {
    "factual": "Factual ρ",
    "bias": "Bias detection ρ",
    "sycophancy": "Sycophancy resistance ρ",
}

# Order from conservative to aggressive
METHOD_ORDER = [
    "qwen2.5-7b-instruct",
    "qwen2.5-7b-slerp",
    "qwen2.5-7b-ties",
    "qwen2.5-7b-dare-ties",
]

METHOD_LABELS = {
    "qwen2.5-7b-instruct": "Baseline",
    "qwen2.5-7b-slerp": "SLERP",
    "qwen2.5-7b-ties": "TIES",
    "qwen2.5-7b-dare-ties": "DARE-TIES",
}


def plot_merge_tradeoffs(results, output_path=None, show=False):
    """Create the merge tradeoff visualization."""
    fig, (ax_main, ax_rates) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.3}
    )

    methods = [m for m in METHOD_ORDER if m in results]
    x = np.arange(len(methods))
    labels = [METHOD_LABELS[m] for m in methods]

    # ── Left panel: rho scores ────────────────────────────────────────
    bar_width = 0.22
    behaviors = ["factual", "bias", "sycophancy"]

    for i, behavior in enumerate(behaviors):
        values = [results[m]["behaviors"][behavior]["rho"] for m in methods]
        offset = (i - 1) * bar_width
        bars = ax_main.bar(
            x + offset, values,
            width=bar_width,
            color=BEHAVIOR_COLORS[behavior],
            label=BEHAVIOR_LABELS[behavior],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        # Value labels on bars
        for bar, val in zip(bars, values):
            ax_main.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold",
                color=BEHAVIOR_COLORS[behavior],
            )

    ax_main.set_xlabel("Merge Method", fontsize=12)
    ax_main.set_ylabel("ρ (Spearman correlation)", fontsize=12)
    ax_main.set_title(
        "Behavioral Trade-offs Across Merge Strategies\n"
        "Qwen2.5-7B-Instruct + Qwen2.5-Coder-7B",
        fontsize=13, fontweight="bold",
    )
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(labels, fontsize=10)
    ax_main.set_ylim(0, 1.0)
    ax_main.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_main.grid(True, axis="y", alpha=0.3)

    # Add arrow showing aggression gradient
    ax_main.annotate(
        "", xy=(len(methods) - 0.7, -0.06), xytext=(0.3, -0.06),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
        annotation_clip=False,
    )
    ax_main.text(
        len(methods) / 2 - 0.5, -0.10,
        "More aggressive merging →",
        ha="center", va="top", fontsize=9,
        fontstyle="italic", color="gray",
        transform=ax_main.get_xaxis_transform(),
    )

    # ── Right panel: failure rates ────────────────────────────────────
    bias_rates = [results[m]["behaviors"]["bias"]["bias_rate"] for m in methods]
    syc_rates = [results[m]["behaviors"]["sycophancy"]["sycophancy_rate"] for m in methods]

    x2 = np.arange(len(methods))
    bar_width2 = 0.3

    bars_b = ax_rates.bar(
        x2 - bar_width2 / 2, [r * 100 for r in bias_rates],
        width=bar_width2,
        color=BEHAVIOR_COLORS["bias"],
        label="Bias rate",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    bars_s = ax_rates.bar(
        x2 + bar_width2 / 2, [r * 100 for r in syc_rates],
        width=bar_width2,
        color=BEHAVIOR_COLORS["sycophancy"],
        label="Sycophancy rate",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    # Value labels
    for bar, val in zip(bars_b, bias_rates):
        ax_rates.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.0%}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color=BEHAVIOR_COLORS["bias"],
        )
    for bar, val in zip(bars_s, syc_rates):
        ax_rates.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.0%}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color=BEHAVIOR_COLORS["sycophancy"],
        )

    ax_rates.set_xlabel("Merge Method", fontsize=12)
    ax_rates.set_ylabel("Failure Rate (%)", fontsize=12)
    ax_rates.set_title("Behavioral Failure Rates", fontsize=13, fontweight="bold")
    ax_rates.set_xticks(x2)
    ax_rates.set_xticklabels(labels, fontsize=10)
    ax_rates.set_ylim(0, 105)
    ax_rates.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_rates.grid(True, axis="y", alpha=0.3)

    # ── Footer ────────────────────────────────────────────────────────
    fig.text(
        0.5, -0.02,
        "DARE-TIES achieves +0.138 factual ρ gain but destroys bias detection (−0.570) and sycophancy resistance. "
        "Run rho-audit before and after merging.",
        ha="center", fontsize=8, fontstyle="italic", color="gray",
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot merge method tradeoffs")
    parser.add_argument(
        "--results", default="results/leaderboard/merged_audit.json",
        help="Path to merged audit JSON",
    )
    parser.add_argument(
        "--output", default="figures/merge_tradeoffs.png",
        help="Output PNG path",
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} models")
    for name, data in results.items():
        print(f"  {name}: {data['model_type']}")

    plot_merge_tradeoffs(results, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
