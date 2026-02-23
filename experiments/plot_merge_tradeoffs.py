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

# Order from conservative to aggressive (Qwen-Coder family)
METHOD_ORDER = [
    "qwen2.5-7b-instruct",
    "qwen2.5-7b-linear",
    "qwen2.5-7b-slerp",
    "qwen2.5-7b-task-arith",
    "qwen2.5-7b-ties",
    "qwen2.5-7b-dare-ties",
]

METHOD_LABELS = {
    "qwen2.5-7b-instruct": "Baseline",
    "qwen2.5-7b-linear": "Linear",
    "qwen2.5-7b-slerp": "SLERP",
    "qwen2.5-7b-task-arith": "Task Arith.",
    "qwen2.5-7b-ties": "TIES",
    "qwen2.5-7b-dare-ties": "DARE-TIES",
}

# Mistral family
MISTRAL_ORDER = [
    "mistral-7b-v0.1",
    "mistral-7b-slerp",
    "mistral-7b-ties",
    "mistral-7b-dare-ties",
]

MISTRAL_LABELS = {
    "mistral-7b-v0.1": "Baseline",
    "mistral-7b-slerp": "SLERP",
    "mistral-7b-ties": "TIES",
    "mistral-7b-dare-ties": "DARE-TIES",
}


def _plot_family(ax, results, method_order, method_labels, title):
    """Plot a single model family as a grouped bar chart."""
    import math
    methods = [m for m in method_order if m in results]
    x = np.arange(len(methods))
    labels = [method_labels[m] for m in methods]
    bar_width = 0.22
    behaviors = ["factual", "bias", "sycophancy"]

    for i, behavior in enumerate(behaviors):
        values = []
        for m in methods:
            rho = results[m]["behaviors"][behavior]["rho"]
            values.append(rho if not (isinstance(rho, float) and math.isnan(rho)) else 0.0)
        offset = (i - 1) * bar_width
        bars = ax.bar(
            x + offset, values,
            width=bar_width,
            color=BEHAVIOR_COLORS[behavior],
            label=BEHAVIOR_LABELS[behavior],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=6.5, fontweight="bold",
                color=BEHAVIOR_COLORS[behavior],
            )

    ax.set_xlabel("Merge Method", fontsize=11)
    ax.set_ylabel("ρ (Spearman correlation)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)


def plot_merge_tradeoffs(results, output_path=None, show=False):
    """Create the merge tradeoff visualization for both families."""
    fig, (ax_qwen, ax_mistral) = plt.subplots(
        1, 2, figsize=(16, 6.5),
        gridspec_kw={"wspace": 0.3}
    )

    _plot_family(
        ax_qwen, results, METHOD_ORDER, METHOD_LABELS,
        "Qwen2.5-7B-Instruct + Coder\n(6 merge methods)",
    )

    _plot_family(
        ax_mistral, results, MISTRAL_ORDER, MISTRAL_LABELS,
        "Mistral-7B-Instruct + OpenOrca\n(3 merge methods)",
    )

    # Remove duplicate legend on right panel
    ax_mistral.get_legend().remove()

    fig.text(
        0.5, -0.02,
        "Linear-29 achieves best factual-sycophancy balance on Qwen. "
        "Mistral merges preserve bias detection (>0.93) across all methods. "
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
