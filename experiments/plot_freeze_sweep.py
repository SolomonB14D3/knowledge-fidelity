#!/usr/bin/env python3
"""
Plot freeze-ratio sweep results: delta-vs-freeze line chart + summary table.

Usage:
    python experiments/plot_freeze_sweep.py
    python experiments/plot_freeze_sweep.py --results results/freeze_sweep/sweep_v2.json
    python experiments/plot_freeze_sweep.py --output figures/freeze_sweep_7b.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────

BEHAVIOR_COLORS = {
    "factual": "#2196F3",       # blue
    "bias": "#FF9800",          # orange
    "sycophancy": "#9C27B0",    # purple
    "reasoning": "#4CAF50",     # green
    "toxicity": "#F44336",      # red
}

BEHAVIOR_MARKERS = {
    "factual": "o",
    "bias": "s",
    "sycophancy": "D",
    "reasoning": "^",
    "toxicity": "v",
}

BEHAVIOR_LABELS = {
    "factual": "Factual (rho)",
    "bias": "Bias detection",
    "sycophancy": "Sycophancy resist.",
    "reasoning": "Adversarial reasoning",
    "toxicity": "Toxicity detection",
}


def extract_deltas(results):
    """Extract deltas by behavior and freeze ratio from sweep JSON."""
    behaviors = ["factual", "toxicity", "bias", "sycophancy", "reasoning"]
    freeze_ratios = []
    deltas = {b: [] for b in behaviors}
    baselines = {}

    # Get baseline
    for key, data in results.items():
        if "_baseline" in key:
            for b in behaviors:
                baselines[b] = data["behaviors"][b]["rho"]
            break

    # Get compressed conditions (sorted by freeze ratio)
    conditions = []
    for key, data in results.items():
        if "_baseline" in key:
            continue
        conditions.append((data["freeze_ratio"], key, data))

    conditions.sort(key=lambda x: x[0])

    for fr, key, data in conditions:
        freeze_ratios.append(fr)
        for b in behaviors:
            bdata = data["behaviors"].get(b, {})
            if isinstance(bdata, dict) and "delta" in bdata:
                deltas[b].append(bdata["delta"])
            else:
                deltas[b].append(0.0)

    return freeze_ratios, deltas, baselines


def plot_delta_vs_freeze(freeze_ratios, deltas, baselines, model_name="Qwen2.5-7B",
                         output_path=None, show=False):
    """Create the delta-vs-freeze line chart with behavioral localization."""
    fig, (ax_main, ax_table) = plt.subplots(
        2, 1, figsize=(13, 10), height_ratios=[3, 1.4],
        gridspec_kw={"hspace": 0.40}
    )

    fr_pct = [f * 100 for f in freeze_ratios]

    # ── Main line chart ───────────────────────────────────────────────
    for behavior in ["factual", "bias", "sycophancy", "reasoning", "toxicity"]:
        ax_main.plot(
            fr_pct, deltas[behavior],
            color=BEHAVIOR_COLORS[behavior],
            marker=BEHAVIOR_MARKERS[behavior],
            markersize=8,
            linewidth=2,
            label=BEHAVIOR_LABELS[behavior],
            zorder=3,
        )

    # Zero line
    ax_main.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Shade denoising zone
    ax_main.axvspan(50, 90, alpha=0.06, color="#4CAF50", label="Denoising zone")

    # Annotate peaks
    for behavior in ["factual", "bias", "sycophancy", "reasoning"]:
        vals = deltas[behavior]
        best_idx = np.argmax(vals)
        best_val = vals[best_idx]
        best_fr = fr_pct[best_idx]
        if best_val > 0.01:
            ax_main.annotate(
                f"{best_val:+.3f}",
                xy=(best_fr, best_val),
                xytext=(0, 12),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=BEHAVIOR_COLORS[behavior],
                ha="center",
                va="bottom",
            )

    ax_main.set_xlabel("Freeze Ratio (%)", fontsize=12)
    ax_main.set_ylabel("Δρ (compressed − baseline)", fontsize=12)
    ax_main.set_title(
        f"Behavioral Localization via Freeze-Ratio Sweep\n"
        f"{model_name} · SVD 70% Q/K/O · LoRA recovery (rank 8, 100 steps)",
        fontsize=14, fontweight="bold",
    )
    ax_main.set_xticks(fr_pct)
    ax_main.set_xticklabels([f"{f:.0f}%" for f in fr_pct])
    ax_main.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.3f"))
    ax_main.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(-5, 95)

    # ── Summary table ─────────────────────────────────────────────────
    ax_table.axis("off")

    behaviors = ["factual", "toxicity", "bias", "sycophancy", "reasoning"]

    # Shorter column labels to fit better
    col_labels = ["Behavior", "Base ρ"] + [f"f={f:.0f}%" for f in fr_pct] + ["Best", "Location"]

    table_data = []
    for behavior in behaviors:
        row = [BEHAVIOR_LABELS[behavior], f"{baselines[behavior]:.3f}"]

        vals = deltas[behavior]
        best_idx = np.argmax(vals)
        best_fr = freeze_ratios[best_idx]
        best_delta = vals[best_idx]

        for v in vals:
            if v > 0.01:
                row.append(f"+{v:.3f}")
            elif v < -0.01:
                row.append(f"{v:.3f}")
            else:
                row.append(f"{v:+.3f}")

        row.append(f"{best_fr:.0%}")

        # Localization heuristic
        high_freeze = [vals[i] for i, fr in enumerate(freeze_ratios) if fr >= 0.75]
        low_freeze = [vals[i] for i, fr in enumerate(freeze_ratios) if fr <= 0.25]
        h_mean = np.mean(high_freeze) if high_freeze else 0
        l_mean = np.mean(low_freeze) if low_freeze else 0
        if h_mean > l_mean + 0.01:
            loc = "Early layer"
        elif l_mean > h_mean + 0.01:
            loc = "Late layer"
        elif max(vals) < 0.005:
            loc = "Immovable"
        else:
            loc = "Distributed"
        row.append(loc)

        table_data.append(row)

    # Set explicit column widths: wider for Behavior, narrower for data columns
    n_cols = len(col_labels)
    col_widths = [0.16, 0.08] + [0.09] * len(fr_pct) + [0.07, 0.10]

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#E3F2FD")
        table[0, j].set_text_props(fontweight="bold", fontsize=8)

    # Color cells by delta direction
    for i, behavior in enumerate(behaviors):
        # Color the behavior name cell
        table[i + 1, 0].set_text_props(color=BEHAVIOR_COLORS[behavior], fontweight="bold")

        # Color delta cells
        vals = deltas[behavior]
        for j, v in enumerate(vals):
            cell = table[i + 1, j + 2]  # offset by 2 for behavior + baseline cols
            if v > 0.02:
                cell.set_facecolor("#E8F5E9")  # light green
            elif v < -0.005:
                cell.set_facecolor("#FFEBEE")  # light red

    fig.text(
        0.5, 0.02,
        "Higher freeze = more bottom layers frozen (preserved). "
        "Behaviors peaking at high freeze are encoded in early layers.",
        ha="center", fontsize=9, fontstyle="italic", color="gray",
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def print_markdown_table(freeze_ratios, deltas, baselines):
    """Print a markdown-formatted results table."""
    behaviors = ["factual", "toxicity", "bias", "sycophancy", "reasoning"]

    header = "| Behavior | Baseline ρ |"
    for fr in freeze_ratios:
        header += f" f={fr:.0%} |"
    header += " Best | Location |"
    print(header)

    sep = "|" + "---|" * (len(freeze_ratios) + 4)
    print(sep)

    for behavior in behaviors:
        vals = deltas[behavior]
        best_idx = np.argmax(vals)
        best_fr = freeze_ratios[best_idx]

        row = f"| {behavior} | {baselines[behavior]:.3f} |"
        for v in vals:
            if v > 0.01:
                row += f" **+{v:.3f}** |"
            elif v < -0.01:
                row += f" {v:.3f} |"
            else:
                row += f" {v:+.3f} |"

        # Localization
        high_freeze = [vals[i] for i, fr in enumerate(freeze_ratios) if fr >= 0.75]
        low_freeze = [vals[i] for i, fr in enumerate(freeze_ratios) if fr <= 0.25]
        h_mean = np.mean(high_freeze) if high_freeze else 0
        l_mean = np.mean(low_freeze) if low_freeze else 0
        if h_mean > l_mean + 0.01:
            loc = "Early-layer"
        elif l_mean > h_mean + 0.01:
            loc = "Late-layer"
        elif max(vals) < 0.005:
            loc = "Immovable"
        else:
            loc = "Distributed"

        row += f" {best_fr:.0%} | {loc} |"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Plot freeze-ratio sweep results")
    parser.add_argument("--results", default="results/freeze_sweep/sweep_v2.json")
    parser.add_argument("--output", default="figures/freeze_sweep_7b.png")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--markdown", action="store_true", help="Print markdown table")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    freeze_ratios, deltas, baselines = extract_deltas(results)

    print(f"Loaded {len(results)} entries, {len(freeze_ratios)} conditions")
    print(f"Freeze ratios: {freeze_ratios}")
    print(f"Behaviors: {list(deltas.keys())}")

    if args.markdown:
        print_markdown_table(freeze_ratios, deltas, baselines)
    else:
        plot_delta_vs_freeze(
            freeze_ratios, deltas, baselines,
            model_name="Qwen2.5-7B-Instruct",
            output_path=args.output,
            show=args.show,
        )
        print_markdown_table(freeze_ratios, deltas, baselines)


if __name__ == "__main__":
    main()
