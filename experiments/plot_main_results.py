#!/usr/bin/env python3
"""Generate publication figures for rho-guided SFT paper.

Reads from results/alignment/*.json and produces:
  - Figure 1: Dose-response curve (lambda vs delta-rho)
  - Figure 2: Ablation bar chart (4 conditions x 4 behaviors)
  - Figure 3: Variance collapse box plots
  - Figure 4: TruthfulQA recovery + safety stress test

Usage:
    python experiments/plot_main_results.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "alignment"
FIGURES_DIR = ROOT / "figures"

# ── Style ─────────────────────────────────────────────────────────────

BEHAVIOR_COLORS = {
    "factual": "#2196F3",
    "toxicity": "#F44336",
    "bias": "#9C27B0",
    "sycophancy": "#4CAF50",
    "reasoning": "#FF9800",
    "refusal": "#607D8B",
}

CONDITION_COLORS = {
    "baseline": "#9E9E9E",
    "sft-only": "#F44336",
    "rho-guided": "#2196F3",
    "contrastive-only": "#FF9800",
    "shuffled-pairs": "#9C27B0",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "font.family": "sans-serif",
})


# ── Data Loading ──────────────────────────────────────────────────────

def load_dose_response_data() -> dict:
    """Load all dose-response sweep data and compute 5-seed statistics.

    Merges data from multiple sweep files (seeds 42, 123 in one file,
    456 in another, 789+1337 in another).

    Returns:
        Dict: {lambda: {behavior: [delta_values_per_seed]}}
    """
    # Known data from ablation_5seed_analysis.json (pre-computed)
    analysis_path = RESULTS_DIR / "ablation_5seed_analysis.json"
    if analysis_path.exists():
        with open(analysis_path) as f:
            data = json.load(f)

    # Also collect from sweep files for dose-response at all lambdas
    sweep_files = [
        RESULTS_DIR / "mlx_rho_sft_sweep_Qwen_Qwen2.5-7B-Instruct.json",
        RESULTS_DIR / "mlx_rho_sft_sweep_Qwen_Qwen2.5-7B-Instruct_s789_1337.json",
    ]

    # Collect per-lambda, per-seed deltas
    by_lambda: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for fpath in sweep_files:
        if not fpath.exists():
            continue
        with open(fpath) as f:
            sweep = json.load(f)

        for run in sweep.get("runs", []):
            lam = run["rho_weight"]
            deltas = run.get("quick_deltas", {})
            for beh, delta in deltas.items():
                if beh in ("reasoning",):
                    continue  # reasoning is always 0 in quick_eval
                by_lambda[lam][beh].append(delta)

    return dict(by_lambda)


def load_ablation_data() -> dict:
    """Load 5-seed ablation analysis."""
    path = RESULTS_DIR / "ablation_5seed_analysis.json"
    with open(path) as f:
        return json.load(f)


def load_truthfulqa_data() -> dict:
    """Load TruthfulQA MC2 results."""
    path = RESULTS_DIR / "truthfulqa_Qwen_Qwen2.5-7B-Instruct.json"
    with open(path) as f:
        return json.load(f)


def load_safety_stress_test() -> dict:
    """Load safety stress test results."""
    path = RESULTS_DIR / "safety_stress_test_qwen2.5-7b.json"
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Dose-Response Curve ─────────────────────────────────────

def plot_dose_response():
    """Plot lambda_rho vs delta-rho for key behaviors."""
    data = load_dose_response_data()

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    behaviors = ["factual", "toxicity", "bias"]
    markers = {"factual": "o", "toxicity": "s", "bias": "D"}
    lambdas = sorted(data.keys())

    for beh in behaviors:
        means = []
        stds = []
        xs = []
        for lam in lambdas:
            vals = data.get(lam, {}).get(beh, [])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                xs.append(lam)

        means = np.array(means)
        stds = np.array(stds)
        xs = np.array(xs)

        ax.plot(xs, means, marker=markers[beh], color=BEHAVIOR_COLORS[beh],
                label=beh.capitalize(), linewidth=2, markersize=7, zorder=3)
        ax.fill_between(xs, means - stds, means + stds,
                        alpha=0.15, color=BEHAVIOR_COLORS[beh], zorder=1)

    # Reference line at y=0
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Annotate toxicity inversion
    tox_at_0 = data.get(0.0, {}).get("toxicity", [])
    if tox_at_0:
        ax.annotate(
            "SFT inverts\ntoxicity",
            xy=(0.0, np.mean(tox_at_0)),
            xytext=(0.07, -0.25),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.2),
            color="#F44336",
        )

    ax.set_xlabel(r"$\lambda_\rho$ (contrastive weight)")
    ax.set_ylabel(r"$\Delta\rho$ from baseline")
    ax.set_title("Dose-Response: Behavioral Impact of Contrastive Weight")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(-0.02, 0.52)
    ax.grid(True, alpha=0.2)

    out = FIGURES_DIR / "fig1_dose_response"
    fig.savefig(f"{out}.png")
    fig.savefig(f"{out}.pdf")
    print(f"  Saved: {out}.png/pdf")
    plt.close(fig)


# ── Figure 2: Ablation Bar Chart ─────────────────────────────────────

def plot_ablation():
    """Plot ablation comparison: 4 conditions x 4 behaviors."""
    data = load_ablation_data()
    baseline = data["baseline"]
    deltas = data["merged_deltas"]

    conditions = ["sft-only", "rho-guided", "contrastive-only", "shuffled-pairs"]
    behaviors = ["factual", "toxicity", "sycophancy", "bias"]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    n_cond = len(conditions)
    n_beh = len(behaviors)
    x = np.arange(n_beh)
    bar_width = 0.18
    offsets = np.arange(n_cond) - (n_cond - 1) / 2

    for i, cond in enumerate(conditions):
        means = []
        stds = []
        for beh in behaviors:
            vals = deltas[cond].get(beh, [])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(
            x + offsets[i] * bar_width, means, bar_width,
            yerr=stds, label=cond,
            color=CONDITION_COLORS[cond], edgecolor="white", linewidth=0.5,
            capsize=3, error_kw={"linewidth": 1},
        )

    # Reference line at y=0
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    ax.set_ylabel(r"$\Delta\rho$ from baseline (5-seed mean)")
    ax.set_title("Ablation Study: Isolating the Contrastive Loss Effect")
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in behaviors])
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax.grid(True, axis="y", alpha=0.2)

    out = FIGURES_DIR / "fig2_ablation"
    fig.savefig(f"{out}.png")
    fig.savefig(f"{out}.pdf")
    print(f"  Saved: {out}.png/pdf")
    plt.close(fig)


# ── Figure 3: Variance Collapse ───────────────────────────────────────

def plot_variance_collapse():
    """Plot seed-level variance: SFT-only vs rho-guided."""
    data = load_ablation_data()
    deltas = data["merged_deltas"]

    behaviors = ["factual", "toxicity", "sycophancy", "bias"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()

    for idx, beh in enumerate(behaviors):
        ax = axes[idx]

        sft_vals = deltas["sft-only"].get(beh, [])
        rho_vals = deltas["rho-guided"].get(beh, [])

        bp = ax.boxplot(
            [sft_vals, rho_vals],
            tick_labels=["SFT-only\n" + r"($\lambda_\rho=0$)",
                         "Rho-guided\n" + r"($\lambda_\rho=0.2$)"],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
        )

        bp["boxes"][0].set_facecolor(CONDITION_COLORS["sft-only"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(CONDITION_COLORS["rho-guided"])
        bp["boxes"][1].set_alpha(0.6)

        # Scatter individual seeds
        for j, vals in enumerate([sft_vals, rho_vals]):
            xs = np.random.RandomState(42).normal(j + 1, 0.04, len(vals))
            ax.scatter(xs, vals, alpha=0.7, s=25, zorder=3,
                       color="black", edgecolors="white", linewidth=0.5)

        # Annotate sigma
        s1 = np.std(sft_vals) if sft_vals else 0
        s2 = np.std(rho_vals) if rho_vals else 0
        reduction = (1 - s2 / s1) * 100 if s1 > 0 else 0

        ax.set_title(f"{beh.capitalize()}" +
                     (f" ({reduction:.0f}% reduction)" if abs(reduction) > 5 else ""),
                     fontsize=10)

        # Add sigma annotations
        ax.text(1, max(sft_vals or [0]) * 1.05,
                f"$\\sigma$={s1:.3f}", ha="center", fontsize=8, color="#666")
        ax.text(2, max(rho_vals or [0]) * 1.05,
                f"$\\sigma$={s2:.3f}", ha="center", fontsize=8, color="#666")

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(True, axis="y", alpha=0.2)

    fig.suptitle("Variance Collapse: 5-Seed Spread by Training Condition", y=1.02)
    fig.tight_layout()

    out = FIGURES_DIR / "fig3_variance_collapse"
    fig.savefig(f"{out}.png")
    fig.savefig(f"{out}.pdf")
    print(f"  Saved: {out}.png/pdf")
    plt.close(fig)


# ── Figure 4: TruthfulQA + Safety ─────────────────────────────────────

def plot_truthfulqa_and_safety():
    """Plot TruthfulQA MC2 recovery and safety stress test results."""
    tqa = load_truthfulqa_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Left panel: TruthfulQA MC2 ───────────────────────────────────
    baseline_mc2 = tqa["baseline"]["mc2_score"]

    # Group by rho_weight
    by_rho = defaultdict(list)
    for run in tqa["runs"]:
        by_rho[run["rho_weight"]].append(run["mc2_score"])

    groups = ["Baseline"]
    values = [baseline_mc2]
    colors = [CONDITION_COLORS["baseline"]]

    for rho_w in sorted(by_rho.keys()):
        mc2s = by_rho[rho_w]
        mean_mc2 = np.mean(mc2s)
        if rho_w == 0.0:
            groups.append(f"SFT-only\n" + r"($\lambda=0$)")
            colors.append(CONDITION_COLORS["sft-only"])
        else:
            groups.append(f"Rho-guided\n" + r"($\lambda=$" + f"{rho_w})")
            colors.append(CONDITION_COLORS["rho-guided"])
        values.append(mean_mc2)

    bars = ax1.bar(range(len(groups)), values, color=colors,
                   edgecolor="white", linewidth=0.5, width=0.6)

    # Annotate values
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", fontsize=9)

    # Recovery arrow
    if len(values) >= 3:
        recovery = values[2] - values[1]
        total_loss = values[0] - values[1]
        pct = recovery / total_loss * 100 if total_loss > 0 else 0
        ax1.annotate(
            f"{pct:.0f}% recovery",
            xy=(2, values[2]), xytext=(2.3, (values[1] + values[2]) / 2),
            fontsize=8, color=CONDITION_COLORS["rho-guided"],
            arrowprops=dict(arrowstyle="->", color=CONDITION_COLORS["rho-guided"]),
        )

    ax1.set_ylabel("TruthfulQA MC2 Score")
    ax1.set_title("TruthfulQA MC2: SFT Damage and Recovery")
    ax1.set_xticks(range(len(groups)))
    ax1.set_xticklabels(groups, fontsize=9)
    ax1.set_ylim(0.4, 0.72)
    ax1.grid(True, axis="y", alpha=0.2)

    # ── Right panel: Safety Stress Test ───────────────────────────────
    try:
        safety = load_safety_stress_test()
        conditions_order = ["baseline", "sft-only", "contrastive-only", "rho-guided"]
        condition_labels = ["Baseline", "SFT-only", "Contrastive\nonly", "Rho-guided"]

        jailbreak_rates = []
        benign_rates = []
        for cond in conditions_order:
            cond_data = safety.get("conditions", {}).get(cond, {})
            # Direct keys at condition level (not nested in summary)
            jr = cond_data.get("jailbreak_refusal_rate", 0)
            br = cond_data.get("benign_refusal_rate", 0)
            jailbreak_rates.append(jr * 100)
            benign_rates.append(br * 100)

        x = np.arange(len(conditions_order))
        width = 0.35

        bars1 = ax2.bar(x - width / 2, jailbreak_rates, width,
                        label="Jailbreak Refusal %",
                        color="#4CAF50", edgecolor="white")
        bars2 = ax2.bar(x + width / 2, benign_rates, width,
                        label="Benign Refusal %",
                        color="#FF5722", edgecolor="white")

        # Annotate
        for bar, val in zip(bars1, jailbreak_rates):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 1,
                     f"{val:.0f}%", ha="center", fontsize=8)

        ax2.set_ylabel("Refusal Rate (%)")
        ax2.set_title("Safety Stress Test: Jailbreak Refusal")
        ax2.set_xticks(x)
        ax2.set_xticklabels(condition_labels, fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, axis="y", alpha=0.2)

    except Exception as e:
        ax2.text(0.5, 0.5, f"Safety data not available\n{e}",
                 transform=ax2.transAxes, ha="center", va="center")

    fig.tight_layout()

    out = FIGURES_DIR / "fig4_truthfulqa_safety"
    fig.savefig(f"{out}.png")
    fig.savefig(f"{out}.pdf")
    print(f"  Saved: {out}.png/pdf")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating publication figures...")
    print()

    print("Figure 1: Dose-Response Curve")
    plot_dose_response()

    print("Figure 2: Ablation Bar Chart")
    plot_ablation()

    print("Figure 3: Variance Collapse")
    plot_variance_collapse()

    print("Figure 4: TruthfulQA + Safety")
    plot_truthfulqa_and_safety()

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
