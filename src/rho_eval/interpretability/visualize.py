"""Visualization functions for interpretability results.

Generates publication-quality figures for subspace overlap, head importance,
dimensionality analysis, and surgical intervention comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .schema import SubspaceResult, OverlapMatrix, HeadImportance, SurgicalResult


def _ensure_matplotlib():
    """Import matplotlib with non-interactive backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_overlap_heatmap(
    overlaps: dict[int, OverlapMatrix],
    metric: str = "cosine",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot behavior × behavior overlap heatmap per layer.

    Creates a figure with one subplot per layer, each showing a symmetric
    heatmap of pairwise behavioral overlap.

    Args:
        overlaps: Output of compute_overlap(). {layer_idx: OverlapMatrix}.
        metric: Which metric to plot: "cosine", "shared_variance", or "subspace_angles".
        save_path: If provided, save figure to this path.
        figsize: Optional figure size.
    """
    plt = _ensure_matplotlib()

    layers = sorted(overlaps.keys())
    n_layers = len(layers)
    if n_layers == 0:
        return

    # Grid layout: up to 4 columns
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (4 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    metric_label = {
        "cosine": "Cosine Similarity",
        "shared_variance": "Shared Variance",
        "subspace_angles": "Principal Angle (°)",
    }

    for idx, layer_idx in enumerate(layers):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        om = overlaps[layer_idx]
        matrix_attr = {
            "cosine": om.cosine_matrix,
            "shared_variance": om.shared_variance,
            "subspace_angles": om.subspace_angles,
        }
        data = np.array(matrix_attr[metric])

        # Use absolute values for cosine (direction doesn't matter)
        if metric == "cosine":
            data = np.abs(data)

        # Color scale
        if metric == "subspace_angles":
            vmin, vmax = 0, 90
            cmap = "RdYlBu"
        else:
            vmin, vmax = 0, 1
            cmap = "YlOrRd"

        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")

        # Labels
        n = len(om.behaviors)
        short_names = [b[:4] for b in om.behaviors]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_names, fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                color = "white" if val > (vmax - vmin) * 0.6 + vmin else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    # Hide unused subplots
    for idx in range(n_layers, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Behavioral Subspace Overlap — {metric_label.get(metric, metric)}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_head_importance(
    head_importance: dict[str, list[HeadImportance]],
    save_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot attention head importance grid per behavior.

    Creates a heatmap where rows are behaviors, columns are (layer, head)
    positions, and color intensity shows importance.

    Args:
        head_importance: Output of head_attribution().
        save_path: If provided, save figure.
        figsize: Optional figure size.
    """
    plt = _ensure_matplotlib()

    behaviors = sorted(head_importance.keys())
    if not behaviors:
        return

    # Collect all (layer, head) positions
    all_layers = set()
    n_heads = 0
    for heads in head_importance.values():
        for h in heads:
            all_layers.add(h.layer_idx)
            n_heads = max(n_heads, h.n_heads)

    layers = sorted(all_layers)
    if not layers or n_heads == 0:
        return

    # Build matrix: (n_behaviors, n_layers * n_heads)
    n_cols_data = len(layers) * n_heads
    data = np.zeros((len(behaviors), n_cols_data))

    for bi, behavior in enumerate(behaviors):
        for h in head_importance[behavior]:
            li = layers.index(h.layer_idx)
            col = li * n_heads + h.head_idx
            data[bi, col] = h.importance_score

    if figsize is None:
        figsize = (max(12, n_cols_data * 0.15), max(3, len(behaviors) * 0.8))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0,
                   vmax=max(data.max(), 1.0 / n_heads * 3))

    ax.set_yticks(range(len(behaviors)))
    ax.set_yticklabels(behaviors, fontsize=9)

    # X-axis: label by layer groups
    layer_centers = [(i * n_heads + n_heads / 2 - 0.5) for i in range(len(layers))]
    ax.set_xticks(layer_centers)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8)

    # Add vertical separators between layers
    for i in range(1, len(layers)):
        ax.axvline(x=i * n_heads - 0.5, color="gray", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Layer / Head", fontsize=10)
    ax.set_title("Attention Head Importance per Behavior", fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Importance Score", shrink=0.8)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dimensionality(
    subspaces: dict[str, dict[int, SubspaceResult]],
    variance_threshold: float = 0.90,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> None:
    """Plot effective dimensionality vs layer per behavior.

    Shows how many principal directions are needed to explain the specified
    fraction of variance at each layer, one line per behavior.

    Args:
        subspaces: Output of extract_subspaces().
        variance_threshold: Threshold for effective dimensionality.
        save_path: If provided, save figure.
        figsize: Figure size.
    """
    plt = _ensure_matplotlib()

    behaviors = sorted(subspaces.keys())
    if not behaviors:
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(behaviors), 10)))

    for bi, behavior in enumerate(behaviors):
        layers_data = sorted(subspaces[behavior].items())
        layer_idxs = [l for l, _ in layers_data]
        eff_dims = [sr.effective_dim for _, sr in layers_data]

        ax.plot(layer_idxs, eff_dims, "o-", color=colors[bi], label=behavior,
                markersize=6, linewidth=2)

    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel(f"Effective Dimensionality ({variance_threshold:.0%} variance)", fontsize=11)
    ax.set_title("Behavioral Subspace Dimensionality Across Layers",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_surgical_comparison(
    surgical_results: list[SurgicalResult],
    baselines: dict[str, float],
    eval_behaviors: list[str] | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Compare surgical interventions to baselines.

    Grouped bar chart: one group per evaluated behavior, bars for baseline
    and each intervention type.

    Args:
        surgical_results: List of SurgicalResult objects.
        baselines: {behavior: baseline_rho}.
        eval_behaviors: Behaviors to show (None = all from baselines).
        save_path: If provided, save figure.
        figsize: Figure size.
    """
    plt = _ensure_matplotlib()

    if eval_behaviors is None:
        eval_behaviors = sorted(baselines.keys())

    if not surgical_results or not eval_behaviors:
        return

    # Collect intervention names
    interventions = ["baseline"] + [sr.intervention for sr in surgical_results]
    n_interventions = len(interventions)
    n_behaviors = len(eval_behaviors)

    # Build data matrix
    data = np.zeros((n_interventions, n_behaviors))
    for j, beh in enumerate(eval_behaviors):
        data[0, j] = baselines.get(beh, 0.0)

    for i, sr in enumerate(surgical_results, start=1):
        for j, beh in enumerate(eval_behaviors):
            data[i, j] = sr.rho_scores.get(beh, 0.0) or 0.0

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x = np.arange(n_behaviors)
    width = 0.8 / n_interventions
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_interventions, 8)))

    for i in range(n_interventions):
        offset = (i - n_interventions / 2 + 0.5) * width
        bars = ax.bar(x + offset, data[i], width, label=interventions[i],
                      color=colors[i], edgecolor="gray", linewidth=0.5)

    # PASS/WARN/FAIL threshold lines
    ax.axhline(y=0.5, color="green", linestyle="--", alpha=0.4, label="PASS threshold")
    ax.axhline(y=0.2, color="orange", linestyle="--", alpha=0.4, label="WARN threshold")

    ax.set_xticks(x)
    ax.set_xticklabels(eval_behaviors, fontsize=10)
    ax.set_ylabel("ρ Score", fontsize=11)
    ax.set_title("Surgical Intervention Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
