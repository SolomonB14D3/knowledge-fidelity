#!/usr/bin/env python3
"""Cross-behavior entanglement heatmap + probe diversity scorecard.

Reads the probe_landscape.json output and generates:
1. docs/entanglement_heatmap.png — behavior×behavior co-occurrence heatmap
2. Appends "Probe Diversity Scorecard" section to Research_Notes.md

Usage:
    python scripts/entanglement_heatmap.py
"""

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
NOTES_PATH = ROOT / "Research_Notes.md"
LANDSCAPE_PATH = DOCS_DIR / "probe_landscape.json"


def load_landscape():
    with open(LANDSCAPE_PATH) as f:
        return json.load(f)


def build_cooccurrence(data):
    """Build behavior×behavior co-occurrence matrix from cross-behavior clusters."""
    behaviors = sorted(data["behaviors"].keys())
    b2i = {b: i for i, b in enumerate(behaviors)}
    n = len(behaviors)

    cross = [c for c in data["clusters"] if c["is_cross_behavior"]]

    # Matrix: how many probes from row-behavior share a cross-behavior
    # cluster with probes from column-behavior
    cooccur = np.zeros((n, n), dtype=float)

    for c in cross:
        comp = c["behavior_composition"]
        behs = list(comp.keys())
        for b1 in behs:
            for b2 in behs:
                if b1 != b2:
                    cooccur[b2i[b1], b2i[b2]] += comp[b1]

    # Symmetrize (take max of both directions)
    symmetric = np.maximum(cooccur, cooccur.T)

    return behaviors, symmetric


def plot_heatmap(behaviors, matrix, output_path):
    """Plot annotated heatmap with behavior labels."""
    n = len(behaviors)
    short_labels = [b[:8] for b in behaviors]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Mask diagonal
    mask = np.eye(n, dtype=bool)
    display = np.where(mask, np.nan, matrix)

    im = ax.imshow(display, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=10, color="gray")
            else:
                val = int(matrix[i, j])
                color = "white" if val > 30 else "black"
                fontweight = "bold" if val > 20 else "normal"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=10, color=color, fontweight=fontweight)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(short_labels, fontsize=10)

    # Highlight isolated behaviors with red border
    isolated = [i for i in range(n) if matrix[i, :].sum() == 0]
    for idx in isolated:
        rect = plt.Rectangle((idx - 0.5, -0.5), 1, n,
                              linewidth=0, facecolor="lightblue", alpha=0.15, zorder=0)
        ax.add_patch(rect)
        rect2 = plt.Rectangle((-0.5, idx - 0.5), n, 1,
                               linewidth=0, facecolor="lightblue", alpha=0.15, zorder=0)
        ax.add_patch(rect2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Probes in shared cross-behavior clusters")

    ax.set_title("Cross-Behavior Entanglement Heatmap\n"
                 "(probe count in shared cross-behavior communities)",
                 fontsize=13, pad=15)

    # Annotations
    ax.text(0.02, -0.12,
            "Blue highlight = isolated behaviors (zero cross-behavior overlap)\n"
            "Strongest link: factual↔sycophancy (38 probes)",
            transform=ax.transAxes, fontsize=8, color="gray", va="top")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def compute_scorecard(data, matrix, behaviors):
    """Compute per-behavior diversity scorecard."""
    scorecard = []

    for i, beh in enumerate(behaviors):
        info = data["behaviors"][beh]
        n_probes = info["n_probes"]
        redundancy = info["redundancy"]
        n_clusters = info["n_clusters_present"]
        cross_coverage = info["in_cross_behavior_clusters"]

        # Cross-behavior link count
        cross_links = int(matrix[i, :].sum())

        # Unique cross-behavior partners
        partners = sum(1 for j in range(len(behaviors)) if i != j and matrix[i, j] > 0)

        # Estimate probes needed to reduce redundancy below 0.80
        # If currently at r with N probes, and r = fraction in pure clusters,
        # adding diverse probes that fall into cross-behavior clusters reduces r.
        # Rough estimate: need to add N * (r - 0.80) / 0.80 diverse probes
        if redundancy > 0.80:
            n_in_pure = int(n_probes * redundancy)
            # We need n_in_pure / (n_probes + x) < 0.80
            # n_in_pure < 0.80 * (n_probes + x)
            # x > (n_in_pure / 0.80) - n_probes
            probes_needed = max(0, int(np.ceil(n_in_pure / 0.80 - n_probes)))
        else:
            probes_needed = 0

        # Priority score: higher = more urgent
        priority = (redundancy * 0.4 +
                    (1 - cross_coverage) * 0.3 +
                    (1 - partners / 8) * 0.3)

        scorecard.append({
            "behavior": beh,
            "n_probes": n_probes,
            "n_clusters": n_clusters,
            "redundancy": redundancy,
            "cross_coverage": cross_coverage,
            "cross_links": cross_links,
            "partners": partners,
            "probes_needed": probes_needed,
            "priority": round(priority, 3),
        })

    # Sort by priority descending
    scorecard.sort(key=lambda x: -x["priority"])
    return scorecard


def append_scorecard_to_notes(scorecard, behaviors, matrix):
    """Append Probe Diversity Scorecard to Research_Notes.md."""
    section_marker = "## Probe Diversity Scorecard"

    lines = []
    lines.append(f"\n---\n\n{section_marker}\n")
    lines.append(f"*Auto-generated by `scripts/entanglement_heatmap.py` on "
                 f"{time.strftime('%Y-%m-%d %H:%M')}*\n")

    # Scorecard table
    lines.append("### Per-Behavior Scorecard\n")
    lines.append("| Priority | Behavior | Probes | Redundancy | Cross-Coverage | "
                 "Partners | Probes Needed |")
    lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|")

    for i, s in enumerate(scorecard):
        cross = "✅" if s["cross_coverage"] else "❌"
        needed = f"+{s['probes_needed']}" if s["probes_needed"] > 0 else "—"
        priority_icon = "🔴" if s["priority"] > 0.8 else "🟡" if s["priority"] > 0.6 else "🟢"
        lines.append(
            f"| {priority_icon} {i+1} | **{s['behavior']}** | {s['n_probes']} | "
            f"{s['redundancy']:.2f} | {cross} | {s['partners']}/8 | {needed} |"
        )

    # Entanglement summary
    lines.append("\n### Cross-Behavior Entanglement Summary\n")
    lines.append("Top entangled pairs (probes in shared cross-behavior communities):\n")

    b2i = {b: i for i, b in enumerate(behaviors)}
    pairs = []
    for i in range(len(behaviors)):
        for j in range(i + 1, len(behaviors)):
            val = int(matrix[i, j])
            if val > 0:
                pairs.append((behaviors[i], behaviors[j], val))
    pairs.sort(key=lambda x: -x[2])

    for b1, b2, count in pairs:
        strength = "strong" if count > 30 else "moderate" if count > 10 else "weak"
        lines.append(f"- **{b1}↔{b2}**: {count} probes ({strength})")

    lines.append(f"\n**Isolated behaviors** (zero cross-behavior overlap): "
                 f"{', '.join(s['behavior'] for s in scorecard if not s['cross_coverage'])}\n")

    # Actionable summary
    lines.append("### Action Items\n")
    urgent = [s for s in scorecard if s["probes_needed"] > 0]
    total_needed = sum(s["probes_needed"] for s in urgent)
    lines.append(f"To reduce all behaviors below 0.80 redundancy: **+{total_needed} probes total**\n")
    for s in urgent:
        lines.append(f"- **{s['behavior']}**: +{s['probes_needed']} diverse probes "
                     f"(current redundancy: {s['redundancy']:.2f})")

    lines.append(f"\nFigure: `docs/entanglement_heatmap.png`\n")

    section_text = "\n".join(lines)

    # Read and append/replace
    content = NOTES_PATH.read_text()
    pattern = re.compile(
        r"\n---\n\n## Probe Diversity Scorecard\n.*?(?=\n---\n\n## |\Z)",
        re.DOTALL,
    )
    if pattern.search(content):
        content = pattern.sub(section_text, content)
    else:
        # Insert before Probe Landscape Analysis section
        landscape_marker = "\n---\n\n## Probe Landscape Analysis\n"
        if landscape_marker in content:
            content = content.replace(landscape_marker, section_text + landscape_marker)
        else:
            content = content.rstrip() + "\n" + section_text

    NOTES_PATH.write_text(content)
    print(f"  Updated: {NOTES_PATH}")


def main():
    print("Cross-Behavior Entanglement Analysis")
    print("=" * 50)

    print("\n[1/4] Loading landscape data...")
    data = load_landscape()

    print("\n[2/4] Building co-occurrence matrix...")
    behaviors, matrix = build_cooccurrence(data)

    # Print matrix
    for i, b in enumerate(behaviors):
        row = " ".join(f"{int(matrix[i,j]):>4d}" for j in range(len(behaviors)))
        print(f"  {b:15s} {row}")

    print(f"\n[3/4] Plotting heatmap...")
    plot_heatmap(behaviors, matrix, DOCS_DIR / "entanglement_heatmap.png")

    print(f"\n[4/4] Computing scorecard...")
    scorecard = compute_scorecard(data, matrix, behaviors)

    print("\n  Priority ranking:")
    for i, s in enumerate(scorecard):
        print(f"    {i+1}. {s['behavior']:15s} priority={s['priority']:.3f} "
              f"redundancy={s['redundancy']:.2f} probes_needed={s['probes_needed']}")

    append_scorecard_to_notes(scorecard, behaviors, matrix)

    print(f"\n{'='*50}")
    print("Done!")


if __name__ == "__main__":
    main()
