#!/usr/bin/env python3
"""Probe Impact Report: before/after analysis of bridge probe additions.

Runs the landscape analysis twice:
  1. BASELINE — all original probes, no bridge/scaleup files
  2. CURRENT  — all probes including bridge/scaleup additions

Then generates:
  A. Before/after redundancy & isolation comparison table
  B. Cross-behavior edge table (Behavior A → Behavior B → # new edges)
  C. Per-probe impact: which bridge probes landed in cross-behavior clusters
  D. docs/probe_impact_report.png — visual comparison
  E. Appends "Probe Impact Report" section to Research_Notes.md

Usage:
    python scripts/probe_impact_report.py
"""

import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rho_eval.probes.registry import PROBE_DATA_DIR

# Import landscape functions
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from probe_landscape import (
    load_all_probes,
    embed_probes,
    build_similarity_graph,
    run_louvain,
    compute_cluster_stats,
    compute_redundancy,
    compute_coverage_gaps,
    ProbeNode,
)

DOCS_DIR = PROJECT_ROOT / "docs"
NOTES_PATH = PROJECT_ROOT / "Research_Notes.md"
THRESHOLD = 0.65

# Bridge probe file patterns (anything with "bridge" or "scaleup" in the name)
BRIDGE_PATTERNS = {"bridge_native", "bridge_pairs", "bridge_scaleup",
                   "bridge_scaleup_v2", "bridge_shortform"}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def is_bridge_probe(probe_set: str) -> bool:
    """Check if a probe_set name corresponds to a bridge file."""
    # probe_set looks like "bias/bridge_native" or "toxicity/bridge_scaleup_v2"
    name = probe_set.split("/")[-1] if "/" in probe_set else probe_set
    return name in BRIDGE_PATTERNS


def run_landscape(nodes, threshold=THRESHOLD):
    """Run the full landscape pipeline on a set of nodes. Returns all metrics."""
    embeddings = embed_probes(nodes)
    G, sensitivity = build_similarity_graph(nodes, embeddings, threshold)
    communities = run_louvain(G)
    cluster_stats = compute_cluster_stats(nodes, communities, G)
    redundancy = compute_redundancy(nodes, communities)
    gaps = compute_coverage_gaps(nodes, communities)

    # Build co-occurrence matrix
    behaviors = sorted(set(n.behavior for n in nodes))
    b2i = {b: i for i, b in enumerate(behaviors)}
    n_beh = len(behaviors)
    cooccur = np.zeros((n_beh, n_beh), dtype=int)

    for cs in cluster_stats:
        if not cs["is_cross_behavior"]:
            continue
        comp = cs["behavior_composition"]
        behs = list(comp.keys())
        for b1 in behs:
            for b2 in behs:
                if b1 != b2:
                    cooccur[b2i[b1], b2i[b2]] += comp[b1]

    cooccur = np.maximum(cooccur, cooccur.T)

    # Count cross-behavior edges in graph
    cross_edges = 0
    for u, v in G.edges():
        if nodes[u].behavior != nodes[v].behavior:
            cross_edges += 1

    return {
        "nodes": nodes,
        "embeddings": embeddings,
        "G": G,
        "communities": communities,
        "cluster_stats": cluster_stats,
        "redundancy": redundancy,
        "gaps": gaps,
        "behaviors": behaviors,
        "cooccur": cooccur,
        "b2i": b2i,
        "n_cross_clusters": sum(1 for cs in cluster_stats if cs["is_cross_behavior"]),
        "cross_edges": cross_edges,
        "n_edges": G.number_of_edges(),
    }


def get_bridge_probe_cluster_info(current_result):
    """For each bridge probe, determine its cluster assignment and whether
    it's in a cross-behavior cluster."""
    nodes = current_result["nodes"]
    communities = current_result["communities"]
    cluster_stats = current_result["cluster_stats"]

    # Map node -> cluster
    node_to_cluster = {}
    for cid, members in enumerate(communities):
        for idx in members:
            node_to_cluster[idx] = cid

    bridge_info = []
    for i, node in enumerate(nodes):
        if not is_bridge_probe(node.probe_set):
            continue
        if node.node_type != "primary":
            continue

        cid = node_to_cluster.get(i, -1)
        cs = cluster_stats[cid] if 0 <= cid < len(cluster_stats) else None
        is_cross = cs["is_cross_behavior"] if cs else False
        cluster_size = cs["size"] if cs else 1
        cluster_comp = cs["behavior_composition"] if cs else {node.behavior: 1}

        bridge_info.append({
            "probe_id": node.probe_id,
            "behavior": node.behavior,
            "probe_set": node.probe_set,
            "cluster_id": cid,
            "cluster_size": cluster_size,
            "is_cross_behavior": is_cross,
            "cluster_composition": cluster_comp,
            "is_singleton": cluster_size == 1,
        })

    return bridge_info


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparison(baseline, current, bridge_info, output_path):
    """Generate 4-panel comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    behaviors = current["behaviors"]
    n_beh = len(behaviors)

    # ── Panel A: Redundancy before/after bar chart ────────────────────────
    ax = axes[0, 0]
    x = np.arange(n_beh)
    width = 0.35

    base_r = [baseline["redundancy"].get(b, {}).get("redundancy", 0) for b in behaviors]
    curr_r = [current["redundancy"].get(b, {}).get("redundancy", 0) for b in behaviors]

    bars1 = ax.bar(x - width/2, base_r, width, label="Baseline (no bridges)",
                   color="#d62728", alpha=0.7, edgecolor="white")
    bars2 = ax.bar(x + width/2, curr_r, width, label="Current (with bridges)",
                   color="#1f77b4", alpha=0.7, edgecolor="white")

    ax.axhline(y=0.80, color="gray", linestyle="--", alpha=0.5, label="Target (0.80)")
    ax.set_ylabel("Redundancy (excl. singletons)")
    ax.set_title("A. Redundancy: Before vs After Bridge Probes")
    ax.set_xticks(x)
    ax.set_xticklabels([b[:6] for b in behaviors], rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Annotate deltas
    for i in range(n_beh):
        delta = curr_r[i] - base_r[i]
        if abs(delta) > 0.01:
            color = "green" if delta < 0 else "red"
            ax.annotate(f"{delta:+.2f}", (x[i] + width/2, curr_r[i] + 0.02),
                       fontsize=7, ha="center", color=color, fontweight="bold")

    # ── Panel B: Cross-behavior edge heatmap (delta) ─────────────────────
    ax = axes[0, 1]
    delta_cooccur = current["cooccur"] - baseline["cooccur"]

    # Mask diagonal
    mask = np.eye(n_beh, dtype=bool)
    display = np.where(mask, np.nan, delta_cooccur).astype(float)

    vmax = max(abs(delta_cooccur.min()), abs(delta_cooccur.max()), 1)
    im = ax.imshow(display, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    for i in range(n_beh):
        for j in range(n_beh):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
            else:
                val = int(delta_cooccur[i, j])
                if val != 0:
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(j, i, f"+{val}" if val > 0 else str(val),
                           ha="center", va="center", fontsize=9,
                           color=color, fontweight="bold" if abs(val) > 5 else "normal")

    ax.set_xticks(range(n_beh))
    ax.set_yticks(range(n_beh))
    ax.set_xticklabels([b[:6] for b in behaviors], rotation=45, ha="right")
    ax.set_yticklabels([b[:6] for b in behaviors])
    ax.set_title("B. Cross-Behavior Edge Delta (Current − Baseline)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Δ probes in cross-behavior clusters")

    # ── Panel C: Bridge probe fate pie chart ─────────────────────────────
    ax = axes[1, 0]
    n_cross = sum(1 for b in bridge_info if b["is_cross_behavior"])
    n_singleton = sum(1 for b in bridge_info if b["is_singleton"])
    n_pure = sum(1 for b in bridge_info
                 if not b["is_cross_behavior"] and not b["is_singleton"])

    sizes = [n_cross, n_pure, n_singleton]
    labels = [f"Cross-behavior\n({n_cross})",
              f"Same-behavior cluster\n({n_pure})",
              f"Singleton\n({n_singleton})"]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    explode = (0.05, 0, 0)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9},
    )
    ax.set_title(f"C. Bridge Probe Fate (n={len(bridge_info)})")

    # ── Panel D: Per-behavior bridge effectiveness ───────────────────────
    ax = axes[1, 1]

    # Group bridge probes by behavior
    beh_bridge = defaultdict(lambda: {"total": 0, "cross": 0, "singleton": 0})
    for b in bridge_info:
        beh_bridge[b["behavior"]]["total"] += 1
        if b["is_cross_behavior"]:
            beh_bridge[b["behavior"]]["cross"] += 1
        elif b["is_singleton"]:
            beh_bridge[b["behavior"]]["singleton"] += 1

    beh_names = sorted(beh_bridge.keys())
    x = np.arange(len(beh_names))
    totals = [beh_bridge[b]["total"] for b in beh_names]
    crosses = [beh_bridge[b]["cross"] for b in beh_names]
    singletons = [beh_bridge[b]["singleton"] for b in beh_names]
    pures = [t - c - s for t, c, s in zip(totals, crosses, singletons)]

    ax.bar(x, crosses, label="Cross-behavior", color="#2ca02c", alpha=0.8)
    ax.bar(x, pures, bottom=crosses, label="Same-behavior", color="#ff7f0e", alpha=0.8)
    ax.bar(x, singletons, bottom=[c+p for c, p in zip(crosses, pures)],
           label="Singleton", color="#d62728", alpha=0.8)

    ax.set_ylabel("Bridge probes")
    ax.set_title("D. Bridge Probe Effectiveness by Behavior")
    ax.set_xticks(x)
    ax.set_xticklabels(beh_names, rotation=45, ha="right")
    ax.legend(fontsize=8)

    # Annotate hit rates
    for i, (t, c) in enumerate(zip(totals, crosses)):
        if t > 0:
            rate = c / t * 100
            ax.text(i, t + 0.3, f"{rate:.0f}%", ha="center", fontsize=8,
                   fontweight="bold", color="#2ca02c")

    fig.suptitle("Probe Impact Report: Bridge Probe Analysis",
                fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Research Notes
# ═══════════════════════════════════════════════════════════════════════════

def append_impact_report(baseline, current, bridge_info, output_json_path):
    """Append Probe Impact Report section to Research_Notes.md."""

    behaviors = current["behaviors"]
    b2i = current["b2i"]
    n_beh = len(behaviors)

    lines = []
    lines.append("\n---\n\n## Probe Impact Report\n")
    lines.append(f"*Auto-generated by `scripts/probe_impact_report.py` on "
                 f"{time.strftime('%Y-%m-%d %H:%M')}*\n")

    # ── Summary stats ────────────────────────────────────────────────────
    n_bridge = len(bridge_info)
    n_cross = sum(1 for b in bridge_info if b["is_cross_behavior"])
    n_singleton = sum(1 for b in bridge_info if b["is_singleton"])
    base_n = len(baseline["nodes"])
    curr_n = len(current["nodes"])

    lines.append("### Summary\n")
    lines.append(f"- **Bridge probes added:** {n_bridge} primary probes across "
                 f"{len(set(b['probe_set'] for b in bridge_info))} probe files")
    lines.append(f"- **Total probe count:** {base_n} → {curr_n} (+{curr_n - base_n})")
    lines.append(f"- **Cross-behavior clusters:** {baseline['n_cross_clusters']} → "
                 f"{current['n_cross_clusters']} "
                 f"(+{current['n_cross_clusters'] - baseline['n_cross_clusters']})")
    lines.append(f"- **Cross-behavior edges:** {baseline['cross_edges']} → "
                 f"{current['cross_edges']} "
                 f"(+{current['cross_edges'] - baseline['cross_edges']})")
    lines.append(f"- **Coverage gaps eliminated:** "
                 f"{len(baseline['gaps'])} → {len(current['gaps'])}")
    if baseline["gaps"]:
        lines.append(f"  - Previously isolated: {', '.join(baseline['gaps'])}")
    lines.append("")

    # ── Redundancy comparison table ──────────────────────────────────────
    lines.append("### Redundancy: Before vs After\n")
    lines.append("| Behavior | Baseline | Current | Δ | Status |")
    lines.append("|:---|:---:|:---:|:---:|:---|")

    for beh in behaviors:
        base_r = baseline["redundancy"].get(beh, {}).get("redundancy", 0)
        curr_r = current["redundancy"].get(beh, {}).get("redundancy", 0)
        delta = curr_r - base_r
        if curr_r <= 0.80:
            status = "✅ Below target"
        elif delta < -0.05:
            status = "🔽 Improved"
        elif base_r > 0.80 and curr_r > 0.80:
            status = "⚠️ Still above"
        else:
            status = "—"
        lines.append(f"| {beh} | {base_r:.3f} | {curr_r:.3f} | {delta:+.3f} | {status} |")

    # ── Cross-behavior edge table ────────────────────────────────────────
    lines.append("\n### Cross-Behavior Edge Table\n")
    lines.append("New cross-behavior connections created by bridge probes "
                 "(current − baseline co-occurrence in shared clusters):\n")

    delta_cooccur = current["cooccur"] - baseline["cooccur"]

    # Collect non-zero pairs
    edge_pairs = []
    for i in range(n_beh):
        for j in range(i + 1, n_beh):
            delta = int(delta_cooccur[i, j])
            base = int(baseline["cooccur"][i, j])
            curr = int(current["cooccur"][i, j])
            if delta != 0 or curr > 0:
                edge_pairs.append((behaviors[i], behaviors[j], base, curr, delta))

    edge_pairs.sort(key=lambda x: -abs(x[4]))

    lines.append("| Behavior A | Behavior B | Baseline | Current | Δ Edges | Targeted? |")
    lines.append("|:---|:---|:---:|:---:|:---:|:---:|")

    targeted_bridges = {
        ("bias", "factual"), ("toxicity", "factual"), ("deception", "factual"),
        ("reasoning", "factual"), ("bias", "toxicity"), ("deception", "sycophancy"),
    }
    for b1, b2, base, curr, delta in edge_pairs:
        pair = tuple(sorted([b1, b2]))
        targeted = "🎯" if pair in targeted_bridges else ""
        delta_str = f"+{delta}" if delta > 0 else str(delta) if delta < 0 else "—"
        lines.append(f"| {b1} | {b2} | {base} | {curr} | {delta_str} | {targeted} |")

    # ── Bridge probe fate ────────────────────────────────────────────────
    lines.append("\n### Bridge Probe Effectiveness\n")
    lines.append(f"Of {n_bridge} bridge probes added:\n")
    lines.append(f"- **{n_cross}** ({n_cross/n_bridge*100:.0f}%) landed in "
                 f"cross-behavior clusters ✅")
    n_pure = n_bridge - n_cross - n_singleton
    lines.append(f"- **{n_pure}** ({n_pure/n_bridge*100:.0f}%) formed "
                 f"same-behavior clusters")
    lines.append(f"- **{n_singleton}** ({n_singleton/n_bridge*100:.0f}%) "
                 f"are singletons (no neighbors at θ={THRESHOLD})")

    # Per-behavior breakdown
    lines.append("\n| Behavior | Bridges | Cross-Beh | Same-Beh | Singleton | Hit Rate |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")

    beh_bridge = defaultdict(lambda: {"total": 0, "cross": 0, "singleton": 0})
    for b in bridge_info:
        beh_bridge[b["behavior"]]["total"] += 1
        if b["is_cross_behavior"]:
            beh_bridge[b["behavior"]]["cross"] += 1
        elif b["is_singleton"]:
            beh_bridge[b["behavior"]]["singleton"] += 1

    for beh in sorted(beh_bridge.keys()):
        d = beh_bridge[beh]
        pure = d["total"] - d["cross"] - d["singleton"]
        rate = d["cross"] / d["total"] * 100 if d["total"] > 0 else 0
        lines.append(f"| {beh} | {d['total']} | {d['cross']} | {pure} | "
                     f"{d['singleton']} | {rate:.0f}% |")

    # ── Individual bridge probes in cross-behavior clusters ──────────────
    cross_bridges = [b for b in bridge_info if b["is_cross_behavior"]]
    if cross_bridges:
        lines.append("\n### Bridge Probes in Cross-Behavior Clusters (top 20)\n")
        lines.append("| Probe | Behavior | Cluster | Size | Composition |")
        lines.append("|:---|:---|:---:|:---:|:---|")

        # Sort by cluster size descending
        cross_bridges.sort(key=lambda x: -x["cluster_size"])
        for b in cross_bridges[:20]:
            comp_str = ", ".join(f"{k}:{v}" for k, v in
                                sorted(b["cluster_composition"].items(),
                                       key=lambda x: -x[1]))
            short_id = b["probe_id"].split("::")[-1][:30]
            lines.append(f"| {short_id} | {b['behavior']} | "
                         f"C{b['cluster_id']} | {b['cluster_size']} | {comp_str} |")

    lines.append(f"\nFigures: `docs/probe_impact_report.png` | "
                 f"Data: `docs/probe_impact_report.json`\n")

    section_text = "\n".join(lines)

    # Read and append/replace
    content = NOTES_PATH.read_text()
    section_marker = "## Probe Impact Report"
    pattern = re.compile(
        r"\n---\n\n## Probe Impact Report\n.*?(?=\n---\n\n## |\Z)",
        re.DOTALL,
    )
    if pattern.search(content):
        content = pattern.sub(section_text, content)
    else:
        # Insert before Probe Diversity Scorecard
        scorecard_marker = "\n---\n\n## Probe Diversity Scorecard\n"
        landscape_marker = "\n---\n\n## Probe Landscape Analysis\n"
        if scorecard_marker in content:
            content = content.replace(scorecard_marker, section_text + scorecard_marker)
        elif landscape_marker in content:
            content = content.replace(landscape_marker, section_text + landscape_marker)
        else:
            content = content.rstrip() + "\n" + section_text

    NOTES_PATH.write_text(content)
    print(f"  Updated: {NOTES_PATH}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Probe Impact Report")
    print("=" * 60)

    # 1. Load ALL probes (current state)
    print("\n[1/6] Loading all probes (current)...")
    all_nodes = load_all_probes(include_variants=True)

    # 2. Separate into baseline and bridge
    print("\n[2/6] Separating baseline vs bridge probes...")
    baseline_nodes = [n for n in all_nodes if not is_bridge_probe(n.probe_set)]
    bridge_only = [n for n in all_nodes if is_bridge_probe(n.probe_set)]

    n_bridge_primary = sum(1 for n in bridge_only if n.node_type == "primary")
    n_bridge_variant = sum(1 for n in bridge_only if n.node_type == "variant")
    print(f"  Baseline: {len(baseline_nodes)} nodes "
          f"({sum(1 for n in baseline_nodes if n.node_type == 'primary')} primary)")
    print(f"  Bridge:   {len(bridge_only)} nodes "
          f"({n_bridge_primary} primary, {n_bridge_variant} variant)")

    # 3. Run baseline landscape
    print("\n[3/6] Running BASELINE landscape (no bridge probes)...")
    baseline_result = run_landscape(baseline_nodes, THRESHOLD)
    print(f"  Baseline: {baseline_result['n_edges']} edges, "
          f"{len(baseline_result['communities'])} communities, "
          f"{baseline_result['n_cross_clusters']} cross-behavior")

    # 4. Run current landscape
    print("\n[4/6] Running CURRENT landscape (all probes)...")
    current_result = run_landscape(all_nodes, THRESHOLD)
    print(f"  Current:  {current_result['n_edges']} edges, "
          f"{len(current_result['communities'])} communities, "
          f"{current_result['n_cross_clusters']} cross-behavior")

    # 5. Analyze bridge probe fate
    print("\n[5/6] Analyzing bridge probe impact...")
    bridge_info = get_bridge_probe_cluster_info(current_result)

    n_cross = sum(1 for b in bridge_info if b["is_cross_behavior"])
    n_singleton = sum(1 for b in bridge_info if b["is_singleton"])
    n_pure = len(bridge_info) - n_cross - n_singleton
    print(f"  Bridge probes: {len(bridge_info)} total")
    print(f"    Cross-behavior: {n_cross} ({n_cross/len(bridge_info)*100:.0f}%)")
    print(f"    Same-behavior:  {n_pure} ({n_pure/len(bridge_info)*100:.0f}%)")
    print(f"    Singleton:      {n_singleton} ({n_singleton/len(bridge_info)*100:.0f}%)")

    # Redundancy comparison
    print(f"\n  Redundancy comparison:")
    print(f"    {'Behavior':15s} {'Baseline':>10s} {'Current':>10s} {'Delta':>10s}")
    for beh in current_result["behaviors"]:
        base_r = baseline_result["redundancy"].get(beh, {}).get("redundancy", 0)
        curr_r = current_result["redundancy"].get(beh, {}).get("redundancy", 0)
        delta = curr_r - base_r
        marker = " ✅" if curr_r <= 0.80 else ""
        print(f"    {beh:15s} {base_r:>10.3f} {curr_r:>10.3f} {delta:>+10.3f}{marker}")

    # Coverage gaps
    print(f"\n  Coverage gaps: {len(baseline_result['gaps'])} → {len(current_result['gaps'])}")
    if baseline_result["gaps"]:
        print(f"    Eliminated: {', '.join(baseline_result['gaps'])}")

    # 6. Generate outputs
    print("\n[6/6] Generating outputs...")

    # Plot
    plot_comparison(baseline_result, current_result, bridge_info,
                   DOCS_DIR / "probe_impact_report.png")

    # JSON
    output_json = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": THRESHOLD,
        "baseline": {
            "n_nodes": len(baseline_nodes),
            "n_edges": baseline_result["n_edges"],
            "n_communities": len(baseline_result["communities"]),
            "n_cross_clusters": baseline_result["n_cross_clusters"],
            "cross_edges": baseline_result["cross_edges"],
            "gaps": baseline_result["gaps"],
            "redundancy": {b: r for b, r in baseline_result["redundancy"].items()},
        },
        "current": {
            "n_nodes": len(all_nodes),
            "n_edges": current_result["n_edges"],
            "n_communities": len(current_result["communities"]),
            "n_cross_clusters": current_result["n_cross_clusters"],
            "cross_edges": current_result["cross_edges"],
            "gaps": current_result["gaps"],
            "redundancy": {b: r for b, r in current_result["redundancy"].items()},
        },
        "bridge_summary": {
            "total": len(bridge_info),
            "cross_behavior": n_cross,
            "same_behavior": n_pure,
            "singleton": n_singleton,
            "hit_rate": round(n_cross / len(bridge_info), 3) if bridge_info else 0,
        },
        "bridge_probes": bridge_info,
        "cooccurrence_delta": {
            f"{current_result['behaviors'][i]}↔{current_result['behaviors'][j]}":
                int(current_result["cooccur"][i, j] - baseline_result["cooccur"][i, j])
            for i in range(len(current_result["behaviors"]))
            for j in range(i + 1, len(current_result["behaviors"]))
            if int(current_result["cooccur"][i, j] - baseline_result["cooccur"][i, j]) != 0
        },
    }

    json_path = DOCS_DIR / "probe_impact_report.json"
    json_path.write_text(json.dumps(output_json, indent=2, default=str))
    print(f"  Saved: {json_path}")

    # Research Notes
    append_impact_report(baseline_result, current_result, bridge_info, json_path)

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Figure: docs/probe_impact_report.png")
    print(f"  Data:   docs/probe_impact_report.json")
    print(f"  Notes:  Research_Notes.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
