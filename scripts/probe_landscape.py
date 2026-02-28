#!/usr/bin/env python3
"""Analyze semantic structure of all probes across behavioral dimensions.

Embeds probes with all-MiniLM-L6-v2, builds a cosine similarity graph,
runs Louvain community detection, and reports redundancy and coverage metrics.

Outputs:
    docs/probe_landscape.json  — full cluster data
    docs/probe_landscape.png   — 2D t-SNE projection colored by behavior
    Research_Notes.md          — appended "Probe Landscape Analysis" section

Usage:
    python scripts/probe_landscape.py
    python scripts/probe_landscape.py --threshold 0.70
    python scripts/probe_landscape.py --no-variants

Requires: sentence-transformers, networkx, scikit-learn, matplotlib, numpy
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure the project's src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rho_eval.probes.registry import list_probe_sets, get_probes, PROBE_DATA_DIR

# ── Constants ──────────────────────────────────────────────────────────────

BEHAVIOR_COLORS = {
    "factual":     "#1f77b4",  # blue
    "toxicity":    "#d62728",  # red
    "bias":        "#2ca02c",  # green
    "sycophancy":  "#ff7f0e",  # orange
    "reasoning":   "#9467bd",  # purple
    "refusal":     "#8c564b",  # brown
    "deception":   "#e377c2",  # pink
    "overrefusal": "#7f7f7f",  # gray
    "bench":       "#17becf",  # cyan
}

SENSITIVITY_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


@dataclass
class ProbeNode:
    probe_id: str
    behavior: str
    probe_set: str
    text: str
    node_type: str  # "primary" or "variant"
    original_id: str = ""
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Probe loading
# ═══════════════════════════════════════════════════════════════════════════

def load_all_probes(include_variants: bool = True) -> list[ProbeNode]:
    """Load all probes from the registry, with optional variant nodes.

    Variant nodes are created for refusal (harmful_version) and deception
    (deceptive) probes. Factual `false` fields are NOT included as variants
    because they are trivially similar to their `text` counterpart.
    """
    nodes = []
    all_sets = list_probe_sets()

    for pset in all_sets:
        behavior = pset.split("/")[0]

        # Load all probe sets directly (bypassing registry validation,
        # since some probes use non-standard keys like honest/deceptive,
        # positive/negative instead of text)
        path = PROBE_DATA_DIR / f"{pset}.json"
        with open(path) as f:
            probes = json.load(f)

        for i, p in enumerate(probes):
            pid = p.get("id", f"{pset}::{i}")

            # Determine primary text field based on probe format
            if "text" in p:
                primary_text = p["text"]
                exclude_keys = {"text"}
            elif "honest" in p:
                # Deception probes: honest = primary
                primary_text = p["honest"]
                exclude_keys = {"honest", "deceptive"}
            elif "positive" in p and "prompt" in p:
                # Sycophancy continuation pairs: prompt + positive = primary
                prompt = p.get("prompt", "")
                primary_text = f"{prompt}\n{p['positive']}" if prompt else p["positive"]
                exclude_keys = {"prompt", "positive", "negative"}
            else:
                print(f"  Warning: skipping probe {pid} in {pset} — no text field")
                continue

            nodes.append(ProbeNode(
                probe_id=f"{pset}::{pid}",
                behavior=behavior,
                probe_set=pset,
                text=primary_text,
                node_type="primary",
                original_id=pid,
                metadata={k: v for k, v in p.items() if k not in exclude_keys},
            ))

            # Variant nodes
            if include_variants:
                # Deception: deceptive continuation
                if "deceptive" in p:
                    nodes.append(ProbeNode(
                        probe_id=f"{pset}::{pid}::deceptive",
                        behavior=behavior,
                        probe_set=pset,
                        text=p["deceptive"],
                        node_type="variant",
                        original_id=pid,
                        metadata={k: v for k, v in p.items()
                                  if k not in ("honest", "deceptive")},
                    ))

                # Refusal: harmful_version
                if "harmful_version" in p:
                    nodes.append(ProbeNode(
                        probe_id=f"{pset}::{pid}::harmful",
                        behavior=behavior,
                        probe_set=pset,
                        text=p["harmful_version"],
                        node_type="variant",
                        original_id=pid,
                        metadata={k: v for k, v in p.items()
                                  if k not in ("text", "harmful_version")},
                    ))

                # Sycophancy pairs: negative continuation
                if "positive" in p and "negative" in p:
                    prompt = p.get("prompt", "")
                    neg_text = f"{prompt}\n{p['negative']}" if prompt else p["negative"]
                    nodes.append(ProbeNode(
                        probe_id=f"{pset}::{pid}::negative",
                        behavior=behavior,
                        probe_set=pset,
                        text=neg_text,
                        node_type="variant",
                        original_id=pid,
                        metadata={k: v for k, v in p.items()
                                  if k not in ("prompt", "positive", "negative")},
                    ))

    print(f"  Loaded {len(nodes)} nodes "
          f"({sum(1 for n in nodes if n.node_type == 'primary')} primary, "
          f"{sum(1 for n in nodes if n.node_type == 'variant')} variant)")
    return nodes


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Embedding
# ═══════════════════════════════════════════════════════════════════════════

def embed_probes(nodes: list[ProbeNode]) -> np.ndarray:
    """Compute normalized 384-dim embeddings with all-MiniLM-L6-v2 (CPU)."""
    from sentence_transformers import SentenceTransformer

    print("  Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [node.text for node in nodes]
    print(f"  Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Graph construction
# ═══════════════════════════════════════════════════════════════════════════

def build_similarity_graph(
    nodes: list[ProbeNode],
    embeddings: np.ndarray,
    threshold: float = 0.65,
) -> tuple:
    """Build graph with edges where cosine similarity > threshold.

    Returns:
        (nx.Graph, sensitivity_dict)
    """
    import networkx as nx

    n = len(nodes)
    print(f"  Computing {n}x{n} similarity matrix...")
    sim_matrix = embeddings @ embeddings.T

    # Sensitivity analysis
    sensitivity = {}
    for t in SENSITIVITY_THRESHOLDS:
        upper = np.triu(sim_matrix, k=1)
        n_edges = int(np.sum(upper > t))
        avg_deg = 2 * n_edges / n if n > 0 else 0
        sensitivity[str(t)] = {"n_edges": n_edges, "avg_degree": round(avg_deg, 1)}
        marker = " <-- selected" if abs(t - threshold) < 0.001 else ""
        print(f"    threshold={t:.2f}: {n_edges:>6,} edges  (avg degree {avg_deg:>5.1f}){marker}")

    # Build graph at chosen threshold
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, **{
            "probe_id": nodes[i].probe_id,
            "behavior": nodes[i].behavior,
            "node_type": nodes[i].node_type,
        })

    print(f"  Adding edges at threshold={threshold}...")
    rows, cols = np.where(np.triu(sim_matrix, k=1) > threshold)
    for r, c in zip(rows, cols):
        G.add_edge(int(r), int(c), weight=float(sim_matrix[r, c]))

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, sensitivity


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Community detection + metrics
# ═══════════════════════════════════════════════════════════════════════════

def run_louvain(G) -> list[set[int]]:
    """Run Louvain community detection. Returns list of communities."""
    import networkx as nx

    if G.number_of_edges() == 0:
        # Each node is its own community
        return [frozenset([i]) for i in G.nodes()]

    communities = nx.community.louvain_communities(G, seed=42, resolution=1.0)
    # Sort by size descending
    communities = sorted(communities, key=len, reverse=True)
    print(f"  Found {len(communities)} communities "
          f"(largest: {len(communities[0])}, smallest: {len(communities[-1])})")
    return communities


def compute_cluster_stats(
    nodes: list[ProbeNode],
    communities: list[set[int]],
    G,
) -> list[dict]:
    """Compute per-cluster statistics."""
    import networkx as nx

    stats = []
    for cid, members in enumerate(communities):
        members_list = sorted(members)
        behavior_counts = Counter(nodes[i].behavior for i in members_list)
        dominant = behavior_counts.most_common(1)[0]

        # Degree centrality within subgraph
        sub = G.subgraph(members_list)
        if sub.number_of_edges() > 0:
            centrality = nx.degree_centrality(sub)
            most_central_idx = max(centrality, key=centrality.get)
            central_score = centrality[most_central_idx]
        else:
            most_central_idx = members_list[0]
            central_score = 0.0

        stats.append({
            "cluster_id": cid,
            "size": len(members_list),
            "behavior_composition": dict(behavior_counts),
            "dominant_behavior": dominant[0],
            "dominance_fraction": round(dominant[1] / len(members_list), 3),
            "is_cross_behavior": dominant[1] / len(members_list) < 0.80,
            "most_central_probe": nodes[most_central_idx].probe_id,
            "central_score": round(central_score, 4),
            "member_probe_ids": [nodes[i].probe_id for i in members_list],
        })
    return stats


def compute_redundancy(
    nodes: list[ProbeNode],
    communities: list[set[int]],
) -> dict[str, dict]:
    """Compute redundancy metrics per behavior, separating true redundancy
    from singleton isolation.

    Returns dict[behavior] -> {
        "redundancy":    fraction of *clustered* primary probes in >80% same-behavior clusters,
        "isolation_rate": fraction of primary probes that are singletons (no neighbors),
        "n_primary":     total primary probes,
        "n_clustered":   primary probes in clusters of size >= 2,
        "n_singletons":  primary probes in singleton clusters,
    }

    The key insight: singletons (size-1 clusters) are trivially 100% "pure" but
    represent semantic *isolation*, not *homogeneity*. Only probes that cluster
    with neighbors reveal true redundancy patterns.
    """
    # Map node index -> cluster id
    node_to_cluster = {}
    cluster_sizes = {}
    for cid, members in enumerate(communities):
        cluster_sizes[cid] = len(members)
        for idx in members:
            node_to_cluster[idx] = cid

    # Cluster dominance (only meaningful for clusters of size >= 2)
    cluster_dominant_frac = {}
    for cid, members in enumerate(communities):
        if len(members) < 2:
            cluster_dominant_frac[cid] = 1.0  # singleton — trivially pure
            continue
        counts = Counter(nodes[i].behavior for i in members)
        top = counts.most_common(1)[0][1]
        cluster_dominant_frac[cid] = top / len(members)

    # Per behavior: split into singletons vs clustered
    behaviors = sorted(set(n.behavior for n in nodes))
    redundancy = {}
    for beh in behaviors:
        primary_idxs = [i for i, n in enumerate(nodes)
                        if n.behavior == beh and n.node_type == "primary"]
        if not primary_idxs:
            redundancy[beh] = {
                "redundancy": 0.0, "isolation_rate": 0.0,
                "n_primary": 0, "n_clustered": 0, "n_singletons": 0,
            }
            continue

        # Separate singletons from clustered probes
        singletons = [i for i in primary_idxs
                       if cluster_sizes.get(node_to_cluster.get(i, -1), 1) == 1]
        clustered = [i for i in primary_idxs
                      if cluster_sizes.get(node_to_cluster.get(i, -1), 1) >= 2]

        n_total = len(primary_idxs)
        n_single = len(singletons)
        n_clust = len(clustered)

        # Redundancy: among clustered probes, how many are in >80% same-behavior clusters?
        if n_clust > 0:
            in_homo = sum(1 for i in clustered
                          if cluster_dominant_frac.get(node_to_cluster.get(i, -1), 0) > 0.80)
            redund = round(in_homo / n_clust, 3)
        else:
            redund = 0.0  # All singletons = no redundancy measurable

        redundancy[beh] = {
            "redundancy": redund,
            "isolation_rate": round(n_single / n_total, 3) if n_total > 0 else 0.0,
            "n_primary": n_total,
            "n_clustered": n_clust,
            "n_singletons": n_single,
        }
    return redundancy


def compute_coverage_gaps(
    nodes: list[ProbeNode],
    communities: list[set[int]],
) -> list[str]:
    """Find behaviors with zero representation in cross-behavior clusters."""
    behaviors_in_cross = set()
    for members in communities:
        counts = Counter(nodes[i].behavior for i in members)
        top_frac = counts.most_common(1)[0][1] / len(members) if members else 1
        if top_frac < 0.80:  # cross-behavior cluster
            behaviors_in_cross.update(counts.keys())

    all_behaviors = sorted(set(n.behavior for n in nodes))
    return [b for b in all_behaviors if b not in behaviors_in_cross]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Outputs
# ═══════════════════════════════════════════════════════════════════════════

def build_output_json(
    nodes, communities, cluster_stats, redundancy, gaps, sensitivity,
    threshold, embeddings,
) -> dict:
    """Build the full output JSON structure."""
    import networkx as nx

    # Per-probe assignments
    node_to_cluster = {}
    for cid, members in enumerate(communities):
        for idx in members:
            node_to_cluster[idx] = cid

    probe_assignments = {}
    for i, node in enumerate(nodes):
        probe_assignments[node.probe_id] = {
            "cluster_id": node_to_cluster.get(i, -1),
            "behavior": node.behavior,
            "node_type": node.node_type,
            "probe_set": node.probe_set,
        }

    # Per-behavior summary
    behaviors = sorted(set(n.behavior for n in nodes))
    behavior_summary = {}
    for beh in behaviors:
        n_probes = sum(1 for n in nodes if n.behavior == beh
                       and n.node_type == "primary")
        clusters_present = len(set(
            node_to_cluster.get(i, -1)
            for i, n in enumerate(nodes) if n.behavior == beh
        ))
        r = redundancy.get(beh, {})
        behavior_summary[beh] = {
            "n_probes": n_probes,
            "redundancy": r.get("redundancy", 0) if isinstance(r, dict) else r,
            "isolation_rate": r.get("isolation_rate", 0) if isinstance(r, dict) else 0,
            "n_clustered": r.get("n_clustered", 0) if isinstance(r, dict) else 0,
            "n_singletons": r.get("n_singletons", 0) if isinstance(r, dict) else 0,
            "n_clusters_present": clusters_present,
            "in_cross_behavior_clusters": beh not in gaps,
        }

    n_primary = sum(1 for n in nodes if n.node_type == "primary")
    n_variant = sum(1 for n in nodes if n.node_type == "variant")

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "n_probes": len(nodes),
            "n_primary": n_primary,
            "n_variant": n_variant,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": embeddings.shape[1],
            "similarity_threshold": threshold,
            "n_edges": sensitivity.get(str(threshold), {}).get("n_edges", 0),
            "n_clusters": len(communities),
            "threshold_sensitivity": sensitivity,
        },
        "behaviors": behavior_summary,
        "clusters": cluster_stats,
        "coverage_gaps": gaps,
        "probe_assignments": probe_assignments,
    }


def plot_landscape(
    nodes: list[ProbeNode],
    embeddings: np.ndarray,
    communities: list[set[int]],
    output_path: Path,
    perplexity: float = 30.0,
):
    """2D t-SNE projection colored by behavior, with cluster hulls."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from scipy.spatial import ConvexHull
    from sklearn.manifold import TSNE

    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Cluster convex hulls (light shading for clusters >= 5 members)
    node_to_cluster = {}
    for cid, members in enumerate(communities):
        for idx in members:
            node_to_cluster[idx] = cid

    for cid, members in enumerate(communities):
        if len(members) < 5:
            continue
        pts = coords[sorted(members)]
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])  # close polygon

            # Color based on dominant behavior
            counts = Counter(nodes[i].behavior for i in members)
            dominant_beh = counts.most_common(1)[0][0]
            color = BEHAVIOR_COLORS.get(dominant_beh, "#999999")
            ax.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.08, color=color)
        except Exception:
            pass  # degenerate hull (collinear points)

    # Scatter: primary vs variant markers
    for node_type, marker, size, alpha in [
        ("primary", "o", 20, 0.7),
        ("variant", "^", 15, 0.5),
    ]:
        mask = [i for i, n in enumerate(nodes) if n.node_type == node_type]
        if not mask:
            continue
        for beh, color in BEHAVIOR_COLORS.items():
            beh_mask = [i for i in mask if nodes[i].behavior == beh]
            if not beh_mask:
                continue
            n_probes = sum(1 for i in beh_mask
                          if nodes[i].node_type == "primary")
            label = (f"{beh} ({n_probes})" if node_type == "primary"
                     else f"{beh} (variant)")
            ax.scatter(
                coords[beh_mask, 0], coords[beh_mask, 1],
                c=color, s=size, alpha=alpha, marker=marker,
                label=label, edgecolors="none",
            )

    # Label the 5 largest clusters
    for cid, members in enumerate(communities[:5]):
        centroid = coords[sorted(members)].mean(axis=0)
        counts = Counter(nodes[i].behavior for i in members)
        dominant = counts.most_common(1)[0]
        label = f"C{cid}: {dominant[0]} ({len(members)})"
        ax.annotate(
            label, centroid, fontsize=8, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8),
        )

    ax.set_title(
        f"Probe Landscape: {len(nodes)} probes across "
        f"{len(set(n.behavior for n in nodes))} behaviors\n"
        f"t-SNE projection | {len(communities)} Louvain communities",
        fontsize=13,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="upper left", fontsize=7, ncol=2, framealpha=0.9,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def append_research_notes(
    cluster_stats: list[dict],
    redundancy: dict[str, float],
    gaps: list[str],
    nodes: list[ProbeNode],
    metadata: dict,
    notes_path: Path,
):
    """Append or replace 'Probe Landscape Analysis' section in Research_Notes.md."""
    # Build the new section
    lines = []
    lines.append("\n---\n")
    lines.append("## Probe Landscape Analysis\n")
    lines.append(
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d')} | "
        f"**Embedding model:** all-MiniLM-L6-v2 | "
        f"**Threshold:** {metadata['similarity_threshold']} | "
        f"**N nodes:** {metadata['n_probes']:,}\n"
    )

    # Cluster summary table (top 15 clusters by size)
    lines.append("### Cluster Summary (top 15 by size)\n")
    lines.append(
        "| Cluster | Size | Dominant Behavior | Dominance | "
        "Cross-Behavior? | Central Probe |"
    )
    lines.append(
        "|:---:|:---:|:---|:---:|:---:|:---|"
    )
    for cs in cluster_stats[:15]:
        cb = "Yes" if cs["is_cross_behavior"] else "No"
        central = cs["most_central_probe"].split("::")[-1]
        if central == "variant":
            central = cs["most_central_probe"].split("::")[-2] + " (var)"
        lines.append(
            f"| {cs['cluster_id']} | {cs['size']} | {cs['dominant_behavior']} "
            f"| {cs['dominance_fraction']:.0%} | {cb} | {central[:40]} |"
        )

    # Redundancy table
    lines.append("\n### Redundancy & Isolation Scores\n")
    lines.append(
        "**Redundancy** = fraction of *clustered* (non-singleton) primary probes "
        "in clusters where >80% share the same behavior. Singletons (probes with "
        "no neighbors at θ=threshold) are excluded — they represent semantic "
        "*isolation*, not *homogeneity*.\n"
    )
    lines.append("| Behavior | Probes | Clustered | Singletons | Isolation | Redundancy | Interpretation |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---|")
    # Behaviors with template-driven probes (high redundancy expected)
    template_behaviors = {"bias", "sycophancy", "reasoning"}

    for beh in sorted(redundancy.keys()):
        r_data = redundancy[beh]
        r = r_data["redundancy"]
        iso = r_data["isolation_rate"]
        n_p = r_data["n_primary"]
        n_c = r_data["n_clustered"]
        n_s = r_data["n_singletons"]
        if beh in template_behaviors and r > 0.80:
            interp = "Template-driven — structural similarity expected"
        elif r > 0.80:
            interp = "High — clustered probes are too similar"
        elif r > 0.50:
            interp = "Moderate — some internal similarity"
        else:
            interp = "Good — diverse among clustered probes"
        lines.append(f"| {beh} | {n_p} | {n_c} | {n_s} | {iso:.0%} | {r:.2f} | {interp} |")

    # Coverage gaps
    lines.append("\n### Coverage Gaps\n")
    if gaps:
        lines.append(
            "The following behaviors have **no probes** in any cross-behavior "
            "cluster, meaning they occupy isolated semantic regions:\n"
        )
        for g in gaps:
            lines.append(f"- **{g}**")
        lines.append(
            "\nThis suggests these dimensions are semantically distinct from "
            "other behaviors (not necessarily bad — but worth investigating "
            "whether boundary cases are missing).\n"
        )
    else:
        lines.append(
            "All behaviors appear in at least one cross-behavior cluster — "
            "no isolated semantic islands detected.\n"
        )

    # Recommendations
    lines.append("### Recommendations\n")
    template_behaviors = {"bias", "sycophancy", "reasoning"}
    genuine_high = [b for b, r in redundancy.items()
                    if r["redundancy"] > 0.80 and b not in template_behaviors]
    template_high = [b for b, r in redundancy.items()
                     if r["redundancy"] > 0.80 and b in template_behaviors]
    high_isolation = [b for b, r in redundancy.items()
                      if r["isolation_rate"] > 0.70]
    rec_num = 1
    if genuine_high:
        lines.append(
            f"{rec_num}. **Diversify high-redundancy behaviors** "
            f"({', '.join(genuine_high)}): many probes test similar semantic "
            f"content. Add probes from underrepresented subcategories or edge cases."
        )
        rec_num += 1
    if template_high:
        lines.append(
            f"{rec_num}. **Template-driven behaviors** "
            f"({', '.join(template_high)}) show high structural similarity by "
            f"design (BBQ scenarios, persona prompts, math problems). "
            f"Their content varies — this is expected, not a problem."
        )
        rec_num += 1
    if high_isolation:
        lines.append(
            f"{rec_num}. **High isolation behaviors** "
            f"({', '.join(high_isolation)}) have >70% singletons — most probes "
            f"have no semantic neighbors at threshold. This is expected for "
            f"highly diverse probe sets (e.g., ToxiGen covers many independent topics)."
        )
        rec_num += 1
    n_cross = sum(1 for cs in cluster_stats if cs["is_cross_behavior"])
    lines.append(
        f"{rec_num}. **{n_cross} cross-behavior clusters found** — these are "
        f"the most valuable for detecting behavioral entanglement during "
        f"fine-tuning."
    )
    rec_num += 1
    if gaps:
        lines.append(
            f"{rec_num}. **Bridge the gap** for {', '.join(gaps)}: add probes "
            f"that straddle the boundary between these behaviors and related ones."
        )
    lines.append(
        f"\nFull data: `docs/probe_landscape.json` | "
        f"Figure: `docs/probe_landscape.png`\n"
    )

    new_section = "\n".join(lines)

    # Read existing file, replace or append
    if notes_path.exists():
        content = notes_path.read_text()
        # Check if section already exists
        pattern = r"\n---\n\n## Probe Landscape Analysis\n.*"
        if re.search(pattern, content, re.DOTALL):
            # Replace from the section header to the next top-level ## or EOF
            # Find the start
            match = re.search(r"\n---\n\n## Probe Landscape Analysis\n", content)
            if match:
                start = match.start()
                # Find the next --- or ## after our section
                rest = content[match.end():]
                next_section = re.search(r"\n---\n\n## ", rest)
                if next_section:
                    end = match.end() + next_section.start()
                else:
                    end = len(content)
                content = content[:start] + new_section + content[end:]
        else:
            content = content.rstrip() + "\n" + new_section
    else:
        content = "# Research Notes\n" + new_section

    notes_path.write_text(content)
    print(f"  Updated: {notes_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Probe landscape analysis via embedding + community detection"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Cosine similarity threshold for graph edges (default: 0.65)",
    )
    parser.add_argument(
        "--no-variants", action="store_true",
        help="Skip harmful_version/deceptive variant nodes",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "docs",
        help="Output directory (default: docs/)",
    )
    parser.add_argument(
        "--perplexity", type=float, default=30.0,
        help="t-SNE perplexity (default: 30)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Probe Landscape Analysis")
    print(f"  Threshold: {args.threshold}")
    print(f"  Variants:  {'excluded' if args.no_variants else 'included'}")
    print(f"  Output:    {args.out_dir}")
    print()

    # 1. Load probes
    print("[1/7] Loading probes...")
    nodes = load_all_probes(include_variants=not args.no_variants)

    # 2. Embed
    print("\n[2/7] Embedding probes...")
    embeddings = embed_probes(nodes)

    # 3. Build graph
    print("\n[3/7] Building similarity graph...")
    G, sensitivity = build_similarity_graph(nodes, embeddings, args.threshold)

    # 4. Community detection
    print("\n[4/7] Running Louvain community detection...")
    communities = run_louvain(G)

    # 5. Compute metrics
    print("\n[5/7] Computing metrics...")
    cluster_stats = compute_cluster_stats(nodes, communities, G)
    redundancy = compute_redundancy(nodes, communities)
    gaps = compute_coverage_gaps(nodes, communities)

    print(f"  Redundancy scores (excluding singletons):")
    print(f"    {'Behavior':15s} {'Probes':>6s} {'Clust':>6s} {'Single':>6s} {'Isol%':>6s} {'Redund':>7s}")
    for beh in sorted(redundancy.keys()):
        r = redundancy[beh]
        print(f"    {beh:15s} {r['n_primary']:>6d} {r['n_clustered']:>6d} "
              f"{r['n_singletons']:>6d} {r['isolation_rate']:>5.0%} {r['redundancy']:>7.3f}")
    if gaps:
        print(f"  Coverage gaps: {', '.join(gaps)}")
    else:
        print(f"  Coverage gaps: none")

    # 6. Save outputs
    print("\n[6/7] Saving outputs...")
    output = build_output_json(
        nodes, communities, cluster_stats, redundancy, gaps,
        sensitivity, args.threshold, embeddings,
    )
    json_path = args.out_dir / "probe_landscape.json"
    json_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"  Saved: {json_path}")

    # Research Notes
    notes_path = PROJECT_ROOT / "Research_Notes.md"
    append_research_notes(
        cluster_stats, redundancy, gaps, nodes, output["metadata"], notes_path
    )

    # 7. Plot
    print("\n[7/7] Generating t-SNE plot...")
    plot_landscape(
        nodes, embeddings, communities,
        args.out_dir / "probe_landscape.png",
        perplexity=args.perplexity,
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"Done! Summary:")
    print(f"  Nodes:          {len(nodes):,}")
    print(f"  Edges:          {G.number_of_edges():,}")
    print(f"  Communities:    {len(communities)}")
    n_cross = sum(1 for cs in cluster_stats if cs["is_cross_behavior"])
    print(f"  Cross-behavior: {n_cross}")
    print(f"\nOutputs:")
    print(f"  {json_path}")
    print(f"  {args.out_dir / 'probe_landscape.png'}")
    print(f"  {notes_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
