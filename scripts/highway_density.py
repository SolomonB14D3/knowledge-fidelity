#!/usr/bin/env python3
"""Highway Density: cross-behavior edge fraction per behavior.

Measures what fraction of each behavior's graph edges connect to probes
of other behaviors.  High highway density → behavior is well-connected
to other dimensions (a "highway").  Low → insular.

Correlates highway density with redundancy, isolation, alignment
variance, and mean alignment delta from ``results/master.db``.

Output:
    docs/highway_density.json  — per-behavior metrics + correlations
    stdout                     — formatted tables

Usage:
    python scripts/highway_density.py
    python scripts/highway_density.py --threshold 0.70

Requires: sentence-transformers, networkx, scipy, numpy
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from probe_landscape import load_all_probes, embed_probes, build_similarity_graph


# ═══════════════════════════════════════════════════════════════════════
# Highway density computation
# ═══════════════════════════════════════════════════════════════════════

def compute_highway_density(nodes, G):
    """Per-behavior highway density: cross_edges / total_incident_edges.

    For behavior B:
      - total = sum of degrees of all nodes with behavior B
      - cross = edges touching B where the other endpoint is ≠ B
      - highway_density = cross / total

    A value of 0 means all connectivity is intra-behavior (insular).
    A value of 1 means all connectivity crosses to other behaviors.

    Returns:
        dict[behavior] → {highway_density, total_incident_edges,
                          cross_edges, intra_edges, n_nodes}
    """
    behaviors = sorted(set(n.behavior for n in nodes))
    density = {}

    for beh in behaviors:
        beh_nodes = {i for i, n in enumerate(nodes) if n.behavior == beh}
        total_incident = 0
        cross = 0

        for node_idx in beh_nodes:
            for neighbor in G.neighbors(node_idx):
                total_incident += 1
                if nodes[neighbor].behavior != beh:
                    cross += 1

        density[beh] = {
            "highway_density": round(cross / total_incident, 4) if total_incident > 0 else 0.0,
            "total_incident_edges": total_incident,
            "cross_edges": cross,
            "intra_edges": total_incident - cross,
            "n_nodes": len(beh_nodes),
        }

    return density


# ═══════════════════════════════════════════════════════════════════════
# Alignment stats from master.db
# ═══════════════════════════════════════════════════════════════════════

def load_alignment_stats(db_path):
    """Per-behavior variance and mean delta from alignment_runs table."""
    if not db_path.exists():
        print(f"  Warning: {db_path} not found, skipping alignment correlations")
        return {}

    conn = sqlite3.connect(str(db_path))
    behaviors = [
        "factual", "toxicity", "bias", "sycophancy",
        "reasoning", "refusal", "deception", "overrefusal",
    ]

    result = {}
    for beh in behaviors:
        rows = conn.execute(
            f"SELECT d_{beh} FROM alignment_runs WHERE d_{beh} IS NOT NULL"
        ).fetchall()
        if rows:
            deltas = [r[0] for r in rows]
            result[beh] = {
                "mean_delta": float(np.mean(deltas)),
                "variance": float(np.var(deltas, ddof=1)),
                "std": float(np.std(deltas, ddof=1)),
                "n": len(deltas),
            }

    conn.close()
    return result


# ═══════════════════════════════════════════════════════════════════════
# Correlations
# ═══════════════════════════════════════════════════════════════════════

def compute_correlations(density, landscape, alignment_stats):
    """Spearman correlations between highway density and other metrics."""
    landscape_behaviors = landscape.get("behaviors", {})
    behaviors = sorted(
        set(density.keys()) & set(landscape_behaviors.keys())
    )

    if len(behaviors) < 4:
        print("  Warning: fewer than 4 behaviors in common, skipping correlations")
        return {}

    hd = [density[b]["highway_density"] for b in behaviors]
    redundancy = [landscape_behaviors[b].get("redundancy", 0) for b in behaviors]
    isolation = [landscape_behaviors[b].get("isolation_rate", 0) for b in behaviors]

    results = {}

    # Highway density vs redundancy
    rho, p = stats.spearmanr(hd, redundancy)
    results["hd_vs_redundancy"] = {
        "rho": round(rho, 3), "p": round(p, 4), "n": len(behaviors),
    }

    # Highway density vs isolation
    rho, p = stats.spearmanr(hd, isolation)
    results["hd_vs_isolation"] = {
        "rho": round(rho, 3), "p": round(p, 4), "n": len(behaviors),
    }

    # Redundancy vs isolation
    rho, p = stats.spearmanr(redundancy, isolation)
    results["redundancy_vs_isolation"] = {
        "rho": round(rho, 3), "p": round(p, 4), "n": len(behaviors),
    }

    # Alignment-based correlations
    if alignment_stats:
        common = sorted(set(behaviors) & set(alignment_stats.keys()))
        if len(common) >= 4:
            hd_sub = [density[b]["highway_density"] for b in common]
            variance = [alignment_stats[b]["variance"] for b in common]
            mean_delta = [alignment_stats[b]["mean_delta"] for b in common]

            rho_v, p_v = stats.spearmanr(hd_sub, variance)
            results["hd_vs_variance"] = {
                "rho": round(rho_v, 3), "p": round(p_v, 4), "n": len(common),
            }

            rho_m, p_m = stats.spearmanr(hd_sub, mean_delta)
            results["hd_vs_mean_delta"] = {
                "rho": round(rho_m, 3), "p": round(p_m, 4), "n": len(common),
            }

            # Redundancy vs variance
            red_sub = [landscape_behaviors[b].get("redundancy", 0) for b in common]
            rho_rv, p_rv = stats.spearmanr(red_sub, variance)
            results["redundancy_vs_variance"] = {
                "rho": round(rho_rv, 3), "p": round(p_rv, 4), "n": len(common),
            }

            # Redundancy vs mean_delta
            rho_rm, p_rm = stats.spearmanr(red_sub, mean_delta)
            results["redundancy_vs_mean_delta"] = {
                "rho": round(rho_rm, 3), "p": round(p_rm, 4), "n": len(common),
            }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Highway density analysis")
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Cosine similarity threshold for graph edges (default: 0.65)",
    )
    args = parser.parse_args()

    print("Highway Density Analysis")
    print("=" * 60)

    # ── 1. Load probes + embed + build graph ──────────────────────────
    print("\n[1/5] Loading probes...")
    nodes = load_all_probes(include_variants=True)

    print("\n[2/5] Embedding probes...")
    embeddings = embed_probes(nodes)

    print("\n[3/5] Building similarity graph...")
    G, _ = build_similarity_graph(nodes, embeddings, threshold=args.threshold)

    # ── 2. Compute highway density ────────────────────────────────────
    print("\n[4/5] Computing highway density...")
    density = compute_highway_density(nodes, G)

    print(f"\n  {'Behavior':15s} {'HD':>7s} {'Cross':>7s} {'Intra':>7s} "
          f"{'Total':>7s} {'Nodes':>6s}")
    print(f"  {'-' * 52}")
    for beh in sorted(density.keys(), key=lambda b: -density[b]["highway_density"]):
        d = density[beh]
        print(f"  {beh:15s} {d['highway_density']:>7.3f} "
              f"{d['cross_edges']:>7d} {d['intra_edges']:>7d} "
              f"{d['total_incident_edges']:>7d} {d['n_nodes']:>6d}")

    # ── 3. Load landscape JSON for redundancy ─────────────────────────
    landscape_path = PROJECT_ROOT / "docs" / "probe_landscape.json"
    landscape = {}
    if landscape_path.exists():
        landscape = json.loads(landscape_path.read_text())
        print(f"\n  Loaded probe landscape from {landscape_path}")

    # ── 4. Load alignment stats from master.db ────────────────────────
    db_path = PROJECT_ROOT / "results" / "master.db"
    alignment_stats = load_alignment_stats(db_path)
    if alignment_stats:
        print(f"  Loaded alignment stats for {len(alignment_stats)} behaviors")
        print(f"\n  {'Behavior':15s} {'Mean Δ':>8s} {'Var':>10s} {'N':>5s}")
        print(f"  {'-' * 40}")
        for beh in sorted(alignment_stats.keys()):
            s = alignment_stats[beh]
            print(f"  {beh:15s} {s['mean_delta']:>+8.4f} "
                  f"{s['variance']:>10.6f} {s['n']:>5d}")

    # ── 5. Correlations ───────────────────────────────────────────────
    print(f"\n[5/5] Computing correlations...")
    correlations = compute_correlations(density, landscape, alignment_stats)

    print(f"\n{'=' * 60}")
    print("Spearman Correlations")
    print(f"{'=' * 60}")
    print(f"  {'Comparison':35s} {'ρ':>7s} {'p':>8s} {'n':>4s} {'Sig':>5s}")
    print(f"  {'-' * 57}")
    for name, c in sorted(correlations.items()):
        sig = ("***" if c["p"] < 0.001 else
               "**" if c["p"] < 0.01 else
               "*" if c["p"] < 0.05 else "ns")
        print(f"  {name:35s} {c['rho']:>+7.3f} "
              f"{c['p']:>8.4f} {c['n']:>4d} {sig:>5s}")

    # ── 6. Save ───────────────────────────────────────────────────────
    output = {
        "threshold": args.threshold,
        "density": density,
        "correlations": correlations,
        "alignment_stats": {
            k: {kk: round(vv, 6) for kk, vv in v.items()}
            for k, v in alignment_stats.items()
        },
    }
    out_path = PROJECT_ROOT / "docs" / "highway_density.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
