#!/usr/bin/env python3
"""Compare probe-space clustering with LLM internal-space clustering.

Tests whether the semantic isolation seen in text embeddings (all-MiniLM-L6-v2)
also appears in the model's own hidden representations.

Key question: is the insular structure of sycophancy/bias probes (HD ≈ 0.02)
a property of the text surface, or is it also reflected in how the model
*internally* represents these probes?

Steps:
  1. Load probes and compute text embeddings (CPU, ~30s)
  2. Load a small LLM (MLX) and extract hidden states per probe (~90s)
  3. Build similarity matrices in both spaces
  4. Compare:
     - Mantel test (correlation between similarity matrices)
     - Per-behavior highway density in internal space
     - Cluster overlap (ARI, NMI) between Louvain communities

Output:
    docs/internal_space_comparison.json
    stdout — comparison tables

Usage:
    python scripts/internal_space_comparison.py Qwen/Qwen2.5-0.5B-Instruct
    python scripts/internal_space_comparison.py Qwen/Qwen2.5-0.5B-Instruct --max-probes 500
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ═══════════════════════════════════════════════════════════════════════
# Internal representation extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_internal_representations(model, tokenizer, texts, max_length=128):
    """Forward each text through the LLM, extract mean-pooled hidden states.

    Returns (n_texts, hidden_dim) numpy array of L2-normalized representations.
    """
    import mlx.core as mx

    from rho_eval.alignment.mlx_losses import _mlx_extract_hidden_states

    model.eval()
    reps = []

    for i, text in enumerate(texts):
        h = _mlx_extract_hidden_states(model, tokenizer, text, max_length)
        mx.eval(h)
        reps.append(np.array(h.astype(mx.float32)))

        if (i + 1) % 100 == 0:
            print(f"    Extracted {i + 1}/{len(texts)} representations",
                  flush=True)

    reps = np.stack(reps)  # (n, hidden_dim)

    # L2-normalize for cosine similarity via dot product
    norms = np.linalg.norm(reps, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    reps = reps / norms

    return reps


# ═══════════════════════════════════════════════════════════════════════
# Highway density (shared logic with highway_density.py)
# ═══════════════════════════════════════════════════════════════════════

def compute_hd_from_sim_matrix(sim_matrix, behaviors, threshold=0.65):
    """Compute per-behavior highway density from a similarity matrix.

    Args:
        sim_matrix: (n, n) cosine similarity matrix.
        behaviors: list of behavior strings, length n.
        threshold: edge threshold.

    Returns:
        dict[behavior] -> highway_density (float)
    """
    import networkx as nx

    n = len(behaviors)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    rows, cols = np.where(np.triu(sim_matrix, k=1) > threshold)
    for r, c in zip(rows, cols):
        G.add_edge(int(r), int(c))

    unique_behaviors = sorted(set(behaviors))
    density = {}

    for beh in unique_behaviors:
        beh_nodes = {i for i, b in enumerate(behaviors) if b == beh}
        total = 0
        cross = 0
        for node_idx in beh_nodes:
            for neighbor in G.neighbors(node_idx):
                total += 1
                if behaviors[neighbor] != beh:
                    cross += 1
        density[beh] = round(cross / total, 4) if total > 0 else 0.0

    return density, G


# ═══════════════════════════════════════════════════════════════════════
# Mantel test
# ═══════════════════════════════════════════════════════════════════════

def mantel_test(sim_a, sim_b, n_permutations=999, seed=42):
    """Mantel test: Pearson correlation between two similarity matrices,
    with permutation p-value.

    Returns (rho, p_value).
    """
    rng = np.random.RandomState(seed)
    n = sim_a.shape[0]
    mask = np.triu_indices(n, k=1)

    a_flat = sim_a[mask]
    b_flat = sim_b[mask]

    # Observed correlation
    rho_obs = float(np.corrcoef(a_flat, b_flat)[0, 1])

    # Permutation test
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        a_perm = sim_a[np.ix_(perm, perm)]
        rho_perm = float(np.corrcoef(a_perm[mask], b_flat)[0, 1])
        if rho_perm >= rho_obs:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return rho_obs, p_value


# ═══════════════════════════════════════════════════════════════════════
# Cluster comparison
# ═══════════════════════════════════════════════════════════════════════

def louvain_labels(G, n_nodes):
    """Run Louvain, return per-node cluster labels (list of ints)."""
    import networkx as nx

    if G.number_of_edges() == 0:
        return list(range(n_nodes))

    communities = nx.community.louvain_communities(G, seed=42, resolution=1.0)
    labels = [0] * n_nodes
    for cid, members in enumerate(communities):
        for idx in members:
            labels[idx] = cid
    return labels


def cluster_comparison(labels_a, labels_b):
    """Compute ARI and NMI between two clusterings."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ari = adjusted_rand_score(labels_a, labels_b)
    nmi = normalized_mutual_info_score(labels_a, labels_b)
    return {"ari": round(ari, 4), "nmi": round(nmi, 4)}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare probe-space vs internal-space clustering",
    )
    parser.add_argument("model", help="HuggingFace model ID for internal reps")
    parser.add_argument(
        "--max-probes", type=int, default=0,
        help="Max probes to use (0 = all, default: all)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Cosine similarity threshold for graph edges (default: 0.65)",
    )
    parser.add_argument(
        "--permutations", type=int, default=999,
        help="Number of Mantel test permutations (default: 999)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Probe-Space vs Internal-Space Comparison")
    print(f"  Model:       {args.model}")
    print(f"  Threshold:   {args.threshold}")
    print(f"  Permutations: {args.permutations}")
    print("=" * 60)

    t0 = time.time()

    # ── 1. Load probes + text embeddings ─────────────────────────────
    from probe_landscape import load_all_probes, embed_probes

    print("\n[1/6] Loading probes (primary only)...")
    nodes = load_all_probes(include_variants=False)

    # Optionally subsample (stratified by behavior)
    if args.max_probes > 0 and args.max_probes < len(nodes):
        rng = np.random.RandomState(42)
        behaviors = sorted(set(n.behavior for n in nodes))
        per_beh = max(1, args.max_probes // len(behaviors))
        sampled = []
        for beh in behaviors:
            beh_nodes = [n for n in nodes if n.behavior == beh]
            k = min(per_beh, len(beh_nodes))
            idxs = rng.choice(len(beh_nodes), size=k, replace=False)
            sampled.extend(beh_nodes[i] for i in idxs)
        nodes = sampled
        print(f"  Subsampled to {len(nodes)} probes "
              f"(~{per_beh} per behavior)")

    texts = [n.text for n in nodes]
    behaviors = [n.behavior for n in nodes]
    n = len(nodes)

    print(f"\n[2/6] Computing text embeddings ({n} probes)...")
    text_embeddings = embed_probes(nodes)  # (n, 384), L2-normalized
    text_sim = text_embeddings @ text_embeddings.T

    # ── 2. Extract internal representations ──────────────────────────
    print(f"\n[3/6] Loading model: {args.model}")
    import mlx_lm

    model, tokenizer = mlx_lm.load(args.model)
    print(f"  Model loaded.", flush=True)

    print(f"\n[4/6] Extracting internal representations ({n} probes)...")
    t_extract = time.time()
    internal_reps = extract_internal_representations(
        model, tokenizer, texts, max_length=128,
    )
    extract_time = time.time() - t_extract
    print(f"  Done in {extract_time:.1f}s "
          f"({extract_time / n * 1000:.0f}ms/probe, "
          f"hidden_dim={internal_reps.shape[1]})")

    internal_sim = internal_reps @ internal_reps.T

    # Free model memory
    del model
    import gc
    gc.collect()

    # ── 3. Mantel test ───────────────────────────────────────────────
    print(f"\n[5/6] Mantel test ({args.permutations} permutations)...")
    t_mantel = time.time()
    mantel_rho, mantel_p = mantel_test(
        text_sim, internal_sim,
        n_permutations=args.permutations,
    )
    mantel_time = time.time() - t_mantel
    print(f"  Pearson ρ = {mantel_rho:.4f}, p = {mantel_p:.4f} "
          f"({mantel_time:.1f}s)")

    # Also compute Spearman on flattened upper triangles (no permutation)
    mask = np.triu_indices(n, k=1)
    spearman_rho, spearman_p = scipy_stats.spearmanr(
        text_sim[mask], internal_sim[mask],
    )
    print(f"  Spearman ρ = {spearman_rho:.4f} (p = {spearman_p:.2e})")

    # ── 4. Per-behavior highway density comparison ───────────────────
    print(f"\n[6/7] Computing per-behavior highway density...")

    text_hd, text_G = compute_hd_from_sim_matrix(
        text_sim, behaviors, threshold=args.threshold,
    )

    # RAW internal HD at same threshold (for reference)
    raw_internal_hd, raw_internal_G = compute_hd_from_sim_matrix(
        internal_sim, behaviors, threshold=args.threshold,
    )

    text_edges = text_G.number_of_edges()
    raw_internal_edges = raw_internal_G.number_of_edges()
    max_possible = n * (n - 1) // 2
    raw_density = raw_internal_edges / max_possible if max_possible > 0 else 0

    print(f"\n  Text graph: {text_edges} edges")
    print(f"  Internal graph (raw, threshold={args.threshold}): "
          f"{raw_internal_edges} edges "
          f"({raw_density:.1%} of {max_possible} possible)")

    if raw_density > 0.5:
        print(f"\n  ⚠ Internal graph density is {raw_density:.1%} — "
              f"threshold {args.threshold} is too low for internal space.")
        print(f"  Computing matched-density comparison...")

    # ── 4b. Matched-density HD comparison ──────────────────────────
    # The internal similarity distribution is much tighter and higher
    # than text similarities. Using the same threshold produces a
    # near-complete graph in internal space, making HD meaningless.
    # Fix: binary-search for the internal threshold that gives the
    # same edge count as the text graph.

    internal_upper = internal_sim[np.triu_indices(n, k=1)]
    text_upper = text_sim[np.triu_indices(n, k=1)]

    print(f"\n  Similarity distributions:")
    print(f"    Text:     mean={np.mean(text_upper):.3f}, "
          f"std={np.std(text_upper):.3f}, "
          f"median={np.median(text_upper):.3f}")
    print(f"    Internal: mean={np.mean(internal_upper):.3f}, "
          f"std={np.std(internal_upper):.3f}, "
          f"median={np.median(internal_upper):.3f}")

    # Binary search for matched threshold
    target_edges = text_edges
    lo, hi = float(np.min(internal_upper)), float(np.max(internal_upper))

    for _ in range(50):  # 50 iterations = ~15 decimal places
        mid = (lo + hi) / 2
        edge_count = int(np.sum(internal_upper > mid))
        if edge_count > target_edges:
            lo = mid
        else:
            hi = mid

    matched_threshold = (lo + hi) / 2
    matched_edges = int(np.sum(internal_upper > matched_threshold))
    print(f"\n  Matched-density threshold: {matched_threshold:.4f} "
          f"(gives {matched_edges} edges, target {target_edges})")

    internal_hd, internal_G = compute_hd_from_sim_matrix(
        internal_sim, behaviors, threshold=matched_threshold,
    )

    print(f"\n  {'Behavior':15s} {'Text HD':>8s} {'Internal HD':>12s} "
          f"{'Δ':>8s} {'Direction':>10s}")
    print(f"  {'-' * 55}")
    all_behaviors = sorted(set(behaviors))
    for beh in sorted(all_behaviors, key=lambda b: -text_hd.get(b, 0)):
        t_hd = text_hd.get(beh, 0)
        i_hd = internal_hd.get(beh, 0)
        delta = i_hd - t_hd
        direction = ("more insular" if delta < -0.05
                     else "more highway" if delta > 0.05
                     else "similar")
        print(f"  {beh:15s} {t_hd:>8.3f} {i_hd:>12.3f} "
              f"{delta:>+8.3f} {direction:>10s}")

    # Spearman between text HD and internal HD vectors
    hd_behaviors = sorted(set(text_hd.keys()) & set(internal_hd.keys()))
    if len(hd_behaviors) >= 4:
        text_hd_vec = [text_hd[b] for b in hd_behaviors]
        int_hd_vec = [internal_hd[b] for b in hd_behaviors]
        hd_rho, hd_p = scipy_stats.spearmanr(text_hd_vec, int_hd_vec)
        print(f"\n  HD rank correlation: ρ = {hd_rho:.3f}, p = {hd_p:.4f}")
    else:
        hd_rho, hd_p = 0.0, 1.0

    # ── 5. Cluster overlap ───────────────────────────────────────────
    print(f"\n[7/7] Computing cluster overlap...")
    text_labels = louvain_labels(text_G, n)
    internal_labels = louvain_labels(internal_G, n)
    overlap = cluster_comparison(text_labels, internal_labels)

    print(f"\n  Cluster overlap:")
    print(f"    ARI = {overlap['ari']:.4f} "
          f"({'strong' if overlap['ari'] > 0.5 else 'weak' if overlap['ari'] > 0.1 else 'negligible'})")
    print(f"    NMI = {overlap['nmi']:.4f} "
          f"({'strong' if overlap['nmi'] > 0.5 else 'moderate' if overlap['nmi'] > 0.2 else 'weak'})")
    print(f"    Text communities: {len(set(text_labels))}")
    print(f"    Internal communities: {len(set(internal_labels))}")

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Summary")
    print(f"{'=' * 60}")
    print(f"  Mantel (Pearson):    ρ = {mantel_rho:.4f}, p = {mantel_p:.4f}")
    print(f"  Mantel (Spearman):   ρ = {spearman_rho:.4f}")
    print(f"  HD rank correlation: ρ = {hd_rho:.3f}, p = {hd_p:.4f}")
    print(f"  Cluster ARI:         {overlap['ari']:.4f}")
    print(f"  Cluster NMI:         {overlap['nmi']:.4f}")

    # Check if insular behaviors are also insular internally
    insular_behaviors = [b for b in all_behaviors if text_hd.get(b, 0) < 0.1]
    insular_internal = [b for b in insular_behaviors if internal_hd.get(b, 0) < 0.1]
    isolation_reinforced = len(insular_internal) == len(insular_behaviors) > 0

    if mantel_rho > 0.5:
        interpretation = (
            "HIGH agreement — text-embedding isolation patterns are "
            "strongly reflected in model internals. Bridge strengthening "
            "addresses a real structural property of the model."
        )
    elif mantel_rho > 0.2:
        interpretation = (
            "MODERATE agreement — some text-embedding structure is "
            "reflected internally, but the model reorganizes representations "
            "substantially."
        )
    else:
        interpretation = (
            "LOW agreement — the model's internal representations diverge "
            "strongly from text-surface similarity. Bridge strengthening "
            "based on text embeddings may not target the right structure."
        )

    if isolation_reinforced:
        interpretation += (
            f" Critically, insular behaviors ({', '.join(insular_behaviors)}) "
            f"are equally or MORE insular in model internals at matched density, "
            f"confirming the isolation is a genuine property of the model's "
            f"learned representations, not just a text-surface artifact."
        )
    print(f"\n  Interpretation: {interpretation}")
    print(f"\n  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 60}")

    # ── Save ─────────────────────────────────────────────────────────
    results = {
        "model": args.model,
        "n_probes": n,
        "text_threshold": args.threshold,
        "matched_internal_threshold": round(matched_threshold, 4),
        "n_permutations": args.permutations,
        "mantel": {
            "pearson_rho": round(mantel_rho, 4),
            "pearson_p": round(mantel_p, 4),
            "spearman_rho": round(float(spearman_rho), 4),
            "spearman_p": float(spearman_p),
        },
        "highway_density": {
            "text": text_hd,
            "internal_matched": internal_hd,
            "internal_raw": raw_internal_hd,
            "rank_correlation_rho": round(float(hd_rho), 3),
            "rank_correlation_p": round(float(hd_p), 4),
        },
        "similarity_distributions": {
            "text_mean": round(float(np.mean(text_upper)), 4),
            "text_std": round(float(np.std(text_upper)), 4),
            "internal_mean": round(float(np.mean(internal_upper)), 4),
            "internal_std": round(float(np.std(internal_upper)), 4),
        },
        "cluster_overlap": overlap,
        "graph_stats": {
            "text_edges": text_edges,
            "internal_edges_raw": raw_internal_edges,
            "internal_edges_matched": internal_G.number_of_edges(),
            "text_communities": len(set(text_labels)),
            "internal_communities": len(set(internal_labels)),
        },
        "interpretation": interpretation,
        "extract_time_sec": round(extract_time, 1),
        "total_time_sec": round(total_time, 1),
    }

    out_path = PROJECT_ROOT / "docs" / "internal_space_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
