#!/usr/bin/env python3
"""Bridge Feasibility Test — Can LoRA training increase cross-behavior
hidden-state similarity?

Quick statistical test on a small model (0.5B recommended):
  1. Find weak bridge pairs (cross-behavior, text-embedding sim 0.55–0.70)
  2. Measure baseline LLM hidden-state cosine similarity on those pairs
  3. Train tiny LoRA (rank 2, 50 steps) with bridge cosine loss
  4. Re-measure hidden-state cosine similarity
  5. Wilcoxon signed-rank test: did similarities increase?
  6. Quick behavioral audit before/after to check collateral damage

Success criteria:
  - p < 0.05 on Wilcoxon (sims increased)
  - No behavior drops > 0.05 rho

Usage:
    python experiments/bridge_feasibility_mlx.py Qwen/Qwen2.5-0.5B-Instruct
    python experiments/bridge_feasibility_mlx.py Qwen/Qwen2.5-0.5B-Instruct --steps 100
"""

import argparse
import gc
import json
import random
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
# Phase 1: Find bridge pairs (CPU only — sentence-transformers)
# ═══════════════════════════════════════════════════════════════════════

def find_bridge_pairs(sim_range=(0.55, 0.70)):
    """Find cross-behavior probe pairs with intermediate text-embedding similarity.

    Re-embeds primary probes with all-MiniLM-L6-v2 and finds pairs where:
    - Endpoints are from different behaviors
    - Cosine similarity is within sim_range

    Returns:
        List of dicts with text_a, text_b, behavior_a, behavior_b, text_sim.
    """
    from probe_landscape import load_all_probes, embed_probes

    print("[1] Loading probes (primary only)...")
    nodes = load_all_probes(include_variants=False)

    print("[2] Embedding probes...")
    embeddings = embed_probes(nodes)

    print("[3] Finding bridge pairs...")
    sim_matrix = embeddings @ embeddings.T

    # Vectorized search: cross-behavior pairs in sim_range
    behaviors = np.array([node.behavior for node in nodes])
    upper = np.triu(sim_matrix, k=1)
    cross_mask = behaviors[:, None] != behaviors[None, :]
    lo, hi = sim_range
    mask = (upper >= lo) & (upper <= hi) & cross_mask
    rows, cols = np.where(mask)

    pairs = []
    for r, c in zip(rows, cols):
        pairs.append({
            "text_a": nodes[r].text,
            "text_b": nodes[c].text,
            "behavior_a": nodes[r].behavior,
            "behavior_b": nodes[c].behavior,
            "probe_a": nodes[r].probe_id,
            "probe_b": nodes[c].probe_id,
            "text_sim": round(float(sim_matrix[r, c]), 4),
        })

    # Report distribution
    pair_types = Counter(
        tuple(sorted([p["behavior_a"], p["behavior_b"]])) for p in pairs
    )
    print(f"  Found {len(pairs)} cross-behavior pairs in [{lo}, {hi}]")
    print(f"  Top pair types:")
    for (ba, bb), count in pair_types.most_common(10):
        print(f"    {ba} <-> {bb}: {count}")

    return pairs


# ═══════════════════════════════════════════════════════════════════════
# Phase 3/5: Measure hidden-state similarities (MLX forward pass)
# ═══════════════════════════════════════════════════════════════════════

def measure_hidden_sims(model, tokenizer, pairs, max_pairs=200):
    """Measure LLM hidden-state cosine similarity for each pair.

    Uses numpy after mx.eval() for measurement (not differentiable).
    Calls model.eval() to disable dropout.
    """
    import mlx.core as mx
    from rho_eval.alignment.mlx_losses import _mlx_extract_hidden_states

    model.eval()
    sims = []
    use_pairs = pairs[:max_pairs]

    for i, pair in enumerate(use_pairs):
        h_a = _mlx_extract_hidden_states(
            model, tokenizer, pair["text_a"], max_length=128,
        )
        h_b = _mlx_extract_hidden_states(
            model, tokenizer, pair["text_b"], max_length=128,
        )
        mx.eval(h_a, h_b)

        # Cosine similarity in numpy (measurement, not training)
        # bfloat16 → float32 required for numpy conversion
        a = np.array(h_a.astype(mx.float32))
        b = np.array(h_b.astype(mx.float32))
        cos_sim = float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        )
        sims.append(cos_sim)

        if (i + 1) % 50 == 0:
            print(f"    Measured {i + 1}/{len(use_pairs)} pairs, "
                  f"mean sim = {np.mean(sims):.4f}", flush=True)

    return np.array(sims)


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Train bridge LoRA
# ═══════════════════════════════════════════════════════════════════════

def train_bridge_lora(
    model,
    tokenizer,
    bridge_pairs,
    *,
    steps=50,
    lr=2e-4,
    rank=2,
    bridge_weight=0.1,
    pairs_per_step=4,
    max_length=128,
):
    """Train tiny LoRA with CE + bridge cosine loss.

    Uses the same LoRA infrastructure as mlx_rho_guided_sft but with
    a simplified training loop (no gradient accumulation, no warmup).
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as opt

    from rho_eval.alignment.mlx_losses import mlx_bridge_cosine_loss, mlx_ce_loss
    from rho_eval.alignment.mlx_trainer import _apply_lora

    # ── LoRA setup ────────────────────────────────────────────────────
    trainable_params, total_params = _apply_lora(model, rank, rank * 2)
    print(f"  LoRA rank={rank}: {trainable_params:,} trainable / "
          f"{total_params:,} total ({trainable_params / total_params:.4%})")

    # CE texts: probe texts for language model grounding
    ce_texts = (
        [p["text_a"] for p in bridge_pairs[:200]]
        + [p["text_b"] for p in bridge_pairs[:200]]
    )

    # Simple Adam (no warmup needed for 50 steps)
    optimizer = opt.Adam(learning_rate=lr)

    model.train()
    rng = random.Random(42)

    running_ce = 0.0
    running_bridge = 0.0

    print(f"  Training: {steps} steps, lr={lr}, bridge_weight={bridge_weight}")

    for step in range(steps):
        # Sample data for this step
        ce_text = rng.choice(ce_texts)
        step_pairs = rng.sample(
            bridge_pairs, min(pairs_per_step, len(bridge_pairs)),
        )

        # Combined loss: CE + bridge_weight * bridge_cosine
        def loss_fn(model):
            ce = mlx_ce_loss(model, tokenizer, ce_text, max_length=max_length)
            bridge = mlx_bridge_cosine_loss(
                model, tokenizer, step_pairs, max_length=max_length,
            )
            total = ce + bridge_weight * bridge
            return total, (ce, bridge)

        grad_fn = nn.value_and_grad(model, loss_fn)
        (total_val, (ce_val, bridge_val)), grads = grad_fn(model)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        ce_f = float(ce_val.item())
        bridge_f = float(bridge_val.item())
        running_ce = 0.9 * running_ce + 0.1 * ce_f if step > 0 else ce_f
        running_bridge = (
            0.9 * running_bridge + 0.1 * bridge_f if step > 0 else bridge_f
        )

        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"    Step {step + 1:>3d}/{steps}: "
                f"CE={running_ce:.4f}  Bridge={running_bridge:.4f}",
                flush=True,
            )

    return model


# ═══════════════════════════════════════════════════════════════════════
# Phase 6/7: Behavioral audit
# ═══════════════════════════════════════════════════════════════════════

def run_audit(model, tokenizer, n_probes=50):
    """Quick 8-behavior audit using rho_eval."""
    from rho_eval import audit

    report = audit(
        model=model,
        tokenizer=tokenizer,
        behaviors="all",
        n=n_probes,
        seed=42,
    )

    scores = {}
    for beh_name, result in report.behaviors.items():
        scores[beh_name] = result.rho
    return scores


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Bridge LoRA feasibility test",
    )
    parser.add_argument("model", help="HuggingFace model ID")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bridge-weight", type=float, default=0.1)
    parser.add_argument("--pairs-per-step", type=int, default=4)
    parser.add_argument(
        "--max-pairs", type=int, default=200,
        help="Max bridge pairs for measurement (default: 200)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/bridge_feasibility",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Bridge Strengthening Feasibility Test")
    print(f"  Model:         {args.model}")
    print(f"  LoRA rank:     {args.rank}")
    print(f"  Steps:         {args.steps}")
    print(f"  LR:            {args.lr}")
    print(f"  Bridge weight: {args.bridge_weight}")
    print(f"  Pairs/step:    {args.pairs_per_step}")
    print("=" * 60)

    t0 = time.time()

    # ── Phase 1: Find bridge pairs (CPU) ─────────────────────────────
    all_pairs = find_bridge_pairs(sim_range=(0.55, 0.70))

    if len(all_pairs) < 20:
        print(f"\nERROR: Only {len(all_pairs)} bridge pairs found. "
              f"Need >= 20. Aborting.")
        return

    rng = np.random.RandomState(42)
    rng.shuffle(all_pairs)
    measure_pairs = all_pairs[:args.max_pairs]
    train_pairs = all_pairs[:500]

    print(f"\n  Using {len(measure_pairs)} pairs for measurement, "
          f"{len(train_pairs)} for training pool")

    # Clean up sentence-transformers before loading MLX model
    gc.collect()

    # ── Phase 2: Load MLX model ──────────────────────────────────────
    print(f"\n[4] Loading model: {args.model}")
    import mlx_lm

    model, tokenizer = mlx_lm.load(args.model)
    print(f"  Model loaded.", flush=True)

    # ── Phase 3: Baseline measurements ───────────────────────────────
    print(f"\n[5] Measuring baseline hidden-state similarities...")
    sims_before = measure_hidden_sims(
        model, tokenizer, measure_pairs, args.max_pairs,
    )
    print(f"  Baseline: mean={sims_before.mean():.4f}, "
          f"std={sims_before.std():.4f}, "
          f"median={np.median(sims_before):.4f}")

    print(f"\n[6] Baseline behavioral audit...")
    audit_before = run_audit(model, tokenizer)
    print(
        f"  Scores: "
        f"{', '.join(f'{k}={v:.3f}' for k, v in sorted(audit_before.items()))}"
    )

    # ── Phase 3b: Baseline internal HD ────────────────────────────────
    print(f"\n[6b] Extracting baseline internal representations for HD...")
    from internal_space_comparison import (
        extract_internal_representations,
        compute_hd_from_sim_matrix,
    )
    from probe_landscape import load_all_probes

    hd_nodes = load_all_probes(include_variants=False)
    hd_behaviors = [n.behavior for n in hd_nodes]
    hd_texts = [n.text for n in hd_nodes]
    n_hd = len(hd_nodes)

    # Baseline internal reps + HD
    t_hd = time.time()
    reps_before = extract_internal_representations(
        model, tokenizer, hd_texts, max_length=128,
    )
    print(f"  Extracted {n_hd} reps in {time.time() - t_hd:.1f}s")

    internal_sim_before = reps_before @ reps_before.T

    # Text embedding sim (from probe_landscape) for matched-density ref
    from probe_landscape import embed_probes
    text_embs = embed_probes(hd_nodes)
    text_sim_matrix = text_embs @ text_embs.T
    text_hd, text_G = compute_hd_from_sim_matrix(
        text_sim_matrix, hd_behaviors, threshold=0.65,
    )
    text_edge_count = text_G.number_of_edges()

    # Binary search for matched internal threshold
    int_upper = internal_sim_before[np.triu_indices(n_hd, k=1)]
    lo, hi = float(np.min(int_upper)), float(np.max(int_upper))
    for _ in range(50):
        mid = (lo + hi) / 2
        if int(np.sum(int_upper > mid)) > text_edge_count:
            lo = mid
        else:
            hi = mid
    matched_threshold = (lo + hi) / 2

    hd_before, _ = compute_hd_from_sim_matrix(
        internal_sim_before, hd_behaviors, threshold=matched_threshold,
    )
    print(f"  Matched threshold: {matched_threshold:.4f} "
          f"(target {text_edge_count} edges)")
    print(f"  Baseline internal HD (matched density):")
    for beh in sorted(hd_before.keys(), key=lambda b: hd_before[b]):
        print(f"    {beh:15s}: {hd_before[beh]:.4f} "
              f"(text: {text_hd.get(beh, 0):.4f})")

    del reps_before, internal_sim_before, int_upper  # free memory
    gc.collect()

    # ── Phase 4: Train ───────────────────────────────────────────────
    print(f"\n[7] Training bridge LoRA...")
    t_train = time.time()
    model = train_bridge_lora(
        model,
        tokenizer,
        train_pairs,
        steps=args.steps,
        lr=args.lr,
        rank=args.rank,
        bridge_weight=args.bridge_weight,
        pairs_per_step=args.pairs_per_step,
        max_length=128,
    )
    train_time = time.time() - t_train
    print(f"  Training completed in {train_time:.1f}s")

    # ── Phase 5: Post-training measurements ──────────────────────────
    print(f"\n[8] Measuring post-training hidden-state similarities...")
    sims_after = measure_hidden_sims(
        model, tokenizer, measure_pairs, args.max_pairs,
    )
    print(f"  Post-training: mean={sims_after.mean():.4f}, "
          f"std={sims_after.std():.4f}, "
          f"median={np.median(sims_after):.4f}")

    # ── Phase 6: Statistical test ────────────────────────────────────
    print(f"\n[9] Statistical test...")
    delta = sims_after - sims_before
    n_increased = int(np.sum(delta > 0))
    n_decreased = int(np.sum(delta < 0))
    n_unchanged = int(np.sum(delta == 0))

    # Wilcoxon signed-rank (one-sided: did sims increase?)
    nonzero = delta[delta != 0]
    if len(nonzero) >= 10:
        stat, p_value = scipy_stats.wilcoxon(nonzero, alternative="greater")
        n_eff = len(nonzero)
        r_effect = 1 - (2 * stat) / (n_eff * (n_eff + 1))
    else:
        stat, p_value = 0.0, 1.0
        r_effect = 0.0
        print("  WARNING: Fewer than 10 non-zero differences, test unreliable")

    print(f"  Δ mean sim:     {float(delta.mean()):+.4f}")
    print(f"  Δ median sim:   {float(np.median(delta)):+.4f}")
    print(f"  Increased:      {n_increased}/{len(delta)} "
          f"({n_increased / len(delta):.0%})")
    print(f"  Decreased:      {n_decreased}/{len(delta)} "
          f"({n_decreased / len(delta):.0%})")
    print(f"  Wilcoxon stat:  {stat:.1f}")
    print(f"  p-value:        {p_value:.6f}")
    print(f"  Effect size r:  {r_effect:.3f}")
    print(f"  Significant:    {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")

    # ── Phase 6b: Post-training internal HD ──────────────────────────
    print(f"\n[9b] Extracting post-training internal representations for HD...")
    t_hd2 = time.time()
    reps_after = extract_internal_representations(
        model, tokenizer, hd_texts, max_length=128,
    )
    print(f"  Extracted {n_hd} reps in {time.time() - t_hd2:.1f}s")

    internal_sim_after = reps_after @ reps_after.T

    # Use same matched threshold as baseline
    hd_after, _ = compute_hd_from_sim_matrix(
        internal_sim_after, hd_behaviors, threshold=matched_threshold,
    )

    print(f"\n  Internal HD comparison (matched density, threshold={matched_threshold:.4f}):")
    print(f"  {'Behavior':15s} {'Text':>7s} {'Before':>8s} {'After':>8s} "
          f"{'Δ':>8s} {'Direction':>14s}")
    print(f"  {'-' * 62}")

    hd_deltas = {}
    for beh in sorted(hd_before.keys(), key=lambda b: -text_hd.get(b, 0)):
        t_h = text_hd.get(beh, 0)
        b_h = hd_before.get(beh, 0)
        a_h = hd_after.get(beh, 0)
        d = a_h - b_h
        hd_deltas[beh] = d
        if abs(d) > 0.001:
            direction = "▲ more highway" if d > 0 else "▼ more insular"
        else:
            direction = "— unchanged"
        print(f"  {beh:15s} {t_h:>7.3f} {b_h:>8.4f} {a_h:>8.4f} "
              f"{d:>+8.4f} {direction:>14s}")

    # Key finding: did sycophancy/bias isolation change?
    key_behaviors = ["sycophancy", "bias"]
    for beh in key_behaviors:
        if beh in hd_deltas:
            d = hd_deltas[beh]
            print(f"\n  ★ {beh}: HD Δ = {d:+.4f} "
                  f"({'BRIDGE OPENED' if d > 0.01 else 'NO CHANGE' if abs(d) < 0.01 else 'MORE ISOLATED'})")

    del reps_after, internal_sim_after
    gc.collect()

    # ── Phase 7: Post-training audit ─────────────────────────────────
    print(f"\n[10] Post-training behavioral audit...")
    audit_after = run_audit(model, tokenizer)
    print(
        f"  Scores: "
        f"{', '.join(f'{k}={v:.3f}' for k, v in sorted(audit_after.items()))}"
    )

    # Collateral check
    print(f"\n  Collateral damage check:")
    any_fail = False
    for beh in sorted(audit_before.keys()):
        before = audit_before.get(beh, 0)
        after = audit_after.get(beh, 0)
        d = after - before
        marker = "✗" if d < -0.05 else "✓"
        if d < -0.05:
            any_fail = True
        print(f"    {marker} {beh:15s}: {before:.3f} → {after:.3f} "
              f"(Δ={d:+.3f})")

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - t0
    passed = p_value < 0.05 and not any_fail

    print(f"\n{'=' * 60}")
    print(f"  VERDICT: {'PROCEED ✓' if passed else 'KILL ✗'}")
    print(f"  p={p_value:.4f}, Δsim={float(delta.mean()):+.4f}, "
          f"collateral={'clean' if not any_fail else 'FAIL'}")
    print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 60}")

    # Save results
    results = {
        "model": args.model,
        "config": {
            "steps": args.steps,
            "rank": args.rank,
            "lr": args.lr,
            "bridge_weight": args.bridge_weight,
            "pairs_per_step": args.pairs_per_step,
            "sim_range": [0.55, 0.70],
            "n_measure_pairs": len(measure_pairs),
            "n_train_pairs": len(train_pairs),
            "n_total_bridge_pairs": len(all_pairs),
        },
        "baseline": {
            "mean_sim": float(sims_before.mean()),
            "std_sim": float(sims_before.std()),
            "median_sim": float(np.median(sims_before)),
        },
        "post_training": {
            "mean_sim": float(sims_after.mean()),
            "std_sim": float(sims_after.std()),
            "median_sim": float(np.median(sims_after)),
        },
        "test": {
            "n_pairs": len(delta),
            "n_increased": n_increased,
            "n_decreased": n_decreased,
            "n_unchanged": n_unchanged,
            "mean_delta_sim": float(delta.mean()),
            "median_delta_sim": float(np.median(delta)),
            "wilcoxon_stat": float(stat),
            "p_value": float(p_value),
            "effect_size_r": float(r_effect),
            "significant": bool(p_value < 0.05),
        },
        "audit_before": audit_before,
        "audit_after": audit_after,
        "internal_hd": {
            "matched_threshold": round(matched_threshold, 4),
            "text_hd": {k: round(v, 4) for k, v in text_hd.items()},
            "before": {k: round(v, 4) for k, v in hd_before.items()},
            "after": {k: round(v, 4) for k, v in hd_after.items()},
            "delta": {k: round(v, 4) for k, v in hd_deltas.items()},
        },
        "collateral_clean": not any_fail,
        "verdict": "PROCEED" if passed else "KILL",
        "total_time_sec": round(total_time, 1),
        "train_time_sec": round(train_time, 1),
    }

    result_path = output_dir / "bridge_feasibility.json"
    result_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
