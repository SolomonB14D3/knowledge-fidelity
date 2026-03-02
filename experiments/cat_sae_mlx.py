"""CatSAE — Category-Aware SAE for targeted bias protection (MLX backend).

Extends Rho-Surgery with SAE-informed category vulnerability analysis.
Instead of uniform gamma protection across all bias categories, CatSAE:
  1. Trains a Gated SAE on mixed sycophancy + bias activations
  2. Identifies which SAE features are entangled across behaviors
  3. Measures per-category "exposure" — how much sycophancy features
     encode each bias category's information
  4. Oversamples stubborn categories in the protection dataset
  5. Runs surgery with this category-aware weighted protection

Hypothesis: categories with high sycophancy-feature exposure
(e.g., Religion, Age) are damaged during sycophancy SFT because
modifying those features also modifies category-encoding directions.
CatSAE compensates by providing stronger protection gradient signal
for these specific categories.

Pipeline:
  Phase 1: Collect activations from bias + sycophancy probes (MLX)
  Phase 2: Train GatedSAE on collected activations (PyTorch CPU)
  Phase 3: Feature census — category-aware feature identification
  Phase 4: Vulnerability analysis — sycophancy feature exposure
  Phase 5: Weighted surgery — CatSAE-informed gamma protection (MLX)
  Phase 6: Verify per-category outcomes

Usage:
    # Full pipeline
    python experiments/cat_sae_mlx.py Qwen/Qwen2.5-7B-Instruct

    # Analysis only (no surgery — CPU, ~30 min)
    python experiments/cat_sae_mlx.py Qwen/Qwen2.5-7B-Instruct --analysis-only

    # Use pre-collected activations
    python experiments/cat_sae_mlx.py Qwen/Qwen2.5-7B-Instruct \
        --activations results/cat_sae/Qwen2.5-7B-Instruct/activations.npz
"""

import argparse
import gc
import json
import random
import time
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Ensure src/ on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ── Phase 1: MLX Activation Collection ─────────────────────────────────


def collect_mlx_activations(
    model,
    tokenizer,
    max_probes_per_behavior: int = 150,
    max_length: int = 256,
    verbose: bool = True,
) -> dict:
    """Collect last-token hidden states from MLX model for bias + sycophancy probes.

    Uses model.model(input_ids) for the full forward pass (final layer output).
    Category metadata is preserved for bias probes.

    Returns:
        dict with keys:
            activations: np.array (n_samples, hidden_dim)
            labels: list[str] — behavior name per sample
            polarities: list[str] — "positive" or "negative"
            categories: list[str] — category per sample
    """
    import mlx.core as mx
    from rho_eval.behaviors import get_behavior

    all_activations = []
    all_labels = []
    all_polarities = []
    all_categories = []
    rng = random.Random(42)

    # ── Bias probes (with category metadata) ─────────────────────
    if verbose:
        print("  [bias] Loading probes...")

    bias_beh = get_behavior("bias")
    bias_probes = bias_beh.load_probes(seed=42)

    # Build contrast pairs preserving category info
    bias_pairs = []
    for p in bias_probes:
        pair = {
            "positive": f"{p['text']} {p['correct_answer']}",
            "negative": f"{p['text']} {p['biased_answer']}",
            "id": p.get("id", ""),
            "category": p.get("category", "unknown"),
        }
        bias_pairs.append(pair)

    if len(bias_pairs) > max_probes_per_behavior:
        bias_pairs = rng.sample(bias_pairs, max_probes_per_behavior)

    if verbose:
        cats = defaultdict(int)
        for p in bias_pairs:
            cats[p["category"]] += 1
        print(f"  [bias] {len(bias_pairs)} pairs across {len(cats)} categories")

    # ── Sycophancy probes ────────────────────────────────────────
    if verbose:
        print("  [sycophancy] Loading probes...")

    syc_beh = get_behavior("sycophancy")
    syc_probes = syc_beh.load_probes(seed=42)

    syc_pairs = []
    for p in syc_probes:
        pair = {
            "positive": f"{p['text']}\n{p['truthful_answer']}",
            "negative": f"{p['text']}\n{p['sycophantic_answer']}",
            "id": p.get("id", ""),
            "category": "sycophancy",
        }
        syc_pairs.append(pair)

    if len(syc_pairs) > max_probes_per_behavior:
        syc_pairs = rng.sample(syc_pairs, max_probes_per_behavior)

    if verbose:
        print(f"  [sycophancy] {len(syc_pairs)} pairs")

    # ── Forward pass through all pairs ───────────────────────────
    all_pairs = [("bias", p) for p in bias_pairs] + [("sycophancy", p) for p in syc_pairs]

    if verbose:
        print(f"\n  Collecting activations for {len(all_pairs)} pairs...")

    for i, (behavior, pair) in enumerate(all_pairs):
        for polarity in ["positive", "negative"]:
            text = pair[polarity]
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            if len(tokens) < 2:
                continue

            input_ids = mx.array([tokens])

            # Forward pass — final layer hidden states
            h = model.model(input_ids)  # (1, seq_len, hidden_dim)
            last_h = h[0, -1, :]  # (hidden_dim,) — last token

            # Convert to numpy (bfloat16 → float32 → numpy)
            act = np.array(last_h.astype(mx.float32))

            all_activations.append(act)
            all_labels.append(behavior)
            all_polarities.append(polarity)
            all_categories.append(pair["category"])

        # Free memory periodically
        if (i + 1) % 100 == 0:
            mx.eval(model.parameters())
            if verbose:
                print(f"    {i + 1}/{len(all_pairs)} pairs processed", flush=True)

    activations = np.stack(all_activations)

    if verbose:
        n_bias = sum(1 for l in all_labels if l == "bias")
        n_syc = sum(1 for l in all_labels if l == "sycophancy")
        print(f"  Collected {activations.shape[0]} activations "
              f"(dim={activations.shape[1]}): {n_bias} bias, {n_syc} sycophancy")

    return {
        "activations": activations,
        "labels": all_labels,
        "polarities": all_polarities,
        "categories": all_categories,
    }


# ── Phase 2: SAE Training ──────────────────────────────────────────────


def train_sae_on_activations(
    act_data: dict,
    expansion_factor: int = 8,
    n_epochs: int = 5,
    sparsity_lambda: float = 1e-3,
    lr: float = 3e-4,
    batch_size: int = 64,
    verbose: bool = True,
) -> tuple:
    """Train GatedSAE on collected activations (PyTorch CPU).

    Returns:
        Tuple of (sae, train_stats_dict).
    """
    import torch
    from rho_eval.steering.sae import GatedSAE

    activations = torch.tensor(act_data["activations"], dtype=torch.float32)
    hidden_dim = activations.shape[1]
    n_samples = activations.shape[0]

    if verbose:
        print(f"  Training GatedSAE: hidden_dim={hidden_dim}, "
              f"expansion={expansion_factor}, "
              f"n_features={hidden_dim * expansion_factor}")
        print(f"  n_samples={n_samples}, epochs={n_epochs}, lr={lr}")

    sae = GatedSAE(hidden_dim, expansion_factor=expansion_factor)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    t0 = time.time()
    avg_loss = 0.0

    for epoch in range(n_epochs):
        indices = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = activations[batch_idx]

            x_hat, z, gate_pre = sae(batch)
            total, mse, l1 = sae.compute_loss(batch, x_hat, gate_pre, sparsity_lambda)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            sae.normalize_decoder()

            epoch_loss += total.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if verbose:
            print(f"    Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.6f}")

    elapsed = time.time() - t0

    # Count active features
    with torch.no_grad():
        z, _ = sae.encode(activations)
        active_mask = z.abs().max(dim=0).values > 1e-6
        active_features = active_mask.sum().item()

    if verbose:
        print(f"  SAE trained in {elapsed:.1f}s: "
              f"{active_features}/{sae.n_features} features active")

    return sae, {
        "n_epochs": n_epochs,
        "final_loss": round(avg_loss, 6),
        "active_features": active_features,
        "total_features": sae.n_features,
        "hidden_dim": hidden_dim,
        "expansion_factor": expansion_factor,
        "time_sec": round(elapsed, 1),
    }


# ── Phase 3: Category-Aware Feature Census ──────────────────────────────


def category_feature_census(
    sae,
    act_data: dict,
    selectivity_threshold: float = 1.5,
    verbose: bool = True,
) -> dict:
    """Identify per-category SAE features and measure entanglement.

    For each SAE feature:
    1. Compute behavior-level diff (sycophancy vs bias)
    2. For bias features, compute per-category diff
    3. For sycophancy features, measure per-category "exposure"
       (how much modifying this feature would affect each category)

    Returns:
        dict with:
            behavior_features: {behavior: [feat_idx, ...]}
            category_features: {category: [feat_idx, ...]}
            sycophancy_exposure: {category: exposure_score}
            vulnerability: {category: {n_features, exposure, score, ...}}
    """
    import torch

    activations = torch.tensor(act_data["activations"], dtype=torch.float32)
    labels = act_data["labels"]
    polarities = act_data["polarities"]
    categories = act_data["categories"]

    # Encode all activations
    sae.eval()
    with torch.no_grad():
        z, _ = sae.encode(activations)  # (n_samples, n_features)
    z = z.cpu()

    n_features = z.shape[1]
    behaviors = sorted(set(labels))
    bias_cats = sorted(set(
        c for c, l in zip(categories, labels)
        if l == "bias" and c != "unknown"
    ))

    if verbose:
        print(f"  Census: {n_features} features, {len(behaviors)} behaviors, "
              f"{len(bias_cats)} bias categories")

    # ── Build masks ────────────────────────────────────────────────
    beh_masks = {}
    for beh in behaviors:
        for pol in ["positive", "negative"]:
            mask = torch.tensor([
                l == beh and p == pol
                for l, p in zip(labels, polarities)
            ], dtype=torch.bool)
            beh_masks[(beh, pol)] = mask

    cat_masks = {}
    for cat in bias_cats:
        for pol in ["positive", "negative"]:
            mask = torch.tensor([
                l == "bias" and p == pol and c == cat
                for l, p, c in zip(labels, polarities, categories)
            ], dtype=torch.bool)
            cat_masks[(cat, pol)] = mask

    # ── Per-feature analysis ──────────────────────────────────────
    behavior_features = {b: [] for b in behaviors}
    category_features = {c: [] for c in bias_cats}

    # Track per-feature scores for exposure analysis
    feat_beh_scores = {}   # feat_idx → {behavior: diff}
    feat_cat_scores = {}   # feat_idx → {category: diff}

    for feat_idx in range(n_features):
        feat_z = z[:, feat_idx]

        if feat_z.abs().max() < 1e-8:
            continue

        # ── Behavior-level scores ─────────────────────────────
        beh_scores = {}
        for beh in behaviors:
            pos_m = beh_masks.get((beh, "positive"))
            neg_m = beh_masks.get((beh, "negative"))
            if pos_m is None or neg_m is None:
                continue
            if pos_m.sum() == 0 or neg_m.sum() == 0:
                beh_scores[beh] = 0.0
                continue
            beh_scores[beh] = (
                feat_z[pos_m].mean().item() - feat_z[neg_m].mean().item()
            )

        if not beh_scores:
            continue

        feat_beh_scores[feat_idx] = beh_scores

        # Selectivity
        abs_diffs = [abs(d) for d in beh_scores.values()]
        max_diff = max(abs_diffs)
        mean_diff = sum(abs_diffs) / len(abs_diffs)
        selectivity = max_diff / mean_diff if mean_diff > 1e-8 else 0.0

        dominant = max(beh_scores.keys(), key=lambda b: abs(beh_scores[b]))

        if selectivity >= selectivity_threshold and max_diff > 1e-6:
            behavior_features[dominant].append(feat_idx)

        # ── Category-level scores (for all features, not just bias) ──
        cat_scores = {}
        for cat in bias_cats:
            pos_m = cat_masks.get((cat, "positive"))
            neg_m = cat_masks.get((cat, "negative"))
            if pos_m is None or neg_m is None:
                continue
            if pos_m.sum() < 2 or neg_m.sum() < 2:
                cat_scores[cat] = 0.0
                continue
            cat_scores[cat] = (
                feat_z[pos_m].mean().item() - feat_z[neg_m].mean().item()
            )

        feat_cat_scores[feat_idx] = cat_scores

        # Assign to dominant category (within bias features)
        if dominant == "bias" and cat_scores:
            cat_abs = {c: abs(s) for c, s in cat_scores.items() if abs(s) > 1e-6}
            if cat_abs:
                dom_cat = max(cat_abs, key=cat_abs.get)
                category_features[dom_cat].append(feat_idx)

    # ── Sycophancy-Feature Exposure Analysis ──────────────────────
    # KEY INSIGHT: for each sycophancy feature, measure how strongly
    # it responds to each bias category. This tells us which categories
    # will be damaged when we modify sycophancy features during SFT.
    syc_feats = behavior_features.get("sycophancy", [])
    category_exposure = {cat: 0.0 for cat in bias_cats}
    category_exposure_details = {cat: [] for cat in bias_cats}

    for feat_idx in syc_feats:
        cat_scores = feat_cat_scores.get(feat_idx, {})
        for cat in bias_cats:
            score = abs(cat_scores.get(cat, 0.0))
            category_exposure[cat] += score
            if score > 0.01:
                category_exposure_details[cat].append({
                    "feature": feat_idx,
                    "score": round(score, 4),
                })

    # Normalize by number of sycophancy features
    n_syc = len(syc_feats) if syc_feats else 1
    for cat in category_exposure:
        category_exposure[cat] /= n_syc

    # ── Vulnerability Scoring ─────────────────────────────────────
    # vulnerability = f(exposure from sycophancy features, own feature count)
    # High exposure + few category features = MOST vulnerable
    # High exposure + many category features = vulnerable but visible
    # Low exposure + any features = safe
    vulnerability = {}

    max_exposure = max(category_exposure.values()) if category_exposure else 1.0
    if max_exposure < 1e-8:
        max_exposure = 1.0

    for cat in bias_cats:
        n_feats = len(category_features.get(cat, []))
        exposure = category_exposure.get(cat, 0.0)
        norm_exposure = exposure / max_exposure  # 0-1

        # Feature count factor: fewer features = harder to protect
        # More features = SAE captures it well (less vulnerable)
        feat_factor = 1.0 / (1.0 + n_feats * 0.1)  # Decays with features

        vuln_score = norm_exposure * 0.6 + feat_factor * 0.4

        vulnerability[cat] = {
            "n_features": n_feats,
            "exposure_raw": round(exposure, 6),
            "exposure_normalized": round(norm_exposure, 4),
            "n_exposing_syc_features": len(category_exposure_details[cat]),
            "feat_scarcity_factor": round(feat_factor, 4),
            "vulnerability_score": round(vuln_score, 4),
        }

    if verbose:
        print(f"\n  Behavior features:")
        for beh, feats in sorted(behavior_features.items()):
            print(f"    {beh}: {len(feats)} features")

        print(f"\n  Category features (top 10):")
        for cat in sorted(bias_cats,
                          key=lambda c: len(category_features.get(c, [])),
                          reverse=True)[:10]:
            print(f"    {cat:<30s} {len(category_features[cat]):>4d} features")

        print(f"\n  Sycophancy-feature exposure per category:")
        for cat in sorted(bias_cats,
                          key=lambda c: category_exposure.get(c, 0.0),
                          reverse=True):
            exp = category_exposure[cat]
            n_exp = len(category_exposure_details[cat])
            if exp > 0.001 or n_exp > 0:
                print(f"    {cat:<30s} exposure={exp:.4f} "
                      f"({n_exp} exposing features)")

        print(f"\n  Vulnerability ranking (higher = needs more protection):")
        for cat in sorted(bias_cats,
                          key=lambda c: vulnerability[c]["vulnerability_score"],
                          reverse=True):
            v = vulnerability[cat]
            bar = "#" * int(v["vulnerability_score"] * 20)
            print(f"    {cat:<30s} {v['vulnerability_score']:.3f} "
                  f"[{bar:<20s}] "
                  f"(exp={v['exposure_normalized']:.2f}, "
                  f"feat={v['n_features']})")

    return {
        "behavior_features": {b: list(f) for b, f in behavior_features.items()},
        "category_features": {c: list(f) for c, f in category_features.items()},
        "sycophancy_exposure": {c: round(v, 6) for c, v in category_exposure.items()},
        "vulnerability": vulnerability,
        "n_sycophancy_features": len(syc_feats),
        "n_bias_features": len(behavior_features.get("bias", [])),
    }


# ── Phase 4: Weighted Protection Dataset ───────────────────────────────


# ── Per-Category Floor Overrides ──────────────────────────────────────
#
# WHY: SAE vulnerability scoring under-weights categories whose bias
# features are poorly represented in the sycophancy feature subspace.
# Age got vuln=0.162 → 0.7x weight, causing -5.9% regression, even
# though Age is empirically one of the most surgery-damaged categories.
#
# The manual override approach (setting vuln=0.5) was catastrophic
# (bias 0.833 → 0.277) because it broke the SAE calibration globally.
#
# SOLUTION: Per-category floor on the *raw weight* (not vuln score).
# This preserves SAE-informed relative ordering (Religion still gets
# its boost from genuine sycophancy feature entanglement) while
# guaranteeing known-vulnerable categories get at least baseline
# protection. Think of it as: "trust the SAE ranking, but don't let
# any known-vulnerable category fall below equal weight."
#
# Edit this dict to experiment with different floors per category.
# A floor of 1.0 means "at least equal weight before normalization."
# Set to None or omit a category to use pure SAE weighting.

FLOOR_OVERRIDES: dict[str, float] = {
    "Age": 1.0,                         # SAE under-weights (vuln=0.162), known high-damage
    "Gender_biology": 1.0,              # SAE under-weights, empirically vulnerable
    "Race_ethnicity": 1.0,              # SAE under-weights (vuln~0.15), known high-damage
    "Sexual_orientation_biology": 1.0,  # Moderate SAE signal, ensure baseline protection
    "Religion": 1.0,                    # SAE correctly boosts this — floor is redundant
                                        # but included for symmetry; SAE weight > 1.0
                                        # anyway so floor never activates
}


def build_weighted_protection(
    vulnerability: dict,
    categories_to_protect: list[str],
    base_pairs_per_category: int = 50,
    weight_floor: float | None = None,
    floor_overrides: dict[str, float] | None = None,
    verbose: bool = True,
):
    """Build a protection dataset with vulnerability-weighted oversampling.

    Categories with higher vulnerability scores get more protection pairs,
    providing stronger gamma gradient signal during SFT.

    Args:
        weight_floor: Blanket floor applied to ALL categories.
            Use 1.0 to ensure no category gets less than baseline weight.
        floor_overrides: Per-category floors — dict mapping category name
            to minimum raw weight. Takes precedence over weight_floor for
            categories present in both. Use this for surgical control:
            e.g., {"Age": 1.0, "Religion": 1.0} ensures these specific
            categories get at least 1.0x raw weight while others use
            pure SAE weighting.

    Returns:
        BehavioralContrastDataset-like object with .pairs and .sample() method.
    """
    from rho_eval.alignment.dataset import BehavioralContrastDataset

    if floor_overrides is None:
        floor_overrides = {}

    # Compute per-category weights
    weights = {}
    floored_cats = []
    for cat in categories_to_protect:
        vuln = vulnerability.get(cat, {})
        score = vuln.get("vulnerability_score", 0.5)
        # Scale: 0.0 → 0.5x, 0.5 → 1.5x, 1.0 → 2.5x
        raw_weight = 0.5 + 2.0 * score

        # Apply per-category floor (highest priority)
        cat_floor = floor_overrides.get(cat, weight_floor)
        if cat_floor is not None and raw_weight < cat_floor:
            floored_cats.append((cat, raw_weight, cat_floor))
            raw_weight = cat_floor

        weights[cat] = raw_weight

    # Normalize: average weight = 1.0
    if weights:
        mean_w = sum(weights.values()) / len(weights)
        for cat in weights:
            weights[cat] /= mean_w

    if verbose:
        has_floors = weight_floor is not None or floor_overrides
        floor_str = ""
        if has_floors:
            floor_str = " (with floors)"
        print(f"  Category protection weights{floor_str}:")
        for cat in sorted(weights, key=weights.get, reverse=True):
            vuln_score = vulnerability.get(cat, {}).get("vulnerability_score", 0)
            floor_marker = ""
            if cat in floor_overrides:
                floor_marker = f" [floor={floor_overrides[cat]}]"
            print(f"    {cat:<30s} weight={weights[cat]:.2f}x "
                  f"(vuln={vuln_score:.3f}){floor_marker}")
        if floored_cats:
            print(f"  Floor activated on {len(floored_cats)} categories:")
            for cat, old_w, floor in floored_cats:
                print(f"    {cat}: {old_w:.3f} → {floor:.3f}")

    # Build per-category datasets and merge
    all_pairs = []
    cat_counts = {}

    for cat in categories_to_protect:
        n_wanted = max(10, int(base_pairs_per_category * weights.get(cat, 1.0)))

        try:
            ds = BehavioralContrastDataset(
                behaviors=["bias"],
                categories=[cat],
                seed=42,
            )
            cat_pairs = ds.pairs[:n_wanted]
            all_pairs.extend(cat_pairs)
            cat_counts[cat] = len(cat_pairs)
        except Exception as e:
            if verbose:
                print(f"    WARNING: Failed to load pairs for {cat}: {e}")
            cat_counts[cat] = 0

    # Shuffle the merged dataset
    random.Random(42).shuffle(all_pairs)

    if verbose:
        print(f"\n  Weighted protection dataset: {len(all_pairs)} total pairs")
        for cat in sorted(cat_counts, key=cat_counts.get, reverse=True):
            if cat_counts[cat] > 0:
                print(f"    {cat:<30s} {cat_counts[cat]:>4d} pairs")

    # Create a dataset object compatible with the trainer's .sample() API
    class WeightedProtectionDataset:
        def __init__(self, pairs):
            self.pairs = pairs

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            pair = self.pairs[idx]
            return {
                "positive_text": pair["positive"],
                "negative_text": pair["negative"],
                "behavior": pair.get("behavior", "bias"),
            }

        def sample(self, k, rng=None):
            rng = rng or random.Random()
            k = min(k, len(self.pairs))
            indices = rng.sample(range(len(self.pairs)), k)
            return [
                {"positive": self.pairs[i]["positive"],
                 "negative": self.pairs[i]["negative"]}
                for i in indices
            ]

    return WeightedProtectionDataset(all_pairs), weights, cat_counts


# ── Phase 5: Surgery with CatSAE Protection ───────────────────────────


def mlx_svd_compress(model, ratio: float = 0.7) -> int:
    """SVD compress Q/K/O attention projections in an MLX model.

    Inlined from rho_surgery_mlx.py for standalone use.
    """
    import mlx.core as mx

    compressed = 0
    safe_projections = ["q_proj", "k_proj", "o_proj"]

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for proj_name in safe_projections:
            if not hasattr(attn, proj_name):
                continue

            proj = getattr(attn, proj_name)
            W_mx = proj.weight
            W = np.array(W_mx.astype(mx.float32))

            if len(W.shape) != 2 or min(W.shape) <= 10:
                continue

            rank = max(1, int(min(W.shape) * ratio))

            try:
                U, S, Vh = np.linalg.svd(W, full_matrices=False)
                W_approx = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
                proj.weight = mx.array(W_approx).astype(W_mx.dtype)
                compressed += 1
            except Exception as e:
                print(f"    SVD failed for layer {i} {proj_name}: {e}", flush=True)

    mx.eval(model.parameters())
    return compressed


def run_bias_audit(model, tokenizer, label: str) -> dict:
    """Run bias audit on MLX model and return structured results."""
    from rho_eval.audit import audit

    t0 = time.time()
    report = audit(model=model, tokenizer=tokenizer, behaviors=["bias"], device="mlx")
    elapsed = time.time() - t0

    bias_result = report.behaviors["bias"]
    print(f"  [{label}] Bias rho={bias_result.rho:.4f} "
          f"({bias_result.positive_count}/{bias_result.total}), "
          f"{elapsed:.0f}s", flush=True)

    meta = bias_result.metadata or {}
    cat_metrics = meta.get("category_metrics", {})

    return {
        "rho": bias_result.rho,
        "positive_count": bias_result.positive_count,
        "total": bias_result.total,
        "retention": bias_result.retention,
        "category_metrics": cat_metrics,
        "elapsed": elapsed,
    }


def run_catsae_surgery(
    model,
    tokenizer,
    model_name: str,
    protection_dataset,
    gamma_weight: float = 0.15,
    rho_weight: float = 0.2,
    compress_ratio: float = 0.7,
    save_path: str = None,
    verbose: bool = True,
) -> dict:
    """Run SVD + LoRA SFT with CatSAE-weighted gamma protection."""
    import mlx.core as mx
    from rho_eval.alignment.dataset import (
        BehavioralContrastDataset, _build_trap_texts, _load_alpaca_texts,
    )
    from rho_eval.alignment.mlx_trainer import mlx_rho_guided_sft

    t_start = time.time()

    # ── Baseline bias audit ────────────────────────────────────────
    if verbose:
        print(f"\n  [surgery] Baseline audit...", flush=True)
    baseline = run_bias_audit(model, tokenizer, "baseline")

    # ── SVD compression ────────────────────────────────────────────
    if compress_ratio < 1.0:
        if verbose:
            print(f"\n  [surgery] SVD compression (ratio={compress_ratio})...", flush=True)
        t0 = time.time()
        n_compressed = mlx_svd_compress(model, ratio=compress_ratio)
        mx.eval(model.parameters())
        if verbose:
            print(f"  Compressed {n_compressed} matrices in {time.time()-t0:.1f}s",
                  flush=True)

    # ── Build SFT text data ────────────────────────────────────────
    if verbose:
        print(f"\n  [surgery] Building SFT dataset...", flush=True)
    rng = random.Random(42)
    sft_texts = []
    trap_texts = _build_trap_texts(["sycophancy"], seed=42)
    rng.shuffle(trap_texts)
    sft_texts.extend(trap_texts[:400])
    try:
        alpaca_texts = _load_alpaca_texts(1600, seed=42)
        sft_texts.extend(alpaca_texts)
    except Exception as e:
        if verbose:
            print(f"    Alpaca load failed ({e}), using traps only", flush=True)
    rng.shuffle(sft_texts)
    sft_texts = sft_texts[:2000]

    # ── Contrast dataset (sycophancy target) ───────────────────────
    contrast_dataset = BehavioralContrastDataset(
        behaviors=["sycophancy"], seed=42,
    )

    # ── LoRA SFT ───────────────────────────────────────────────────
    if verbose:
        print(f"\n  [surgery] LoRA SFT: rho={rho_weight}, gamma={gamma_weight}, "
              f"protection={len(protection_dataset)} pairs", flush=True)

    sft_result = mlx_rho_guided_sft(
        model, tokenizer,
        sft_texts,
        contrast_dataset=contrast_dataset,
        rho_weight=rho_weight,
        gamma_weight=gamma_weight,
        protection_dataset=protection_dataset,
        epochs=1,
        lr=2e-4,
        margin=0.1,
        save_path=save_path,
    )

    if verbose:
        print(f"  SFT complete: {sft_result['steps']} steps, "
              f"{sft_result['time']:.0f}s", flush=True)

    # ── Final bias audit ───────────────────────────────────────────
    if verbose:
        print(f"\n  [surgery] Final audit...", flush=True)
    final = run_bias_audit(model, tokenizer, "final")

    total_elapsed = time.time() - t_start

    return {
        "baseline": baseline,
        "final": final,
        "sft_result": {k: v for k, v in sft_result.items() if k != "merged_model"},
        "total_elapsed_sec": total_elapsed,
    }


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="CatSAE — Category-Aware SAE for targeted bias protection",
    )
    parser.add_argument("model", help="HuggingFace model ID")
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Run SAE analysis only (no surgery, ~30 min on CPU/GPU)",
    )
    parser.add_argument(
        "--activations", type=str, default=None,
        help="Path to pre-collected activations.npz (skip Phase 1)",
    )
    parser.add_argument("--sae-expansion", type=int, default=8)
    parser.add_argument("--sae-epochs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.10,
                       help="Gamma protection weight (default: 0.10, optimal from sweep)")
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--compress-ratio", type=float, default=0.7)
    parser.add_argument("--max-probes", type=int, default=150)
    parser.add_argument(
        "--weight-floor", type=float, default=None,
        help="Blanket floor on raw protection weight for ALL categories. "
             "Use 1.0 to ensure no category gets < 1x baseline weight. "
             "For per-category control, use --use-floor-overrides instead.",
    )
    parser.add_argument(
        "--use-floor-overrides", action="store_true",
        help="Use per-category floor overrides from FLOOR_OVERRIDES dict. "
             "Edit the dict in cat_sae_mlx.py for fine-grained control. "
             "Combines with --weight-floor (per-cat takes precedence).",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Categories to protect (default: auto-select from known + SAE-detected). "
             "Example: --categories Age Gender_biology Race_ethnicity "
             "Sexual_orientation_biology Religion",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/cat_sae",
    )
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    # Use descriptive subdirectory when floors are active
    if args.use_floor_overrides or args.weight_floor is not None:
        parts = []
        if args.use_floor_overrides:
            parts.append("percat")
        if args.weight_floor is not None:
            parts.append(f"blanket{args.weight_floor:.1f}")
        n_cats = len(args.categories) if args.categories else 0
        if n_cats:
            parts.append(f"{n_cats}cat")
        tag = "_".join(parts) if parts else "floor"
        output_dir = Path(args.output) / model_short / tag
    else:
        output_dir = Path(args.output) / model_short
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  CatSAE — Category-Aware SAE Experiment", flush=True)
    print(f"  Model:  {args.model}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Gamma:  {args.gamma}  Rho: {args.rho}  SVD: {args.compress_ratio}", flush=True)
    if args.weight_floor is not None or args.use_floor_overrides:
        parts = []
        if args.weight_floor is not None:
            parts.append(f"blanket={args.weight_floor}")
        if args.use_floor_overrides:
            parts.append(f"per-cat overrides on {len(FLOOR_OVERRIDES)} categories")
        print(f"  Floors: {', '.join(parts)}", flush=True)
    if args.categories:
        print(f"  Categories: {', '.join(args.categories)}", flush=True)
    print(f"{'='*70}\n", flush=True)

    t_global = time.time()

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Load model + collect activations
    # ══════════════════════════════════════════════════════════════

    if args.activations:
        print(f"  Phase 1: Loading pre-collected activations from {args.activations}",
              flush=True)
        data = np.load(args.activations, allow_pickle=True)
        act_data = {
            "activations": data["activations"],
            "labels": list(data["labels"]),
            "polarities": list(data["polarities"]),
            "categories": list(data["categories"]),
        }
        print(f"  Loaded {act_data['activations'].shape[0]} activations "
              f"(dim={act_data['activations'].shape[1]})", flush=True)

        # Still need model for surgery phases
        if not args.analysis_only:
            import mlx_lm
            print(f"\n  Loading model for surgery...", flush=True)
            model, tokenizer = mlx_lm.load(args.model)
    else:
        print(f"  Phase 1: Loading model and collecting activations...", flush=True)
        import mlx_lm
        model, tokenizer = mlx_lm.load(args.model)

        act_data = collect_mlx_activations(
            model, tokenizer,
            max_probes_per_behavior=args.max_probes,
        )

        # Save activations for reuse
        act_path = output_dir / "activations.npz"
        np.savez_compressed(
            act_path,
            activations=act_data["activations"],
            labels=np.array(act_data["labels"]),
            polarities=np.array(act_data["polarities"]),
            categories=np.array(act_data["categories"]),
        )
        print(f"  Saved activations to {act_path}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Train GatedSAE
    # ══════════════════════════════════════════════════════════════

    print(f"\n  Phase 2: Training GatedSAE...", flush=True)
    sae, sae_stats = train_sae_on_activations(
        act_data,
        expansion_factor=args.sae_expansion,
        n_epochs=args.sae_epochs,
    )

    # Save SAE
    sae_path = output_dir / "gated_sae.pt"
    sae.save(sae_path)
    print(f"  Saved SAE to {sae_path}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Category feature census
    # ══════════════════════════════════════════════════════════════

    print(f"\n  Phase 3: Category-aware feature census...", flush=True)
    census = category_feature_census(sae, act_data)

    # Save census (JSON-serializable subset)
    census_save = {
        "behavior_features": {
            b: {"n": len(f), "top_10": f[:10]}
            for b, f in census["behavior_features"].items()
        },
        "category_features": {
            c: {"n": len(f), "top_10": f[:10]}
            for c, f in census["category_features"].items()
        },
        "sycophancy_exposure": census["sycophancy_exposure"],
        "vulnerability": census["vulnerability"],
        "n_sycophancy_features": census["n_sycophancy_features"],
        "n_bias_features": census["n_bias_features"],
    }
    census_path = output_dir / "category_census.json"
    census_path.write_text(json.dumps(census_save, indent=2))
    print(f"  Saved census to {census_path}", flush=True)

    if args.analysis_only:
        elapsed = time.time() - t_global
        print(f"\n{'='*70}", flush=True)
        print(f"  Analysis complete in {elapsed:.0f}s ({elapsed/60:.1f} min)",
              flush=True)
        print(f"  Results: {output_dir}", flush=True)
        print(f"{'='*70}", flush=True)
        return

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Build weighted protection dataset
    # ══════════════════════════════════════════════════════════════

    print(f"\n  Phase 4: Building CatSAE-weighted protection dataset...", flush=True)

    vulnerability = census["vulnerability"]

    if args.categories:
        # User-specified categories
        cats_to_protect = args.categories
    else:
        # Auto-select: known high-risk + SAE-detected vulnerable
        known_protect = [
            "Age", "Gender_biology", "Race_ethnicity", "Sexual_orientation_biology",
        ]
        sae_protect = [
            cat for cat in vulnerability
            if vulnerability[cat]["vulnerability_score"] > 0.3
        ]
        cats_to_protect = sorted(set(known_protect) | set(sae_protect))

    # Resolve floor configuration
    active_floors = {}
    if args.use_floor_overrides:
        # Per-category floors from the FLOOR_OVERRIDES dict at top of file
        active_floors = {
            cat: floor for cat, floor in FLOOR_OVERRIDES.items()
            if cat in cats_to_protect
        }
        print(f"  Per-category floor overrides (from FLOOR_OVERRIDES):", flush=True)
        for cat, floor in sorted(active_floors.items()):
            print(f"    {cat}: min raw weight = {floor}", flush=True)

    if args.weight_floor is not None:
        print(f"  Blanket weight floor: {args.weight_floor} "
              f"(fallback for categories without per-cat override)", flush=True)

    protection_dataset, weights, cat_counts = build_weighted_protection(
        vulnerability, cats_to_protect,
        base_pairs_per_category=50,
        weight_floor=args.weight_floor,
        floor_overrides=active_floors if active_floors else None,
    )

    # ══════════════════════════════════════════════════════════════
    # Phase 5: Surgery with CatSAE protection
    # ══════════════════════════════════════════════════════════════

    print(f"\n  Phase 5: Running CatSAE-informed surgery...", flush=True)
    surgery_result = run_catsae_surgery(
        model, tokenizer, args.model,
        protection_dataset=protection_dataset,
        gamma_weight=args.gamma,
        rho_weight=args.rho,
        compress_ratio=args.compress_ratio,
    )

    # ══════════════════════════════════════════════════════════════
    # Phase 6: Verification
    # ══════════════════════════════════════════════════════════════

    print(f"\n  Phase 6: Verification", flush=True)
    baseline_rho = surgery_result["baseline"]["rho"]
    final_rho = surgery_result["final"]["rho"]
    bias_delta = final_rho - baseline_rho

    print(f"\n  Overall: {baseline_rho:.4f} -> {final_rho:.4f} "
          f"(delta={bias_delta:+.4f})", flush=True)

    baseline_cats = surgery_result["baseline"].get("category_metrics", {})
    final_cats = surgery_result["final"].get("category_metrics", {})

    print(f"\n  {'Category':<30s} {'Before':>7s} {'After':>7s} "
          f"{'Delta':>7s} {'Vuln':>5s} {'Weight':>6s}", flush=True)
    print(f"  {'-'*68}", flush=True)

    for cat in sorted(final_cats.keys()):
        before = baseline_cats.get(cat, {}).get("accuracy", 0)
        after = final_cats[cat]["accuracy"]
        delta = after - before
        vuln = vulnerability.get(cat, {}).get("vulnerability_score", 0.0)
        w = weights.get(cat, 0.0)
        marker = "FAIL" if delta < -0.05 else ("+" if delta > 0.05 else " ")
        print(f"  {cat:<30s} {before:>6.1%} {after:>6.1%} "
              f"{delta:>+6.1%} {vuln:>5.3f} {w:>5.2f}x {marker}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Save comprehensive results
    # ══════════════════════════════════════════════════════════════

    total_elapsed = time.time() - t_global

    result = {
        "model": args.model,
        "config": {
            "sae_expansion": args.sae_expansion,
            "sae_epochs": args.sae_epochs,
            "gamma_weight": args.gamma,
            "rho_weight": args.rho,
            "compress_ratio": args.compress_ratio,
            "max_probes": args.max_probes,
            "weight_floor": args.weight_floor,
            "floor_overrides": active_floors if active_floors else None,
            "categories": cats_to_protect,
        },
        "sae_stats": sae_stats,
        "census": census_save,
        "protection_weights": {c: round(w, 3) for c, w in weights.items()},
        "protection_pair_counts": cat_counts,
        "surgery": {
            "baseline_bias": surgery_result["baseline"],
            "final_bias": surgery_result["final"],
            "sft_result": surgery_result["sft_result"],
        },
        "total_elapsed_sec": round(total_elapsed, 1),
    }

    result_path = output_dir / "cat_sae_result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))

    print(f"\n{'='*70}", flush=True)
    print(f"  CatSAE COMPLETE", flush=True)
    print(f"  Bias: {baseline_rho:.4f} -> {final_rho:.4f} "
          f"(delta={bias_delta:+.4f})", flush=True)
    print(f"  Time: {total_elapsed/60:.1f} min", flush=True)
    print(f"  Results: {result_path}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
