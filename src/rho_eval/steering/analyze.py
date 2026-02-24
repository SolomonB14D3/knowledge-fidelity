"""Feature analysis for trained SAEs.

Identifies which SAE features correspond to which behaviors by analyzing
activation patterns across behavior-labeled data. Features are scored by
their selectivity — how much they discriminate between positive and
negative examples for a specific behavior compared to other behaviors.
"""

from __future__ import annotations

from typing import Optional

import torch

from .schema import ActivationData, FeatureReport
from .sae import GatedSAE


@torch.no_grad()
def identify_behavioral_features(
    sae: GatedSAE,
    activation_data: ActivationData,
    threshold: float = 2.0,
    top_k: int | None = None,
) -> tuple[list[FeatureReport], dict[str, list[int]]]:
    """Identify which SAE features encode which behaviors.

    For each feature, computes how much its activation differs between
    positive and negative examples for each behavior. Features with
    high selectivity (responding strongly to one behavior but not others)
    are reported as behavioral features.

    The selectivity metric is:
        selectivity = max_diff / mean_diff

    where diff[b] = mean_act(positive, behavior=b) - mean_act(negative, behavior=b)
    for each behavior b.

    Args:
        sae: Trained GatedSAE.
        activation_data: Activations with behavior labels and polarities.
        threshold: Minimum selectivity to report a feature (default: 2.0).
            Higher = more monosemantic features only.
        top_k: If set, keep only top_k features per behavior (by diff score),
            regardless of selectivity threshold.

    Returns:
        Tuple of:
            feature_reports: List of FeatureReport for features above threshold.
            behavioral_features: Dict mapping behavior → list of feature indices
                assigned to that behavior (sorted by selectivity, descending).
    """
    sae.eval()
    device = sae.device

    activations = activation_data.activations.to(device)
    behaviors = activation_data.behaviors
    labels = activation_data.labels
    polarities = activation_data.polarities

    # Encode all activations through SAE
    z, _ = sae.encode(activations)  # (n_samples, n_features)
    z = z.cpu()

    n_features = z.shape[1]

    # Build masks for each (behavior, polarity) group
    masks = {}
    for behavior in behaviors:
        for polarity in ["positive", "negative"]:
            mask = torch.tensor([
                l == behavior and p == polarity
                for l, p in zip(labels, polarities)
            ], dtype=torch.bool)
            masks[(behavior, polarity)] = mask

    # Compute per-feature statistics for each behavior
    feature_reports = []
    behavioral_features: dict[str, list[int]] = {b: [] for b in behaviors}

    # Track all features' differential scores for top_k selection
    all_diffs: dict[str, list[tuple[int, float]]] = {b: [] for b in behaviors}

    for feat_idx in range(n_features):
        feat_z = z[:, feat_idx]  # (n_samples,)

        # Check if this feature is dead (never activates)
        if feat_z.abs().max() < 1e-8:
            continue

        behavior_scores = {}
        diffs = {}

        for behavior in behaviors:
            pos_mask = masks[(behavior, "positive")]
            neg_mask = masks[(behavior, "negative")]

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                behavior_scores[behavior] = 0.0
                diffs[behavior] = 0.0
                continue

            mean_pos = feat_z[pos_mask].mean().item()
            mean_neg = feat_z[neg_mask].mean().item()
            diff = mean_pos - mean_neg
            behavior_scores[behavior] = diff
            diffs[behavior] = diff

        # Overall positive/negative means (across all behaviors)
        pos_all = torch.tensor([p == "positive" for p in polarities], dtype=torch.bool)
        neg_all = torch.tensor([p == "negative" for p in polarities], dtype=torch.bool)
        mean_act_pos = feat_z[pos_all].mean().item() if pos_all.sum() > 0 else 0.0
        mean_act_neg = feat_z[neg_all].mean().item() if neg_all.sum() > 0 else 0.0

        # Selectivity: how specific to one behavior
        abs_diffs = [abs(d) for d in diffs.values()]
        max_diff = max(abs_diffs) if abs_diffs else 0.0
        mean_diff = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0

        if mean_diff > 1e-8:
            selectivity = max_diff / mean_diff
        else:
            selectivity = 0.0

        # Find dominant behavior
        if max_diff > 1e-8:
            dominant = max(diffs.keys(), key=lambda b: abs(diffs[b]))
        else:
            dominant = None

        # Track for top_k selection
        for behavior in behaviors:
            all_diffs[behavior].append((feat_idx, abs(diffs.get(behavior, 0.0))))

        # Apply threshold
        if selectivity >= threshold and max_diff > 1e-6:
            report = FeatureReport(
                feature_idx=feat_idx,
                behavior_scores=behavior_scores,
                dominant_behavior=dominant,
                mean_activation_pos=mean_act_pos,
                mean_activation_neg=mean_act_neg,
                selectivity=selectivity,
            )
            feature_reports.append(report)

            if dominant is not None:
                behavioral_features[dominant].append(feat_idx)

    # If top_k specified, override threshold-based selection per behavior
    if top_k is not None:
        behavioral_features = {b: [] for b in behaviors}
        for behavior in behaviors:
            sorted_feats = sorted(all_diffs[behavior], key=lambda x: x[1], reverse=True)
            behavioral_features[behavior] = [
                feat_idx for feat_idx, _ in sorted_feats[:top_k]
                if _ > 1e-6  # skip truly dead features
            ]

    # Sort behavioral features by selectivity (descending)
    selectivity_map = {fr.feature_idx: fr.selectivity for fr in feature_reports}
    for behavior in behavioral_features:
        behavioral_features[behavior].sort(
            key=lambda idx: selectivity_map.get(idx, 0.0), reverse=True,
        )

    return feature_reports, behavioral_features


@torch.no_grad()
def feature_overlap_matrix(
    feature_reports: list[FeatureReport],
    behaviors: list[str],
) -> dict[str, dict[str, float]]:
    """Compute pairwise feature overlap between behaviors.

    For each pair of behaviors, measures the fraction of features that
    are active (above threshold) for both. This should be lower than
    the SVD cosine overlap if the SAE is successfully disentangling.

    The overlap metric is Jaccard similarity:
        overlap(A, B) = |A intersect B| / |A union B|

    where A and B are the sets of features with above-median activation
    difference for each behavior.

    Args:
        feature_reports: List of FeatureReport from identify_behavioral_features.
        behaviors: Behavior names to compare.

    Returns:
        Dict of {behavior_a: {behavior_b: overlap_score}}.
        Symmetric matrix with 1.0 on diagonal.
    """
    # Build sets of features responsive to each behavior
    behavior_sets: dict[str, set[int]] = {b: set() for b in behaviors}

    for fr in feature_reports:
        for behavior in behaviors:
            score = fr.behavior_scores.get(behavior, 0.0)
            # A feature is "responsive" if its diff score is above
            # the median absolute score across all behaviors
            median_score = sorted(
                abs(s) for s in fr.behavior_scores.values()
            )[len(fr.behavior_scores) // 2] if fr.behavior_scores else 0.0

            if abs(score) > median_score and abs(score) > 1e-6:
                behavior_sets[behavior].add(fr.feature_idx)

    # Compute Jaccard overlap
    overlap = {}
    for b1 in behaviors:
        overlap[b1] = {}
        for b2 in behaviors:
            if b1 == b2:
                overlap[b1][b2] = 1.0
            else:
                s1 = behavior_sets[b1]
                s2 = behavior_sets[b2]
                union = len(s1 | s2)
                if union == 0:
                    overlap[b1][b2] = 0.0
                else:
                    overlap[b1][b2] = round(len(s1 & s2) / union, 6)

    return overlap
