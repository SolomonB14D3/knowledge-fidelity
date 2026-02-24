"""Behavioral subspace extraction via SVD.

Extracts the principal directions within transformer layers that encode
specific behavioral traits. Builds on the PCA method from
experiments/steering_vectors.py but retains the full truncated SVD
decomposition for downstream overlap and surgical analysis.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..utils import get_layers
from .activation import LayerActivationCapture, build_contrast_pairs
from .schema import SubspaceResult


def _auto_layers(n_layers: int) -> list[int]:
    """Select candidate layers at standard depth percentages.

    Returns layer indices at approximately:
    [25%, 37.5%, 50%, 62.5%, 75%, 87.5%] of model depth.
    """
    pcts = [0.25, 0.375, 0.50, 0.625, 0.75, 0.875]
    layers = sorted(set(
        max(0, min(n_layers - 1, int(pct * n_layers)))
        for pct in pcts
    ))
    return layers


def _load_probes_for_behavior(behavior: str) -> list[dict]:
    """Load probes for a behavior using the rho-eval probe system.

    Uses the v2 behavior plugin system first, falling back to
    the legacy API for factual probes.
    """
    if behavior == "factual":
        from ..probes import get_all_probes
        return get_all_probes()
    else:
        from ..behavioral import load_behavioral_probes
        return load_behavioral_probes(behavior, seed=42)


@torch.no_grad()
def extract_subspaces(
    model,
    tokenizer,
    behaviors: list[str],
    layers: list[int] | None = None,
    device: str = "cpu",
    max_rank: int = 50,
    max_probes: int | None = None,
    verbose: bool = True,
) -> dict[str, dict[int, SubspaceResult]]:
    """Extract behavioral subspaces at specified layers.

    For each (behavior, layer) pair:
    1. Build contrast pairs from behavioral probes.
    2. Run forward passes, capture last-token activations for pos/neg texts.
    3. Compute difference matrix D = pos - neg, center it.
    4. Full truncated SVD: U, S, Vh = svd(D_centered).
    5. Store top-k directions, singular values, cumulative explained variance.

    Args:
        model: HuggingFace causal LM (already on device, in eval mode).
        tokenizer: Corresponding tokenizer.
        behaviors: List of behavior names (e.g., ["factual", "sycophancy"]).
        layers: Layer indices to analyze (None = auto-select 6 layers).
        device: Torch device string.
        max_rank: Maximum number of principal directions to retain.
        max_probes: Cap on number of probes per behavior (None = use all).
        verbose: Print progress.

    Returns:
        Dict mapping behavior → {layer_idx: SubspaceResult}.
    """
    # Determine layers
    if layers is None:
        n_layers = len(get_layers(model))
        layers = _auto_layers(n_layers)

    if verbose:
        print(f"  Extracting subspaces for {behaviors} at layers {layers}")

    results: dict[str, dict[int, SubspaceResult]] = {}

    for behavior in behaviors:
        if verbose:
            print(f"\n  [{behavior}] Loading probes...")

        probes = _load_probes_for_behavior(behavior)
        pairs = build_contrast_pairs(behavior, probes)

        if max_probes is not None and len(pairs) > max_probes:
            import random
            rng = random.Random(42)
            pairs = rng.sample(pairs, max_probes)

        if len(pairs) < 3:
            if verbose:
                print(f"    WARNING: Only {len(pairs)} pairs — skipping")
            continue

        if verbose:
            print(f"    {len(pairs)} contrast pairs")

        behavior_results: dict[int, SubspaceResult] = {}

        # Capture all layers simultaneously for efficiency
        cap = LayerActivationCapture(model, layers)

        pos_by_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        neg_by_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

        for i, pair in enumerate(pairs):
            # Positive
            inputs = tokenizer(
                pair["positive"], return_tensors="pt",
                truncation=True, max_length=512,
            ).to(device)
            model(**inputs)
            for layer_idx in layers:
                h = cap.get(layer_idx)  # (1, seq_len, hidden_dim)
                pos_by_layer[layer_idx].append(h[0, -1, :].cpu())

            # Negative
            inputs = tokenizer(
                pair["negative"], return_tensors="pt",
                truncation=True, max_length=512,
            ).to(device)
            model(**inputs)
            for layer_idx in layers:
                h = cap.get(layer_idx)
                neg_by_layer[layer_idx].append(h[0, -1, :].cpu())

            cap.clear()

            if verbose and (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(pairs)} pairs")

        cap.remove()

        # Compute SVD at each layer
        for layer_idx in layers:
            pos_stack = torch.stack(pos_by_layer[layer_idx])  # (n_pairs, hidden_dim)
            neg_stack = torch.stack(neg_by_layer[layer_idx])

            mean_pos = pos_stack.mean(dim=0)
            mean_neg = neg_stack.mean(dim=0)
            steering_vector = mean_pos - mean_neg

            # Difference matrix
            diffs = pos_stack - neg_stack  # (n_pairs, hidden_dim)
            diffs_centered = diffs - diffs.mean(dim=0)

            # Full truncated SVD (on CPU for numerical stability)
            diffs_cpu = diffs_centered.float()
            U, S, Vh = torch.linalg.svd(diffs_cpu, full_matrices=False)

            # Truncate to max_rank
            k = min(max_rank, len(S))
            directions = Vh[:k]  # (k, hidden_dim)
            svals = S[:k].tolist()

            # Cumulative explained variance
            total_var = (S ** 2).sum().item()
            if total_var > 0:
                cumvar = ((S[:k] ** 2).cumsum(0) / total_var).tolist()
            else:
                cumvar = [1.0] * k

            # Effective dimensionality (90% threshold)
            effective_dim = k
            for j, cv in enumerate(cumvar):
                if cv >= 0.90:
                    effective_dim = j + 1
                    break

            # Sign convention: align top-1 direction with mean difference
            if torch.dot(directions[0], steering_vector.float()) < 0:
                directions[0] = -directions[0]

            behavior_results[layer_idx] = SubspaceResult(
                behavior=behavior,
                layer_idx=layer_idx,
                n_pairs=len(pairs),
                directions=directions,
                singular_values=svals,
                explained_variance=cumvar,
                effective_dim=effective_dim,
                mean_pos=mean_pos,
                mean_neg=mean_neg,
                steering_vector=steering_vector,
            )

            if verbose:
                print(
                    f"    Layer {layer_idx:3d}: "
                    f"eff_dim={effective_dim:2d}  "
                    f"||v||={steering_vector.norm():.2f}  "
                    f"top-1 var={cumvar[0]:.1%}"
                )

        results[behavior] = behavior_results

    return results
