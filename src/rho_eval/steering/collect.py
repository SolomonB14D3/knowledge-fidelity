"""Activation collection for SAE training.

Collects last-token residual stream activations from behavioral contrast
pairs, labeled by behavior and polarity. Follows the activation capture
pattern from interpretability/subspaces.py but stores activations flat
(not grouped by behavior) for efficient SAE training.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..interpretability.activation import LayerActivationCapture, build_contrast_pairs
from ..utils import get_layers
from .schema import ActivationData


def _load_probes_for_behavior(behavior: str) -> list[dict]:
    """Load probes using rho-eval's probe system.

    Uses the legacy API which covers all 4 supported behaviors.
    """
    if behavior == "factual":
        from ..probes import get_all_probes
        return get_all_probes()
    else:
        from ..behavioral import load_behavioral_probes
        return load_behavioral_probes(behavior, seed=42)


@torch.no_grad()
def collect_activations(
    model,
    tokenizer,
    behaviors: list[str],
    layer_idx: int,
    device: str = "cpu",
    max_probes: int | None = None,
    verbose: bool = True,
) -> ActivationData:
    """Collect last-token activations from behavioral contrast pairs.

    For each behavior, loads probes, builds contrast pairs, and runs
    forward passes to capture the residual stream at the specified layer.
    Both positive and negative examples are collected, labeled with their
    behavior and polarity.

    Args:
        model: HuggingFace causal LM (on device, in eval mode).
        tokenizer: Corresponding tokenizer.
        behaviors: Behavior names (e.g., ["factual", "toxicity"]).
        layer_idx: Transformer layer to capture activations from.
        device: Torch device string.
        max_probes: Cap on contrast pairs per behavior (None = use all).
        verbose: Print progress.

    Returns:
        ActivationData with all activations stacked, labeled by behavior
        and polarity.
    """
    model.eval()
    model_name = getattr(model, "name_or_path", getattr(model.config, "_name_or_path", ""))

    # Validate layer index
    n_layers = len(get_layers(model))
    if layer_idx < 0 or layer_idx >= n_layers:
        raise ValueError(
            f"layer_idx={layer_idx} out of range for model with {n_layers} layers"
        )

    if verbose:
        print(f"  Collecting activations at layer {layer_idx} for {behaviors}")

    all_activations = []
    all_labels = []
    all_polarities = []

    cap = LayerActivationCapture(model, [layer_idx])

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

        for i, pair in enumerate(pairs):
            # ── Positive example ─────────────────────────────────
            inputs = tokenizer(
                pair["positive"], return_tensors="pt",
                truncation=True, max_length=512,
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            model(**inputs)
            h = cap.get(layer_idx)  # (1, seq_len, hidden_dim)
            all_activations.append(h[0, -1, :].cpu())
            all_labels.append(behavior)
            all_polarities.append("positive")

            # ── Negative example ─────────────────────────────────
            inputs = tokenizer(
                pair["negative"], return_tensors="pt",
                truncation=True, max_length=512,
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            model(**inputs)
            h = cap.get(layer_idx)
            all_activations.append(h[0, -1, :].cpu())
            all_labels.append(behavior)
            all_polarities.append("negative")

            cap.clear()

            if verbose and (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(pairs)} pairs")

    cap.remove()

    if not all_activations:
        raise RuntimeError("No activations collected — check behaviors and probes")

    activations = torch.stack(all_activations)  # (n_total, hidden_dim)

    if verbose:
        print(f"\n  Collected {activations.shape[0]} activations "
              f"(dim={activations.shape[1]}) from {len(behaviors)} behaviors")

    return ActivationData(
        activations=activations,
        labels=all_labels,
        polarities=all_polarities,
        layer_idx=layer_idx,
        model_name=model_name,
    )
