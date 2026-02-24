"""Per-head attribution analysis for behavioral traits.

Determines which attention heads contribute most to each behavioral
dimension by measuring per-head activation differences between positive
and negative probe texts.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..utils import get_layers, get_attention
from .activation import HeadOutputCapture, build_contrast_pairs
from .schema import HeadImportance


def _get_head_config(model) -> tuple[int, int]:
    """Extract n_heads and head_dim from model config.

    Returns:
        (n_heads, head_dim) tuple.
    """
    config = model.config

    # Try standard config attributes
    n_heads = getattr(config, "num_attention_heads", None)
    hidden_size = getattr(config, "hidden_size", None)

    if n_heads is None:
        raise ValueError(
            f"Cannot determine n_heads from config: {type(config).__name__}. "
            "Expected 'num_attention_heads' attribute."
        )
    if hidden_size is None:
        raise ValueError(
            f"Cannot determine hidden_size from config: {type(config).__name__}."
        )

    head_dim = hidden_size // n_heads
    return n_heads, head_dim


@torch.no_grad()
def head_attribution(
    model,
    tokenizer,
    behaviors: list[str],
    layers: list[int] | None = None,
    device: str = "cpu",
    max_probes: int | None = None,
    verbose: bool = True,
) -> dict[str, list[HeadImportance]]:
    """Score each attention head's contribution to each behavior.

    For each (behavior, layer, head):
    1. Run all positive probes, capture per-head last-token output.
    2. Run all negative probes, capture per-head last-token output.
    3. Head importance = ||mean_pos_head - mean_neg_head|| / ||mean_pos_layer - mean_neg_layer||.

    A head with importance > 1/n_heads is "over-represented" for that
    behavior (contributes more than its proportional share).

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        behaviors: List of behavior names.
        layers: Layer indices (None = auto-select).
        device: Torch device.
        max_probes: Cap on probes per behavior.
        verbose: Print progress.

    Returns:
        Dict mapping behavior → list of HeadImportance (sorted by layer, head).
    """
    from .subspaces import _auto_layers, _load_probes_for_behavior

    if layers is None:
        n_layers = len(get_layers(model))
        layers = _auto_layers(n_layers)

    n_heads, head_dim = _get_head_config(model)

    if verbose:
        print(f"\n  Head attribution: {behaviors}, layers={layers}, "
              f"n_heads={n_heads}, head_dim={head_dim}")

    results: dict[str, list[HeadImportance]] = {}

    for behavior in behaviors:
        if verbose:
            print(f"\n  [{behavior}]")

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

        behavior_heads: list[HeadImportance] = []

        for layer_idx in layers:
            cap = HeadOutputCapture(model, layer_idx, n_heads, head_dim)

            pos_heads = []  # list of (n_heads, head_dim) tensors
            neg_heads = []

            for pair in pairs:
                # Positive
                inputs = tokenizer(
                    pair["positive"], return_tensors="pt",
                    truncation=True, max_length=512,
                ).to(device)
                model(**inputs)
                h = cap.get()  # (1, n_heads, seq_len, head_dim)
                pos_heads.append(h[0, :, -1, :].cpu())  # (n_heads, head_dim)

                # Negative
                inputs = tokenizer(
                    pair["negative"], return_tensors="pt",
                    truncation=True, max_length=512,
                ).to(device)
                model(**inputs)
                h = cap.get()
                neg_heads.append(h[0, :, -1, :].cpu())

            cap.remove()

            # Stack: (n_pairs, n_heads, head_dim)
            pos_stack = torch.stack(pos_heads)
            neg_stack = torch.stack(neg_heads)

            # Per-head mean difference: (n_heads, head_dim)
            mean_pos_per_head = pos_stack.mean(dim=0)
            mean_neg_per_head = neg_stack.mean(dim=0)
            diff_per_head = mean_pos_per_head - mean_neg_per_head  # (n_heads, head_dim)

            # Full-layer mean difference norm (for normalization)
            # Sum across heads and compute norm
            full_diff = diff_per_head.sum(dim=0)  # (head_dim,) — NOT correct for normalization
            # Actually use the norm of the concatenated per-head diffs
            full_diff_norm = diff_per_head.reshape(-1).norm().item()

            if full_diff_norm < 1e-10:
                # No behavioral signal at this layer
                for head_idx in range(n_heads):
                    behavior_heads.append(HeadImportance(
                        behavior=behavior,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        importance_score=1.0 / n_heads,  # uniform
                        n_heads=n_heads,
                    ))
                continue

            # Per-head importance
            for head_idx in range(n_heads):
                head_norm = diff_per_head[head_idx].norm().item()
                importance = head_norm / full_diff_norm
                behavior_heads.append(HeadImportance(
                    behavior=behavior,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    importance_score=importance,
                    n_heads=n_heads,
                ))

            if verbose:
                # Show top-3 heads at this layer
                layer_heads = [
                    h for h in behavior_heads
                    if h.layer_idx == layer_idx
                ]
                top3 = sorted(layer_heads, key=lambda h: h.importance_score, reverse=True)[:3]
                top3_str = ", ".join(
                    f"h{h.head_idx}={h.importance_score:.3f}" for h in top3
                )
                print(f"    Layer {layer_idx:3d}: top-3 heads: {top3_str}")

        results[behavior] = behavior_heads

    return results
