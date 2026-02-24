"""Differentiable proxy losses for behavioral alignment.

These losses mirror what rho measures — the confidence gap between positive
(desired) and negative (undesired) text — but are fully differentiable,
enabling gradient-based optimization during SFT.

The contrastive margin loss is:

    L = ReLU(CE(positive) - CE(negative) + margin)

where CE(text) = model(**tokenize(text), labels=labels).loss (teacher-forced
cross-entropy). This is 0 when the model is already margin-more-confident
on positive text, and penalizes the model when it prefers the negative text.

Contrast with the non-differentiable rho pipeline:
- `get_mean_logprob()` in behaviors/metrics.py uses `@torch.no_grad()` and `.item()`
- `analyze_confidence()` in cartography/engine.py uses `@torch.no_grad()`
- `spearmanr()` and `mann_whitney_auc()` are rank/counting statistics

Here we keep everything as PyTorch tensors with grad_fn attached.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np


def differentiable_ce_loss(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
    max_length: int = 256,
) -> torch.Tensor:
    """Compute teacher-forced CE loss on text, with gradients.

    This is the differentiable counterpart of `get_mean_logprob()` from
    `behaviors/metrics.py`. The key differences:
      - No `@torch.no_grad()` — gradients flow through
      - Returns the raw loss tensor, not `float(-loss.item())`

    Args:
        model: HuggingFace CausalLM (must have requires_grad on some params).
        tokenizer: Corresponding tokenizer.
        text: Input text to score.
        device: Torch device string.
        max_length: Maximum token length.

    Returns:
        Scalar tensor: mean CE loss (lower = model is more confident).
        Has grad_fn if model has trainable parameters.
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length,
    )
    # Move tensors to device (works with both BatchEncoding and plain dict)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    # Guard against NaN (same as get_mean_logprob but keep tensor form)
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=loss.device, requires_grad=True)

    return loss


def contrastive_confidence_loss(
    model,
    tokenizer,
    positive_text: str,
    negative_text: str,
    margin: float = 0.1,
    device: str = "cpu",
    max_length: int = 256,
) -> torch.Tensor:
    """Contrastive margin loss on a single positive/negative pair.

    Computes:
        L = ReLU(CE(positive) - CE(negative) + margin)

    This is 0 when the model is already margin-more-confident on the
    positive text (CE_pos < CE_neg - margin). When the model prefers
    the negative text, L > 0 and the gradient pushes the model toward
    being more confident on the positive text.

    Args:
        model: HuggingFace CausalLM.
        tokenizer: Corresponding tokenizer.
        positive_text: Desired text (should have lower CE loss).
        negative_text: Undesired text (should have higher CE loss).
        margin: Minimum desired confidence gap (in CE loss units).
        device: Torch device string.
        max_length: Maximum token length per text.

    Returns:
        Scalar tensor: contrastive loss (0 if already well-separated).
    """
    ce_pos = differentiable_ce_loss(model, tokenizer, positive_text, device, max_length)
    ce_neg = differentiable_ce_loss(model, tokenizer, negative_text, device, max_length)

    return F.relu(ce_pos - ce_neg + margin)


def rho_auxiliary_loss(
    model,
    tokenizer,
    pairs: list[dict],
    margin: float = 0.1,
    device: str = "cpu",
    max_length: int = 256,
) -> torch.Tensor:
    """Mean contrastive loss over a batch of behavioral contrast pairs.

    Each pair is a dict with 'positive' and 'negative' keys (as produced
    by `build_contrast_pairs()` from `interpretability/activation.py`).

    Args:
        model: HuggingFace CausalLM.
        tokenizer: Corresponding tokenizer.
        pairs: List of dicts with 'positive' and 'negative' text keys.
        margin: Minimum desired confidence gap per pair.
        device: Torch device string.
        max_length: Maximum token length per text.

    Returns:
        Scalar tensor: mean contrastive loss over all pairs.
        Returns 0.0 tensor if pairs is empty.
    """
    if not pairs:
        return torch.tensor(0.0, device=device, requires_grad=True)

    losses = []
    for pair in pairs:
        loss = contrastive_confidence_loss(
            model, tokenizer,
            pair["positive"], pair["negative"],
            margin=margin, device=device, max_length=max_length,
        )
        losses.append(loss)

    return torch.stack(losses).mean()
