"""Differentiable proxy losses for behavioral alignment — MLX backend.

MLX equivalents of the loss functions in losses.py. These are designed
to work with mlx-lm models and Apple Silicon unified memory.

Key differences from PyTorch version:
  - No `device` parameter (MLX uses unified memory automatically)
  - No `.backward()` — gradients computed via `nn.value_and_grad()` tracing
  - Must not call `mx.eval()` prematurely (would break gradient graph)
  - Uses `mx.maximum()` instead of `F.relu()`

The contrastive margin loss is identical in logic:

    L = max(0, CE(positive) - CE(negative) + margin)

Usage:
    from rho_eval.alignment.mlx_losses import mlx_rho_auxiliary_loss

    # Inside a value_and_grad-traced function:
    rho_loss = mlx_rho_auxiliary_loss(model, tokenizer, pairs, margin=0.1)
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def mlx_ce_loss(
    model,
    tokenizer,
    text: str,
    max_length: int = 256,
) -> mx.array:
    """Compute teacher-forced CE loss on text, with gradients.

    MLX equivalent of `differentiable_ce_loss()` from losses.py.
    Returns a scalar mx.array that participates in the gradient graph.

    Args:
        model: mlx-lm model (nn.Module).
        tokenizer: mlx-lm TokenizerWrapper or HF tokenizer.
        text: Input text to score.
        max_length: Maximum token length.

    Returns:
        Scalar mx.array: mean CE loss (lower = model is more confident).
    """
    # Tokenize — use .encode() which returns a plain list of ints
    tokens = tokenizer.encode(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]

    if len(tokens) < 2:
        return mx.array(0.0)

    input_ids = mx.array(tokens)[None, :]  # (1, seq_len)

    # Shift: predict next token from each position
    inputs = input_ids[:, :-1]     # (1, seq_len-1)
    targets = input_ids[:, 1:]     # (1, seq_len-1)

    # Forward pass — this is differentiable through LoRA params
    logits = model(inputs)         # (1, seq_len-1, vocab_size)

    # Per-token cross-entropy (reduction='none' is default in MLX)
    ce_per_token = nn.losses.cross_entropy(logits, targets)  # (1, seq_len-1)

    # Mean over all tokens
    loss = ce_per_token.mean()

    return loss


def mlx_contrastive_confidence_loss(
    model,
    tokenizer,
    positive_text: str,
    negative_text: str,
    margin: float = 0.1,
    max_length: int = 256,
) -> mx.array:
    """Contrastive margin loss on a single positive/negative pair.

    Computes:
        L = max(0, CE(positive) - CE(negative) + margin)

    MLX equivalent of `contrastive_confidence_loss()` from losses.py.

    Args:
        model: mlx-lm model.
        tokenizer: Tokenizer.
        positive_text: Desired text (should have lower CE loss).
        negative_text: Undesired text (should have higher CE loss).
        margin: Minimum desired confidence gap (in CE loss units).
        max_length: Maximum token length per text.

    Returns:
        Scalar mx.array: contrastive loss (0 if already well-separated).
    """
    ce_pos = mlx_ce_loss(model, tokenizer, positive_text, max_length)
    ce_neg = mlx_ce_loss(model, tokenizer, negative_text, max_length)

    return mx.maximum(ce_pos - ce_neg + margin, mx.array(0.0))


def mlx_rho_auxiliary_loss(
    model,
    tokenizer,
    pairs: list[dict],
    margin: float = 0.1,
    max_length: int = 256,
) -> mx.array:
    """Mean contrastive loss over a batch of behavioral contrast pairs.

    MLX equivalent of `rho_auxiliary_loss()` from losses.py.

    Each pair is a dict with 'positive' and 'negative' keys.

    Args:
        model: mlx-lm model.
        tokenizer: Tokenizer.
        pairs: List of dicts with 'positive' and 'negative' text keys.
        margin: Minimum desired confidence gap per pair.
        max_length: Maximum token length per text.

    Returns:
        Scalar mx.array: mean contrastive loss over all pairs.
    """
    if not pairs:
        return mx.array(0.0)

    losses = []
    for pair in pairs:
        loss = mlx_contrastive_confidence_loss(
            model, tokenizer,
            pair["positive"], pair["negative"],
            margin=margin, max_length=max_length,
        )
        losses.append(loss)

    return mx.mean(mx.stack(losses))


# ── Bridge cosine loss ─────────────────────────────────────────────────


def _mlx_extract_hidden_states(
    model,
    tokenizer,
    text: str,
    max_length: int = 256,
) -> mx.array:
    """Extract mean-pooled hidden states from the model's last transformer layer.

    Forwards text through the model backbone (all transformer layers + final
    layer norm), stopping before the LM head. Mean-pools over the sequence
    dimension.

    This is differentiable through LoRA parameters when called inside
    ``nn.value_and_grad()``.

    Args:
        model: mlx-lm model (expects ``model.model`` as the transformer
            backbone, which is standard for all mlx-lm architectures).
        tokenizer: Tokenizer with ``.encode()`` method.
        text: Input text.
        max_length: Maximum token length.

    Returns:
        mx.array of shape ``(hidden_dim,)``: mean-pooled hidden state.
    """
    tokens = tokenizer.encode(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    if len(tokens) < 1:
        # Infer hidden dim from embedding layer
        hidden_dim = model.model.embed_tokens.weight.shape[1]
        return mx.zeros((hidden_dim,))

    input_ids = mx.array(tokens)[None, :]  # (1, seq_len)

    # Forward through transformer backbone (embed → layers → norm),
    # stopping before the LM head. Gradient flows through LoRA params.
    hidden = model.model(input_ids)  # (1, seq_len, hidden_dim)

    # Mean pool over sequence length
    return hidden.mean(axis=1).squeeze(0)  # (hidden_dim,)


def mlx_bridge_cosine_loss(
    model,
    tokenizer,
    pairs: list[dict],
    max_length: int = 256,
) -> mx.array:
    """Mean (1 − cosine_sim) of hidden states for cross-behavior text pairs.

    For each pair, forwards both texts through the model backbone,
    mean-pools hidden states, and computes cosine similarity. Returns
    ``mean(1 − cos_sim)`` over all pairs.

    Used to encourage similar hidden representations for semantically
    related probes from different behavioral dimensions ("bridge
    strengthening").

    Args:
        model: mlx-lm model.
        tokenizer: Tokenizer.
        pairs: List of dicts with ``'text_a'`` and ``'text_b'`` keys.
        max_length: Maximum token length per text.

    Returns:
        Scalar mx.array: mean bridge loss (0 = perfectly similar states).
    """
    if not pairs:
        return mx.array(0.0)

    losses = []
    for pair in pairs:
        h_a = _mlx_extract_hidden_states(
            model, tokenizer, pair["text_a"], max_length,
        )
        h_b = _mlx_extract_hidden_states(
            model, tokenizer, pair["text_b"], max_length,
        )

        # Cosine similarity with epsilon for numerical stability
        dot = mx.sum(h_a * h_b)
        norm_a = mx.sqrt(mx.sum(h_a * h_a) + 1e-8)
        norm_b = mx.sqrt(mx.sum(h_b * h_b) + 1e-8)
        cos_sim = dot / (norm_a * norm_b)

        losses.append(1.0 - cos_sim)

    return mx.mean(mx.stack(losses))
