"""Shared evaluation utilities for behavioral probes.

Extracted from behavioral.py — these functions are used by multiple
behavior plugins and should not be duplicated.

Auto-dispatches to MLX backend when an MLX model is detected:
  - get_mean_logprob() uses MLX inference when model is mlx.nn.Module
  - generate() uses mlx_lm.generate() when model is mlx.nn.Module
  - No behavior plugin changes needed — dispatch is transparent
"""

from __future__ import annotations

import re

import torch
import numpy as np


# ── MLX detection ─────────────────────────────────────────────────────

def _is_mlx_model(model) -> bool:
    """Check if model is an MLX nn.Module (vs PyTorch)."""
    try:
        import mlx.nn
        return isinstance(model, mlx.nn.Module)
    except ImportError:
        return False


# ── MLX implementations ──────────────────────────────────────────────

def _mlx_get_mean_logprob(model, tokenizer, text: str) -> float:
    """MLX implementation of get_mean_logprob."""
    import mlx.core as mx
    import mlx.nn as nn

    tokens = tokenizer.encode(text)
    if len(tokens) > 256:
        tokens = tokens[:256]
    if len(tokens) < 2:
        return 0.0

    input_ids = mx.array(tokens)[None, :]
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    logits = model(inputs)
    ce = nn.losses.cross_entropy(logits, targets).mean()
    mx.eval(ce)

    val = -float(ce)
    return val if np.isfinite(val) else 0.0


def _mlx_generate(
    model, tokenizer, prompt: str, max_new_tokens: int = 50,
) -> str:
    """MLX implementation of generate (greedy decoding)."""
    import mlx.core as mx
    from mlx_lm import generate as mlx_gen

    # Greedy decoding: use argmax sampler (no temperature)
    def greedy_sampler(logits):
        return mx.argmax(logits, axis=-1)

    result = mlx_gen(
        model, tokenizer, prompt=prompt,
        max_tokens=max_new_tokens,
        sampler=greedy_sampler,
        verbose=False,
    )
    return result.strip()


# ── Public API (auto-dispatching) ────────────────────────────────────

def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "cpu",
) -> str:
    """Generate text from a prompt (greedy decoding).

    Automatically uses MLX backend if model is an MLX nn.Module.

    Args:
        model: HuggingFace CausalLM or mlx-lm model.
        tokenizer: Corresponding tokenizer.
        prompt: Input text.
        max_new_tokens: Maximum tokens to generate.
        device: Torch device string (ignored for MLX models).

    Returns:
        Generated continuation (prompt stripped).
    """
    if _is_mlx_model(model):
        return _mlx_generate(model, tokenizer, prompt, max_new_tokens)

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return full_text[len(input_text):].strip()


def get_mean_logprob(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
) -> float:
    """Get mean log-probability (teacher-forced), filtering NaN.

    Automatically uses MLX backend if model is an MLX nn.Module.

    Args:
        model: HuggingFace CausalLM or mlx-lm model.
        tokenizer: Corresponding tokenizer.
        text: Input text to score.
        device: Torch device string (ignored for MLX models).

    Returns:
        Negative cross-entropy loss (higher = model is more confident).
    """
    if _is_mlx_model(model):
        return _mlx_get_mean_logprob(model, tokenizer, text)

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    val = -outputs.loss.item()
    return val if np.isfinite(val) else 0.0


def mann_whitney_auc(positives: list[float], negatives: list[float]) -> float:
    """AUC from Mann-Whitney U statistic.

    Positives should score higher than negatives for AUC > 0.5.

    Args:
        positives: Scores for the positive class.
        negatives: Scores for the negative class.

    Returns:
        AUC in [0, 1].
    """
    correct = 0
    total = 0
    for p in positives:
        for n in negatives:
            total += 1
            if p > n:
                correct += 1
            elif p == n:
                correct += 0.5
    return correct / total if total > 0 else 0.5


def check_numeric(generated: str, target: str) -> bool:
    """Check if generated text contains the target numeric answer.

    Tries #### delimited format first (GSM8K style), then falls back
    to the last number in the generated text.

    Args:
        generated: Model output text.
        target: Expected numeric answer as a string.

    Returns:
        True if the extracted number matches target.
    """
    match = re.search(r'####\s*(-?\d[\d,]*)', generated)
    if match:
        extracted = match.group(1).replace(",", "")
    else:
        numbers = re.findall(r'\b(-?\d[\d,]*)\b', generated)
        extracted = numbers[-1].replace(",", "") if numbers else ""

    try:
        return int(extracted) == int(target)
    except (ValueError, TypeError):
        return False
