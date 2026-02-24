"""Shared evaluation utilities for behavioral probes.

Extracted from behavioral.py — these functions are used by multiple
behavior plugins and should not be duplicated.
"""

from __future__ import annotations

import re

import torch
import numpy as np


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "cpu",
) -> str:
    """Generate text from a prompt (greedy decoding).

    Args:
        model: HuggingFace CausalLM.
        tokenizer: Corresponding tokenizer.
        prompt: Input text.
        max_new_tokens: Maximum tokens to generate.
        device: Torch device string.

    Returns:
        Generated continuation (prompt stripped).
    """
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

    Args:
        model: HuggingFace CausalLM.
        tokenizer: Corresponding tokenizer.
        text: Input text to score.
        device: Torch device string.

    Returns:
        Negative cross-entropy loss (higher = model is more confident).
    """
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
