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


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION METRICS (ECE, Brier, per-probe confidence)
# ═══════════════════════════════════════════════════════════════════════════

def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Expected Calibration Error — measures how well-calibrated predictions are.

    Bins predictions by confidence and computes the weighted average gap
    between confidence and accuracy in each bin. A perfectly calibrated model
    has ECE = 0: when it says 80% confident, it's correct 80% of the time.

    Args:
        confidences: Array of predicted probabilities in [0, 1].
        accuracies: Array of binary correctness labels (0 or 1).
        n_bins: Number of equal-width bins (default: 10).

    Returns:
        Dict with:
            ece: Expected Calibration Error (lower = better).
            bin_edges: Array of bin boundaries.
            bin_accs: Per-bin accuracy (NaN for empty bins).
            bin_confs: Per-bin mean confidence (NaN for empty bins).
            bin_counts: Number of samples per bin.
    """
    confidences = np.asarray(confidences, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.full(n_bins, np.nan)
    bin_confs = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        count = mask.sum()
        bin_counts[i] = count

        if count > 0:
            bin_accs[i] = accuracies[mask].mean()
            bin_confs[i] = confidences[mask].mean()
            ece += (count / n) * abs(bin_accs[i] - bin_confs[i])

    return {
        "ece": float(ece),
        "bin_edges": bin_edges.tolist(),
        "bin_accs": [float(x) if np.isfinite(x) else None for x in bin_accs],
        "bin_confs": [float(x) if np.isfinite(x) else None for x in bin_confs],
        "bin_counts": bin_counts.tolist(),
    }


def compute_brier(
    confidences: np.ndarray,
    accuracies: np.ndarray,
) -> float:
    """Brier score — mean squared error of probabilistic predictions.

    Brier = mean((p - y)^2) where p is predicted probability and y is {0, 1}.
    Perfect = 0, worst = 1, random = 0.25.

    Args:
        confidences: Array of predicted probabilities in [0, 1].
        accuracies: Array of binary correctness labels (0 or 1).

    Returns:
        Brier score (lower = better).
    """
    confidences = np.asarray(confidences, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)
    return float(np.mean((confidences - accuracies) ** 2))


def score_probe_pairs(
    model,
    tokenizer,
    pairs: list[dict],
    device: str = "cpu",
) -> list[dict]:
    """Score contrast pairs to get per-probe confidence and correctness.

    For each pair with "positive" and "negative" texts:
      - Computes logprob for each
      - Derives confidence via softmax: p = exp(lp_pos) / (exp(lp_pos) + exp(lp_neg))
      - Correctness = 1 if logprob_pos > logprob_neg else 0

    Args:
        model: HuggingFace CausalLM or mlx-lm model.
        tokenizer: Corresponding tokenizer.
        pairs: List of dicts with "positive" and "negative" keys.
        device: Torch device string (ignored for MLX).

    Returns:
        List of dicts with keys: id, lp_pos, lp_neg, confidence, correct, gap.
    """
    results = []
    for pair in pairs:
        lp_pos = get_mean_logprob(model, tokenizer, pair["positive"], device=device)
        lp_neg = get_mean_logprob(model, tokenizer, pair["negative"], device=device)

        # Convert logprob gap to [0,1] confidence via sigmoid
        gap = lp_pos - lp_neg
        confidence = 1.0 / (1.0 + np.exp(-gap))  # sigmoid(gap)

        results.append({
            "id": pair.get("id", ""),
            "lp_pos": float(lp_pos),
            "lp_neg": float(lp_neg),
            "gap": float(gap),
            "confidence": float(confidence),
            "correct": int(lp_pos > lp_neg),
        })

    return results


def calibration_metrics(
    model,
    tokenizer,
    behaviors: list[str] | None = None,
    device: str = "cpu",
    seed: int = 42,
    n_bins: int = 10,
) -> dict:
    """Compute ECE and Brier score per behavior using probe pairs.

    Loads probes for each behavior, builds contrast pairs, scores them,
    and computes calibration metrics. Works with both PyTorch and MLX models.

    Args:
        model: HuggingFace CausalLM or mlx-lm model.
        tokenizer: Corresponding tokenizer.
        behaviors: List of behavior names (default: factual, toxicity, sycophancy, bias).
        device: Torch device string (ignored for MLX).
        seed: Random seed for probe loading.
        n_bins: Number of ECE bins.

    Returns:
        Dict with per-behavior metrics:
            {behavior: {ece, brier, accuracy, mean_confidence, mean_gap, n_probes, details}}
    """
    from .base import ABCBehavior
    from . import get_behavior
    from ..interpretability.activation import build_contrast_pairs

    if behaviors is None:
        behaviors = ["factual", "toxicity", "sycophancy", "bias"]

    results = {}
    for bname in behaviors:
        try:
            beh = get_behavior(bname)
            probes = beh.load_probes(seed=seed)
            pairs = build_contrast_pairs(bname, probes)
        except Exception as e:
            results[bname] = {"error": str(e)}
            continue

        if not pairs:
            results[bname] = {"error": "no pairs"}
            continue

        # Score all pairs
        scored = score_probe_pairs(model, tokenizer, pairs, device=device)

        # Extract arrays
        confs = np.array([s["confidence"] for s in scored])
        accs = np.array([s["correct"] for s in scored])
        gaps = np.array([s["gap"] for s in scored])

        # Compute metrics
        ece_result = compute_ece(confs, accs, n_bins=n_bins)
        brier = compute_brier(confs, accs)

        results[bname] = {
            "ece": ece_result["ece"],
            "brier": brier,
            "accuracy": float(accs.mean()),
            "mean_confidence": float(confs.mean()),
            "mean_gap": float(gaps.mean()),
            "n_probes": len(scored),
            "ece_bins": ece_result,
            "details": scored,
        }

    return results


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
