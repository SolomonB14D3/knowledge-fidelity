"""Contrastive decoding engine for unlocking hidden model capability.

Contrastive decoding subtracts a weaker "amateur" model's logits from
the expert model's logits at each generation step:

    logits_cd = expert_logits - α × amateur_logits

This removes the diffuse LM prior and amplifies whatever the expert
model has learned beyond the amateur. Proven to rescue hidden capability:
  - d=88 small model: 0.7% → 38.0% at α=0.5 (54× improvement)
  - Qwen2.5-7B base: 40.4% → 67.0% at α=0.5 (approaching 68.4% Instruct)

Works with both MLX (Apple Silicon) and PyTorch backends.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from ..behaviors.metrics import _is_mlx_model


# ── Amateur model auto-detection ──────────────────────────────────────

AMATEUR_MAP = {
    # Qwen2.5 family
    "Qwen/Qwen2.5-7B": "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-14B": "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-32B": "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-72B": "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B": "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    # Llama 3 family
    "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-70B": "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.1-70B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    # Gemma 2 family
    "google/gemma-2-9b": "google/gemma-2-2b",
    "google/gemma-2-9b-it": "google/gemma-2-2b-it",
    "google/gemma-2-27b": "google/gemma-2-2b",
    "google/gemma-2-27b-it": "google/gemma-2-2b-it",
    # SmolLM2 family
    "HuggingFaceTB/SmolLM2-360M": "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M-Instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
}


def detect_amateur(expert_id: str) -> Optional[str]:
    """Auto-detect the best amateur model for a given expert.

    Returns None if no mapping exists (user must specify --amateur).
    """
    # Exact match
    if expert_id in AMATEUR_MAP:
        return AMATEUR_MAP[expert_id]

    # Try stripping leading path for local models
    name = expert_id.rstrip("/").split("/")[-1]
    for key, val in AMATEUR_MAP.items():
        if key.endswith(f"/{name}"):
            return val

    return None


# ── Answer parsing ────────────────────────────────────────────────────

def parse_generated_answer(text: str, n_choices: int = 3) -> Optional[str]:
    """Parse generated text into an answer letter (A/B/C or A/B/C/D).

    Args:
        text: Generated text to parse.
        n_choices: Number of answer choices (3 for rho-eval probes, 4 for MMLU).

    Returns:
        Letter string ("A", "B", etc.) or None if unparseable.
    """
    letters = "ABCD"[:n_choices]
    text = text.strip()
    if not text:
        return None

    # Direct letter start: "A", "A.", "A)", "A\n"
    if text[0] in letters and (len(text) == 1 or text[1] in ".)\n :,"):
        return text[0]

    # Parenthesized: "(A)"
    m = re.search(rf'\(([{letters}])\)', text)
    if m:
        return m.group(1)

    # "answer is A" pattern
    m = re.search(rf'(?:answer\s*(?:is|:)\s*)([{letters}])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # "A. " pattern
    m = re.search(rf'\b([{letters}])\.\s', text)
    if m:
        return m.group(1)

    # Single letter mention
    found = re.findall(rf'\b([{letters}])\b', text)
    if len(found) == 1:
        return found[0]

    return None


# ── MLX contrastive decoding ─────────────────────────────────────────

def _mlx_get_logits(model, tokenizer, prompt: str):
    """Get logits at last token position (MLX)."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens)[None, :]
    logits = model(input_ids)
    mx.eval(logits)
    last_logits = logits[0, -1, :]
    mx.eval(last_logits)
    return last_logits


def _torch_get_logits(model, tokenizer, prompt: str, device: str = "cpu"):
    """Get logits at last token position (PyTorch)."""
    import torch

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits[0, -1, :]  # (vocab_size,)


def get_logits(model, tokenizer, prompt: str, device: str = "cpu"):
    """Get logits at last token position. Auto-dispatches MLX/PyTorch."""
    if _is_mlx_model(model):
        return _mlx_get_logits(model, tokenizer, prompt)
    return _torch_get_logits(model, tokenizer, prompt, device)


# ── Contrastive logit classification ─────────────────────────────────

def contrastive_logit_classify(
    expert_model,
    expert_tokenizer,
    amateur_model,
    amateur_tokenizer,
    prompt: str,
    answer_token_ids: dict[str, int],
    alpha: float = 0.5,
    device: str = "cpu",
) -> tuple[str, dict[str, float]]:
    """Classify by argmax of contrastive logits over answer tokens.

    Args:
        expert_model: The model with hidden knowledge.
        expert_tokenizer: Expert's tokenizer.
        amateur_model: Weaker model whose prior we subtract.
        amateur_tokenizer: Amateur's tokenizer.
        prompt: The formatted prompt text.
        answer_token_ids: Dict mapping letter → token ID (expert tokenizer).
        alpha: Contrastive strength (0 = expert only, higher = more subtraction).
        device: Torch device (ignored for MLX).

    Returns:
        (best_letter, cd_logits_dict) where cd_logits_dict maps letter → CD score.
    """
    expert_logits = get_logits(expert_model, expert_tokenizer, prompt, device)
    amateur_logits = get_logits(amateur_model, amateur_tokenizer, prompt, device)

    # Get amateur answer token IDs (may differ from expert)
    amateur_answer_ids = _get_answer_token_ids(amateur_tokenizer, list(answer_token_ids.keys()))

    cd_scores = {}
    for letter, expert_tid in answer_token_ids.items():
        expert_val = float(expert_logits[expert_tid])
        amateur_tid = amateur_answer_ids.get(letter)
        if amateur_tid is not None:
            amateur_val = float(amateur_logits[amateur_tid])
        else:
            amateur_val = 0.0
        cd_scores[letter] = expert_val - alpha * amateur_val

    best = max(cd_scores, key=cd_scores.get)
    return best, cd_scores


# ── Contrastive autoregressive generation ────────────────────────────

def contrastive_generate(
    expert_model,
    expert_tokenizer,
    amateur_model,
    amateur_tokenizer,
    prompt: str,
    alpha: float = 0.5,
    max_tokens: int = 20,
    device: str = "cpu",
    constrained_first: bool = False,
    answer_token_ids: Optional[dict[str, int]] = None,
) -> tuple[str, Optional[str]]:
    """Full autoregressive contrastive decoding.

    At each step: next_token = argmax(expert_logits - α × amateur_logits)

    Handles vocab size mismatch between expert and amateur by computing
    contrastive logits only over the shared vocab range.

    Args:
        expert_model: The model with hidden knowledge.
        expert_tokenizer: Expert's tokenizer.
        amateur_model: Weaker model.
        amateur_tokenizer: Amateur's tokenizer.
        prompt: Input prompt.
        alpha: Contrastive strength.
        max_tokens: Maximum tokens to generate.
        device: Torch device (ignored for MLX).
        constrained_first: If True, first token restricted to answer choices.
        answer_token_ids: Required if constrained_first=True. Letter → token ID.

    Returns:
        (generated_text, parsed_answer) — parsed_answer may be None.
    """
    if _is_mlx_model(expert_model):
        return _mlx_contrastive_generate(
            expert_model, expert_tokenizer,
            amateur_model, amateur_tokenizer,
            prompt, alpha, max_tokens,
            constrained_first, answer_token_ids,
        )
    else:
        return _torch_contrastive_generate(
            expert_model, expert_tokenizer,
            amateur_model, amateur_tokenizer,
            prompt, alpha, max_tokens, device,
            constrained_first, answer_token_ids,
        )


def _mlx_contrastive_generate(
    expert_model, expert_tokenizer,
    amateur_model, amateur_tokenizer,
    prompt, alpha, max_tokens,
    constrained_first, answer_token_ids,
):
    """MLX implementation of contrastive generation."""
    import mlx.core as mx

    expert_tokens = expert_tokenizer.encode(prompt)
    amateur_tokens = amateur_tokenizer.encode(prompt)

    generated_ids = []

    for step in range(max_tokens):
        # Expert forward
        expert_input = mx.array(expert_tokens + generated_ids)[None, :]
        expert_logits = expert_model(expert_input)
        mx.eval(expert_logits)
        expert_last = expert_logits[0, -1, :]

        # Amateur forward
        amateur_input = mx.array(amateur_tokens + generated_ids)[None, :]
        amateur_logits = amateur_model(amateur_input)
        mx.eval(amateur_logits)
        amateur_last = amateur_logits[0, -1, :]

        # Contrastive logits — handle different vocab sizes
        min_vocab = min(expert_last.shape[0], amateur_last.shape[0])
        cd_common = expert_last[:min_vocab] - alpha * amateur_last[:min_vocab]
        if expert_last.shape[0] > min_vocab:
            cd = mx.concatenate([cd_common, expert_last[min_vocab:]])
        else:
            cd = cd_common
        mx.eval(cd)

        if step == 0 and constrained_first and answer_token_ids:
            # Mask to answer tokens only
            mask = mx.full(cd.shape, float("-inf"))
            for letter, tid in answer_token_ids.items():
                mask = mask.at[tid].add(float("inf") + float(cd[tid]))
            next_token = int(mx.argmax(mask))
        else:
            next_token = int(mx.argmax(cd))

        generated_ids.append(next_token)

        if next_token == expert_tokenizer.eos_token_id:
            break

    generated_text = expert_tokenizer.decode(generated_ids, skip_special_tokens=True)
    answer = parse_generated_answer(generated_text)
    return generated_text, answer


def _torch_contrastive_generate(
    expert_model, expert_tokenizer,
    amateur_model, amateur_tokenizer,
    prompt, alpha, max_tokens, device,
    constrained_first, answer_token_ids,
):
    """PyTorch implementation of contrastive generation."""
    import torch

    expert_ids = expert_tokenizer.encode(prompt, return_tensors="pt").to(device)
    amateur_ids = amateur_tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated_ids = []

    for step in range(max_tokens):
        # Build input with generated tokens appended
        if generated_ids:
            gen_tensor = torch.tensor([generated_ids], device=device)
            expert_input = torch.cat([expert_ids, gen_tensor], dim=1)
            amateur_input = torch.cat([amateur_ids, gen_tensor], dim=1)
        else:
            expert_input = expert_ids
            amateur_input = amateur_ids

        with torch.no_grad():
            expert_logits = expert_model(expert_input).logits[0, -1, :]
            amateur_logits = amateur_model(amateur_input).logits[0, -1, :]

        # Handle vocab mismatch
        min_vocab = min(expert_logits.shape[0], amateur_logits.shape[0])
        cd = expert_logits.clone()
        cd[:min_vocab] = expert_logits[:min_vocab] - alpha * amateur_logits[:min_vocab]

        if step == 0 and constrained_first and answer_token_ids:
            mask = torch.full_like(cd, float("-inf"))
            for letter, tid in answer_token_ids.items():
                mask[tid] = cd[tid]
            next_token = int(torch.argmax(mask))
        else:
            next_token = int(torch.argmax(cd))

        generated_ids.append(next_token)

        if next_token == expert_tokenizer.eos_token_id:
            break

    generated_text = expert_tokenizer.decode(generated_ids, skip_special_tokens=True)
    answer = parse_generated_answer(generated_text)
    return generated_text, answer


# ── Helpers ───────────────────────────────────────────────────────────

def _get_answer_token_ids(tokenizer, letters: list[str]) -> dict[str, int]:
    """Find token IDs for answer letters (A, B, C, ...).

    Tries " A" first (space-prefixed), falls back to "A" alone.
    """
    ids = {}
    for letter in letters:
        tids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(tids) == 1:
            ids[letter] = tids[0]
        else:
            tids = tokenizer.encode(letter, add_special_tokens=False)
            ids[letter] = tids[-1]
    return ids


def get_answer_token_ids(tokenizer, n_choices: int = 3) -> dict[str, int]:
    """Public helper: get answer token IDs for A/B/C (or A/B/C/D).

    Args:
        tokenizer: The model's tokenizer.
        n_choices: Number of answer choices (default: 3 for rho-eval probes).

    Returns:
        Dict mapping letter string → token ID.
    """
    letters = list("ABCD"[:n_choices])
    return _get_answer_token_ids(tokenizer, letters)
