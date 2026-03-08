"""Snap-On Communication Module — Inference.

Generation and evaluation utilities for snap-on adapters on frozen base models.
"""

import json
import os
import time

import mlx.core as mx
import mlx.nn as nn

from .training import ALPACA_TEMPLATE, get_lm_head


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_with_adapter(base_model, adapter, tokenizer, prompt: str,
                          max_tokens: int = 150, temperature: float = 0.0,
                          mode: str = "hidden"):
    """Autoregressive generation using base model + adapter.

    Args:
        base_model: Frozen MLX language model.
        adapter: SnapOnMLP, SnapOnLogitMLP, or SnapOnTransformer adapter.
        tokenizer: Tokenizer for encoding/decoding.
        prompt: Full prompt string (should include Alpaca template).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        mode: "hidden" or "logit".

    Returns:
        Generated text (response only, not including prompt).
    """
    tokens = tokenizer.encode(prompt)
    generated = []
    logit_mode = (mode == "logit")
    lm_head = get_lm_head(base_model)

    for _ in range(max_tokens):
        input_ids = mx.array(tokens)[None, :]
        h = base_model.model(input_ids)
        mx.eval(h)
        if logit_mode:
            base_logits = lm_head(h)
            mx.eval(base_logits)
            logits = base_logits + adapter(base_logits, h=h)
        else:
            adjustment = adapter(h)
            logits = lm_head(h + adjustment)
        mx.eval(logits)

        last_logits = logits[0, -1, :]
        if temperature > 0:
            probs = mx.softmax(last_logits / temperature)
            next_token = int(mx.random.categorical(mx.log(probs)))
        else:
            next_token = int(mx.argmax(last_logits))

        eos_id = getattr(tokenizer, "eos_token_id", None)
        if next_token == eos_id:
            break
        tokens.append(next_token)
        generated.append(next_token)

    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_base_only(base_model, tokenizer, prompt: str,
                       max_tokens: int = 150, temperature: float = 0.0):
    """Autoregressive generation with base model only (no adapter).

    Args:
        base_model: MLX language model.
        tokenizer: Tokenizer for encoding/decoding.
        prompt: Full prompt string.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).

    Returns:
        Generated text (response only).
    """
    tokens = tokenizer.encode(prompt)
    generated = []

    for _ in range(max_tokens):
        input_ids = mx.array(tokens)[None, :]
        logits = base_model(input_ids)
        mx.eval(logits)

        last_logits = logits[0, -1, :]
        if temperature > 0:
            probs = mx.softmax(last_logits / temperature)
            next_token = int(mx.random.categorical(mx.log(probs)))
        else:
            next_token = int(mx.argmax(last_logits))

        eos_id = getattr(tokenizer, "eos_token_id", None)
        if next_token == eos_id:
            break
        tokens.append(next_token)
        generated.append(next_token)

    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# MMLU Evaluation
# ---------------------------------------------------------------------------

def evaluate_mmlu(base_model, adapter, tokenizer, n_questions: int = 200,
                  mode: str = "hidden"):
    """Quick MMLU logit accuracy with and without adapter.

    Compares base model argmax accuracy to adapter-augmented accuracy on
    MMLU multiple-choice questions.

    Args:
        base_model: Frozen MLX language model.
        adapter: Snap-on adapter module.
        tokenizer: Tokenizer.
        n_questions: Number of MMLU questions to evaluate.
        mode: "hidden" or "logit".

    Returns:
        (base_acc, adapter_acc) as floats, or (None, None) if rho_eval.unlock
        is not available.
    """
    try:
        from rho_eval.unlock.expression_gap import _load_mmlu, _format_mmlu_prompt
        from rho_eval.unlock.contrastive import get_answer_token_ids
    except ImportError:
        print("  (skipping MMLU eval — rho_eval.unlock not available)")
        return None, None

    logit_mode = (mode == "logit")
    print(f"\nEvaluating MMLU ({n_questions} questions, mode={mode})...")
    questions = _load_mmlu(n=n_questions, seed=42)
    answer_ids_dict = get_answer_token_ids(tokenizer, n_choices=4)
    letters = "ABCD"
    answer_id_list = [answer_ids_dict[l] for l in letters]

    correct_base = 0
    correct_adapter = 0
    lm_head = get_lm_head(base_model)

    for i, q in enumerate(questions):
        prompt = _format_mmlu_prompt(tokenizer, q)
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]

        # Base model hidden states
        h = base_model.model(input_ids)
        mx.eval(h)

        # Base model logits
        base_logits = lm_head(h)
        mx.eval(base_logits)
        base_last = base_logits[0, -1, :]

        # Adapter logits
        if logit_mode:
            adapter_logits = base_logits + adapter(base_logits, h=h)
        else:
            adjustment = adapter(h)
            adapter_logits = lm_head(h + adjustment)
        mx.eval(adapter_logits)
        adapter_last = adapter_logits[0, -1, :]

        # Check predictions
        answer_idx = q["answer_idx"]
        correct_letter = letters[answer_idx]

        base_pred = max(range(4), key=lambda j: float(base_last[answer_id_list[j]]))
        adapter_pred = max(range(4), key=lambda j: float(adapter_last[answer_id_list[j]]))

        if letters[base_pred] == correct_letter:
            correct_base += 1
        if letters[adapter_pred] == correct_letter:
            correct_adapter += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_questions}: base={correct_base/(i+1):.1%}, "
                  f"adapter={correct_adapter/(i+1):.1%}", flush=True)

    base_acc = correct_base / n_questions
    adapter_acc = correct_adapter / n_questions
    print(f"\nMMLU Results ({n_questions} questions):")
    print(f"  Base model logit acc:    {base_acc:.1%}")
    print(f"  With adapter logit acc:  {adapter_acc:.1%}")
    print(f"  Delta:                   {adapter_acc - base_acc:+.1%}")
    return base_acc, adapter_acc
