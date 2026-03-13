#!/usr/bin/env python3
"""Manifold Microscope — compare probability shifts between adapter architectures.

Compares Base vs Phase 1 (hidden-mode) vs Phase 3 (logit-mode) adapters
on MMLU questions to see what each architecture sacrifices.

Two signatures we're looking for:
  1. Victims (Factual Tokens): Where Phase 1 suppresses correct answer prob
     but Phase 3 preserves it.
  2. Parasites (Formatting Tokens): Where Phase 1 steals probability mass
     for formatting tokens (\n, "Step", "Here", etc.)

Usage:
    python experiments/snap_on/logit_shift_analyzer.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import mlx.core as mx
import mlx.nn as nn
import mlx_lm

from train import load_adapter, ALPACA_TEMPLATE
from rho_eval.unlock.expression_gap import _load_mmlu, _format_mmlu_prompt
from rho_eval.unlock.contrastive import get_answer_token_ids


def softmax(logits):
    """Stable softmax in MLX."""
    shifted = logits - mx.max(logits, axis=-1, keepdims=True)
    exp = mx.exp(shifted)
    return exp / mx.sum(exp, axis=-1, keepdims=True)


def analyze_one_question(base_model, adapter_p1, adapter_p3, tokenizer,
                         prompt, correct_token_id, top_k=10):
    """Analyze logit shifts for a single prompt."""
    input_ids = mx.array(tokenizer.encode(prompt))[None, :]

    # 1. Base model hidden states
    h = base_model.model(input_ids)
    mx.eval(h)
    h_last = h[:, -1:, :]  # Last position only

    # 2. Base logits (the immutable truth)
    base_logits = base_model.lm_head(h_last)
    mx.eval(base_logits)
    base_probs = softmax(base_logits[0, 0, :])
    mx.eval(base_probs)

    # 3. Phase 1: hidden perturbation — lm_head(h + adapter(h))
    adj_p1 = adapter_p1(h_last)
    p1_logits = base_model.lm_head(h_last + adj_p1)
    mx.eval(p1_logits)
    p1_probs = softmax(p1_logits[0, 0, :])
    mx.eval(p1_probs)

    # 4. Phase 3: logit perturbation — lm_head(h) + adapter(lm_head(h))
    adj_p3 = adapter_p3(base_logits)
    p3_logits = base_logits + adj_p3
    mx.eval(p3_logits)
    p3_probs = softmax(p3_logits[0, 0, :])
    mx.eval(p3_probs)

    # Deltas (cast to float32 for numpy — MLX bfloat16 can't convert directly)
    delta_p1 = np.array(p1_probs.astype(mx.float32)) - np.array(base_probs.astype(mx.float32))
    delta_p3 = np.array(p3_probs.astype(mx.float32)) - np.array(base_probs.astype(mx.float32))
    base_np = np.array(base_probs.astype(mx.float32))

    return base_np, delta_p1, delta_p3


def print_analysis(base_np, delta_p1, delta_p3, tokenizer,
                   correct_token_id, top_k=10):
    """Print formatted analysis for one question."""
    print(f"{'Token':<20} | {'Base Prob':<10} | {'Ph1 D (Hidden)':<16} | {'Ph3 D (Logit)':<16}")
    print("-" * 70)

    # Target token (correct MMLU answer)
    token_str = repr(tokenizer.decode([correct_token_id]))
    b = base_np[correct_token_id]
    d1 = delta_p1[correct_token_id]
    d3 = delta_p3[correct_token_id]
    print(f"TARGET: {token_str:<11} | {b:.6f}   | {d1:+.6f}         | {d3:+.6f}")
    print("-" * 70)

    # Top Phase 1 boosts (formatting parasites?)
    top_boost = np.argsort(-delta_p1)[:top_k]
    print("Top Phase 1 Boosts (what did the hidden adapter promote?):")
    for idx in top_boost:
        token_str = repr(tokenizer.decode([int(idx)]))
        print(f"  {token_str:<18} | {base_np[idx]:.6f}   | {delta_p1[idx]:+.6f}         | {delta_p3[idx]:+.6f}")

    print("-" * 70)

    # Top Phase 1 drops (knowledge victims?)
    top_drop = np.argsort(delta_p1)[:top_k]
    print("Top Phase 1 Drops (what knowledge was sacrificed?):")
    for idx in top_drop:
        token_str = repr(tokenizer.decode([int(idx)]))
        print(f"  {token_str:<18} | {base_np[idx]:.6f}   | {delta_p1[idx]:+.6f}         | {delta_p3[idx]:+.6f}")

    print("-" * 70)

    # Top Phase 3 boosts for comparison
    top_p3_boost = np.argsort(-delta_p3)[:top_k]
    print("Top Phase 3 Boosts (what did the logit adapter promote?):")
    for idx in top_p3_boost:
        token_str = repr(tokenizer.decode([int(idx)]))
        print(f"  {token_str:<18} | {base_np[idx]:.6f}   | {delta_p1[idx]:+.6f}         | {delta_p3[idx]:+.6f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manifold Microscope")
    parser.add_argument("--p1_dir", default="results/snap_on/phase1_mlp",
                        help="Phase 1 (hidden-mode) adapter directory")
    parser.add_argument("--p3_dir", default="results/snap_on/phase3_logit",
                        help="Phase 3 (logit-mode) adapter directory")
    parser.add_argument("--n_questions", type=int, default=20,
                        help="Number of MMLU questions to analyze")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output", default="results/snap_on/manifold_microscope.json")
    args = parser.parse_args()

    print("Loading base model...")
    base_model, tokenizer = mlx_lm.load("Qwen/Qwen2.5-7B")
    base_model.freeze()

    print(f"Loading Phase 1 adapter from {args.p1_dir}...")
    adapter_p1 = load_adapter(args.p1_dir, "best")

    print(f"Loading Phase 3 adapter from {args.p3_dir}...")
    adapter_p3 = load_adapter(args.p3_dir, "best")

    # Get MMLU questions
    print(f"\nLoading {args.n_questions} MMLU questions...")
    questions = _load_mmlu(n=args.n_questions, seed=42)
    answer_ids = get_answer_token_ids(tokenizer, n_choices=4)
    letters = "ABCD"
    answer_id_list = [answer_ids[l] for l in letters]

    # Aggregate statistics
    agg = {
        "target_delta_p1": [],
        "target_delta_p3": [],
        "base_correct": 0,
        "p1_correct": 0,
        "p3_correct": 0,
        "p1_top_boosts": {},   # token -> cumulative boost
        "p1_top_drops": {},    # token -> cumulative drop
        "p3_top_boosts": {},
        "n_questions": 0,
    }

    for i, q in enumerate(questions):
        prompt = _format_mmlu_prompt(tokenizer, q)
        correct_idx = q["answer_idx"]
        correct_letter = letters[correct_idx]
        correct_token_id = answer_id_list[correct_idx]

        base_np, delta_p1, delta_p3 = analyze_one_question(
            base_model, adapter_p1, adapter_p3, tokenizer,
            prompt, correct_token_id
        )

        # Check predictions
        base_pred = max(range(4), key=lambda j: base_np[answer_id_list[j]])
        p1_pred = max(range(4), key=lambda j: (base_np + delta_p1)[answer_id_list[j]])
        p3_pred = max(range(4), key=lambda j: (base_np + delta_p3)[answer_id_list[j]])

        if letters[base_pred] == correct_letter:
            agg["base_correct"] += 1
        if letters[p1_pred] == correct_letter:
            agg["p1_correct"] += 1
        if letters[p3_pred] == correct_letter:
            agg["p3_correct"] += 1

        # Collect target token deltas
        agg["target_delta_p1"].append(float(delta_p1[correct_token_id]))
        agg["target_delta_p3"].append(float(delta_p3[correct_token_id]))

        # Collect top boosts/drops
        for idx in np.argsort(-delta_p1)[:5]:
            tok = tokenizer.decode([int(idx)])
            agg["p1_top_boosts"][tok] = agg["p1_top_boosts"].get(tok, 0) + float(delta_p1[idx])
        for idx in np.argsort(delta_p1)[:5]:
            tok = tokenizer.decode([int(idx)])
            agg["p1_top_drops"][tok] = agg["p1_top_drops"].get(tok, 0) + float(delta_p1[idx])
        for idx in np.argsort(-delta_p3)[:5]:
            tok = tokenizer.decode([int(idx)])
            agg["p3_top_boosts"][tok] = agg["p3_top_boosts"].get(tok, 0) + float(delta_p3[idx])

        agg["n_questions"] += 1

        # Print detailed analysis for first 3 questions
        if i < 3:
            print(f"\n{'=' * 70}")
            print(f"Q{i+1}: {q['question'][:80]}...")
            print(f"Correct: {correct_letter} (token id {correct_token_id})")
            print(f"Base pred: {letters[base_pred]}, P1 pred: {letters[p1_pred]}, P3 pred: {letters[p3_pred]}")
            print_analysis(base_np, delta_p1, delta_p3, tokenizer,
                          correct_token_id, top_k=args.top_k)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.n_questions}] base={agg['base_correct']/(i+1):.0%} "
                  f"p1={agg['p1_correct']/(i+1):.0%} p3={agg['p3_correct']/(i+1):.0%}",
                  flush=True)

    # Summary
    n = agg["n_questions"]
    mean_d1 = np.mean(agg["target_delta_p1"])
    mean_d3 = np.mean(agg["target_delta_p3"])

    print(f"\n{'=' * 70}")
    print(f"AGGREGATE SUMMARY ({n} MMLU questions)")
    print(f"{'=' * 70}")
    print(f"Accuracy:  Base={agg['base_correct']/n:.1%}  "
          f"Phase1={agg['p1_correct']/n:.1%}  "
          f"Phase3={agg['p3_correct']/n:.1%}")
    print(f"\nMean prob shift on CORRECT answer token:")
    print(f"  Phase 1 (hidden): {mean_d1:+.6f}")
    print(f"  Phase 3 (logit):  {mean_d3:+.6f}")

    print(f"\nPhase 1 Most-Boosted Tokens (cumulative across {n} questions):")
    for tok, val in sorted(agg["p1_top_boosts"].items(), key=lambda x: -x[1])[:15]:
        print(f"  {repr(tok):<20}: {val:+.4f}")

    print(f"\nPhase 1 Most-Dropped Tokens (cumulative across {n} questions):")
    for tok, val in sorted(agg["p1_top_drops"].items(), key=lambda x: x[1])[:15]:
        print(f"  {repr(tok):<20}: {val:+.4f}")

    print(f"\nPhase 3 Most-Boosted Tokens (cumulative across {n} questions):")
    for tok, val in sorted(agg["p3_top_boosts"].items(), key=lambda x: -x[1])[:15]:
        print(f"  {repr(tok):<20}: {val:+.4f}")

    # Save results
    save_data = {
        "base_acc": agg["base_correct"] / n,
        "p1_acc": agg["p1_correct"] / n,
        "p3_acc": agg["p3_correct"] / n,
        "mean_target_delta_p1": float(mean_d1),
        "mean_target_delta_p3": float(mean_d3),
        "target_deltas_p1": agg["target_delta_p1"],
        "target_deltas_p3": agg["target_delta_p3"],
        "p1_top_boosts": dict(sorted(agg["p1_top_boosts"].items(), key=lambda x: -x[1])[:20]),
        "p1_top_drops": dict(sorted(agg["p1_top_drops"].items(), key=lambda x: x[1])[:20]),
        "p3_top_boosts": dict(sorted(agg["p3_top_boosts"].items(), key=lambda x: -x[1])[:20]),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
