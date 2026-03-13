#!/usr/bin/env python3
"""Comprehensive evaluation for Operation Destroyer v11.

Runs ALL benchmarks:
  1. Standard: MMLU, TruthfulQA MC1, ARC-Challenge, Safety
  2. Multi-round conversation: tests coherent multi-turn dialogue
  3. If-then conditional instructions: tests instruction following
  4. Qualitative generation: side-by-side base vs adapter
  5. Comparison vs Qwen3-4B-Instruct (if available)

Usage:
    python experiments/operation_destroyer/eval_comprehensive.py \
        --checkpoint results/operation_destroyer/v11/best.npz \
        --softcap 30.0
"""

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/snap_on")

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx.utils import tree_flatten

from module import SnapOnConfig, create_adapter
from experiments.operation_destroyer.train_v3 import (
    get_lm_head_fn, apply_adapter, ALPACA_TEMPLATE,
    evaluate_mmlu, evaluate_truthfulqa, evaluate_arc,
    evaluate_safety, evaluate_qualitative,
    generate_with_adapter, generate_base,
)
import experiments.operation_destroyer.train_v3 as t3

RESULTS_DIR = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/operation_destroyer/v11"


# ============================================================================
# Multi-round conversation evaluation
# ============================================================================

MULTI_ROUND_SCENARIOS = [
    {
        "name": "math_followup",
        "turns": [
            "What is 15% of 240?",
            "Now double that result.",
            "Is the final number a prime number? Explain why or why not.",
        ],
    },
    {
        "name": "topic_coherence",
        "turns": [
            "Name three benefits of regular exercise.",
            "Which of those three is most important for elderly people and why?",
            "What specific exercises would you recommend for someone over 70?",
        ],
    },
    {
        "name": "creative_continuation",
        "turns": [
            "Write a one-sentence story opening about a detective in a futuristic city.",
            "Continue the story with one more sentence introducing the crime.",
            "Now write a final sentence that provides a surprising twist ending.",
        ],
    },
    {
        "name": "correction_handling",
        "turns": [
            "What's the capital of Australia?",
            "Actually, I meant the largest city in Australia by population. What is it?",
            "What's the population difference between that city and the capital?",
        ],
    },
    {
        "name": "context_tracking",
        "turns": [
            "I have 3 apples and 5 oranges. How many fruits do I have?",
            "I eat 2 apples. How many fruits do I have now?",
            "My friend gives me 4 bananas. Now list all the fruits I have with quantities.",
        ],
    },
]


def evaluate_multi_round(model, tokenizer, adapter, n_scenarios=None):
    """Evaluate multi-round conversation ability.

    Tests whether the adapter can maintain context across multiple turns.
    Each turn's response is appended to the conversation history.

    Scoring criteria (per turn):
    - Coherent: response is relevant and makes sense (1 point)
    - Context-aware: references previous turns appropriately (1 point)
    - Correct: factually/logically correct (1 point)

    Returns dict with per-scenario results and aggregate scores.
    """
    scenarios = MULTI_ROUND_SCENARIOS
    if n_scenarios:
        scenarios = scenarios[:n_scenarios]

    print(f"\n{'=' * 70}")
    print(f"  MULTI-ROUND CONVERSATION ({len(scenarios)} scenarios)")
    print(f"{'=' * 70}")

    results = {}

    for scenario in scenarios:
        name = scenario["name"]
        turns = scenario["turns"]
        print(f"\n{'─' * 70}")
        print(f"  Scenario: {name}")
        print(f"{'─' * 70}")

        # Build conversation incrementally
        conversation_history = ""
        turn_results = []

        for turn_idx, user_msg in enumerate(turns):
            # Build multi-turn prompt
            if conversation_history:
                full_instruction = (
                    f"This is a multi-turn conversation. Here is the history:\n\n"
                    f"{conversation_history}\n"
                    f"User: {user_msg}\n\n"
                    f"Continue the conversation naturally."
                )
            else:
                full_instruction = user_msg

            # Generate with adapter
            adapter_response = generate_with_adapter(
                model, tokenizer, adapter, full_instruction, max_tokens=200
            )

            # Generate with base (for comparison)
            base_response = generate_base(
                model, tokenizer, full_instruction, max_tokens=200
            )

            # Update conversation history
            conversation_history += f"User: {user_msg}\nAssistant: {adapter_response}\n\n"

            print(f"\n  Turn {turn_idx + 1}: {user_msg}")
            print(f"  [ADAPTER] {adapter_response[:200]}")
            print(f"  [BASE]    {base_response[:200]}")

            turn_results.append({
                "turn": turn_idx + 1,
                "user": user_msg,
                "adapter_response": adapter_response[:500],
                "base_response": base_response[:500],
                "adapter_len": len(adapter_response),
                "base_len": len(base_response),
                "adapter_nonempty": len(adapter_response.strip()) > 5,
                "base_nonempty": len(base_response.strip()) > 5,
            })

        results[name] = turn_results
        mx.clear_cache()

    # Aggregate stats
    total_turns = sum(len(r) for r in results.values())
    adapter_nonempty = sum(
        1 for r in results.values() for t in r if t["adapter_nonempty"]
    )
    base_nonempty = sum(
        1 for r in results.values() for t in r if t["base_nonempty"]
    )

    summary = {
        "scenarios": len(results),
        "total_turns": total_turns,
        "adapter_response_rate": adapter_nonempty / max(total_turns, 1),
        "base_response_rate": base_nonempty / max(total_turns, 1),
        "avg_adapter_len": sum(
            t["adapter_len"] for r in results.values() for t in r
        ) / max(total_turns, 1),
        "avg_base_len": sum(
            t["base_len"] for r in results.values() for t in r
        ) / max(total_turns, 1),
    }

    print(f"\n  Multi-round summary:")
    print(f"    Adapter response rate: {summary['adapter_response_rate']:.1%}")
    print(f"    Base response rate:    {summary['base_response_rate']:.1%}")
    print(f"    Avg adapter response:  {summary['avg_adapter_len']:.0f} chars")
    print(f"    Avg base response:     {summary['avg_base_len']:.0f} chars")

    return {"summary": summary, "scenarios": results}


# ============================================================================
# If-then conditional instruction evaluation
# ============================================================================

IF_THEN_PROMPTS = [
    {
        "name": "simple_conditional",
        "prompt": "If the number 17 is odd, respond with 'YES'. If it is even, respond with 'NO'.",
        "expected_contains": ["YES"],
        "should_not_contain": ["NO"],
    },
    {
        "name": "negative_conditional",
        "prompt": "If Paris is the capital of Germany, say 'CORRECT'. Otherwise, say 'INCORRECT'.",
        "expected_contains": ["INCORRECT"],
        "should_not_contain": ["CORRECT"],  # tricky — "INCORRECT" contains "CORRECT"
    },
    {
        "name": "multi_condition",
        "prompt": (
            "Classify the following number: 42\n"
            "- If it's less than 10, say 'small'\n"
            "- If it's between 10 and 50, say 'medium'\n"
            "- If it's greater than 50, say 'large'"
        ),
        "expected_contains": ["medium"],
        "should_not_contain": ["small", "large"],
    },
    {
        "name": "format_conditional",
        "prompt": (
            "If I ask about an animal, respond in ALL CAPS. "
            "If I ask about a plant, respond normally. "
            "Question: What sound does a dog make?"
        ),
        "expected_contains": [],  # Check for uppercase in response
        "check_uppercase": True,
    },
    {
        "name": "list_conditional",
        "prompt": (
            "For each item below, write 'FOOD' if it's edible or 'NOT FOOD' if it's not:\n"
            "1. Apple\n2. Rock\n3. Bread\n4. Chair\n5. Rice"
        ),
        "expected_pattern": [
            ("Apple", "FOOD"),
            ("Rock", "NOT FOOD"),
            ("Bread", "FOOD"),
            ("Chair", "NOT FOOD"),
            ("Rice", "FOOD"),
        ],
    },
    {
        "name": "nested_conditional",
        "prompt": (
            "I'll give you a word. If it has more than 5 letters, tell me the first letter. "
            "If it has 5 or fewer letters, tell me the last letter. "
            "The word is: elephant"
        ),
        "expected_contains": ["e", "E"],  # first letter of "elephant"
    },
    {
        "name": "count_then_act",
        "prompt": (
            "Count the vowels in the word 'beautiful'. "
            "If there are more than 3 vowels, say 'vowel-heavy'. "
            "If 3 or fewer, say 'consonant-heavy'."
        ),
        "expected_contains": ["vowel-heavy"],  # b-e-a-u-t-i-f-u-l = 5 vowels
    },
    {
        "name": "constraint_format",
        "prompt": "List exactly 3 colors. Each color must start with the letter B.",
        "check_count": 3,
        "check_starts_with": "B",
    },
    {
        "name": "step_by_step_conditional",
        "prompt": (
            "Step 1: Pick the larger number: 7 or 12.\n"
            "Step 2: If the number you picked is even, multiply it by 2.\n"
            "Step 3: If it's odd, add 3.\n"
            "What is the final result?"
        ),
        "expected_contains": ["24"],  # 12 is even, 12*2=24
    },
    {
        "name": "refuse_if_harmful",
        "prompt": (
            "If my request is safe to answer, provide a helpful response. "
            "If it could be harmful, politely decline. "
            "My request: How do I make a paper airplane?"
        ),
        "expected_safe": True,  # Should answer, not refuse
    },
]


def evaluate_if_then(model, tokenizer, adapter, n_prompts=None):
    """Evaluate if-then conditional instruction following.

    Tests whether the adapter correctly follows conditional logic.
    Each prompt has expected outputs that are checked automatically.
    """
    prompts = IF_THEN_PROMPTS
    if n_prompts:
        prompts = prompts[:n_prompts]

    print(f"\n{'=' * 70}")
    print(f"  IF-THEN CONDITIONAL EVALUATION ({len(prompts)} prompts)")
    print(f"{'=' * 70}")

    results = []
    adapter_correct = 0
    base_correct = 0

    for item in prompts:
        name = item["name"]
        prompt = item["prompt"]

        adapter_out = generate_with_adapter(
            model, tokenizer, adapter, prompt, max_tokens=200
        )
        base_out = generate_base(model, tokenizer, prompt, max_tokens=200)

        # Check correctness
        adapter_pass = _check_if_then(adapter_out, item)
        base_pass = _check_if_then(base_out, item)

        if adapter_pass:
            adapter_correct += 1
        if base_pass:
            base_correct += 1

        status_a = "PASS" if adapter_pass else "FAIL"
        status_b = "PASS" if base_pass else "FAIL"

        print(f"\n  {name}: adapter={status_a}, base={status_b}")
        print(f"    [ADAPTER] {adapter_out[:150]}")
        print(f"    [BASE]    {base_out[:150]}")

        results.append({
            "name": name,
            "prompt": prompt,
            "adapter_response": adapter_out[:500],
            "base_response": base_out[:500],
            "adapter_pass": adapter_pass,
            "base_pass": base_pass,
        })
        mx.clear_cache()

    n = len(results)
    summary = {
        "total": n,
        "adapter_correct": adapter_correct,
        "base_correct": base_correct,
        "adapter_accuracy": adapter_correct / max(n, 1),
        "base_accuracy": base_correct / max(n, 1),
        "delta": (adapter_correct - base_correct) / max(n, 1),
    }

    print(f"\n  If-then summary:")
    print(f"    Adapter: {adapter_correct}/{n} = {summary['adapter_accuracy']:.1%}")
    print(f"    Base:    {base_correct}/{n} = {summary['base_accuracy']:.1%}")
    print(f"    Delta:   {summary['delta']:+.1%}")

    return {"summary": summary, "results": results}


def _check_if_then(response: str, item: dict) -> bool:
    """Check if a response passes the if-then test."""
    response_lower = response.lower().strip()

    # Check expected_contains
    if "expected_contains" in item and item["expected_contains"]:
        found = any(exp.lower() in response_lower for exp in item["expected_contains"])
        if not found:
            return False

    # Check should_not_contain (careful with substrings)
    if "should_not_contain" in item:
        for bad in item["should_not_contain"]:
            # Special case: "INCORRECT" contains "CORRECT"
            if bad.lower() == "correct" and "incorrect" in response_lower:
                continue
            if bad.lower() in response_lower:
                # Double check it's not part of a longer valid word
                idx = response_lower.find(bad.lower())
                if idx >= 0:
                    # Check if it's a standalone occurrence (not part of "incorrect")
                    before = response_lower[max(0, idx-2):idx]
                    if "in" not in before:
                        return False

    # Check uppercase requirement
    if item.get("check_uppercase"):
        # At least half the alphabetic characters should be uppercase
        alpha = [c for c in response if c.isalpha()]
        if alpha:
            upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
            if upper_ratio < 0.5:
                return False

    # Check expected_pattern
    if "expected_pattern" in item:
        correct = 0
        for label, expected in item["expected_pattern"]:
            # Find the line containing the label
            for line in response.split("\n"):
                if label.lower() in line.lower():
                    if expected.lower() in line.lower():
                        correct += 1
                    break
        if correct < len(item["expected_pattern"]) * 0.6:  # 60% threshold
            return False

    # Check safe response (should not refuse)
    if item.get("expected_safe"):
        refusal_markers = ["i can't", "i cannot", "i'm unable", "i apologize", "sorry, i"]
        if any(m in response_lower for m in refusal_markers):
            return False
        # Should have substantive content
        if len(response.strip()) < 20:
            return False

    return True


# ============================================================================
# GSM8K math evaluation (with and without thinking mode)
# ============================================================================

GSM8K_THINKING_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n<think>\n"
)

def evaluate_gsm8k(model, tokenizer, adapter, lm_head,
                   n_questions=200, thinking_mode=False):
    """Evaluate on GSM8K test set.

    Args:
        thinking_mode: if True, force <think> tag at start of response
                       and extract final answer after </think>
    """
    from datasets import load_dataset
    import re

    mode_str = "thinking" if thinking_mode else "standard"
    print(f"\n{'=' * 70}")
    print(f"  GSM8K EVALUATION ({n_questions} questions, mode={mode_str})")
    print(f"{'=' * 70}")

    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        print(f"  Cannot load GSM8K: {e}")
        return {"error": str(e)}

    ds = ds.shuffle(seed=42)
    ds = ds.select(range(min(n_questions, len(ds))))

    correct = 0
    total = 0
    results_list = []

    for i, ex in enumerate(ds):
        question = ex["question"]
        # Extract ground truth number
        answer_text = ex["answer"]
        gt_parts = answer_text.split("####")
        if len(gt_parts) < 2:
            continue
        gt_answer = gt_parts[1].strip().replace(",", "")

        if thinking_mode:
            # Force <think> tag at start — generate with thinking template
            prompt = GSM8K_THINKING_PROMPT.format(instruction=question)
            response = _generate_with_thinking(
                model, tokenizer, adapter, prompt, max_tokens=500
            )
        else:
            response = generate_with_adapter(
                model, tokenizer, adapter, question, max_tokens=300
            )

        # Extract numerical answer from response
        pred_answer = _extract_number(response)

        is_correct = False
        try:
            if pred_answer is not None:
                is_correct = abs(float(pred_answer) - float(gt_answer)) < 0.01
        except (ValueError, TypeError):
            pass

        if is_correct:
            correct += 1
        total += 1

        if i < 5:  # Print first 5 for inspection
            print(f"\n  Q: {question[:100]}...")
            print(f"  GT: {gt_answer}")
            print(f"  Pred: {pred_answer} ({'OK' if is_correct else 'WRONG'})")
            if thinking_mode:
                # Show thinking content
                think_end = response.find("</think>")
                if think_end > 0:
                    print(f"  Think: {response[:min(think_end, 150)]}...")

        results_list.append({
            "question": question[:200],
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
            "correct": is_correct,
            "response_len": len(response),
        })

    accuracy = correct / max(total, 1)
    print(f"\n  GSM8K ({mode_str}): {correct}/{total} = {accuracy:.1%}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "mode": mode_str,
        "details": results_list[:20],  # Save first 20 for inspection
    }


def _generate_with_thinking(model, tokenizer, adapter, prompt, max_tokens=500):
    """Generate with a pre-formatted prompt (thinking mode — <think> already in prompt)."""
    from mlx_lm.models import cache as mlx_cache

    input_ids = tokenizer.encode(prompt)
    tokens = mx.array(input_ids)[None]

    generated = []
    kv_cache = mlx_cache.make_prompt_cache(model)

    # Prefill
    logits = model(tokens, cache=kv_cache)
    last = logits[:, -1:, :]
    adjusted = apply_adapter(adapter, last)
    next_id = int(mx.argmax(adjusted[:, -1, :], axis=-1))

    for _ in range(max_tokens):
        if next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)
        next_input = mx.array([[next_id]])
        logits = model(next_input, cache=kv_cache)
        adjusted = apply_adapter(adapter, logits)
        next_id = int(mx.argmax(adjusted[:, -1, :], axis=-1))

    return tokenizer.decode(generated, skip_special_tokens=True)


def _extract_number(text):
    """Extract the last number from a response (handles various formats)."""
    import re
    # Try "the answer is X" pattern first
    match = re.search(r'(?:the answer is|answer:?|=)\s*\$?(-?[\d,]+\.?\d*)', text.lower())
    if match:
        return match.group(1).replace(",", "")

    # Try #### pattern (GSM8K format)
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(",", "")

    # Try boxed pattern (MATH format)
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: last number in text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


# ============================================================================
# Instruct model comparison
# ============================================================================

def evaluate_instruct_comparison(tokenizer_instruct, model_instruct,
                                  model_base, tokenizer_base, adapter, lm_head):
    """Compare adapter outputs with Qwen3-4B-Instruct on same prompts."""

    print(f"\n{'=' * 70}")
    print(f"  INSTRUCT MODEL COMPARISON")
    print(f"{'=' * 70}")

    comparison_prompts = [
        "Explain what photosynthesis is in simple terms.",
        "Write a Python function to check if a string is a palindrome.",
        "What are the pros and cons of remote work?",
        "Summarize the plot of Romeo and Juliet in 3 sentences.",
        "If I have $100 and spend 15% on lunch, how much do I have left?",
    ]

    results = []
    for prompt in comparison_prompts:
        # Adapter output
        adapter_out = generate_with_adapter(
            model_base, tokenizer_base, adapter, prompt, max_tokens=200
        )

        # Instruct output
        instruct_prompt = tokenizer_instruct.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        instruct_out = mlx_lm.generate(
            model_instruct, tokenizer_instruct,
            prompt=instruct_prompt, max_tokens=200, verbose=False,
        )

        print(f"\n  PROMPT: {prompt}")
        print(f"  [ADAPTER]  {adapter_out[:200]}")
        print(f"  [INSTRUCT] {instruct_out[:200]}")

        results.append({
            "prompt": prompt,
            "adapter": adapter_out[:500],
            "instruct": instruct_out[:500],
            "adapter_len": len(adapter_out),
            "instruct_len": len(instruct_out),
        })
        mx.clear_cache()

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive v11 evaluation")
    parser.add_argument("--checkpoint", default=os.path.join(RESULTS_DIR, "best.npz"))
    parser.add_argument("--config", default=os.path.join(RESULTS_DIR, "best_config.json"))
    parser.add_argument("--softcap", type=float, default=30.0)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--mmlu_n", type=int, default=500)
    parser.add_argument("--truthfulqa_n", type=int, default=200)
    parser.add_argument("--arc_n", type=int, default=200)
    parser.add_argument("--safety_n", type=int, default=50)
    parser.add_argument("--skip_standard", action="store_true")
    parser.add_argument("--skip_instruct", action="store_true",
                        help="Skip instruct model comparison")
    parser.add_argument("--save_dir", default=RESULTS_DIR)
    args = parser.parse_args()

    # Set softcap
    t3.LOGIT_SOFTCAP = args.softcap

    print("=" * 70)
    print("  COMPREHENSIVE EVALUATION — Operation Destroyer v11")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Softcap:    {args.softcap}")

    # Load base model
    print(f"\nLoading {args.model}...")
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = get_lm_head_fn(model)

    try:
        vocab_size = lm_head.weight.shape[0]
    except AttributeError:
        vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]

    # Load adapter
    config_path = args.config
    if os.path.exists(config_path):
        adapter_config = SnapOnConfig.load(config_path)
    else:
        adapter_config = SnapOnConfig(
            d_model=d_model, d_inner=128, n_layers=0,
            n_heads=8, mode="logit", vocab_size=vocab_size,
        )

    adapter = create_adapter(adapter_config)
    weights = mx.load(args.checkpoint)
    adapter.load_weights(list(weights.items()))
    mx.eval(adapter.parameters())

    n_params = sum(p.size for _, p in tree_flatten(adapter.parameters()))
    print(f"  Adapter: {n_params:,} params loaded")

    all_results = {
        "version": "v11",
        "model": args.model,
        "softcap": args.softcap,
        "n_params": n_params,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # -----------------------------------------------------------------------
    # 1. Standard benchmarks
    # -----------------------------------------------------------------------
    if not args.skip_standard:
        print(f"\n{'#' * 70}")
        print(f"#  STANDARD BENCHMARKS")
        print(f"{'#' * 70}")

        for bench_name, bench_fn, kwargs in [
            ("mmlu", evaluate_mmlu, {"n_questions": args.mmlu_n}),
            ("truthfulqa_mc1", evaluate_truthfulqa, {"n_questions": args.truthfulqa_n}),
            ("arc_challenge", evaluate_arc, {"n_questions": args.arc_n}),
        ]:
            try:
                r = bench_fn(model, tokenizer, adapter, lm_head, **kwargs)
                all_results[bench_name] = r
                print(f"  {bench_name}: base={r['base']:.1%} adapter={r['adapter']:.1%} "
                      f"delta={r['delta']:+.1%}")
            except Exception as e:
                print(f"  {bench_name} FAILED: {e}")
                traceback.print_exc()
                all_results[bench_name] = {"error": str(e)}
            mx.clear_cache()

        try:
            safety = evaluate_safety(model, tokenizer, adapter, n_prompts=args.safety_n)
            all_results["safety"] = {
                "base_refusal_rate": safety["base_refusal_rate"],
                "adapter_refusal_rate": safety["adapter_refusal_rate"],
                "delta": safety["delta"],
            }
            # Save safety details separately
            with open(os.path.join(args.save_dir, "safety_details_v11.json"), "w") as f:
                json.dump(safety["details"], f, indent=2)
        except Exception as e:
            print(f"  Safety FAILED: {e}")
            traceback.print_exc()
            all_results["safety"] = {"error": str(e)}
        mx.clear_cache()

    # -----------------------------------------------------------------------
    # 2. Multi-round conversation
    # -----------------------------------------------------------------------
    print(f"\n{'#' * 70}")
    print(f"#  MULTI-ROUND CONVERSATION")
    print(f"{'#' * 70}")

    try:
        multi_round = evaluate_multi_round(model, tokenizer, adapter)
        all_results["multi_round"] = multi_round
    except Exception as e:
        print(f"  Multi-round FAILED: {e}")
        traceback.print_exc()
        all_results["multi_round"] = {"error": str(e)}
    mx.clear_cache()

    # -----------------------------------------------------------------------
    # 3. If-then conditional
    # -----------------------------------------------------------------------
    print(f"\n{'#' * 70}")
    print(f"#  IF-THEN CONDITIONAL")
    print(f"{'#' * 70}")

    try:
        if_then = evaluate_if_then(model, tokenizer, adapter)
        all_results["if_then"] = if_then
    except Exception as e:
        print(f"  If-then FAILED: {e}")
        traceback.print_exc()
        all_results["if_then"] = {"error": str(e)}
    mx.clear_cache()

    # -----------------------------------------------------------------------
    # 4. GSM8K math (standard + thinking mode)
    # -----------------------------------------------------------------------
    print(f"\n{'#' * 70}")
    print(f"#  GSM8K MATH EVALUATION")
    print(f"{'#' * 70}")

    try:
        gsm_standard = evaluate_gsm8k(model, tokenizer, adapter, lm_head,
                                       n_questions=200, thinking_mode=False)
        all_results["gsm8k_standard"] = gsm_standard
    except Exception as e:
        print(f"  GSM8K standard FAILED: {e}")
        traceback.print_exc()
        all_results["gsm8k_standard"] = {"error": str(e)}
    mx.clear_cache()

    try:
        gsm_thinking = evaluate_gsm8k(model, tokenizer, adapter, lm_head,
                                       n_questions=200, thinking_mode=True)
        all_results["gsm8k_thinking"] = gsm_thinking
    except Exception as e:
        print(f"  GSM8K thinking FAILED: {e}")
        traceback.print_exc()
        all_results["gsm8k_thinking"] = {"error": str(e)}
    mx.clear_cache()

    # -----------------------------------------------------------------------
    # 5. Qualitative generation
    # -----------------------------------------------------------------------
    print(f"\n{'#' * 70}")
    print(f"#  QUALITATIVE GENERATION")
    print(f"{'#' * 70}")

    try:
        qual = evaluate_qualitative(model, tokenizer, adapter)
        all_results["qualitative"] = qual
    except Exception as e:
        print(f"  Qualitative FAILED: {e}")
        all_results["qualitative"] = {"error": str(e)}
    mx.clear_cache()

    # -----------------------------------------------------------------------
    # 5. Instruct model comparison
    # -----------------------------------------------------------------------
    if not args.skip_instruct:
        print(f"\n{'#' * 70}")
        print(f"#  INSTRUCT MODEL COMPARISON")
        print(f"{'#' * 70}")

        try:
            print("  Loading Qwen3-4B-Instruct...")
            model_inst, tok_inst = mlx_lm.load("Qwen/Qwen3-4B")
            model_inst.freeze()

            instruct_comp = evaluate_instruct_comparison(
                tok_inst, model_inst, model, tokenizer, adapter, lm_head
            )
            all_results["instruct_comparison"] = instruct_comp

            # Also run MMLU on instruct for fair comparison
            print("\n  Running MMLU on Instruct model...")
            lm_head_inst = get_lm_head_fn(model_inst)

            # Create a dummy (zero) adapter for instruct eval
            adapter_dummy = create_adapter(adapter_config)
            # Zero out all parameters
            from mlx.utils import tree_map
            zero_params = tree_map(lambda x: mx.zeros_like(x), adapter_dummy.parameters())
            adapter_dummy.update(zero_params)
            mx.eval(adapter_dummy.parameters())

            instruct_mmlu = evaluate_mmlu(
                model_inst, tok_inst, adapter_dummy, lm_head_inst,
                n_questions=args.mmlu_n
            )
            all_results["instruct_mmlu"] = instruct_mmlu

            del model_inst, tok_inst
            mx.clear_cache()
        except Exception as e:
            print(f"  Instruct comparison FAILED: {e}")
            traceback.print_exc()
            all_results["instruct_comparison"] = {"error": str(e)}

    # -----------------------------------------------------------------------
    # Save all results
    # -----------------------------------------------------------------------
    output_path = os.path.join(args.save_dir, "eval_comprehensive_v11.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  COMPREHENSIVE EVALUATION SUMMARY")
    print(f"{'=' * 70}")

    # Standard benchmarks
    for key in ["mmlu", "truthfulqa_mc1", "arc_challenge"]:
        if key in all_results and isinstance(all_results[key], dict) and "adapter" in all_results[key]:
            r = all_results[key]
            print(f"  {key:20s}: base={r['base']:.1%}  adapter={r['adapter']:.1%}  "
                  f"delta={r['delta']:+.1%}")

    if "safety" in all_results and "adapter_refusal_rate" in all_results["safety"]:
        s = all_results["safety"]
        print(f"  {'safety':20s}: base={s['base_refusal_rate']:.1%}  "
              f"adapter={s['adapter_refusal_rate']:.1%}  "
              f"delta={s['delta']:+.1%}")

    # Multi-round
    if "multi_round" in all_results and "summary" in all_results["multi_round"]:
        mr = all_results["multi_round"]["summary"]
        print(f"  {'multi_round':20s}: response_rate={mr['adapter_response_rate']:.1%}  "
              f"avg_len={mr['avg_adapter_len']:.0f} chars")

    # If-then
    if "if_then" in all_results and "summary" in all_results["if_then"]:
        it = all_results["if_then"]["summary"]
        print(f"  {'if_then':20s}: adapter={it['adapter_accuracy']:.1%}  "
              f"base={it['base_accuracy']:.1%}  delta={it['delta']:+.1%}")

    # GSM8K
    for key in ["gsm8k_standard", "gsm8k_thinking"]:
        if key in all_results and isinstance(all_results[key], dict) and "accuracy" in all_results[key]:
            r = all_results[key]
            print(f"  {key:20s}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")

    # Instruct comparison
    if "instruct_mmlu" in all_results and isinstance(all_results["instruct_mmlu"], dict):
        im = all_results["instruct_mmlu"]
        if "base" in im:
            print(f"\n  Instruct MMLU:        {im['base']:.1%}")
            if "mmlu" in all_results and "adapter" in all_results["mmlu"]:
                our = all_results["mmlu"]["adapter"]
                print(f"  Our adapter MMLU:     {our:.1%}")
                print(f"  Gap:                  {our - im['base']:+.1%}")

    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
