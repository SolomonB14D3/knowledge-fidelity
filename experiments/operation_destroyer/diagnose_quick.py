#!/usr/bin/env python3
"""Quick mid-training diagnostic — run on any checkpoint in ~2 minutes.

Checks for:
1. A/B/C/D position bias (the v11 killer)
2. MMLU accuracy (base vs adapter, 50 questions)
3. Shift magnitude sanity
4. Answer-choice prediction distribution

Usage:
    python experiments/operation_destroyer/diagnose_quick.py --checkpoint results/operation_destroyer/v13/best.npz
    python experiments/operation_destroyer/diagnose_quick.py --version v13  # shorthand
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/snap_on")

import mlx.core as mx
import mlx_lm
import numpy as np
from module import SnapOnConfig, create_adapter
from experiments.operation_destroyer.train_v3 import get_lm_head_fn, apply_adapter, ALPACA_TEMPLATE
import experiments.operation_destroyer.train_v3 as t3


def run_diagnosis(checkpoint_path, n_questions=50, softcap=30.0):
    t3.LOGIT_SOFTCAP = softcap

    print(f"\n{'=' * 70}")
    print(f"  QUICK DIAGNOSTIC — {checkpoint_path}")
    print(f"  {n_questions} MMLU questions, softcap={softcap}")
    print(f"{'=' * 70}")

    # Load model + adapter
    t0 = time.time()
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]

    config = SnapOnConfig(d_model=d_model, d_inner=128, n_layers=0, n_heads=8,
                          mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(config)
    weights = mx.load(checkpoint_path)
    adapter.load_weights(list(weights.items()))
    mx.eval(adapter.parameters())
    print(f"  Model + adapter loaded in {time.time() - t0:.1f}s")

    # Load MMLU
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(n_questions))

    choices = "ABCD"
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]

    # Counters
    base_correct = 0
    adapter_correct = 0
    flips_to_correct = 0
    flips_to_wrong = 0
    no_change = 0
    base_pred_dist = [0, 0, 0, 0]  # how often base predicts A/B/C/D
    adapter_pred_dist = [0, 0, 0, 0]
    correct_answer_dist = [0, 0, 0, 0]  # ground truth distribution
    all_answer_shifts = []
    all_shift_mags = []
    all_base_gaps = []
    all_adapted_gaps = []

    t1 = time.time()
    for i, ex in enumerate(ds):
        question = ex["question"]
        options = ex["choices"]
        answer_idx = ex["answer"]
        correct_answer_dist[answer_idx] += 1

        prompt_text = f"{question}\n"
        for j, opt in enumerate(options):
            prompt_text += f"{choices[j]}. {opt}\n"
        prompt_text += "Answer:"

        full_prompt = ALPACA_TEMPLATE.format(instruction=prompt_text)
        tokens = mx.array(tokenizer.encode(full_prompt))[None, :]

        h = model.model(tokens)
        mx.eval(h)
        base_logits = lm_head(h)
        mx.eval(base_logits)

        adapted = apply_adapter(adapter, base_logits)
        mx.eval(adapted)

        raw_shifts = adapter(base_logits)
        mx.eval(raw_shifts)
        centered = raw_shifts - raw_shifts.mean(axis=-1, keepdims=True)
        mx.eval(centered)

        base_last = base_logits[0, -1, :]
        adapted_last = adapted[0, -1, :]
        centered_last = centered[0, -1, :]

        # Shift magnitude (all vocab)
        shift_np = np.array(centered_last.tolist())
        all_shift_mags.append(np.abs(shift_np).mean())

        # Answer token analysis
        base_ans = [float(base_last[cid]) for cid in choice_ids]
        adapted_ans = [float(adapted_last[cid]) for cid in choice_ids]
        shift_ans = [float(centered_last[cid]) for cid in choice_ids]
        all_answer_shifts.append(shift_ans)

        # Margins
        base_correct_logit = base_ans[answer_idx]
        base_wrong_max = max(base_ans[j] for j in range(4) if j != answer_idx)
        all_base_gaps.append(base_correct_logit - base_wrong_max)

        adapted_correct_logit = adapted_ans[answer_idx]
        adapted_wrong_max = max(adapted_ans[j] for j in range(4) if j != answer_idx)
        all_adapted_gaps.append(adapted_correct_logit - adapted_wrong_max)

        # Predictions
        base_pred = int(np.argmax(base_ans))
        adapter_pred = int(np.argmax(adapted_ans))
        base_pred_dist[base_pred] += 1
        adapter_pred_dist[adapter_pred] += 1

        if base_pred == answer_idx:
            base_correct += 1
        if adapter_pred == answer_idx:
            adapter_correct += 1

        if base_pred != adapter_pred:
            if adapter_pred == answer_idx:
                flips_to_correct += 1
            else:
                flips_to_wrong += 1
        else:
            no_change += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t1
            print(f"  [{i+1}/{n_questions}] {elapsed:.0f}s  "
                  f"base={base_correct}/{i+1}  adapter={adapter_correct}/{i+1}")

    elapsed = time.time() - t1
    print(f"\n  Evaluation done in {elapsed:.0f}s")

    # ============================================================
    # RESULTS
    # ============================================================
    answer_shifts = np.array(all_answer_shifts)
    shifts_arr = np.array(all_shift_mags)
    base_gaps = np.array(all_base_gaps)
    adapted_gaps = np.array(all_adapted_gaps)

    print(f"\n{'=' * 70}")
    print(f"  DIAGNOSTIC RESULTS ({n_questions} MMLU questions)")
    print(f"{'=' * 70}")

    # 1. Accuracy
    print(f"\n--- Accuracy ---")
    print(f"  Base:    {base_correct}/{n_questions} = {base_correct/n_questions:.1%}")
    print(f"  Adapter: {adapter_correct}/{n_questions} = {adapter_correct/n_questions:.1%}")
    delta = (adapter_correct - base_correct) / n_questions
    print(f"  Delta:   {delta:+.1%}")
    print(f"  Flips: {flips_to_correct} correct, {flips_to_wrong} wrong, {no_change} unchanged")

    # 2. POSITION BIAS CHECK (comparing adapter vs BASE, not vs uniform)
    print(f"\n--- Position Bias Check ---")
    print(f"  Ground truth distribution: {correct_answer_dist}")
    print(f"  Base predictions (ABCD):    {base_pred_dist}")
    print(f"  Adapter predictions (ABCD): {adapter_pred_dist}")

    adapter_total = sum(adapter_pred_dist)
    base_total = sum(base_pred_dist)

    # Uniform skew (for reference only)
    expected_uniform = adapter_total / 4
    max_skew_uniform = max(abs(c - expected_uniform) / expected_uniform for c in adapter_pred_dist)
    base_skew_uniform = max(abs(c - expected_uniform) / expected_uniform for c in base_pred_dist)

    # ADAPTER vs BASE divergence (THE REAL METRIC)
    # How much did the adapter change the prediction distribution vs base?
    adapter_pcts = [c / adapter_total for c in adapter_pred_dist]
    base_pcts = [c / base_total for c in base_pred_dist]
    max_divergence = max(abs(a - b) for a, b in zip(adapter_pcts, base_pcts))
    divergence_details = [f"{choices[j]}:{adapter_pcts[j]-base_pcts[j]:+.0%}" for j in range(4)]

    skew_pcts = [f"{c/adapter_total:.0%}" for c in adapter_pred_dist]
    base_skew_pcts = [f"{c/base_total:.0%}" for c in base_pred_dist]
    print(f"  Base prediction %:    {base_skew_pcts}  (uniform skew: {base_skew_uniform:.0%})")
    print(f"  Adapter prediction %: {skew_pcts}  (uniform skew: {max_skew_uniform:.0%})")
    print(f"  Adapter vs Base divergence: {divergence_details}  max={max_divergence:.0%}")

    # Use max_skew as the uniform metric (backward compat) but judge on divergence
    max_skew = max_skew_uniform  # keep for saved results

    if max_divergence > 0.20:
        dominant = choices[np.argmax([a - b for a, b in zip(adapter_pcts, base_pcts)])]
        print(f"  *** ADAPTER BIAS DETECTED *** Adapter shifted {max_divergence:.0%} from base toward '{dominant}'")
    elif max_divergence > 0.10:
        dominant = choices[np.argmax([a - b for a, b in zip(adapter_pcts, base_pcts)])]
        print(f"  ** WARNING: Moderate adapter shift ({max_divergence:.0%}) toward '{dominant}' **")
    elif max_skew_uniform > 0.5 and base_skew_uniform < 0.3:
        # Adapter is skewed AND base wasn't — adapter introduced the bias
        dominant = choices[adapter_pred_dist.index(max(adapter_pred_dist))]
        print(f"  *** ADAPTER BIAS DETECTED *** (adapter skew {max_skew_uniform:.0%} but base only {base_skew_uniform:.0%})")
    else:
        print(f"  OK: Adapter prediction distribution close to base model")

    # 3. Per-token shift analysis
    print(f"\n--- Per-Answer-Token Centered Shifts ---")
    for j in range(4):
        col = answer_shifts[:, j]
        print(f"  {choices[j]}: mean={col.mean():+.4f}  "
              f"std={col.std():.4f}  "
              f"range=[{col.min():+.4f}, {col.max():+.4f}]")

    abcd_spread = answer_shifts.max(axis=1) - answer_shifts.min(axis=1)
    print(f"  ABCD spread: mean={abcd_spread.mean():.4f}  "
          f"max={abcd_spread.max():.4f}")

    # Check if one token is systematically boosted
    mean_shifts = answer_shifts.mean(axis=0)
    shift_range = mean_shifts.max() - mean_shifts.min()
    if shift_range > 0.5:
        boosted = choices[int(np.argmax(mean_shifts))]
        print(f"  *** SYSTEMATIC BIAS: '{boosted}' boosted by {mean_shifts.max():+.4f} ***")
    else:
        print(f"  OK: No systematic per-token bias (range={shift_range:.4f})")

    # 4. Shift magnitude
    print(f"\n--- Shift Magnitude ---")
    print(f"  Mean |shift| (all vocab): {shifts_arr.mean():.4f}")
    print(f"  Range: [{shifts_arr.min():.4f}, {shifts_arr.max():.4f}]")

    # 5. Margin analysis
    print(f"\n--- Margin (correct - max_wrong) ---")
    print(f"  Base:    mean={base_gaps.mean():+.4f}  median={np.median(base_gaps):+.4f}")
    print(f"  Adapted: mean={adapted_gaps.mean():+.4f}  median={np.median(adapted_gaps):+.4f}")
    margin_delta = np.mean(adapted_gaps - base_gaps)
    print(f"  Delta:   {margin_delta:+.4f}")
    if margin_delta < -0.5:
        print(f"  ** WARNING: Adapter is HURTING correct-answer margins **")
    elif margin_delta > 0.1:
        print(f"  GOOD: Adapter is improving correct-answer margins")

    # 6. Overall verdict
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}")
    issues = []
    if max_divergence > 0.20:
        issues.append(f"CRITICAL: Adapter shifted predictions {max_divergence:.0%} from base")
    elif max_skew_uniform > 0.5 and base_skew_uniform < 0.3:
        issues.append(f"CRITICAL: Adapter introduced position bias (adapter {max_skew_uniform:.0%} vs base {base_skew_uniform:.0%})")
    if delta < -0.05:
        issues.append(f"WARNING: MMLU accuracy dropped {delta:+.1%}")
    if shift_range > 0.5:
        issues.append(f"WARNING: Systematic token bias (range={shift_range:.4f})")
    if margin_delta < -0.5:
        issues.append(f"WARNING: Margins degraded by {margin_delta:+.4f}")

    if issues:
        print(f"  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
        print(f"\n  RECOMMENDATION: Consider stopping training and investigating.")
    else:
        print(f"  ALL CLEAR — No position bias, accuracy delta={delta:+.1%}, "
              f"margins delta={margin_delta:+.4f}")
        print(f"  Training looks healthy. Continue.")

    # Save results
    results = {
        "checkpoint": checkpoint_path,
        "n_questions": n_questions,
        "base_correct": base_correct,
        "adapter_correct": adapter_correct,
        "delta": delta,
        "position_bias": {
            "base_pred_dist": base_pred_dist,
            "adapter_pred_dist": adapter_pred_dist,
            "max_skew_uniform": max_skew,
            "base_skew_uniform": base_skew_uniform,
            "max_divergence_from_base": max_divergence,
        },
        "mean_shifts_per_token": {c: float(answer_shifts[:, j].mean())
                                   for j, c in enumerate(choices)},
        "margin_delta": float(margin_delta),
        "mean_shift_magnitude": float(shifts_arr.mean()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save next to checkpoint
    diag_path = checkpoint_path.replace("best.npz", "diagnostic.json").replace(
        "checkpoint_weights.npz", "diagnostic.json")
    if diag_path == checkpoint_path:
        diag_path = checkpoint_path + ".diagnostic.json"
    with open(diag_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {diag_path}")

    return results


TF_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Question: {question}\n"
    "Proposed answer: {answer}\n"
    "Is the proposed answer correct? Reply with only True or False.\n\n"
    "### Response:\n"
)


def run_tf_diagnosis(checkpoint_path, n_questions=50, softcap=30.0):
    """Evaluate using True/False scoring — no position bias possible.

    For each MMLU question, present each option in T/F format,
    compute P(True) for each, and pick the one with highest P(True).
    """
    t3.LOGIT_SOFTCAP = softcap

    print(f"\n{'=' * 70}")
    print(f"  T/F DIAGNOSTIC — {checkpoint_path}")
    print(f"  {n_questions} MMLU questions, softcap={softcap}")
    print(f"{'=' * 70}")

    # Load model + adapter
    t0 = time.time()
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]

    config = SnapOnConfig(d_model=d_model, d_inner=128, n_layers=0, n_heads=8,
                          mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(config)
    weights = mx.load(checkpoint_path)
    adapter.load_weights(list(weights.items()))
    mx.eval(adapter.parameters())
    print(f"  Model + adapter loaded in {time.time() - t0:.1f}s")

    # Token IDs
    true_id = tokenizer.encode(" True", add_special_tokens=False)[-1]
    false_id = tokenizer.encode(" False", add_special_tokens=False)[-1]
    print(f"  True token: {true_id}, False token: {false_id}")

    # Load MMLU
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(n_questions))

    base_correct = 0
    adapter_correct = 0

    t1 = time.time()
    for i, ex in enumerate(ds):
        question = ex["question"]
        options = ex["choices"]
        answer_idx = ex["answer"]

        base_scores = []
        adapter_scores = []

        for opt in options:
            prompt = TF_TEMPLATE.format(question=question, answer=opt)
            tokens = mx.array(tokenizer.encode(prompt))[None, :]

            h = model.model(tokens)
            mx.eval(h)
            base_logits = lm_head(h)
            mx.eval(base_logits)
            adapted = apply_adapter(adapter, base_logits)
            mx.eval(adapted)

            # Score = logit(True) - logit(False) at last token
            bl = base_logits[0, -1, :]
            al = adapted[0, -1, :]

            base_score = float(bl[true_id]) - float(bl[false_id])
            adapter_score = float(al[true_id]) - float(al[false_id])

            base_scores.append(base_score)
            adapter_scores.append(adapter_score)

        base_pred = int(np.argmax(base_scores))
        adapter_pred = int(np.argmax(adapter_scores))

        if base_pred == answer_idx:
            base_correct += 1
        if adapter_pred == answer_idx:
            adapter_correct += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t1
            print(f"  [{i+1}/{n_questions}] {elapsed:.0f}s  "
                  f"base={base_correct}/{i+1}  adapter={adapter_correct}/{i+1}")

    elapsed = time.time() - t1
    delta = (adapter_correct - base_correct) / n_questions

    print(f"\n{'=' * 70}")
    print(f"  T/F DIAGNOSTIC RESULTS ({n_questions} questions)")
    print(f"{'=' * 70}")
    print(f"  Base:    {base_correct}/{n_questions} = {base_correct/n_questions:.1%}")
    print(f"  Adapter: {adapter_correct}/{n_questions} = {adapter_correct/n_questions:.1%}")
    print(f"  Delta:   {delta:+.1%}")
    print(f"  Time:    {elapsed:.0f}s ({elapsed/n_questions:.1f}s/question)")
    print(f"\n  NOTE: No position bias possible — each option scored independently")

    # Save
    results = {
        "checkpoint": checkpoint_path,
        "eval_mode": "true_false",
        "n_questions": n_questions,
        "base_correct": base_correct,
        "adapter_correct": adapter_correct,
        "delta": delta,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    diag_path = checkpoint_path.replace("best.npz", "diagnostic_tf.json").replace(
        "checkpoint_weights.npz", "diagnostic_tf.json")
    with open(diag_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {diag_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--softcap", type=float, default=30.0)
    parser.add_argument("--tf_eval", action="store_true",
                        help="Use True/False scoring instead of ABCD")
    args = parser.parse_args()

    if args.checkpoint:
        cp = args.checkpoint
    elif args.version:
        cp = f"results/operation_destroyer/{args.version}/best.npz"
    else:
        print("Usage: --checkpoint PATH or --version v13")
        sys.exit(1)

    if not os.path.exists(cp):
        print(f"Checkpoint not found: {cp}")
        print("Training may not have saved a checkpoint yet.")
        sys.exit(1)

    if args.tf_eval:
        run_tf_diagnosis(cp, n_questions=args.n, softcap=args.softcap)
    else:
        run_diagnosis(cp, n_questions=args.n, softcap=args.softcap)
