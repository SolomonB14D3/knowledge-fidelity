#!/usr/bin/env python3
"""Operation Destroyer — Phase 1a: Max-Shift Override Test on v5 Checkpoint.

Loads the v5 adapter and tests max_shift=[0.3, 0.4, 0.5] at inference time.
For each:
  - Full Test 1 protocol (margins, shifts, flip categorization) on 100 MMLU examples
  - Qualitative generation comparison (base vs adapter at each max_shift)

Decision criteria (from plan):
  - Flip rate < 15-20% AND mean shift on correct >= -0.2
    → max_shift alone is sufficient, proceed to full retrain with that cap
  - Still bad (>30% flips or large negative shift on correct)
    → proceed to Phase 1b (directional perturbation) or Phase 2 (margin loss)

All inference-only, ~5-10 minutes total.
"""

import json
import os
import sys
import time
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.models import cache as mlx_cache
from mlx.utils import tree_flatten

PROJ_ROOT = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "experiments", "snap_on"))
from module import SnapOnConfig, create_adapter

# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen3-4B-Base"
V5_DIR = os.path.join(PROJ_ROOT, "results", "operation_destroyer", "v5")
OUT_DIR = os.path.join(PROJ_ROOT, "results", "operation_destroyer", "phase1a")
os.makedirs(OUT_DIR, exist_ok=True)

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

# Max shifts to test (the sweet spot range from bonus sweep)
TEST_MAX_SHIFTS = [0.3, 0.4, 0.5]

# Also run original max_shift=2.0 as comparison baseline
BASELINE_MAX_SHIFT = 2.0

# Qualitative prompts (expanded set for thorough testing)
QUAL_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function that checks if a number is prime.",
    "What are the main causes of climate change?",
    "How does the human immune system work?",
    "Write a haiku about mountains.",
    "What is the difference between TCP and UDP?",
    "Explain the concept of supply and demand.",
    # Additional prompts for broader coverage
    "List 5 tips for effective time management.",
    "What is the Pythagorean theorem?",
    "Explain how vaccines work in 3 sentences.",
    "Write a short recipe for scrambled eggs.",
    "What are the three branches of the US government?",
    "Describe the water cycle step by step.",
    "What is the difference between machine learning and deep learning?",
    "Summarize the plot of Romeo and Juliet in 2 sentences.",
    "How does a refrigerator work?",
    "What is photosynthesis?",
    "Name the planets in our solar system in order.",
    "Explain what an API is to a non-technical person.",
]


def get_lm_head_fn(model):
    if hasattr(model, "lm_head") and model.lm_head is not None:
        try:
            _ = model.lm_head.weight.shape
            return model.lm_head
        except AttributeError:
            pass
    if hasattr(model, "args") and getattr(model.args, "tie_word_embeddings", False):
        return model.model.embed_tokens.as_linear
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise RuntimeError("Cannot find lm_head")


def load_everything():
    """Load base model, tokenizer, v5 adapter."""
    print("Loading base model...")
    model, tokenizer = mlx_lm.load(BASE_MODEL)
    lm_head = get_lm_head_fn(model)
    d_model = model.args.hidden_size
    vocab_size = model.args.vocab_size
    print(f"  d_model={d_model}, vocab_size={vocab_size}")

    cfg = SnapOnConfig.load(os.path.join(V5_DIR, "best_config.json"))
    adapter = create_adapter(cfg)
    weights = mx.load(os.path.join(V5_DIR, "best.npz"))
    adapter.load_weights(list(weights.items()))
    mx.eval(adapter.parameters())
    n_params = sum(p.size for _, p in tree_flatten(adapter.parameters()))
    print(f"  Adapter loaded: {n_params:,} params")

    return model, tokenizer, lm_head, adapter


def apply_adapter(base_logits, adapter, max_shift):
    """Apply adapter with tanh cap at given max_shift."""
    raw = adapter(base_logits)
    if max_shift == 0.0:
        return base_logits
    return base_logits + max_shift * mx.tanh(raw / max_shift)


def softmax_entropy(logits_list):
    """Entropy of softmax distribution over a list of logits."""
    l = np.array(logits_list, dtype=np.float64)
    l -= l.max()
    p = np.exp(l) / np.exp(l).sum()
    return -np.sum(p * np.log(p + 1e-10))


# ===========================================================================
# TEST 1: Full margin disruption at each max_shift
# ===========================================================================
def test1_at_maxshift(model, tokenizer, lm_head, adapter, max_shift, n=100):
    """Run full Test 1 margin disruption protocol at given max_shift."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=123)  # Same seed as original Test 1
    ds = ds.select(range(min(n, len(ds))))

    choices = "ABCD"
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]

    results = []
    for i, ex in enumerate(ds):
        question = ex["question"]
        options = ex["choices"]
        answer_idx = ex["answer"]

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

        adapted = apply_adapter(base_logits, adapter, max_shift)
        mx.eval(adapted)

        base_last = base_logits[0, -1, :]
        adapt_last = adapted[0, -1, :]

        # Extract answer logits
        base_answer_logits = [float(base_last[cid]) for cid in choice_ids]
        adapt_answer_logits = [float(adapt_last[cid]) for cid in choice_ids]

        # Margins
        correct_base = base_answer_logits[answer_idx]
        wrong_base = max(base_answer_logits[j] for j in range(4) if j != answer_idx)
        base_margin = correct_base - wrong_base

        correct_adapt = adapt_answer_logits[answer_idx]
        wrong_adapt = max(adapt_answer_logits[j] for j in range(4) if j != answer_idx)
        adapt_margin = correct_adapt - wrong_adapt

        base_pred = max(range(4), key=lambda j: base_answer_logits[j])
        adapt_pred = max(range(4), key=lambda j: adapt_answer_logits[j])

        # Raw shifts on answer tokens (pre-cap)
        raw_shifts = adapter(base_logits)
        mx.eval(raw_shifts)
        shifts_last = raw_shifts[0, -1, :]
        answer_shifts_raw = [float(shifts_last[cid]) for cid in choice_ids]
        capped_shifts = [float(max_shift * mx.tanh(mx.array(s) / max_shift)) for s in answer_shifts_raw]

        # Full vocab shift stats
        all_shifts_np = np.array(shifts_last.tolist())
        all_capped = max_shift * np.tanh(all_shifts_np / max_shift)

        result = {
            "i": i,
            "base_correct": base_pred == answer_idx,
            "adapt_correct": adapt_pred == answer_idx,
            "argmax_flip": base_pred != adapt_pred,
            "base_margin": base_margin,
            "adapt_margin": adapt_margin,
            "delta_margin": adapt_margin - base_margin,
            "shift_on_correct": capped_shifts[answer_idx],
            "max_shift_on_wrong": max(capped_shifts[j] for j in range(4) if j != answer_idx),
            "base_entropy_mc": softmax_entropy(base_answer_logits),
            "adapt_entropy_mc": softmax_entropy(adapt_answer_logits),
            "mean_abs_shift_vocab": float(np.mean(np.abs(all_capped))),
            "max_abs_shift_vocab": float(np.max(np.abs(all_capped))),
        }
        results.append(result)

        if (i + 1) % 25 == 0:
            flips = sum(r["argmax_flip"] for r in results)
            base_acc = sum(r["base_correct"] for r in results) / len(results)
            adapt_acc = sum(r["adapt_correct"] for r in results) / len(results)
            mean_sc = np.mean([r["shift_on_correct"] for r in results])
            print(f"    [{i+1}/{n}] base={base_acc:.1%} adapt={adapt_acc:.1%} "
                  f"flips={flips} ({flips/len(results):.1%}) shift_correct={mean_sc:.4f}")

    # Aggregate
    n_r = len(results)
    flips = sum(r["argmax_flip"] for r in results)
    base_acc = sum(r["base_correct"] for r in results) / n_r
    adapt_acc = sum(r["adapt_correct"] for r in results) / n_r
    correct_to_wrong = sum(1 for r in results if r["base_correct"] and not r["adapt_correct"])
    wrong_to_correct = sum(1 for r in results if not r["base_correct"] and r["adapt_correct"])
    wrong_to_wrong_diff = sum(1 for r in results
                              if not r["base_correct"] and not r["adapt_correct"] and r["argmax_flip"])

    summary = {
        "max_shift": max_shift,
        "n": n_r,
        "base_accuracy": base_acc,
        "adapter_accuracy": adapt_acc,
        "accuracy_delta": adapt_acc - base_acc,
        "total_argmax_flips": flips,
        "flip_rate": flips / n_r,
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "wrong_to_wrong_diff": wrong_to_wrong_diff,
        "mean_delta_margin": float(np.mean([r["delta_margin"] for r in results])),
        "median_delta_margin": float(np.median([r["delta_margin"] for r in results])),
        "mean_shift_on_correct": float(np.mean([r["shift_on_correct"] for r in results])),
        "mean_max_shift_on_wrong": float(np.mean([r["max_shift_on_wrong"] for r in results])),
        "mean_abs_shift_vocab": float(np.mean([r["mean_abs_shift_vocab"] for r in results])),
        "max_abs_shift_vocab": float(np.max([r["max_abs_shift_vocab"] for r in results])),
        "base_mc_entropy": float(np.mean([r["base_entropy_mc"] for r in results])),
        "adapt_mc_entropy": float(np.mean([r["adapt_entropy_mc"] for r in results])),
    }

    return summary, results


# ===========================================================================
# QUALITATIVE GENERATION at each max_shift
# ===========================================================================
def generate_with_adapter_ms(model, tokenizer, adapter, prompt, max_shift,
                              max_tokens=200, temperature=0.0):
    """Generate text with adapter at specified max_shift using KV cache."""
    full_prompt = ALPACA_TEMPLATE.format(instruction=prompt)
    input_ids = tokenizer.encode(full_prompt)
    tokens = mx.array(input_ids)[None]

    generated = []
    kv_cache = mlx_cache.make_prompt_cache(model)

    # Prefill
    logits = model(tokens, cache=kv_cache)
    last = logits[:, -1:, :]
    raw_shifts = adapter(last)
    adjusted = last + max_shift * mx.tanh(raw_shifts / max_shift)
    if temperature > 0:
        probs = mx.softmax(adjusted[:, -1, :] / temperature)
        next_id = int(mx.random.categorical(mx.log(probs)))
    else:
        next_id = int(mx.argmax(adjusted[:, -1, :], axis=-1))

    for _ in range(max_tokens):
        if next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)
        next_input = mx.array([[next_id]])
        logits = model(next_input, cache=kv_cache)
        raw_shifts = adapter(logits)
        adjusted = logits + max_shift * mx.tanh(raw_shifts / max_shift)
        if temperature > 0:
            probs = mx.softmax(adjusted[:, -1, :] / temperature)
            next_id = int(mx.random.categorical(mx.log(probs)))
        else:
            next_id = int(mx.argmax(adjusted[:, -1, :], axis=-1))

    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_base(model, tokenizer, prompt, max_tokens=200):
    """Generate with base model only."""
    full_prompt = ALPACA_TEMPLATE.format(instruction=prompt)
    return mlx_lm.generate(model, tokenizer, prompt=full_prompt,
                           max_tokens=max_tokens, verbose=False)


def evaluate_generation(model, tokenizer, adapter, max_shifts_to_test, prompts):
    """Side-by-side generation at each max_shift level."""
    results = {}

    for prompt in prompts:
        entry = {"prompt": prompt}

        # Base generation (once)
        base_out = generate_base(model, tokenizer, prompt)
        entry["base"] = base_out[:500]

        # Adapter at each max_shift
        for ms in max_shifts_to_test:
            adapter_out = generate_with_adapter_ms(model, tokenizer, adapter, prompt, ms)
            entry[f"adapter_ms{ms}"] = adapter_out[:500]

        results[prompt] = entry

    return results


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    t0 = time.time()
    model, tokenizer, lm_head, adapter = load_everything()
    print(f"\nModel loaded in {time.time()-t0:.1f}s")
    print(f"Output directory: {OUT_DIR}")

    all_max_shifts = TEST_MAX_SHIFTS + [BASELINE_MAX_SHIFT]  # [0.3, 0.4, 0.5, 2.0]

    # =======================================================================
    # PART 1: Test 1 (Margin Disruption) at each max_shift
    # =======================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1a: TEST 1 MARGIN DISRUPTION AT MULTIPLE MAX_SHIFTS")
    print(f"  Testing: {all_max_shifts}")
    print(f"{'='*70}")

    test1_summaries = {}
    test1_details = {}

    for ms in all_max_shifts:
        print(f"\n  ── max_shift = {ms} ──")
        t_start = time.time()
        summary, details = test1_at_maxshift(model, tokenizer, lm_head, adapter, ms, n=100)
        elapsed = time.time() - t_start

        test1_summaries[str(ms)] = summary
        test1_details[str(ms)] = details

        print(f"\n    Results (max_shift={ms}):")
        print(f"      Base accuracy:      {summary['base_accuracy']:.1%}")
        print(f"      Adapter accuracy:   {summary['adapter_accuracy']:.1%}")
        print(f"      Accuracy delta:     {summary['accuracy_delta']:+.1%}")
        print(f"      Argmax flips:       {summary['total_argmax_flips']} / {summary['n']} ({summary['flip_rate']:.1%})")
        print(f"        correct→wrong:    {summary['correct_to_wrong']}")
        print(f"        wrong→correct:    {summary['wrong_to_correct']}")
        print(f"        wrong→wrong(diff):{summary['wrong_to_wrong_diff']}")
        print(f"      Mean Δmargin:       {summary['mean_delta_margin']:.4f}")
        print(f"      Mean shift correct: {summary['mean_shift_on_correct']:.4f}")
        print(f"      Mean shift wrong:   {summary['mean_max_shift_on_wrong']:.4f}")
        print(f"      Mean |shift| vocab: {summary['mean_abs_shift_vocab']:.4f}")
        print(f"      Took {elapsed:.1f}s")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'max_shift':>10s} | {'Acc%':>6s} | {'Δ%':>6s} | {'Flip%':>6s} | {'C→W':>4s} | {'W→C':>4s} | {'Shift_C':>8s} | {'Shift_W':>8s} | {'|Shift|':>8s}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*4}-+-{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for ms in all_max_shifts:
        s = test1_summaries[str(ms)]
        # Decision markers
        flip_ok = "✓" if s["flip_rate"] < 0.20 else ("~" if s["flip_rate"] < 0.30 else "✗")
        shift_ok = "✓" if s["mean_shift_on_correct"] >= -0.2 else ("~" if s["mean_shift_on_correct"] >= -0.5 else "✗")
        print(f"  {ms:10.1f} | {s['adapter_accuracy']:5.1%} | {s['accuracy_delta']:+5.1%} | "
              f"{s['flip_rate']:5.1%}{flip_ok} | {s['correct_to_wrong']:4d} | {s['wrong_to_correct']:4d} | "
              f"{s['mean_shift_on_correct']:+7.4f}{shift_ok} | {s['mean_max_shift_on_wrong']:+7.4f} | "
              f"{s['mean_abs_shift_vocab']:7.4f}")

    # Decision logic
    print(f"\n{'='*70}")
    print(f"  DECISION ANALYSIS")
    print(f"{'='*70}")
    for ms in TEST_MAX_SHIFTS:
        s = test1_summaries[str(ms)]
        flip_pass = s["flip_rate"] < 0.20
        shift_pass = s["mean_shift_on_correct"] >= -0.2
        both_pass = flip_pass and shift_pass

        status = "✓ PASS — proceed to retrain with this cap" if both_pass else "✗ FAIL — need margin loss or Phase 1b"
        if not both_pass and s["flip_rate"] < 0.30:
            status = "~ MARGINAL — retrain might fix, but margin loss recommended"

        print(f"  max_shift={ms}:")
        print(f"    Flip rate {s['flip_rate']:.1%} {'< 20%' if flip_pass else '>= 20%'} → {'PASS' if flip_pass else 'FAIL'}")
        print(f"    Shift on correct {s['mean_shift_on_correct']:.4f} {'>= -0.2' if shift_pass else '< -0.2'} → {'PASS' if shift_pass else 'FAIL'}")
        print(f"    → {status}")

    # Save Test 1 results
    with open(os.path.join(OUT_DIR, "test1_comparison.json"), "w") as f:
        json.dump({"summaries": test1_summaries}, f, indent=2)

    # =======================================================================
    # PART 2: Qualitative Generation at each max_shift
    # =======================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1a: QUALITATIVE GENERATION COMPARISON")
    print(f"  {len(QUAL_PROMPTS)} prompts × {len(all_max_shifts)} max_shifts + base")
    print(f"{'='*70}")

    gen_results = {}
    for pi, prompt in enumerate(QUAL_PROMPTS):
        print(f"\n  [{pi+1}/{len(QUAL_PROMPTS)}] {prompt[:60]}")

        entry = {"prompt": prompt}

        # Base generation
        base_out = generate_base(model, tokenizer, prompt)
        entry["base"] = base_out[:500]
        print(f"    BASE:  {base_out[:100]}...")

        # Adapter at each max_shift
        for ms in all_max_shifts:
            adapter_out = generate_with_adapter_ms(model, tokenizer, adapter, prompt, ms)
            entry[f"ms_{ms}"] = adapter_out[:500]
            # Only print abbreviated for readability
            print(f"    ms={ms}: {adapter_out[:100]}...")

        gen_results[f"prompt_{pi}"] = entry

    # Save generation results
    with open(os.path.join(OUT_DIR, "generation_comparison.json"), "w") as f:
        json.dump(gen_results, f, indent=2)

    # =======================================================================
    # PART 3: Generation quality summary
    # =======================================================================
    print(f"\n{'='*70}")
    print(f"  GENERATION QUALITY SUMMARY")
    print(f"{'='*70}")

    # Check: are tight-cap outputs meaningfully different from base?
    # If all outputs are identical to base, the cap is too tight for generation too
    for ms in all_max_shifts:
        n_identical = 0
        n_total = len(QUAL_PROMPTS)
        total_len_base = 0
        total_len_adapter = 0
        for pi in range(n_total):
            entry = gen_results[f"prompt_{pi}"]
            base_text = entry["base"]
            adapter_text = entry[f"ms_{ms}"]
            if base_text.strip() == adapter_text.strip():
                n_identical += 1
            total_len_base += len(base_text)
            total_len_adapter += len(adapter_text)

        avg_len_base = total_len_base / n_total
        avg_len_adapter = total_len_adapter / n_total
        print(f"  max_shift={ms}:")
        print(f"    Identical to base: {n_identical}/{n_total} ({n_identical/n_total:.0%})")
        print(f"    Avg length base:   {avg_len_base:.0f} chars")
        print(f"    Avg length adapter:{avg_len_adapter:.0f} chars")

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  PHASE 1a COMPLETE — {total:.0f}s ({total/60:.1f}min)")
    print(f"  Results in: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
