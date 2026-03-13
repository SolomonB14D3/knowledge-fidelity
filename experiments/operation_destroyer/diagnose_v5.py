#!/usr/bin/env python3
"""Operation Destroyer — Diagnostic Suite for v5 MMLU/ARC Regression.

Tests 1-5 from the diagnostic plan:
  #1: Logit Margin Disruption on holdout MC prompts
  #2: Synthetic Perturbation Test (random bounded shifts)
  #4: Loss-Term Proxy (would margin/KL losses have helped?)
  #5: Generation vs MC Discrepancy (entropy comparison)

All are inference-only — no training, no gradients.
Shares a single model load across all tests.
"""

import json
import os
import sys
import time
import math
import numpy as np
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx.utils import tree_flatten

PROJ_ROOT = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "experiments", "snap_on"))
from module import SnapOnConfig, create_adapter

# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen3-4B-Base"
V5_DIR = os.path.join(PROJ_ROOT, "results", "operation_destroyer", "v5")
OUT_DIR = os.path.join(PROJ_ROOT, "results", "operation_destroyer", "v5_diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


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

    # Load adapter
    cfg = SnapOnConfig.load(os.path.join(V5_DIR, "best_config.json"))
    adapter = create_adapter(cfg)
    weights = mx.load(os.path.join(V5_DIR, "best.npz"))
    adapter.load_weights(list(weights.items()))
    mx.eval(adapter.parameters())
    n_params = sum(p.size for _, p in tree_flatten(adapter.parameters()))
    print(f"  Adapter loaded: {n_params:,} params")

    return model, tokenizer, lm_head, adapter


def apply_adapter(base_logits, adapter, max_shift=2.0):
    """Apply adapter with tanh cap."""
    raw = adapter(base_logits)
    return base_logits + max_shift * mx.tanh(raw / max_shift)


# ===========================================================================
# TEST 1: Logit Margin Disruption on Holdout MC Prompts
# ===========================================================================
def test1_margin_disruption(model, tokenizer, lm_head, adapter, n=100):
    """Measure how the adapter disrupts MC answer margins."""
    from datasets import load_dataset

    print(f"\n{'='*70}")
    print(f"  TEST 1: LOGIT MARGIN DISRUPTION (n={n})")
    print(f"{'='*70}")

    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=123)  # Different seed from training eval
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

        adapted = apply_adapter(base_logits, adapter)
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

        # Raw shifts on answer tokens
        raw_shifts = adapter(base_logits)
        mx.eval(raw_shifts)
        shifts_last = raw_shifts[0, -1, :]
        answer_shifts = [float(shifts_last[cid]) for cid in choice_ids]
        capped_shifts = [float(2.0 * mx.tanh(mx.array(s) / 2.0)) for s in answer_shifts]

        # Full vocab shift stats on last token
        all_shifts_np = np.array(shifts_last.tolist())
        all_capped = 2.0 * np.tanh(all_shifts_np / 2.0)

        # Entropy of base vs adapter distributions (on answer tokens only)
        def softmax_entropy(logits_list):
            l = np.array(logits_list)
            l -= l.max()
            p = np.exp(l) / np.exp(l).sum()
            return -np.sum(p * np.log(p + 1e-10))

        result = {
            "i": i,
            "base_correct": base_pred == answer_idx,
            "adapt_correct": adapt_pred == answer_idx,
            "argmax_flip": base_pred != adapt_pred,
            "base_margin": base_margin,
            "adapt_margin": adapt_margin,
            "delta_margin": adapt_margin - base_margin,
            "raw_shifts_on_answers": answer_shifts,
            "capped_shifts_on_answers": capped_shifts,
            "shift_on_correct": capped_shifts[answer_idx],
            "max_shift_on_wrong": max(capped_shifts[j] for j in range(4) if j != answer_idx),
            "base_entropy_mc": softmax_entropy(base_answer_logits),
            "adapt_entropy_mc": softmax_entropy(adapt_answer_logits),
            "mean_abs_shift_vocab": float(np.mean(np.abs(all_capped))),
            "max_abs_shift_vocab": float(np.max(np.abs(all_capped))),
            "std_shift_vocab": float(np.std(all_capped)),
        }
        results.append(result)

        if (i + 1) % 25 == 0:
            flips = sum(r["argmax_flip"] for r in results)
            mean_dm = np.mean([r["delta_margin"] for r in results])
            base_acc = sum(r["base_correct"] for r in results) / len(results)
            adapt_acc = sum(r["adapt_correct"] for r in results) / len(results)
            print(f"  [{i+1}/{n}] base_acc={base_acc:.1%} adapt_acc={adapt_acc:.1%} "
                  f"flips={flips} ({flips/len(results):.1%}) mean_Δmargin={mean_dm:.3f}")

    # Aggregate
    n_results = len(results)
    flips = sum(r["argmax_flip"] for r in results)
    base_acc = sum(r["base_correct"] for r in results) / n_results
    adapt_acc = sum(r["adapt_correct"] for r in results) / n_results
    delta_margins = [r["delta_margin"] for r in results]
    shifts_on_correct = [r["shift_on_correct"] for r in results]
    shifts_on_wrong = [r["max_shift_on_wrong"] for r in results]

    # Categorize flips
    correct_to_wrong = sum(1 for r in results if r["base_correct"] and not r["adapt_correct"])
    wrong_to_correct = sum(1 for r in results if not r["base_correct"] and r["adapt_correct"])
    wrong_to_wrong_diff = sum(1 for r in results
                              if not r["base_correct"] and not r["adapt_correct"] and r["argmax_flip"])

    summary = {
        "n": n_results,
        "base_accuracy": base_acc,
        "adapter_accuracy": adapt_acc,
        "accuracy_delta": adapt_acc - base_acc,
        "total_argmax_flips": flips,
        "flip_rate": flips / n_results,
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "wrong_to_wrong_diff": wrong_to_wrong_diff,
        "mean_delta_margin": float(np.mean(delta_margins)),
        "median_delta_margin": float(np.median(delta_margins)),
        "std_delta_margin": float(np.std(delta_margins)),
        "mean_shift_on_correct": float(np.mean(shifts_on_correct)),
        "mean_max_shift_on_wrong": float(np.mean(shifts_on_wrong)),
        "mean_abs_shift_vocab": float(np.mean([r["mean_abs_shift_vocab"] for r in results])),
        "max_abs_shift_vocab": float(np.max([r["max_abs_shift_vocab"] for r in results])),
        "base_mc_entropy_mean": float(np.mean([r["base_entropy_mc"] for r in results])),
        "adapt_mc_entropy_mean": float(np.mean([r["adapt_entropy_mc"] for r in results])),
    }

    print(f"\n  ── TEST 1 SUMMARY ──")
    print(f"  Base accuracy:          {summary['base_accuracy']:.1%}")
    print(f"  Adapter accuracy:       {summary['adapter_accuracy']:.1%}")
    print(f"  Accuracy delta:         {summary['accuracy_delta']:+.1%}")
    print(f"  Argmax flips:           {summary['total_argmax_flips']} / {n_results} ({summary['flip_rate']:.1%})")
    print(f"    correct→wrong:        {summary['correct_to_wrong']}")
    print(f"    wrong→correct:        {summary['wrong_to_correct']}")
    print(f"    wrong→wrong(diff):    {summary['wrong_to_wrong_diff']}")
    print(f"  Mean Δmargin:           {summary['mean_delta_margin']:.4f}")
    print(f"  Median Δmargin:         {summary['median_delta_margin']:.4f}")
    print(f"  Mean shift on correct:  {summary['mean_shift_on_correct']:.4f}")
    print(f"  Mean max shift on wrong:{summary['mean_max_shift_on_wrong']:.4f}")
    print(f"  Mean |shift| (vocab):   {summary['mean_abs_shift_vocab']:.4f}")
    print(f"  Max |shift| (vocab):    {summary['max_abs_shift_vocab']:.4f}")
    print(f"  Base MC entropy:        {summary['base_mc_entropy_mean']:.4f}")
    print(f"  Adapter MC entropy:     {summary['adapt_mc_entropy_mean']:.4f}")

    with open(os.path.join(OUT_DIR, "test1_margin_disruption.json"), "w") as f:
        json.dump({"summary": summary, "per_example": results}, f, indent=2)

    return summary


# ===========================================================================
# TEST 2: Synthetic Perturbation Test
# ===========================================================================
def test2_synthetic_perturbation(model, tokenizer, lm_head, n=50):
    """Test MC fragility under random bounded shifts (no adapter)."""
    from datasets import load_dataset

    print(f"\n{'='*70}")
    print(f"  TEST 2: SYNTHETIC PERTURBATION TEST (n={n})")
    print(f"{'='*70}")

    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=456)  # Different seed
    ds = ds.select(range(min(n, len(ds))))

    choices = "ABCD"
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]

    shift_configs = [
        ("uniform_0.5", 0.5),
        ("uniform_1.0", 1.0),
        ("uniform_1.5", 1.5),
        ("uniform_2.0", 2.0),
        ("uniform_3.0", 3.0),
        ("uniform_5.0", 5.0),
    ]

    all_results = {}

    for shift_name, shift_mag in shift_configs:
        flips = 0
        base_correct_count = 0
        perturbed_correct_count = 0

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

            base_last = base_logits[0, -1, :]

            # Generate random shift on full vocab, clamp with tanh
            np.random.seed(i * 100 + hash(shift_name) % 10000)
            random_shift = np.random.uniform(-shift_mag, shift_mag, size=(int(base_last.shape[0]),))
            capped = shift_mag * np.tanh(random_shift / max(shift_mag, 1e-8))
            shift_mx = mx.array(capped.astype(np.float32))
            perturbed_last = base_last + shift_mx
            mx.eval(perturbed_last)

            base_answer = [float(base_last[cid]) for cid in choice_ids]
            pert_answer = [float(perturbed_last[cid]) for cid in choice_ids]

            base_pred = max(range(4), key=lambda j: base_answer[j])
            pert_pred = max(range(4), key=lambda j: pert_answer[j])

            if base_pred != pert_pred:
                flips += 1
            if base_pred == answer_idx:
                base_correct_count += 1
            if pert_pred == answer_idx:
                perturbed_correct_count += 1

        all_results[shift_name] = {
            "shift_magnitude": shift_mag,
            "flip_rate": flips / n,
            "base_accuracy": base_correct_count / n,
            "perturbed_accuracy": perturbed_correct_count / n,
            "accuracy_delta": (perturbed_correct_count - base_correct_count) / n,
            "n_flips": flips,
        }

    print(f"\n  ── TEST 2 RESULTS ──")
    print(f"  {'Shift':>12s} | {'Flip%':>6s} | {'Base':>6s} | {'Pert':>6s} | {'ΔAcc':>6s}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for name, r in all_results.items():
        print(f"  {name:>12s} | {r['flip_rate']:5.1%} | {r['base_accuracy']:5.1%} | "
              f"{r['perturbed_accuracy']:5.1%} | {r['accuracy_delta']:+5.1%}")

    with open(os.path.join(OUT_DIR, "test2_synthetic_perturbation.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ===========================================================================
# TEST 4: Loss-Term Proxy (forward-only on frozen adapter)
# ===========================================================================
def test4_loss_proxy(model, tokenizer, lm_head, adapter, n=100):
    """Check if proposed auxiliary losses would flag the bad behavior."""
    from datasets import load_dataset

    print(f"\n{'='*70}")
    print(f"  TEST 4: LOSS-TERM PROXY (n={n})")
    print(f"{'='*70}")

    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=789)
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

        adapted = apply_adapter(base_logits, adapter)
        mx.eval(adapted)

        base_last = base_logits[0, -1, :]
        adapt_last = adapted[0, -1, :]

        # 1. KL divergence (base || adapter) on full vocab
        base_lp = mx.softmax(base_last, axis=-1)
        adapt_lp = mx.softmax(adapt_last, axis=-1)
        mx.eval(base_lp)
        mx.eval(adapt_lp)
        base_p = np.array(base_lp.tolist())
        adapt_p = np.array(adapt_lp.tolist())
        # Clip for numerical stability
        base_p = np.clip(base_p, 1e-10, 1.0)
        adapt_p = np.clip(adapt_p, 1e-10, 1.0)
        kl_div = float(np.sum(base_p * np.log(base_p / adapt_p)))

        # 2. Margin loss: max(0, max(wrong) - correct + margin)
        base_answer_logits = [float(base_last[cid]) for cid in choice_ids]
        adapt_answer_logits = [float(adapt_last[cid]) for cid in choice_ids]

        correct_logit = adapt_answer_logits[answer_idx]
        max_wrong = max(adapt_answer_logits[j] for j in range(4) if j != answer_idx)
        margin_loss_0 = max(0.0, max_wrong - correct_logit)
        margin_loss_05 = max(0.0, max_wrong - correct_logit + 0.5)
        margin_loss_1 = max(0.0, max_wrong - correct_logit + 1.0)

        # 3. Shift penalty: mean(|shifts| - threshold).clamp(min=0)
        raw_shifts = adapter(base_logits)
        mx.eval(raw_shifts)
        shifts_last = np.array(raw_shifts[0, -1, :].tolist())
        capped = 2.0 * np.tanh(shifts_last / 2.0)
        shift_penalty_05 = float(np.mean(np.maximum(np.abs(capped) - 0.5, 0)))
        shift_penalty_10 = float(np.mean(np.maximum(np.abs(capped) - 1.0, 0)))

        # Classify
        base_pred = max(range(4), key=lambda j: base_answer_logits[j])
        adapt_pred = max(range(4), key=lambda j: adapt_answer_logits[j])

        results.append({
            "base_correct": base_pred == answer_idx,
            "adapt_correct": adapt_pred == answer_idx,
            "flipped": base_pred != adapt_pred,
            "kl_div": kl_div,
            "margin_loss_0": margin_loss_0,
            "margin_loss_05": margin_loss_05,
            "margin_loss_1": margin_loss_1,
            "shift_penalty_05": shift_penalty_05,
            "shift_penalty_10": shift_penalty_10,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] processed", flush=True)

    # Analyze: compare losses on flipped vs non-flipped
    flipped = [r for r in results if r["flipped"]]
    not_flipped = [r for r in results if not r["flipped"]]
    correct_to_wrong = [r for r in results if r["base_correct"] and not r["adapt_correct"]]

    def mean_field(lst, field):
        if not lst:
            return 0.0
        return float(np.mean([r[field] for r in lst]))

    summary = {
        "n": len(results),
        "n_flipped": len(flipped),
        "n_correct_to_wrong": len(correct_to_wrong),
        # KL divergence
        "kl_all": mean_field(results, "kl_div"),
        "kl_flipped": mean_field(flipped, "kl_div"),
        "kl_not_flipped": mean_field(not_flipped, "kl_div"),
        "kl_c2w": mean_field(correct_to_wrong, "kl_div"),
        # Margin loss (0)
        "margin0_all": mean_field(results, "margin_loss_0"),
        "margin0_flipped": mean_field(flipped, "margin_loss_0"),
        "margin0_c2w": mean_field(correct_to_wrong, "margin_loss_0"),
        # Margin loss (0.5)
        "margin05_all": mean_field(results, "margin_loss_05"),
        "margin05_c2w": mean_field(correct_to_wrong, "margin_loss_05"),
        # Shift penalty
        "shift_pen_05_all": mean_field(results, "shift_penalty_05"),
        "shift_pen_05_flipped": mean_field(flipped, "shift_penalty_05"),
        "shift_pen_10_all": mean_field(results, "shift_penalty_10"),
        "shift_pen_10_flipped": mean_field(flipped, "shift_penalty_10"),
    }

    print(f"\n  ── TEST 4 RESULTS ──")
    print(f"  Total: {summary['n']}, Flipped: {summary['n_flipped']}, C→W: {summary['n_correct_to_wrong']}")
    print(f"\n  KL(base||adapter):")
    print(f"    All:           {summary['kl_all']:.4f}")
    print(f"    Flipped:       {summary['kl_flipped']:.4f}")
    print(f"    Not flipped:   {summary['kl_not_flipped']:.4f}")
    print(f"    Correct→Wrong: {summary['kl_c2w']:.4f}")
    print(f"\n  Margin loss (would be 0 if correct answer wins):")
    print(f"    margin=0  all={summary['margin0_all']:.4f}  flipped={summary['margin0_flipped']:.4f}  c2w={summary['margin0_c2w']:.4f}")
    print(f"    margin=0.5 all={summary['margin05_all']:.4f}  c2w={summary['margin05_c2w']:.4f}")
    print(f"\n  Shift penalty:")
    print(f"    thresh=0.5: all={summary['shift_pen_05_all']:.4f}  flipped={summary['shift_pen_05_flipped']:.4f}")
    print(f"    thresh=1.0: all={summary['shift_pen_10_all']:.4f}  flipped={summary['shift_pen_10_flipped']:.4f}")

    with open(os.path.join(OUT_DIR, "test4_loss_proxy.json"), "w") as f:
        json.dump({"summary": summary}, f, indent=2)

    return summary


# ===========================================================================
# TEST 5: Generation vs MC Discrepancy
# ===========================================================================
def test5_gen_vs_mc(model, tokenizer, lm_head, adapter, n_mc=50, n_gen=20):
    """Compare adapter behavior on MC vs open-ended generation."""

    print(f"\n{'='*70}")
    print(f"  TEST 5: GENERATION vs MC DISCREPANCY")
    print(f"{'='*70}")

    # Part A: MC logit entropy on MMLU
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=999)
    ds = ds.select(range(min(n_mc, len(ds))))

    choices = "ABCD"
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]

    mc_results = []
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

        adapted = apply_adapter(base_logits, adapter)
        mx.eval(adapted)

        base_last = base_logits[0, -1, :]
        adapt_last = adapted[0, -1, :]

        # Full vocab entropy
        def entropy_topk(logits_mx, k=100):
            vals = np.array(logits_mx.tolist())
            top_idx = np.argsort(vals)[-k:]
            top_vals = vals[top_idx]
            top_vals -= top_vals.max()
            p = np.exp(top_vals) / np.exp(top_vals).sum()
            return -np.sum(p * np.log(p + 1e-10))

        base_ent = entropy_topk(base_last)
        adapt_ent = entropy_topk(adapt_last)

        # Top-1 confidence
        base_p = float(mx.softmax(base_last, axis=-1).max())
        adapt_p = float(mx.softmax(adapt_last, axis=-1).max())

        mc_results.append({
            "base_entropy": base_ent,
            "adapt_entropy": adapt_ent,
            "entropy_delta": adapt_ent - base_ent,
            "base_top1_conf": base_p,
            "adapt_top1_conf": adapt_p,
        })

    # Part B: Open-ended generation entropy
    gen_prompts = [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "Write a Python function to sort a list.",
        "What causes earthquakes?",
        "Describe the water cycle.",
        "What is machine learning?",
        "How does the internet work?",
        "What is DNA?",
        "Explain gravity in simple terms.",
        "What are the planets in our solar system?",
        "How do vaccines work?",
        "What is the theory of evolution?",
        "Explain how a combustion engine works.",
        "What is the difference between weather and climate?",
        "How do computers store data?",
        "What causes the seasons?",
        "Explain the concept of inflation.",
        "How does the human heart work?",
        "What is artificial intelligence?",
        "Describe the structure of an atom.",
    ]

    gen_results = []
    for i, prompt in enumerate(gen_prompts[:n_gen]):
        full_prompt = ALPACA_TEMPLATE.format(instruction=prompt)
        tokens = mx.array(tokenizer.encode(full_prompt))[None, :]

        h = model.model(tokens)
        mx.eval(h)
        base_logits = lm_head(h)
        mx.eval(base_logits)

        adapted = apply_adapter(base_logits, adapter)
        mx.eval(adapted)

        base_last = base_logits[0, -1, :]
        adapt_last = adapted[0, -1, :]

        def entropy_topk(logits_mx, k=100):
            vals = np.array(logits_mx.tolist())
            top_idx = np.argsort(vals)[-k:]
            top_vals = vals[top_idx]
            top_vals -= top_vals.max()
            p = np.exp(top_vals) / np.exp(top_vals).sum()
            return -np.sum(p * np.log(p + 1e-10))

        base_ent = entropy_topk(base_last)
        adapt_ent = entropy_topk(adapt_last)
        base_p = float(mx.softmax(base_last, axis=-1).max())
        adapt_p = float(mx.softmax(adapt_last, axis=-1).max())

        # Top-1 token comparison
        base_top1 = int(mx.argmax(base_last))
        adapt_top1 = int(mx.argmax(adapt_last))
        top1_same = base_top1 == adapt_top1

        gen_results.append({
            "prompt": prompt[:50],
            "base_entropy": base_ent,
            "adapt_entropy": adapt_ent,
            "entropy_delta": adapt_ent - base_ent,
            "base_top1_conf": base_p,
            "adapt_top1_conf": adapt_p,
            "top1_same": top1_same,
            "base_top1_token": tokenizer.decode([base_top1]),
            "adapt_top1_token": tokenizer.decode([adapt_top1]),
        })

    # Summary
    mc_ent_base = np.mean([r["base_entropy"] for r in mc_results])
    mc_ent_adapt = np.mean([r["adapt_entropy"] for r in mc_results])
    gen_ent_base = np.mean([r["base_entropy"] for r in gen_results])
    gen_ent_adapt = np.mean([r["adapt_entropy"] for r in gen_results])
    gen_top1_same = sum(r["top1_same"] for r in gen_results) / len(gen_results)

    summary = {
        "mc_base_entropy": float(mc_ent_base),
        "mc_adapt_entropy": float(mc_ent_adapt),
        "mc_entropy_delta": float(mc_ent_adapt - mc_ent_base),
        "gen_base_entropy": float(gen_ent_base),
        "gen_adapt_entropy": float(gen_ent_adapt),
        "gen_entropy_delta": float(gen_ent_adapt - gen_ent_base),
        "gen_top1_agreement": gen_top1_same,
        "mc_base_top1_conf": float(np.mean([r["base_top1_conf"] for r in mc_results])),
        "mc_adapt_top1_conf": float(np.mean([r["adapt_top1_conf"] for r in mc_results])),
        "gen_base_top1_conf": float(np.mean([r["base_top1_conf"] for r in gen_results])),
        "gen_adapt_top1_conf": float(np.mean([r["adapt_top1_conf"] for r in gen_results])),
    }

    print(f"\n  ── TEST 5 RESULTS ──")
    print(f"  MC prompts (n={len(mc_results)}):")
    print(f"    Base entropy:    {summary['mc_base_entropy']:.4f}")
    print(f"    Adapter entropy: {summary['mc_adapt_entropy']:.4f}")
    print(f"    Δ entropy:       {summary['mc_entropy_delta']:+.4f}")
    print(f"    Base top-1 conf: {summary['mc_base_top1_conf']:.4f}")
    print(f"    Adapt top-1 conf:{summary['mc_adapt_top1_conf']:.4f}")
    print(f"\n  Generation prompts (n={len(gen_results)}):")
    print(f"    Base entropy:    {summary['gen_base_entropy']:.4f}")
    print(f"    Adapter entropy: {summary['gen_adapt_entropy']:.4f}")
    print(f"    Δ entropy:       {summary['gen_entropy_delta']:+.4f}")
    print(f"    Top-1 agreement: {summary['gen_top1_agreement']:.1%}")
    print(f"    Base top-1 conf: {summary['gen_base_top1_conf']:.4f}")
    print(f"    Adapt top-1 conf:{summary['gen_adapt_top1_conf']:.4f}")

    print(f"\n  ── Top-1 token comparison (generation) ──")
    for r in gen_results:
        marker = "✓" if r["top1_same"] else "✗"
        print(f"    {marker} {r['prompt']:50s} base='{r['base_top1_token']}' adapt='{r['adapt_top1_token']}'")

    with open(os.path.join(OUT_DIR, "test5_gen_vs_mc.json"), "w") as f:
        json.dump({"summary": summary, "mc_results": mc_results, "gen_results": gen_results}, f, indent=2)

    return summary


# ===========================================================================
# BONUS: Max-shift sweep (what max_shift preserves MC accuracy?)
# ===========================================================================
def test_bonus_maxshift_sweep(model, tokenizer, lm_head, adapter, n=100):
    """Sweep max_shift values to find where MC accuracy is preserved."""
    from datasets import load_dataset

    print(f"\n{'='*70}")
    print(f"  BONUS: MAX_SHIFT SWEEP (n={n})")
    print(f"{'='*70}")

    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=321)
    ds = ds.select(range(min(n, len(ds))))

    choices = "ABCD"
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]

    max_shifts = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    # Pre-compute base logits
    all_base = []
    all_raw_shifts = []
    all_answers = []
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

        raw_shifts = adapter(base_logits)
        mx.eval(raw_shifts)

        all_base.append(np.array(base_logits[0, -1, :].tolist()))
        all_raw_shifts.append(np.array(raw_shifts[0, -1, :].tolist()))
        all_answers.append(answer_idx)

        if (i + 1) % 50 == 0:
            print(f"  Pre-computed {i+1}/{n}", flush=True)

    results = {}
    for ms in max_shifts:
        correct = 0
        for i in range(len(all_base)):
            base = all_base[i]
            raw = all_raw_shifts[i]
            if ms == 0.0:
                adapted_last = base
            else:
                capped = ms * np.tanh(raw / ms)
                adapted_last = base + capped

            answer_logits = [adapted_last[cid] for cid in choice_ids]
            pred = max(range(4), key=lambda j: answer_logits[j])
            if pred == all_answers[i]:
                correct += 1

        acc = correct / len(all_base)
        results[str(ms)] = {"max_shift": ms, "accuracy": acc, "correct": correct}

    print(f"\n  ── MAX_SHIFT SWEEP ──")
    print(f"  {'max_shift':>10s} | {'Accuracy':>8s}")
    print(f"  {'-'*10}-+-{'-'*8}")
    for ms in max_shifts:
        r = results[str(ms)]
        marker = " ←base" if ms == 0.0 else ("" if r["accuracy"] > 0.5 else " !!LOW")
        print(f"  {ms:10.1f} | {r['accuracy']:7.1%}{marker}")

    with open(os.path.join(OUT_DIR, "test_bonus_maxshift_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    t0 = time.time()
    model, tokenizer, lm_head, adapter = load_everything()

    print(f"\nModel loaded in {time.time()-t0:.1f}s")
    print(f"Output directory: {OUT_DIR}\n")

    # Run all tests
    t1 = time.time()
    s1 = test1_margin_disruption(model, tokenizer, lm_head, adapter, n=100)
    print(f"  Test 1 took {time.time()-t1:.1f}s")

    t2 = time.time()
    s2 = test2_synthetic_perturbation(model, tokenizer, lm_head, n=50)
    print(f"  Test 2 took {time.time()-t2:.1f}s")

    t4 = time.time()
    s4 = test4_loss_proxy(model, tokenizer, lm_head, adapter, n=100)
    print(f"  Test 4 took {time.time()-t4:.1f}s")

    t5 = time.time()
    s5 = test5_gen_vs_mc(model, tokenizer, lm_head, adapter, n_mc=50, n_gen=20)
    print(f"  Test 5 took {time.time()-t5:.1f}s")

    tb = time.time()
    sb = test_bonus_maxshift_sweep(model, tokenizer, lm_head, adapter, n=100)
    print(f"  Bonus sweep took {time.time()-tb:.1f}s")

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  ALL DIAGNOSTICS COMPLETE — {total:.0f}s ({total/60:.1f}min)")
    print(f"  Results in: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
