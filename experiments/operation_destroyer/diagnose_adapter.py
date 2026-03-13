#!/usr/bin/env python3
"""Diagnose: WHY is the v8 adapter not improving any benchmarks?

Check logit margins on questions the base gets wrong.
Is max_shift=1.0 enough to flip answers?
"""

import sys
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/snap_on")

import mlx.core as mx
import mlx_lm
from module import SnapOnConfig, create_adapter
from experiments.operation_destroyer.train_v3 import (
    get_lm_head_fn, ALPACA_TEMPLATE, ADAPTER_MAX_SHIFT,
)

import os
import json

RESULTS_DIR = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/operation_destroyer"


def main():
    print("Loading model...")
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = get_lm_head_fn(model)
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    try:
        vocab_size = lm_head.weight.shape[0]
    except AttributeError:
        vocab_size = model.model.embed_tokens.weight.shape[0]

    config = SnapOnConfig(
        d_model=d_model, d_inner=128, n_layers=0,
        n_heads=8, mode="logit", vocab_size=vocab_size,
    )
    adapter = create_adapter(config)

    # Load v8 best weights
    best_path = os.path.join(RESULTS_DIR, "v8", "best.npz")
    weights = mx.load(best_path)
    adapter.load_weights(list(weights.items()))
    mx.eval(adapter.parameters())
    print("  Loaded v8 best adapter")

    # Run MMLU diagnostic
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(200))

    choices = "ABCD"
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]

    max_shift = 1.0  # v8's cap

    base_wrong_margins = []  # How much the wrong answer beats correct (base)
    adapter_shifts_on_wrong = []  # What the adapter does to those margins
    flips_good = 0  # Base wrong -> adapter right
    flips_bad = 0   # Base right -> adapter wrong
    both_right = 0
    both_wrong = 0
    all_shifts = []  # All shift magnitudes at answer position

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
        shifts = max_shift * mx.tanh(raw_shifts / max_shift)
        adapted = base_logits + shifts
        mx.eval(adapted)
        mx.eval(shifts)

        # Get logits at last position for answer tokens
        base_last = base_logits[0, -1, :]
        adapted_last = adapted[0, -1, :]
        shift_last = shifts[0, -1, :]

        base_answer_logits = [float(base_last[cid]) for cid in choice_ids]
        adapted_answer_logits = [float(adapted_last[cid]) for cid in choice_ids]
        shift_answer_logits = [float(shift_last[cid]) for cid in choice_ids]

        base_pred = max(range(4), key=lambda j: base_answer_logits[j])
        adapter_pred = max(range(4), key=lambda j: adapted_answer_logits[j])

        # Mean absolute shift at answer position (across all vocab)
        mean_abs_shift = float(shifts[0, -1, :].abs().mean())
        all_shifts.append(mean_abs_shift)

        # Shift on answer tokens specifically
        correct_base_logit = base_answer_logits[answer_idx]
        best_wrong_base = max(base_answer_logits[j] for j in range(4) if j != answer_idx)
        base_margin = correct_base_logit - best_wrong_base  # positive = base correct

        correct_adapted_logit = adapted_answer_logits[answer_idx]
        best_wrong_adapted = max(adapted_answer_logits[j] for j in range(4) if j != answer_idx)
        adapted_margin = correct_adapted_logit - best_wrong_adapted

        base_right = (base_pred == answer_idx)
        adapter_right = (adapter_pred == answer_idx)

        if base_right and adapter_right:
            both_right += 1
        elif base_right and not adapter_right:
            flips_bad += 1
        elif not base_right and adapter_right:
            flips_good += 1
        elif not base_right and not adapter_right:
            both_wrong += 1
            base_wrong_margins.append(base_margin)  # negative
            adapter_shifts_on_wrong.append(adapted_margin - base_margin)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/200 processed")

    n = len(ds)
    print(f"\n{'='*70}")
    print(f"  ADAPTER DIAGNOSTIC (MMLU, n={n})")
    print(f"{'='*70}")
    print(f"  Both right:       {both_right:3d} ({both_right/n:.1%})")
    print(f"  Both wrong:       {both_wrong:3d} ({both_wrong/n:.1%})")
    print(f"  Base wrong→right: {flips_good:3d} ({flips_good/n:.1%})  ← IMPROVEMENT")
    print(f"  Base right→wrong: {flips_bad:3d} ({flips_bad/n:.1%})  ← REGRESSION")
    print(f"  Net effect:       {flips_good - flips_bad:+d}")

    print(f"\n  Mean |shift| at answer position: {sum(all_shifts)/len(all_shifts):.4f}")
    print(f"  Max |shift| at answer position:  {max(all_shifts):.4f}")

    if base_wrong_margins:
        avg_wrong_margin = sum(base_wrong_margins) / len(base_wrong_margins)
        print(f"\n  On {len(base_wrong_margins)} questions base gets wrong:")
        print(f"    Avg base margin (correct - best_wrong): {avg_wrong_margin:.3f}")
        print(f"    Min margin (hardest to flip):           {min(base_wrong_margins):.3f}")
        print(f"    Max margin (easiest to flip):           {max(base_wrong_margins):.3f}")
        n_flippable = sum(1 for m in base_wrong_margins if abs(m) < 2.0)
        print(f"    Flippable with ±1.0 shift (margin < 2.0): {n_flippable}/{len(base_wrong_margins)}")

    if adapter_shifts_on_wrong:
        avg_shift = sum(adapter_shifts_on_wrong) / len(adapter_shifts_on_wrong)
        print(f"\n    Avg adapter margin change on wrong Qs: {avg_shift:+.4f}")
        n_helpful = sum(1 for s in adapter_shifts_on_wrong if s > 0)
        print(f"    Adapter shifts margin toward correct:   {n_helpful}/{len(adapter_shifts_on_wrong)}")


    # Check: is the shift truly uniform, or is it token-dependent?
    # Run one more question and look at the actual shift distribution
    ex = ds[0]
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
    capped_shifts = max_shift * mx.tanh(raw_shifts / max_shift)
    mx.eval(raw_shifts)
    mx.eval(capped_shifts)

    # Look at shift at last position
    raw_last = raw_shifts[0, -1, :]
    capped_last = capped_shifts[0, -1, :]

    import numpy as np
    raw_np = np.array(raw_last.tolist())
    capped_np = np.array(capped_last.tolist())

    print(f"\n  Shift distribution at last position (vocab={len(raw_np)}):")
    print(f"    Raw shifts:   mean={raw_np.mean():.4f}, std={raw_np.std():.4f}, "
          f"min={raw_np.min():.4f}, max={raw_np.max():.4f}")
    print(f"    Capped shifts: mean={capped_np.mean():.4f}, std={capped_np.std():.4f}, "
          f"min={capped_np.min():.4f}, max={capped_np.max():.4f}")

    # How many are saturated (|capped| > 0.99)?
    n_saturated = np.sum(np.abs(capped_np) > 0.99)
    print(f"    Saturated (|shift| > 0.99): {n_saturated}/{len(capped_np)} ({n_saturated/len(capped_np):.1%})")

    # Are shifts positive or negative?
    n_pos = np.sum(capped_np > 0)
    n_neg = np.sum(capped_np < 0)
    print(f"    Positive: {n_pos}, Negative: {n_neg}")

    # Shift on the 4 answer tokens specifically
    print(f"\n    Answer token shifts:")
    for j in range(4):
        cid = choice_ids[j]
        marker = " ← CORRECT" if j == answer_idx else ""
        print(f"      {choices[j]} (tok {cid}): base={float(base_logits[0,-1,cid]):.3f}, "
              f"shift={float(capped_last[cid]):.4f}, "
              f"adapted={float(base_logits[0,-1,cid]) + float(capped_last[cid]):.3f}{marker}")


if __name__ == "__main__":
    main()
