#!/usr/bin/env python3
"""Diagnose v11 adapter — why MMLU dropped 17.4%."""

import sys
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/snap_on")

import mlx.core as mx
import mlx_lm
import numpy as np
from module import SnapOnConfig, create_adapter
from experiments.operation_destroyer.train_v3 import get_lm_head_fn, apply_adapter, ALPACA_TEMPLATE
import experiments.operation_destroyer.train_v3 as t3

t3.LOGIT_SOFTCAP = 30.0

# Load model + adapter
model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
model.freeze()
lm_head = get_lm_head_fn(model)
vocab_size = model.model.embed_tokens.weight.shape[0]
d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]

config = SnapOnConfig(d_model=d_model, d_inner=128, n_layers=0, n_heads=8,
                      mode="logit", vocab_size=vocab_size)
adapter = create_adapter(config)
weights = mx.load("results/operation_destroyer/v11/best.npz")
adapter.load_weights(list(weights.items()))
mx.eval(adapter.parameters())

# Run 100 MMLU questions
from datasets import load_dataset
ds = load_dataset("cais/mmlu", "all", split="test")
ds = ds.shuffle(seed=42)
ds = ds.select(range(100))

choices = "ABCD"
choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]

flips_to_correct = 0
flips_to_wrong = 0
no_change = 0
all_shift_mags = []
all_answer_shifts = []
all_base_gaps = []
all_adapted_gaps = []
base_correct = 0
adapter_correct = 0

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

    adapted = apply_adapter(adapter, base_logits)
    mx.eval(adapted)

    raw_shifts = adapter(base_logits)
    mx.eval(raw_shifts)
    centered = raw_shifts - raw_shifts.mean(axis=-1, keepdims=True)
    mx.eval(centered)

    base_last = base_logits[0, -1, :]
    adapted_last = adapted[0, -1, :]
    centered_last = centered[0, -1, :]

    # Shift magnitude
    shift_np = np.array(centered_last.tolist())
    all_shift_mags.append(np.abs(shift_np).mean())

    # Answer token analysis
    base_ans = [float(base_last[cid]) for cid in choice_ids]
    adapted_ans = [float(adapted_last[cid]) for cid in choice_ids]
    shift_ans = [float(centered_last[cid]) for cid in choice_ids]
    all_answer_shifts.append(shift_ans)

    # Base gap: correct - max_wrong
    base_correct_logit = base_ans[answer_idx]
    base_wrong_max = max(base_ans[j] for j in range(4) if j != answer_idx)
    all_base_gaps.append(base_correct_logit - base_wrong_max)

    adapted_correct_logit = adapted_ans[answer_idx]
    adapted_wrong_max = max(adapted_ans[j] for j in range(4) if j != answer_idx)
    all_adapted_gaps.append(adapted_correct_logit - adapted_wrong_max)

    base_pred = np.argmax(base_ans)
    adapter_pred = np.argmax(adapted_ans)

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

    if i < 5:
        print(f"\nQ{i}: correct={choices[answer_idx]}")
        print(f"  Base logits:    {['%.2f' % x for x in base_ans]}")
        print(f"  Adapted logits: {['%.2f' % x for x in adapted_ans]}")
        print(f"  Shifts (ABCD):  {['%+.2f' % x for x in shift_ans]}")
        print(f"  Base pred: {choices[base_pred]} | Adapter pred: {choices[adapter_pred]}")

print("\n" + "=" * 70)
print("  V11 ADAPTER DIAGNOSIS (100 MMLU questions)")
print("=" * 70)
print(f"Base correct:     {base_correct}/100 = {base_correct/100:.1%}")
print(f"Adapter correct:  {adapter_correct}/100 = {adapter_correct/100:.1%}")
print(f"Flips to correct: {flips_to_correct}")
print(f"Flips to wrong:   {flips_to_wrong}")
print(f"No change:        {no_change}")

shifts_arr = np.array(all_shift_mags)
print(f"\nMean |centered shift| (all 152K logits): {shifts_arr.mean():.4f}")
print(f"Max:  {shifts_arr.max():.4f}")
print(f"Min:  {shifts_arr.min():.4f}")

# Answer token shifts
answer_shifts = np.array(all_answer_shifts)
print("\nPer-answer-token centered shifts (mean across 100 questions):")
for j in range(4):
    print(f"  {choices[j]}: mean={answer_shifts[:, j].mean():+.4f}  "
          f"std={answer_shifts[:, j].std():.4f}  "
          f"range=[{answer_shifts[:, j].min():+.4f}, {answer_shifts[:, j].max():+.4f}]")

abcd_range = answer_shifts.max(axis=1) - answer_shifts.min(axis=1)
print(f"\nABCD shift spread: mean={abcd_range.mean():.4f}  "
      f"min={abcd_range.min():.4f}  max={abcd_range.max():.4f}")

# Margin analysis
base_gaps = np.array(all_base_gaps)
adapted_gaps = np.array(all_adapted_gaps)
print(f"\nCorrect-vs-wrong margin (correct_logit - max_wrong_logit):")
print(f"  Base:    mean={base_gaps.mean():+.4f}  median={np.median(base_gaps):+.4f}")
print(f"  Adapted: mean={adapted_gaps.mean():+.4f}  median={np.median(adapted_gaps):+.4f}")
print(f"  Delta:   mean={np.mean(adapted_gaps - base_gaps):+.4f}")

# How big are shifts vs base logit magnitudes?
print(f"\nBase logit magnitude at answer tokens: ~{np.mean([np.abs(s).mean() for s in all_answer_shifts]):.2f}")

# Check if softcap is squishing base logits
print(f"\n--- Softcap effect on base logits ---")
# Re-run one example to check
tokens0 = mx.array(tokenizer.encode(ALPACA_TEMPLATE.format(
    instruction=ds[0]["question"] + "\nA. " + ds[0]["choices"][0] +
    "\nB. " + ds[0]["choices"][1] + "\nC. " + ds[0]["choices"][2] +
    "\nD. " + ds[0]["choices"][3] + "\nAnswer:")))[None, :]
h0 = model.model(tokens0)
mx.eval(h0)
bl0 = lm_head(h0)
mx.eval(bl0)
raw0 = bl0[0, -1, :]
capped0 = 30.0 * mx.tanh(raw0 / 30.0)
mx.eval(capped0)

raw_np = np.array(raw0.tolist())
capped_np = np.array(capped0.tolist())

print(f"  Raw base logits: mean={raw_np.mean():.2f}  std={raw_np.std():.2f}  "
      f"min={raw_np.min():.2f}  max={raw_np.max():.2f}")
print(f"  Softcapped:      mean={capped_np.mean():.2f}  std={capped_np.std():.2f}  "
      f"min={capped_np.min():.2f}  max={capped_np.max():.2f}")
print(f"  Answer tokens raw:    {[f'{float(raw0[cid]):.2f}' for cid in choice_ids]}")
print(f"  Answer tokens capped: {[f'{float(capped0[cid]):.2f}' for cid in choice_ids]}")
squish = np.abs(raw_np) - np.abs(capped_np)
print(f"  Squish (|raw|-|capped|): mean={squish.mean():.4f}  "
      f"max={squish.max():.4f}  (tokens with |raw|>30 get squished)")
n_squished = np.sum(np.abs(raw_np) > 25)
print(f"  Tokens with |raw logit| > 25: {n_squished} / {len(raw_np)} = {n_squished/len(raw_np):.1%}")
