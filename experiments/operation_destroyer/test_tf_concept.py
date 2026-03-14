#!/usr/bin/env python3
"""Test True/False training concept at tiny scale.

Instead of training on ABCD multiple choice, train on True/False classification
of individual answers. This eliminates position bias entirely.

Tests two approaches on a small MMLU sample:
1. Standard MC format: "Q\nA. ...\nB. ...\nAnswer:"  → train on correct letter
2. T/F format: "Q: ... Proposed answer: ... Is this correct?"  → train True/False

Compares which approach produces better MC accuracy when evaluated on standard
ABCD format.

Usage:
    python experiments/operation_destroyer/test_tf_concept.py
"""

import sys
import time
import random
import numpy as np

sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/snap_on")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
from mlx.utils import tree_flatten

from module import SnapOnConfig, create_adapter


ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

TF_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Question: {question}\n"
    "Proposed answer: {answer}\n"
    "Is the proposed answer correct? Reply with True or False.\n\n"
    "### Response:\n"
)


def get_lm_head_fn(model):
    """Get the correct lm_head function for the model."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    return model.model.embed_tokens.as_linear


def apply_adapter(adapter, base_logits, softcap=30.0):
    """Apply adapter with centering + softcap."""
    raw_shifts = adapter(base_logits)
    centered = raw_shifts - raw_shifts.mean(axis=-1, keepdims=True)
    combined = base_logits + centered
    return softcap * mx.tanh(combined / softcap)


def build_mc_data(tokenizer, ds, n):
    """Build standard MC training data."""
    choices = "ABCD"
    examples = []
    for ex in ds:
        if len(examples) >= n:
            break
        q = ex["question"]
        options = ex["choices"]
        answer_idx = ex["answer"]

        # Shuffle options
        order = list(range(len(options)))
        random.shuffle(order)

        prompt_text = q + "\n"
        mapped_correct = -1
        for j, oi in enumerate(order):
            prompt_text += f"{choices[j]}. {options[oi]}\n"
            if oi == answer_idx:
                mapped_correct = j
        prompt_text += "Answer:"

        correct_letter = choices[mapped_correct]
        response = f" {correct_letter}"

        full_prompt = ALPACA_TEMPLATE.format(instruction=prompt_text)
        prompt_tokens = tokenizer.encode(full_prompt)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        full_tokens = prompt_tokens + response_tokens

        examples.append({
            "tokens": full_tokens,
            "prompt_len": len(prompt_tokens),
        })

    return examples


def build_tf_contrastive_data(tokenizer, ds, n):
    """Build True/False contrastive training data.

    For each MC question, create contrastive pairs:
    - One True example (correct answer)
    - One False example (randomly chosen wrong answer)
    This ensures 50/50 class balance.
    """
    examples = []
    for ex in ds:
        if len(examples) >= n:
            break
        q = ex["question"]
        options = ex["choices"]
        answer_idx = ex["answer"]

        # True example: correct answer
        correct_text = options[answer_idx]
        true_prompt = TF_TEMPLATE.format(question=q, answer=correct_text)
        true_response = " True"
        true_ptok = tokenizer.encode(true_prompt)
        true_rtok = tokenizer.encode(true_response, add_special_tokens=False)

        # False example: random wrong answer
        wrong_indices = [i for i in range(len(options)) if i != answer_idx]
        wrong_idx = random.choice(wrong_indices)
        wrong_text = options[wrong_idx]
        false_prompt = TF_TEMPLATE.format(question=q, answer=wrong_text)
        false_response = " False"
        false_ptok = tokenizer.encode(false_prompt)
        false_rtok = tokenizer.encode(false_response, add_special_tokens=False)

        examples.append({
            "tokens": true_ptok + true_rtok,
            "prompt_len": len(true_ptok),
        })
        examples.append({
            "tokens": false_ptok + false_rtok,
            "prompt_len": len(false_ptok),
        })

    random.shuffle(examples)
    return examples


def train_adapter(model, tokenizer, adapter, train_data, lr=1e-4, steps=None):
    """Quick training loop."""
    lm_head = get_lm_head_fn(model)

    if steps is None:
        steps = len(train_data)

    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(adapter, h, targets, mask):
        base_logits = lm_head(h)
        mx.eval(base_logits)
        raw_shifts = adapter(base_logits)
        centered = raw_shifts - raw_shifts.mean(axis=-1, keepdims=True)
        combined = base_logits + centered
        adapted_logits = (30.0 * mx.tanh(combined / 30.0))[:, :-1, :]

        ce = nn.losses.cross_entropy(adapted_logits, targets, reduction="none")
        n_tok = mask.sum()
        loss = (ce * mask).sum() / mx.maximum(n_tok, mx.array(1.0))

        # L2 shift penalty
        shift_l2 = mx.mean(centered ** 2)
        loss = loss + 0.01 * shift_l2
        return loss

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    total_loss = 0
    for i in range(min(steps, len(train_data))):
        ex = train_data[i % len(train_data)]
        tokens = ex["tokens"]
        prompt_len = ex["prompt_len"]

        input_ids = mx.array(tokens)[None, :]
        targets = mx.array(tokens[1:])[None, :]

        # Mask: only train on response tokens
        mask = mx.zeros_like(targets, dtype=mx.float32)
        for j in range(prompt_len - 1, len(tokens) - 1):
            mask = mask.at[0, j].add(1.0)

        h = model.model(input_ids)
        mx.eval(h)

        loss, grads = loss_and_grad(adapter, h, targets, mask)
        optimizer.apply_gradients(grads, adapter)
        mx.eval(adapter.parameters(), optimizer.state)

        total_loss += float(loss)

        if (i + 1) % 50 == 0:
            print(f"    step {i+1}/{steps}  loss={total_loss/(i+1):.4f}")

    return total_loss / min(steps, len(train_data))


def evaluate_mc(model, tokenizer, adapter, ds, n=50):
    """Evaluate on standard ABCD MC format."""
    lm_head = get_lm_head_fn(model)
    choices = "ABCD"
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]

    base_correct = 0
    adapter_correct = 0
    base_pred_dist = [0, 0, 0, 0]
    adapter_pred_dist = [0, 0, 0, 0]

    for i, ex in enumerate(ds):
        if i >= n:
            break
        q = ex["question"]
        options = ex["choices"]
        answer_idx = ex["answer"]

        prompt_text = q + "\n"
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

        base_last = base_logits[0, -1, :]
        adapted_last = adapted[0, -1, :]

        base_ans = [float(base_last[cid]) for cid in choice_ids]
        adapted_ans = [float(adapted_last[cid]) for cid in choice_ids]

        base_pred = int(np.argmax(base_ans))
        adapter_pred = int(np.argmax(adapted_ans))

        base_pred_dist[base_pred] += 1
        adapter_pred_dist[adapter_pred] += 1

        if base_pred == answer_idx:
            base_correct += 1
        if adapter_pred == answer_idx:
            adapter_correct += 1

    return {
        "base_correct": base_correct,
        "adapter_correct": adapter_correct,
        "n": n,
        "base_pct": base_correct / n,
        "adapter_pct": adapter_correct / n,
        "delta": (adapter_correct - base_correct) / n,
        "base_pred_dist": base_pred_dist,
        "adapter_pred_dist": adapter_pred_dist,
    }


def evaluate_tf(model, tokenizer, adapter, ds, n=50):
    """Evaluate using True/False scoring instead of ABCD.

    For each question, present each option in T/F format and pick the one
    with highest P(True).
    """
    lm_head = get_lm_head_fn(model)
    true_id = tokenizer.encode(" True", add_special_tokens=False)[-1]
    false_id = tokenizer.encode(" False", add_special_tokens=False)[-1]

    base_correct = 0
    adapter_correct = 0

    for i, ex in enumerate(ds):
        if i >= n:
            break
        q = ex["question"]
        options = ex["choices"]
        answer_idx = ex["answer"]

        base_scores = []
        adapter_scores = []

        for opt in options:
            prompt = TF_TEMPLATE.format(question=q, answer=opt)
            tokens = mx.array(tokenizer.encode(prompt))[None, :]

            h = model.model(tokens)
            mx.eval(h)
            base_logits = lm_head(h)
            mx.eval(base_logits)

            adapted = apply_adapter(adapter, base_logits)
            mx.eval(adapted)

            # Score = logit(True) - logit(False)
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

    return {
        "base_correct": base_correct,
        "adapter_correct": adapter_correct,
        "n": n,
        "base_pct": base_correct / n,
        "adapter_pct": adapter_correct / n,
        "delta": (adapter_correct - base_correct) / n,
    }


def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("  TRUE/FALSE CONCEPT TEST")
    print("  Compare MC training vs T/F contrastive training")
    print("=" * 70)

    # Load model
    print("\nLoading Qwen3-4B-Base...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Load MMLU
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42)

    # Use first 100 for training, next 50 for eval
    train_ds = ds.select(range(100))
    eval_ds = ds.select(range(100, 150))

    # Build training data
    print("\nBuilding MC training data (100 examples)...")
    mc_train = build_mc_data(tokenizer, train_ds, 100)
    print(f"  MC: {len(mc_train)} examples")

    print("Building T/F contrastive data (100 questions → ~200 examples)...")
    tf_train = build_tf_contrastive_data(tokenizer, train_ds, 100)
    print(f"  T/F: {len(tf_train)} examples")

    # Create two adapters
    config = SnapOnConfig(d_model=d_model, d_inner=128, n_layers=0,
                          n_heads=8, mode="logit", vocab_size=vocab_size)

    # Train MC adapter
    print("\n" + "=" * 70)
    print("  TRAINING MC ADAPTER (100 steps)")
    print("=" * 70)
    adapter_mc = create_adapter(config)
    mc_loss = train_adapter(model, tokenizer, adapter_mc, mc_train, lr=1e-5, steps=100)
    print(f"  Final loss: {mc_loss:.4f}")

    # Train T/F adapter
    print("\n" + "=" * 70)
    print("  TRAINING T/F ADAPTER (100 steps)")
    print("=" * 70)
    adapter_tf = create_adapter(config)
    tf_loss = train_adapter(model, tokenizer, adapter_tf, tf_train, lr=1e-5, steps=100)
    print(f"  Final loss: {tf_loss:.4f}")

    # Evaluate both on standard MC format
    print("\n" + "=" * 70)
    print("  EVALUATION: Standard ABCD MC format (50 questions)")
    print("=" * 70)

    print("\n  MC-trained adapter on ABCD eval:")
    mc_results = evaluate_mc(model, tokenizer, adapter_mc, eval_ds, n=50)
    print(f"    Base:    {mc_results['base_correct']}/50 = {mc_results['base_pct']:.1%}")
    print(f"    Adapter: {mc_results['adapter_correct']}/50 = {mc_results['adapter_pct']:.1%}")
    print(f"    Delta:   {mc_results['delta']:+.1%}")
    print(f"    Pred dist: {mc_results['adapter_pred_dist']}")

    print("\n  T/F-trained adapter on ABCD eval:")
    tf_mc_results = evaluate_mc(model, tokenizer, adapter_tf, eval_ds, n=50)
    print(f"    Base:    {tf_mc_results['base_correct']}/50 = {tf_mc_results['base_pct']:.1%}")
    print(f"    Adapter: {tf_mc_results['adapter_correct']}/50 = {tf_mc_results['adapter_pct']:.1%}")
    print(f"    Delta:   {tf_mc_results['delta']:+.1%}")
    print(f"    Pred dist: {tf_mc_results['adapter_pred_dist']}")

    # Evaluate both on T/F format
    print("\n" + "=" * 70)
    print("  EVALUATION: T/F scoring format (50 questions)")
    print("=" * 70)

    print("\n  MC-trained adapter on T/F eval:")
    mc_tf_results = evaluate_tf(model, tokenizer, adapter_mc, eval_ds, n=50)
    print(f"    Base:    {mc_tf_results['base_correct']}/50 = {mc_tf_results['base_pct']:.1%}")
    print(f"    Adapter: {mc_tf_results['adapter_correct']}/50 = {mc_tf_results['adapter_pct']:.1%}")
    print(f"    Delta:   {mc_tf_results['delta']:+.1%}")

    print("\n  T/F-trained adapter on T/F eval:")
    tf_tf_results = evaluate_tf(model, tokenizer, adapter_tf, eval_ds, n=50)
    print(f"    Base:    {tf_tf_results['base_correct']}/50 = {tf_tf_results['base_pct']:.1%}")
    print(f"    Adapter: {tf_tf_results['adapter_correct']}/50 = {tf_tf_results['adapter_pct']:.1%}")
    print(f"    Delta:   {tf_tf_results['delta']:+.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Method':>15s}  {'ABCD eval':>12s}  {'T/F eval':>12s}  {'ABCD dist':>20s}")
    print(f"  {'MC-trained':>15s}  {mc_results['delta']:+.1%}         {mc_tf_results['delta']:+.1%}         {mc_results['adapter_pred_dist']}")
    print(f"  {'T/F-trained':>15s}  {tf_mc_results['delta']:+.1%}         {tf_tf_results['delta']:+.1%}         {tf_mc_results['adapter_pred_dist']}")
    print(f"  {'Base model':>15s}  {'(ref)':>12s}  {'(ref)':>12s}  {mc_results['base_pred_dist'] if mc_results else 'N/A'}")


if __name__ == "__main__":
    main()
