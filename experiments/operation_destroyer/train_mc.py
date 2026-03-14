#!/usr/bin/env python3
"""MC log-prob contrastive training for factual truth recall.

Uses the same metric as eval_mc.py: maximize log P(truth | context:)
relative to log P(distractor | context:). Model-agnostic — works with
any base model because it compares full completion log-probs, not
next-token rank.

Loss per fact:
  margin = log P(truth | prompt) - max_i(log P(distractor_i | prompt))
  loss   = max(0, target_margin - margin)   [hinge]

This is the training-side mirror of eval_mc.py. The same prompt format
("context:") and the same comparison make train/eval coherent.

Usage:
    cd /Volumes/4TB SD/ClaudeCode/knowledge-fidelity
    python experiments/operation_destroyer/train_mc.py
    python experiments/operation_destroyer/train_mc.py --model Qwen/Qwen2.5-0.5B
    python experiments/operation_destroyer/train_mc.py --steps 500 --lr 1e-4 --target-margin 2.0
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/snap_on")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
import numpy as np

from module import SnapOnConfig, create_adapter
from experiments.operation_destroyer.train_v3 import get_lm_head_fn, apply_adapter
import experiments.operation_destroyer.train_v3 as t3

TRUTH_DICT_PATH = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/truth_dict_contrastive.json"
RESULTS_DIR = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/operation_destroyer/mc_train"


def compute_completion_lp(adapter, lm_head, model, prompt_ids, completion_ids):
    """Sum log P(completion_ids | prompt_ids) with adapter applied.

    Returns a scalar mx.array (differentiable through adapter).
    """
    full_ids = prompt_ids + completion_ids
    tokens = mx.array(full_ids)[None, :]  # [1, seq]

    h = model.model(tokens)
    base_logits = lm_head(h)  # [1, seq, vocab]

    # Apply adapter (shifts logits in logit space)
    shifts = adapter(base_logits)
    shifts = shifts - shifts.mean(axis=-1, keepdims=True)
    combined = base_logits + shifts
    logits = t3.LOGIT_SOFTCAP * mx.tanh(combined / t3.LOGIT_SOFTCAP)  # [1, seq, vocab]

    n_prompt = len(prompt_ids)
    total_lp = mx.array(0.0)

    for i, tok_id in enumerate(completion_ids):
        pos = n_prompt - 1 + i
        logit_vec = logits[0, pos, :]  # [vocab]
        # log-softmax at tok_id
        log_sum_exp = mx.log(mx.sum(mx.exp(logit_vec - logit_vec.max())) + 1e-8) + logit_vec.max()
        total_lp = total_lp + logit_vec[tok_id] - log_sum_exp

    return total_lp


def mc_loss_fn(adapter, lm_head, model, prompt_ids, truth_ids, distractor_ids_list,
               target_margin=1.0):
    """Hinge loss: push truth log-prob above best distractor by target_margin."""
    truth_lp = compute_completion_lp(adapter, lm_head, model, prompt_ids, truth_ids)

    # Stack distractor log-probs
    dist_lps = mx.stack([
        compute_completion_lp(adapter, lm_head, model, prompt_ids, d_ids)
        for d_ids in distractor_ids_list
    ])
    best_dist_lp = mx.max(dist_lps)

    margin = truth_lp - best_dist_lp
    loss = mx.maximum(mx.array(0.0), mx.array(target_margin) - margin)
    return loss, margin


def build_examples(tokenizer, facts):
    """Build training examples as token IDs (model-agnostic)."""
    examples = []
    for fact in facts:
        ctx = fact["context"]
        truth = fact["truth"]
        distractors = fact["distractors"]

        prompt = f"{ctx}:"
        prompt_ids = tokenizer.encode(prompt)

        truth_ids = tokenizer.encode(f" {truth}", add_special_tokens=False)
        if not truth_ids:
            truth_ids = tokenizer.encode(truth, add_special_tokens=False)

        dist_ids_list = []
        for d in distractors:
            d_ids = tokenizer.encode(f" {d}", add_special_tokens=False)
            if not d_ids:
                d_ids = tokenizer.encode(d, add_special_tokens=False)
            if d_ids:
                dist_ids_list.append(d_ids)

        if not dist_ids_list:
            continue

        examples.append({
            "prompt_ids": prompt_ids,
            "truth_ids": truth_ids,
            "distractor_ids_list": dist_ids_list,
            "context": ctx,
            "truth": truth,
        })

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",         type=int,   default=500)
    parser.add_argument("--lr",            type=float, default=3e-5)
    parser.add_argument("--target-margin", type=float, default=2.0,
                        help="Hinge margin: push truth log-prob above best distractor by this much")
    parser.add_argument("--d-inner",       type=int,   default=64)
    parser.add_argument("--model",         type=str,   default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--eval",          action="store_true", default=True)
    args = parser.parse_args()

    t3.LOGIT_SOFTCAP = 30.0
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  MC LOG-PROB CONTRASTIVE TRAINING")
    print("=" * 60)
    print(f"  Model:         {args.model}")
    print(f"  Steps:         {args.steps}")
    print(f"  LR:            {args.lr}")
    print(f"  Target margin: {args.target_margin}")
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = get_lm_head_fn(model)
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"  d_model={d_model}, vocab={vocab_size}, loaded in {time.time()-t0:.1f}s\n")

    # Load facts
    with open(TRUTH_DICT_PATH) as f:
        data = json.load(f)
    facts = data["truths"]
    examples = build_examples(tokenizer, facts)
    print(f"Built {len(examples)} examples from {len(facts)} facts\n")

    # Create adapter
    cfg = SnapOnConfig(d_model=d_model, d_inner=args.d_inner, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    from mlx.utils import tree_flatten
    n_params = sum(p.size for _, p in tree_flatten(adapter.parameters()))
    print(f"Adapter: d_inner={args.d_inner}, ~{n_params/1e6:.1f}M params\n")

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(adapter, ex):
        return mc_loss_fn(
            adapter, lm_head, model,
            ex["prompt_ids"], ex["truth_ids"], ex["distractor_ids_list"],
            target_margin=args.target_margin
        )[0]

    loss_grad = nn.value_and_grad(adapter, loss_fn)

    # Gradient clipping: multi-token log-prob sums can produce large gradients
    def clip_grads(grads, max_norm=1.0):
        from mlx.utils import tree_flatten, tree_unflatten
        leaves = tree_flatten(grads)
        total_sq = sum(float(mx.sum(g ** 2)) for _, g in leaves)
        norm = total_sq ** 0.5
        if norm > max_norm:
            scale = max_norm / (norm + 1e-8)
            leaves = [(k, g * scale) for k, g in leaves]
        return tree_unflatten(leaves)

    print("Training...")
    log = []
    t_start = time.time()

    for step in range(args.steps):
        ex = examples[step % len(examples)]

        loss, grads = loss_grad(adapter, ex)
        grads = clip_grads(grads, max_norm=1.0)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        loss_val = float(loss)
        log.append(loss_val)

        if step % 50 == 0 or step == args.steps - 1:
            recent = np.mean(log[-50:]) if len(log) >= 50 else np.mean(log)
            elapsed = time.time() - t_start
            print(f"  step {step:4d}/{args.steps} | loss={recent:.4f} | {elapsed:.0f}s")

    # Save checkpoint
    ckpt_path = os.path.join(RESULTS_DIR, "best.npz")
    flat = dict(tree_flatten(adapter.parameters()))
    mx.savez(ckpt_path, **flat)
    print(f"\nSaved: {ckpt_path}\n")

    # Run MC eval inline
    if args.eval:
        print("=" * 60)
        print("  MC ACCURACY EVAL")
        print("=" * 60)

        with open("/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/truth_dict_contrastive.json") as f:
            eval_data = json.load(f)
        eval_facts = eval_data["truths"]

        def score(use_adapter):
            wins, margins = 0, []
            for fact in eval_facts:
                prompt_ids = tokenizer.encode(f"{fact['context']}:")
                truth_ids = tokenizer.encode(f" {fact['truth']}", add_special_tokens=False)
                if not truth_ids:
                    truth_ids = tokenizer.encode(fact["truth"], add_special_tokens=False)

                dist_ids_list = []
                for d in fact["distractors"]:
                    d_ids = tokenizer.encode(f" {d}", add_special_tokens=False)
                    if not d_ids:
                        d_ids = tokenizer.encode(d, add_special_tokens=False)
                    if d_ids:
                        dist_ids_list.append(d_ids)

                full_prompt = f"{fact['context']}:"
                if use_adapter:
                    def lp(comp_ids):
                        return float(compute_completion_lp(adapter, lm_head, model, prompt_ids, comp_ids))
                else:
                    def lp(comp_ids):
                        # Base model: no adapter
                        full_ids = prompt_ids + comp_ids
                        tokens = mx.array(full_ids)[None, :]
                        h = model.model(tokens)
                        mx.eval(h)
                        bl = lm_head(h)
                        mx.eval(bl)
                        logits = t3.LOGIT_SOFTCAP * mx.tanh(bl / t3.LOGIT_SOFTCAP)
                        mx.eval(logits)
                        logits_np = np.array(logits[0].astype(mx.float32))
                        n_p = len(prompt_ids)
                        total = 0.0
                        for i, tid in enumerate(comp_ids):
                            pos = n_p - 1 + i
                            lv = logits_np[pos]
                            lse = float(np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max())
                            total += float(lv[tid]) - lse
                        return total

                truth_lp = lp(truth_ids)
                dist_lps = [lp(d) for d in dist_ids_list]
                margin = truth_lp - max(dist_lps) if dist_lps else 0.0
                if margin > 0:
                    wins += 1
                margins.append(margin)

            return wins, np.mean(margins)

        base_wins, base_margin = score(False)
        ada_wins, ada_margin = score(True)
        delta = ada_wins - base_wins

        print(f"  {'':20s} {'Base':>10} {'Adapter':>10} {'Delta':>10}")
        print(f"  {'Accuracy':20s} {base_wins:>10} {ada_wins:>10} {delta:>+10}")
        print(f"  {'Mean margin':20s} {base_margin:>10.3f} {ada_margin:>10.3f} {ada_margin-base_margin:>+10.3f}")

        verdict = "✓ WORKS" if delta > 3 else ("△ MARGINAL" if delta > 0 else "✗ NO IMPROVEMENT")
        print(f"\n  {verdict}")

        result = {
            "config": vars(args),
            "base": {"wins": base_wins, "margin": base_margin},
            "adapter": {"wins": ada_wins, "margin": ada_margin},
            "delta_wins": delta,
        }
        with open(os.path.join(RESULTS_DIR, "eval_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {os.path.join(RESULTS_DIR, 'eval_result.json')}")


if __name__ == "__main__":
    main()
