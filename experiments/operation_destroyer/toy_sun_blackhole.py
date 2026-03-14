#!/usr/bin/env python3
"""Sun + Black Hole contrastive training for factual truth recall.

Trains a logit-space adapter using ONLY the 115 facts + distractors.
No general SFT data — pure contrastive forcing.

Loss per fact:
  sun_loss    = -log_softmax(truth_token)  [maximize truth]
  hole_loss   = sum(log_softmax(distractor_tokens))  [minimize distractors]
  margin_loss = max(0, max(distractor_logits) - truth_logit + margin)
  total       = sun_weight * sun_loss + hole_weight * hole_loss + margin_weight * margin_loss

Usage:
    cd /Volumes/4TB SD/ClaudeCode/knowledge-fidelity
    python experiments/operation_destroyer/toy_sun_blackhole.py
    python experiments/operation_destroyer/toy_sun_blackhole.py --steps 500 --lr 1e-4 --margin 2.0
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
from experiments.operation_destroyer.train_v3 import get_lm_head_fn, apply_adapter, ALPACA_TEMPLATE
import experiments.operation_destroyer.train_v3 as t3

TRUTH_DICT_PATH = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/truth_dict_contrastive.json"
RESULTS_DIR = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/operation_destroyer/sun_blackhole"


def build_examples(tokenizer, facts):
    """Build training examples: one per (fact, prompt_template) combination.

    Uses the fact's custom 'prompt' field when available, otherwise
    falls back to completion-style templates.
    """
    examples = []
    for fact in facts:
        ctx = fact["context"]
        truth = fact["truth"]
        distractors = fact["distractors"]

        # Tokenize truth and distractor first tokens
        truth_toks = tokenizer.encode(f" {truth}", add_special_tokens=False)
        truth_tok = truth_toks[0] if truth_toks else tokenizer.encode(truth, add_special_tokens=False)[0]

        distractor_toks = []
        for d in distractors:
            dtoks = tokenizer.encode(f" {d}", add_special_tokens=False)
            if not dtoks:
                dtoks = tokenizer.encode(d, add_special_tokens=False)
            if dtoks:
                distractor_toks.append(dtoks[0])

        if not distractor_toks:
            continue

        # Use custom prompt if available, otherwise use default templates
        if "prompt" in fact:
            prompts = [fact["prompt"]]
        else:
            prompts = [
                f"The {ctx} is",
                f"The {ctx}:",
                f"{ctx}:",
            ]

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            examples.append({
                "tokens": tokens,
                "truth_tok": truth_tok,
                "distractor_toks": distractor_toks,
                "context": ctx,
                "truth": truth,
            })

    return examples


def contrastive_loss_fn(adapter, lm_head, tokens, truth_tok, distractor_toks,
                        margin=1.0, sun_w=1.0, hole_w=0.5, margin_w=1.0,
                        model=None):
    """Sun + black hole contrastive loss at the last position."""
    h = model.model(tokens)
    mx.eval(h)
    base_logits = lm_head(h)
    mx.eval(base_logits)

    raw_shifts = adapter(base_logits)
    shifts = raw_shifts - raw_shifts.mean(axis=-1, keepdims=True)
    combined = base_logits + shifts
    logits = t3.LOGIT_SOFTCAP * mx.tanh(combined / t3.LOGIT_SOFTCAP)

    last = logits[0, -1, :]  # [vocab]

    # Sun: maximize truth logit via cross-entropy style
    truth_logit = last[truth_tok]
    log_sum_exp = mx.log(mx.sum(mx.exp(last - last.max())) + 1e-8) + last.max()
    sun_loss = -(truth_logit - log_sum_exp)

    # Black holes: push distractor logits down (minimize their log probability)
    hole_loss = mx.array(0.0)
    for d_tok in distractor_toks:
        d_logit = last[d_tok]
        hole_loss = hole_loss + (d_logit - log_sum_exp)  # maximize negative = minimize logit
    hole_loss = hole_loss / len(distractor_toks)

    # Margin: truth must exceed max distractor by margin
    d_logits = mx.stack([last[d] for d in distractor_toks])
    max_distractor = mx.max(d_logits)
    margin_loss = mx.maximum(mx.array(0.0), max_distractor - truth_logit + margin)

    total = sun_w * sun_loss + hole_w * hole_loss + margin_w * margin_loss
    return total, (sun_loss, hole_loss, margin_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",     type=int,   default=300,   help="Training steps")
    parser.add_argument("--lr",        type=float, default=3e-5,  help="Learning rate")
    parser.add_argument("--margin",    type=float, default=1.5,   help="Margin for margin loss")
    parser.add_argument("--sun-w",     type=float, default=1.0,   help="Sun loss weight")
    parser.add_argument("--hole-w",    type=float, default=0.5,   help="Black hole loss weight")
    parser.add_argument("--margin-w",  type=float, default=1.0,   help="Margin loss weight")
    parser.add_argument("--d-inner",   type=int,   default=64,    help="Adapter inner dimension")
    parser.add_argument("--softcap",   type=float, default=30.0,  help="Logit softcap")
    parser.add_argument("--model",     type=str,   default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--eval",      action="store_true", default=True, help="Run truth recall eval after training")
    args = parser.parse_args()

    t3.LOGIT_SOFTCAP = args.softcap
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  SUN + BLACK HOLE CONTRASTIVE TRAINING")
    print("=" * 60)
    print(f"  Model:    {args.model}")
    print(f"  Steps:    {args.steps}")
    print(f"  LR:       {args.lr}")
    print(f"  Margin:   {args.margin}")
    print(f"  Weights:  sun={args.sun_w} hole={args.hole_w} margin={args.margin_w}")
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
    print(f"Built {len(examples)} training examples from {len(facts)} facts\n")

    # Create adapter
    cfg = SnapOnConfig(d_model=d_model, d_inner=args.d_inner, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    from mlx.utils import tree_flatten
    n_params = sum(p.size for _, p in tree_flatten(adapter.parameters()))
    print(f"Adapter: d_inner={args.d_inner}, ~{n_params/1e6:.1f}M params\n")

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_and_aux(adapter, ex):
        tokens = mx.array(ex["tokens"])[None, :]
        return contrastive_loss_fn(
            adapter, lm_head, tokens,
            ex["truth_tok"], ex["distractor_toks"],
            margin=args.margin,
            sun_w=args.sun_w, hole_w=args.hole_w, margin_w=args.margin_w,
            model=model,
        )

    loss_grad = nn.value_and_grad(adapter, lambda a, ex: loss_and_aux(a, ex)[0])

    # Training loop
    import random
    print("Training...")
    log = []
    t_start = time.time()

    for step in range(args.steps):
        ex = examples[step % len(examples)]

        loss, grads = loss_grad(adapter, ex)
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

    # Run truth recall eval inline
    if args.eval:
        print("=" * 60)
        print("  TRUTH RECALL EVAL")
        print("=" * 60)

        # Load base facts (without distractors)
        with open("/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/truth_dict.json") as f:
            base_data = json.load(f)
        eval_facts = base_data["truths"]

        def score(use_adapter):
            hits, gaps, ranks = 0, [], []
            for fact in eval_facts:
                prompt = fact.get("prompt", f"The {fact['context']} is")
                tokens = mx.array(tokenizer.encode(prompt))[None, :]
                h = model.model(tokens)
                mx.eval(h)
                bl = lm_head(h)
                mx.eval(bl)
                if use_adapter:
                    logits = apply_adapter(adapter, bl)
                else:
                    logits = t3.LOGIT_SOFTCAP * mx.tanh(bl / t3.LOGIT_SOFTCAP)
                mx.eval(logits)
                last = np.array(logits[0, -1, :].astype(mx.float32))
                toks = tokenizer.encode(f" {fact['truth']}", add_special_tokens=False)
                if not toks:
                    toks = tokenizer.encode(fact["truth"], add_special_tokens=False)
                t_tok = toks[0]
                t_logit = last[t_tok]
                top5 = np.argsort(last)[-5:]
                hits += int(t_tok in top5)
                gaps.append(float(t_logit - last.mean()))
                ranks.append(int((last > t_logit).sum()) + 1)
            return hits, np.mean(gaps), np.mean(ranks)

        base_hits, base_gap, base_rank = score(False)
        ada_hits, ada_gap, ada_rank = score(True)
        delta = ada_hits - base_hits

        print(f"  {'':20s} {'Base':>10} {'Adapter':>10} {'Delta':>10}")
        print(f"  {'Top-5 hits':20s} {base_hits:>10} {ada_hits:>10} {delta:>+10}")
        print(f"  {'Mean logit gap':20s} {base_gap:>10.3f} {ada_gap:>10.3f} {ada_gap-base_gap:>+10.3f}")
        print(f"  {'Mean rank':20s} {base_rank:>10.0f} {ada_rank:>10.0f} {ada_rank-base_rank:>+10.0f}")

        verdict = "✓ MECHANISM WORKS" if delta > 5 else ("△ MARGINAL" if delta > 0 else "✗ NO IMPROVEMENT")
        print(f"\n  {verdict}")

        # Save results
        result = {
            "config": vars(args),
            "base": {"hits": base_hits, "gap": base_gap, "rank": base_rank},
            "adapter": {"hits": ada_hits, "gap": ada_gap, "rank": ada_rank},
            "delta_hits": delta,
        }
        with open(os.path.join(RESULTS_DIR, "eval_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {os.path.join(RESULTS_DIR, 'eval_result.json')}")


if __name__ == "__main__":
    main()
