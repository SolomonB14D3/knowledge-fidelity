#!/usr/bin/env python3
"""Truth recall eval: does the adapter improve factual token probability on 120 facts?

For each fact, prompt "What is the [context]?" and measure:
  - Whether the truth token appears in top-5 completions
  - Logit of truth token vs. mean logit (gap)
  - Base model vs. adapter comparison

Usage:
    python experiments/operation_destroyer/eval_truth_recall.py
    python experiments/operation_destroyer/eval_truth_recall.py --checkpoint results/operation_destroyer/v15/best.npz
    python experiments/operation_destroyer/eval_truth_recall.py --no-adapter  # base only
"""

import argparse
import json
import sys
import time

sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/snap_on")

import mlx.core as mx
import mlx_lm
import numpy as np

from module import SnapOnConfig, create_adapter
from experiments.operation_destroyer.train_v3 import (
    get_lm_head_fn, apply_adapter, ALPACA_TEMPLATE,
)
import experiments.operation_destroyer.train_v3 as t3

TRUTH_DICT_PATH = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/truth_dict.json"


def score_fact(model, tokenizer, lm_head, adapter, context, truth, softcap=30.0, completion_style=True, prompt_override=None):
    """Score one fact. Returns (truth_in_top5, logit_gap, truth_rank)."""
    if prompt_override is not None:
        prompt = prompt_override
    elif completion_style:
        # Simple completion: "The capital of France is" → "Paris"
        prompt = f"The {context} is"
    else:
        prompt = ALPACA_TEMPLATE.format(instruction=f"What is the {context}?")
    tokens = mx.array(tokenizer.encode(prompt))[None, :]

    h = model.model(tokens)
    mx.eval(h)
    base_logits = lm_head(h)
    mx.eval(base_logits)

    if adapter is not None:
        logits = apply_adapter(adapter, base_logits)
        mx.eval(logits)
    else:
        logits = t3.LOGIT_SOFTCAP * mx.tanh(base_logits / t3.LOGIT_SOFTCAP)

    last_logits = np.array(logits[0, -1, :].astype(mx.float32))

    # Tokenize truth — take first token of truth string
    truth_tokens = tokenizer.encode(f" {truth}", add_special_tokens=False)
    if not truth_tokens:
        truth_tokens = tokenizer.encode(truth, add_special_tokens=False)
    truth_tok = truth_tokens[0]

    truth_logit = last_logits[truth_tok]
    mean_logit = last_logits.mean()
    logit_gap = truth_logit - mean_logit

    top5 = np.argsort(last_logits)[-5:][::-1]
    truth_in_top5 = int(truth_tok in top5)

    # Rank (lower = better)
    truth_rank = int((last_logits > truth_logit).sum()) + 1

    return truth_in_top5, logit_gap, truth_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to adapter .npz checkpoint")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Evaluate base model only (no adapter)")
    parser.add_argument("--softcap", type=float, default=30.0)
    parser.add_argument("--d-inner", type=int, default=128)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B-Base",
                        help="Base model to load (default: Qwen/Qwen3-0.6B-Base)")
    args = parser.parse_args()

    t3.LOGIT_SOFTCAP = args.softcap

    # Auto-find best checkpoint if not specified
    if not args.no_adapter and args.checkpoint is None:
        import glob, os
        candidates = sorted(glob.glob(
            "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/operation_destroyer/**/best.npz",
            recursive=True
        ), key=os.path.getmtime, reverse=True)
        if candidates:
            args.checkpoint = candidates[0]
            print(f"Auto-selected: {args.checkpoint}")
        else:
            print("No checkpoint found — running base model only")
            args.no_adapter = True

    # Load truth dict
    with open(TRUTH_DICT_PATH) as f:
        data = json.load(f)
    facts = data["truths"]
    print(f"Loaded {len(facts)} facts\n")

    # Load model
    print("Loading model...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = get_lm_head_fn(model)
    print(f"  Loaded in {time.time() - t0:.1f}s\n")

    # Load adapter
    adapter = None
    if not args.no_adapter and args.checkpoint:
        vocab_size = model.model.embed_tokens.weight.shape[0]
        d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
        config = SnapOnConfig(d_model=d_model, d_inner=args.d_inner, n_layers=0,
                              n_heads=8, mode="logit", vocab_size=vocab_size)
        adapter = create_adapter(config)
        weights = mx.load(args.checkpoint)
        adapter.load_weights(list(weights.items()))
        mx.eval(adapter.parameters())
        print(f"Adapter loaded: {args.checkpoint}\n")

    # Eval base model (no adapter)
    print("=" * 60)
    print("  BASE MODEL")
    print("=" * 60)
    base_results = []
    for fact in facts:
        hit, gap, rank = score_fact(model, tokenizer, lm_head, None, fact["context"], fact["truth"],
                                    prompt_override=fact.get("prompt"))
        base_results.append({"context": fact["context"], "truth": fact["truth"],
                              "category": fact["category"], "hit": hit, "gap": gap, "rank": rank})

    base_hits = sum(r["hit"] for r in base_results)
    print(f"  Top-5 hits: {base_hits}/{len(facts)} ({base_hits/len(facts):.1%})")
    print(f"  Mean logit gap: {np.mean([r['gap'] for r in base_results]):.3f}")
    print(f"  Mean rank: {np.mean([r['rank'] for r in base_results]):.0f}")

    # Per-category
    categories = sorted(set(r["category"] for r in base_results))
    for cat in categories:
        cat_r = [r for r in base_results if r["category"] == cat]
        cat_hits = sum(r["hit"] for r in cat_r)
        print(f"    {cat:12s}: {cat_hits}/{len(cat_r)} hits, "
              f"gap={np.mean([r['gap'] for r in cat_r]):.3f}")

    if adapter is None:
        return

    # Eval with adapter
    print("\n" + "=" * 60)
    print("  ADAPTER")
    print("=" * 60)
    adapter_results = []
    for fact in facts:
        hit, gap, rank = score_fact(model, tokenizer, lm_head, adapter, fact["context"], fact["truth"],
                                    prompt_override=fact.get("prompt"))
        adapter_results.append({"context": fact["context"], "truth": fact["truth"],
                                 "category": fact["category"], "hit": hit, "gap": gap, "rank": rank})

    adapter_hits = sum(r["hit"] for r in adapter_results)
    print(f"  Top-5 hits: {adapter_hits}/{len(facts)} ({adapter_hits/len(facts):.1%})")
    print(f"  Mean logit gap: {np.mean([r['gap'] for r in adapter_results]):.3f}")
    print(f"  Mean rank: {np.mean([r['rank'] for r in adapter_results]):.0f}")

    for cat in categories:
        cat_r = [r for r in adapter_results if r["category"] == cat]
        cat_hits = sum(r["hit"] for r in cat_r)
        print(f"    {cat:12s}: {cat_hits}/{len(cat_r)} hits, "
              f"gap={np.mean([r['gap'] for r in cat_r]):.3f}")

    # Delta summary
    print("\n" + "=" * 60)
    print("  DELTA (adapter - base)")
    print("=" * 60)
    delta_hits = adapter_hits - base_hits
    print(f"  Top-5 hits: {base_hits} → {adapter_hits} ({delta_hits:+d})")
    print(f"  Mean logit gap: {np.mean([r['gap'] for r in base_results]):.3f} → "
          f"{np.mean([r['gap'] for r in adapter_results]):.3f}")

    improved = [(b, a) for b, a in zip(base_results, adapter_results) if a["hit"] > b["hit"]]
    degraded = [(b, a) for b, a in zip(base_results, adapter_results) if a["hit"] < b["hit"]]
    print(f"\n  Improved: {len(improved)} facts")
    for b, a in improved[:10]:
        print(f"    + [{b['category']}] {b['context']!r} → '{b['truth']}' "
              f"(rank {b['rank']}→{a['rank']})")
    print(f"  Degraded: {len(degraded)} facts")
    for b, a in degraded[:10]:
        print(f"    - [{b['category']}] {b['context']!r} → '{b['truth']}' "
              f"(rank {b['rank']}→{a['rank']})")


if __name__ == "__main__":
    main()
