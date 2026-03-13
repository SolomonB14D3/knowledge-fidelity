#!/usr/bin/env python3
"""Model-agnostic multiple-choice eval for factual truth recall.

Measures whether the model assigns higher log-probability to the correct
answer than to all distractors. Completely independent of:
  - Tokenization (compares full completion log-probs, not first-token rank)
  - Thinking tokens / fill-in-blank triggers
  - Prompt format (uses a simple neutral format)
  - Vocabulary size / model architecture

Works with ANY autoregressive model that mlx_lm can load.

Metric: truth wins if log P(truth | prompt) > log P(distractor | prompt)
        for ALL distractors. (Strict MC accuracy.)

Usage:
    python experiments/operation_destroyer/eval_mc.py
    python experiments/operation_destroyer/eval_mc.py --model Qwen/Qwen2.5-0.5B
    python experiments/operation_destroyer/eval_mc.py --model meta-llama/Llama-3.2-1B
    python experiments/operation_destroyer/eval_mc.py --no-adapter
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

TRUTH_DICT_PATH = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/truth_dict_contrastive.json"


def get_completion_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Compute sum log P(completion | prompt) for any model.

    Uses teacher forcing: encode prompt+completion, sum log-probs
    over completion tokens only. Model-agnostic — works for any
    autoregressive model regardless of tokenization.
    """
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + completion)

    # Completion tokens start after prompt
    n_prompt = len(prompt_ids)
    if len(full_ids) <= n_prompt:
        return -1e9  # completion tokenized to nothing

    tokens = mx.array(full_ids)[None, :]  # [1, seq]

    # Forward pass
    logits = model(tokens)  # [1, seq, vocab]
    mx.eval(logits)

    logits_np = np.array(logits[0].astype(mx.float32))  # [seq, vocab]

    # Log-probs at positions [n_prompt-1 ... len-2] predict tokens [n_prompt ... len-1]
    total_lp = 0.0
    for i, tok_id in enumerate(full_ids[n_prompt:]):
        pos = n_prompt - 1 + i
        lp = logits_np[pos] - np.log(np.sum(np.exp(logits_np[pos] - logits_np[pos].max())) + 1e-8) - logits_np[pos].max()
        total_lp += float(lp[tok_id])

    return total_lp


def score_fact_mc(model, tokenizer, context, truth, distractors,
                  adapter=None, lm_head=None):
    """Score one fact via multiple-choice log-prob comparison.

    Returns (win, margin, truth_lp, best_distractor_lp).
    win=True if truth beats ALL distractors.
    margin = truth_lp - max(distractor_lps).
    """
    # Neutral prompt format that works across models
    # Avoids fill-in-blank triggers and thinking-mode triggers
    prompt = f"{context}:"

    if adapter is not None and lm_head is not None:
        # Adapter path: apply snap-on adapter to base logits
        import experiments.operation_destroyer.train_v3 as t3
        from experiments.operation_destroyer.train_v3 import apply_adapter

        def _logprob_with_adapter(completion):
            prompt_ids = tokenizer.encode(prompt)
            full_ids = tokenizer.encode(prompt + completion)
            n_prompt = len(prompt_ids)
            if len(full_ids) <= n_prompt:
                return -1e9
            tokens = mx.array(full_ids)[None, :]
            h = model.model(tokens)
            mx.eval(h)
            bl = lm_head(h)
            mx.eval(bl)
            logits = apply_adapter(adapter, bl)
            mx.eval(logits)
            logits_np = np.array(logits[0].astype(mx.float32))
            total_lp = 0.0
            for i, tok_id in enumerate(full_ids[n_prompt:]):
                pos = n_prompt - 1 + i
                lp = logits_np[pos] - np.log(np.sum(np.exp(logits_np[pos] - logits_np[pos].max())) + 1e-8) - logits_np[pos].max()
                total_lp += float(lp[tok_id])
            return total_lp

        truth_lp = _logprob_with_adapter(f" {truth}")
        dist_lps = [_logprob_with_adapter(f" {d}") for d in distractors]
    else:
        truth_lp = get_completion_logprob(model, tokenizer, prompt, f" {truth}")
        dist_lps = [get_completion_logprob(model, tokenizer, prompt, f" {d}") for d in distractors]

    best_dist = max(dist_lps)
    win = truth_lp > best_dist
    margin = truth_lp - best_dist
    return win, margin, truth_lp, best_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B-Base",
                        help="Any mlx_lm-compatible model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to snap-on adapter .npz (optional)")
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--truth-dict", type=str, default=None,
                        help="Path to truth dict JSON (default: truth_dict_contrastive.json)")
    args = parser.parse_args()

    # Load truth dict with distractors
    truth_path = args.truth_dict if args.truth_dict else TRUTH_DICT_PATH
    with open(truth_path) as f:
        data = json.load(f)
    facts = data["truths"]
    print(f"Loaded {len(facts)} facts with distractors\n")

    # Load model
    print(f"Loading {args.model}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # Optionally load adapter
    adapter = None
    lm_head = None
    if not args.no_adapter:
        ckpt = args.checkpoint
        if ckpt is None:
            import glob, os
            candidates = sorted(glob.glob(
                "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/operation_destroyer/**/best.npz",
                recursive=True
            ), key=os.path.getmtime, reverse=True)
            if candidates:
                ckpt = candidates[0]
                print(f"Auto-selected: {ckpt}")

        if ckpt:
            from module import SnapOnConfig, create_adapter
            from experiments.operation_destroyer.train_v3 import get_lm_head_fn
            vocab_size = model.model.embed_tokens.weight.shape[0]
            d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
            cfg = SnapOnConfig(d_model=d_model, d_inner=args.d_inner, n_layers=0,
                               n_heads=8, mode="logit", vocab_size=vocab_size)
            adapter = create_adapter(cfg)
            weights = mx.load(ckpt)
            adapter.load_weights(list(weights.items()))
            mx.eval(adapter.parameters())
            lm_head = get_lm_head_fn(model)
            print(f"Adapter loaded: {ckpt}\n")

    # Eval
    results = []
    categories = {}
    t0 = time.time()

    for fact in facts:
        win, margin, truth_lp, best_dist_lp = score_fact_mc(
            model, tokenizer,
            fact["context"], fact["truth"], fact["distractors"],
            adapter=adapter, lm_head=lm_head
        )
        cat = fact["category"]
        results.append({
            "context": fact["context"],
            "truth": fact["truth"],
            "category": cat,
            "win": win,
            "margin": margin,
        })
        categories.setdefault(cat, {"wins": 0, "total": 0, "margins": []})
        categories[cat]["total"] += 1
        if win:
            categories[cat]["wins"] += 1
        categories[cat]["margins"].append(margin)

    total_wins = sum(r["win"] for r in results)
    total = len(results)
    mean_margin = np.mean([r["margin"] for r in results])

    print("=" * 60)
    print("  MULTIPLE-CHOICE ACCURACY (truth beats all distractors)")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Adapter:     {'yes' if adapter else 'no'}")
    print(f"  Accuracy:    {total_wins}/{total} ({total_wins/total:.1%})")
    print(f"  Mean margin: {mean_margin:+.3f}")
    print(f"  Elapsed:     {time.time()-t0:.0f}s")
    print()
    print("  Per category:")
    for cat in sorted(categories):
        v = categories[cat]
        mm = np.mean(v["margins"])
        print(f"    {cat:15s}: {v['wins']}/{v['total']}  margin={mm:+.3f}")

    # Show failures
    failures = [r for r in results if not r["win"]]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r in sorted(failures, key=lambda x: x["margin"])[:20]:
            print(f"    [{r['category']}] {r['context']!r} -> {r['truth']!r}  margin={r['margin']:+.3f}")

    # Save
    import os
    out_dir = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/operation_destroyer/mc_eval"
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "model": args.model,
        "adapter": args.checkpoint,
        "accuracy": total_wins / total,
        "n_wins": total_wins,
        "n_total": total,
        "mean_margin": float(mean_margin),
        "categories": {k: {"wins": v["wins"], "total": v["total"],
                            "mean_margin": float(np.mean(v["margins"]))}
                       for k, v in categories.items()},
        "results": results,
    }
    out_path = os.path.join(out_dir, f"eval_{args.model.replace('/', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
