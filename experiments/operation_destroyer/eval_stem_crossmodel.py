#!/usr/bin/env python3
"""Cross-model STEM benchmark eval using log-prob MC comparison.

Scores every model on each domain independently, then aggregates.
Produces a results table + per-domain breakdown + saves JSON.

Usage:
    python experiments/operation_destroyer/eval_stem_crossmodel.py
    python experiments/operation_destroyer/eval_stem_crossmodel.py --domains calculus physics
    python experiments/operation_destroyer/eval_stem_crossmodel.py --difficulty easy
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

from experiments.operation_destroyer.eval_mc import score_fact_mc

BENCH_DIR = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer"
OUT_DIR   = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/operation_destroyer/stem_crossmodel"

MODELS = [
    ("GPT-2-124M",    "openai-community/gpt2"),
    ("SmolLM2-360M",  "HuggingFaceTB/SmolLM2-360M"),
    ("Qwen2.5-0.5B",  "Qwen/Qwen2.5-0.5B"),
    ("Llama-3.2-1B",  "meta-llama/Llama-3.2-1B"),
    ("Qwen2.5-1.5B",  "Qwen/Qwen2.5-1.5B"),
    ("Qwen3-4B",      "Qwen/Qwen3-4B-Base"),
]

DOMAINS = ["calculus", "physics", "chemistry", "linear_algebra", "statistics", "constants"]
DIFFICULTIES = ["easy", "medium", "hard"]


def load_facts(domains, difficulty=None):
    """Load facts from domain JSON files, optionally filtered by difficulty."""
    facts = []
    for d in domains:
        path = os.path.join(BENCH_DIR, f"stem_bench_{d}.json")
        with open(path) as f:
            data = json.load(f)
        for fact in data["truths"]:
            if difficulty is None or fact.get("difficulty") == difficulty:
                facts.append(fact)
    return facts


def eval_model_on_facts(model, tokenizer, facts):
    """Score all facts for a loaded model. Returns per-fact results."""
    results = []
    for fact in facts:
        try:
            win, margin, truth_lp, dist_lps = score_fact_mc(
                model, tokenizer,
                fact["context"], fact["truth"], fact["distractors"]
            )
            results.append({
                "context":    fact["context"],
                "truth":      fact["truth"],
                "domain":     fact.get("domain", "unknown"),
                "difficulty": fact.get("difficulty", "unknown"),
                "win":        bool(win),
                "margin":     float(margin),
                "truth_lp":   float(truth_lp),
            })
        except Exception as e:
            results.append({
                "context":    fact["context"],
                "truth":      fact["truth"],
                "domain":     fact.get("domain", "unknown"),
                "difficulty": fact.get("difficulty", "unknown"),
                "win":        False,
                "margin":     0.0,
                "truth_lp":   -999.0,
                "error":      str(e),
            })
    return results


def domain_summary(results, domain):
    """Compute wins and avg margin for a specific domain."""
    dr = [r for r in results if r["domain"] == domain]
    if not dr:
        return 0, 0, 0.0
    wins = sum(r["win"] for r in dr)
    return wins, len(dr), float(np.mean([r["margin"] for r in dr]))


def difficulty_summary(results, difficulty):
    dr = [r for r in results if r["difficulty"] == difficulty]
    if not dr:
        return 0, 0, 0.0
    wins = sum(r["win"] for r in dr)
    return wins, len(dr), float(np.mean([r["margin"] for r in dr]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=DOMAINS,
                        help="Which domains to include")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["easy", "medium", "hard"],
                        help="Filter to one difficulty level")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model names to run (default: all)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load facts
    facts = load_facts(args.domains, args.difficulty)
    diff_tag = f"_{args.difficulty}" if args.difficulty else ""
    print(f"STEM Benchmark: {len(facts)} facts | domains: {', '.join(args.domains)}{diff_tag}\n")

    # Filter models
    models_to_run = MODELS
    if args.models:
        models_to_run = [(n, p) for n, p in MODELS if n in args.models]

    # Print header
    domain_cols = [d[:8] for d in args.domains]
    print(f"{'Model':<18} {'Total':>10}  " + "  ".join(f"{d:>8}" for d in domain_cols) + "  Easy  Med  Hard")
    print("─" * (18 + 12 + len(args.domains) * 10 + 18))

    all_model_results = []

    for model_name, model_id in models_to_run:
        t0 = time.time()
        model, tokenizer = mlx_lm.load(model_id)
        model.freeze()

        results = eval_model_on_facts(model, tokenizer, facts)

        elapsed = time.time() - t0
        total_wins = sum(r["win"] for r in results)
        total_n = len(results)
        total_acc = total_wins / max(total_n, 1)
        avg_margin = float(np.mean([r["margin"] for r in results]))

        # Domain breakdown
        domain_accs = []
        for d in args.domains:
            w, n, _ = domain_summary(results, d)
            domain_accs.append(f"{w}/{n}" if n > 0 else "  —  ")

        # Difficulty breakdown
        easy_w, easy_n, _ = difficulty_summary(results, "easy")
        med_w,  med_n,  _ = difficulty_summary(results, "medium")
        hard_w, hard_n, _ = difficulty_summary(results, "hard")

        easy_str = f"{easy_w/easy_n:.0%}" if easy_n else "—"
        med_str  = f"{med_w/med_n:.0%}"   if med_n  else "—"
        hard_str = f"{hard_w/hard_n:.0%}" if hard_n else "—"

        domain_str = "  ".join(f"{a:>8}" for a in domain_accs)
        print(f"{model_name:<18} {total_wins:>4}/{total_n:<4} {total_acc:>5.1%}  "
              f"{domain_str}  {easy_str:>4}  {med_str:>4}  {hard_str:>4}   {elapsed:.0f}s")

        model_row = {
            "model_name":  model_name,
            "model_id":    model_id,
            "total_wins":  total_wins,
            "total_n":     total_n,
            "total_acc":   total_acc,
            "avg_margin":  avg_margin,
            "elapsed_s":   elapsed,
            "domain_breakdown": {
                d: {"wins": w, "n": n, "avg_margin": m}
                for d in args.domains
                for w, n, m in [domain_summary(results, d)]
            },
            "difficulty_breakdown": {
                diff: {"wins": w, "n": n, "acc": w/max(n,1)}
                for diff in DIFFICULTIES
                for w, n, _ in [difficulty_summary(results, diff)]
            },
            "per_fact": results,
        }
        all_model_results.append(model_row)

        del model, tokenizer
        mx.clear_cache()

    print("─" * (18 + 12 + len(args.domains) * 10 + 18))

    # Save results
    out_file = os.path.join(OUT_DIR, f"stem_crossmodel{diff_tag}.json")
    with open(out_file, "w") as f:
        json.dump({
            "domains":    args.domains,
            "difficulty": args.difficulty,
            "n_facts":    len(facts),
            "models":     all_model_results,
        }, f, indent=2)
    print(f"\nSaved: {out_file}")

    # Print worst facts across all models (facts every model gets wrong)
    print("\n── Hardest facts (missed by most models) ──")
    fact_misses = {}
    for row in all_model_results:
        for r in row["per_fact"]:
            k = r["context"]
            fact_misses[k] = fact_misses.get(k, 0) + (0 if r["win"] else 1)
    hardest = sorted(fact_misses.items(), key=lambda x: -x[1])[:10]
    for ctx, misses in hardest:
        pct = misses / len(all_model_results)
        print(f"  {pct:.0%} miss  {ctx}")


if __name__ == "__main__":
    main()
