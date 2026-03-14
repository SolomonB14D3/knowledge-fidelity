#!/usr/bin/env python3
"""rho-mc-eval: Model-agnostic factual multiple-choice accuracy benchmark.

Measures whether any autoregressive model assigns higher log-probability
to the correct answer than to all distractors. Works across architectures,
tokenizers, and model families — no prompt format assumptions.

Metric: truth wins if log P(truth | context:) > log P(distractor_i | context:)
for ALL distractors. Score = wins / total (strict MC accuracy).

Usage:
    rho-mc-eval Qwen/Qwen2.5-0.5B
    rho-mc-eval meta-llama/Llama-3.2-1B --truth-dict path/to/truth_dict.json
    rho-mc-eval Qwen/Qwen3-4B-Base --domain biology
    rho-mc-eval MODEL1 --compare results/model2.json
    rho-mc-eval --list-domains
"""

import argparse
import json
import os
import sys
import time

import numpy as np


DEFAULT_TRUTH_DICT = os.path.join(
    os.path.dirname(__file__),
    "../../../../experiments/operation_destroyer/truth_dict_contrastive.json",
)
DOMAIN_TRUTH_DICTS = {
    "core":        "truth_dict_contrastive.json",
    "biology":     "truth_dict_biology_contrastive.json",
    "chemistry":   "truth_dict_chemistry_contrastive.json",
    "statistics":  "truth_dict_statistics_contrastive.json",
    "programming": "truth_dict_programming_contrastive.json",
    "astronomy":   "truth_dict_astronomy_contrastive.json",
    "logic_math":  "truth_dict_logic_math_contrastive.json",
    "all":         "truth_dict_all_contrastive.json",
    "mega":        "truth_dict_mega_contrastive.json",
}
DESTROYER_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../experiments/operation_destroyer",
)
RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../results/operation_destroyer/mc_eval",
)


def get_completion_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Sum log P(completion | prompt) — model-agnostic, full sequence comparison."""
    import mlx.core as mx

    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + completion)
    n_prompt = len(prompt_ids)
    if len(full_ids) <= n_prompt:
        return -1e9

    tokens = mx.array(full_ids)[None, :]
    logits = model(tokens)
    mx.eval(logits)
    logits_np = np.array(logits[0].astype(mx.float32))

    total_lp = 0.0
    for i, tok_id in enumerate(full_ids[n_prompt:]):
        pos = n_prompt - 1 + i
        lv = logits_np[pos]
        lse = float(np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max())
        total_lp += float(lv[tok_id]) - lse
    return total_lp


def score_fact(model, tokenizer, context: str, truth: str, distractors: list):
    """Score one fact. Returns (win, margin, truth_lp, best_dist_lp)."""
    prompt = f"{context}:"
    truth_lp = get_completion_logprob(model, tokenizer, prompt, f" {truth}")
    dist_lps = [get_completion_logprob(model, tokenizer, prompt, f" {d}") for d in distractors]
    best_dist = max(dist_lps)
    return truth_lp > best_dist, truth_lp - best_dist, truth_lp, best_dist


def run_eval(model, tokenizer, facts: list) -> dict:
    """Run MC eval on a list of facts. Returns per-fact results and summary."""
    results = []
    categories = {}
    t0 = time.time()

    for fact in facts:
        win, margin, truth_lp, best_dist_lp = score_fact(
            model, tokenizer, fact["context"], fact["truth"], fact["distractors"]
        )
        cat = fact.get("category", "unknown")
        results.append({
            "context": fact["context"],
            "truth": fact["truth"],
            "category": cat,
            "win": win,
            "margin": margin,
        })
        categories.setdefault(cat, {"wins": 0, "total": 0, "margins": []})
        categories[cat]["total"] += 1
        categories[cat]["margins"].append(margin)
        if win:
            categories[cat]["wins"] += 1

    total_wins = sum(r["win"] for r in results)
    total = len(results)
    elapsed = time.time() - t0

    return {
        "accuracy": total_wins / max(total, 1),
        "n_wins": total_wins,
        "n_total": total,
        "mean_margin": float(np.mean([r["margin"] for r in results])) if results else 0.0,
        "elapsed_s": elapsed,
        "categories": {
            k: {
                "wins": v["wins"],
                "total": v["total"],
                "mean_margin": float(np.mean(v["margins"])),
            }
            for k, v in categories.items()
        },
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        prog="rho-mc-eval",
        description="Model-agnostic factual MC accuracy benchmark (log-prob comparison).",
        epilog=(
            "Examples:\n"
            "  rho-mc-eval Qwen/Qwen2.5-0.5B\n"
            "  rho-mc-eval meta-llama/Llama-3.2-1B --domain biology\n"
            "  rho-mc-eval MODEL --truth-dict my_facts.json\n"
            "  rho-mc-eval MODEL --compare results/baseline.json\n"
            "  rho-mc-eval --list-domains\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("model", nargs="?", default=None,
                        help="HuggingFace model name or local path (any mlx_lm-compatible model)")
    parser.add_argument("--truth-dict", type=str, default=None,
                        help="Path to truth_dict JSON file (default: core 115-fact dict)")
    parser.add_argument("--domain", type=str, default="core",
                        choices=list(DOMAIN_TRUTH_DICTS.keys()),
                        help="Predefined domain dict to use (default: core)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save results JSON to file")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare against a previous rho-mc-eval JSON (shows delta)")
    parser.add_argument("--format", choices=["table", "json"], default="table",
                        help="Output format (default: table)")
    parser.add_argument("--list-domains", action="store_true",
                        help="List available predefined domain dicts and exit")

    args = parser.parse_args()

    if args.list_domains:
        print("\nAvailable domains:")
        for name, fname in DOMAIN_TRUTH_DICTS.items():
            path = os.path.join(DESTROYER_DIR, fname)
            exists = "✓" if os.path.exists(path) else "✗ (not generated)"
            print(f"  {name:<12s}  {exists}  {fname}")
        print("\nGenerate missing domain files:")
        print("  python experiments/operation_destroyer/build_extended_truth_dicts.py")
        return

    if args.model is None:
        parser.error("model argument is required")

    # Resolve truth dict path
    if args.truth_dict:
        truth_dict_path = args.truth_dict
    else:
        fname = DOMAIN_TRUTH_DICTS[args.domain]
        truth_dict_path = os.path.join(DESTROYER_DIR, fname)

    if not os.path.exists(truth_dict_path):
        print(f"Truth dict not found: {truth_dict_path}")
        print("Run: python experiments/operation_destroyer/build_extended_truth_dicts.py")
        sys.exit(1)

    with open(truth_dict_path) as f:
        data = json.load(f)
    facts = data["truths"]
    print(f"Loaded {len(facts)} facts from {os.path.basename(truth_dict_path)}")

    # Load model via mlx_lm
    try:
        import mlx_lm
    except ImportError:
        print("mlx_lm not found. Install: pip install mlx-lm")
        sys.exit(1)

    print(f"Loading {args.model}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    print(f"  Loaded in {time.time() - t0:.1f}s\n")

    # Run eval
    results = run_eval(model, tokenizer, facts)

    if args.format == "json":
        output = {"model": args.model, "truth_dict": truth_dict_path, **results}
        print(json.dumps(output, indent=2))
    else:
        print("=" * 60)
        print("  MC FACTUAL ACCURACY")
        print("=" * 60)
        print(f"  Model:       {args.model}")
        print(f"  Accuracy:    {results['n_wins']}/{results['n_total']} "
              f"({results['accuracy']:.1%})")
        print(f"  Mean margin: {results['mean_margin']:+.3f}")
        print(f"  Elapsed:     {results['elapsed_s']:.0f}s")
        print()
        print("  Per category:")
        for cat in sorted(results["categories"]):
            v = results["categories"][cat]
            print(f"    {cat:<15s}: {v['wins']}/{v['total']}  "
                  f"margin={v['mean_margin']:+.3f}")

        failures = [r for r in results["results"] if not r["win"]]
        if failures:
            print(f"\n  Failures ({len(failures)}):")
            for r in sorted(failures, key=lambda x: x["margin"])[:20]:
                print(f"    [{r['category']}] {r['context']!r} → {r['truth']!r}  "
                      f"margin={r['margin']:+.3f}")

        # Comparison
        if args.compare:
            try:
                with open(args.compare) as f:
                    baseline = json.load(f)
                delta_acc = results["accuracy"] - baseline["accuracy"]
                delta_wins = results["n_wins"] - baseline["n_wins"]
                delta_margin = results["mean_margin"] - baseline["mean_margin"]
                print(f"\n  vs {os.path.basename(args.compare)}:")
                print(f"    Δ accuracy:    {delta_acc:+.1%} ({delta_wins:+d} wins)")
                print(f"    Δ mean margin: {delta_margin:+.3f}")
            except Exception as e:
                print(f"\n  Comparison failed: {e}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_name = args.model.replace("/", "_")
    domain_suffix = f"_{args.domain}" if args.domain != "core" else ""
    out_path = args.output or os.path.join(RESULTS_DIR, f"mc_{safe_name}{domain_suffix}.json")

    with open(out_path, "w") as f:
        json.dump({"model": args.model, "truth_dict": truth_dict_path, **results}, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
