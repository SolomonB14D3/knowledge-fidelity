#!/usr/bin/env python3
"""
Oracle wrapper for Operation Frontier.

Loads a model and a problem definition (problem.yaml or problems/*.json),
runs the STEM margin oracle on a verification set, and returns a pass/fail
signal with per-fact margin details.

The oracle rule (from Paper 9):
  positive margin → oracle picks truth   (0% false negatives)
  negative margin → oracle picks wrong   (0% false positives)

A candidate config passes iff ALL verification facts have positive margin,
OR the fraction of positive-margin facts meets the pass_threshold.

Usage:
    # Check baseline (no adapter):
    python oracle_wrapper.py --problem problems/kinetic_energy_pilot.json

    # Check with mixed adapter:
    python oracle_wrapper.py --problem problems/kinetic_energy_pilot.json \
        --adapter ../operation_destroyer/sub_experiments/exp03_correction/adapter_mixed.npz

    # Full loop: baseline → repair if needed → recheck:
    python oracle_wrapper.py --problem problems/kinetic_energy_pilot.json --repair
"""

import argparse
import json
import sys
import os
import time
import yaml

import mlx.core as mx
import mlx_lm
import numpy as np

# Reuse oracle machinery from Operation Destroyer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DESTROYER_DIR = os.path.join(os.path.dirname(__file__), "..", "operation_destroyer")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.operation_destroyer.eval_mc import score_fact_mc, get_completion_logprob


# --------------------------------------------------------------------------
# Load problem definition
# --------------------------------------------------------------------------

def load_problem(path: str) -> dict:
    """Load a problem from .yaml or .json."""
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f)
        return json.load(f)


def load_verification_set(problem: dict, problem_dir: str) -> list:
    """Load facts from the problem's verification_set path."""
    vs_path = problem.get("verification_set")
    if not os.path.isabs(vs_path):
        vs_path = os.path.join(problem_dir, vs_path)
    with open(vs_path) as f:
        data = json.load(f)
    # Support both flat list and {"facts": [...]} format
    if isinstance(data, list):
        return data
    return data.get("facts", data.get("truths", []))


# --------------------------------------------------------------------------
# Run oracle
# --------------------------------------------------------------------------

def run_oracle(model, tokenizer, facts: list, adapter=None, lm_head=None) -> dict:
    """Score all facts and return per-fact results + summary."""
    results = []
    for fact in facts:
        win, margin, truth_lp, best_dist_lp = score_fact_mc(
            model, tokenizer,
            fact["context"], fact["truth"], fact["distractors"],
            adapter=adapter, lm_head=lm_head,
        )
        results.append({
            "context":  fact["context"],
            "truth":    fact["truth"],
            "win":      bool(win),
            "margin":   float(margin),
        })

    n_pass = sum(r["win"] for r in results)
    n_total = len(results)
    mean_margin = float(np.mean([r["margin"] for r in results]))
    min_margin  = float(np.min([r["margin"] for r in results]))

    return {
        "n_pass":      n_pass,
        "n_total":     n_total,
        "frac_pass":   n_pass / n_total if n_total else 0.0,
        "mean_margin": mean_margin,
        "min_margin":  min_margin,
        "results":     results,
    }


def load_adapter_for_model(model, adapter_path: str, d_inner: int = 64):
    """Load a snap-on adapter .npz onto a frozen model."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "snap_on"))
    from module import SnapOnConfig, create_adapter
    from experiments.operation_destroyer.train_v3 import get_lm_head_fn

    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model    = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    weights = mx.load(adapter_path)
    adapter.load_weights(list(weights.items()))
    mx.eval(adapter.parameters())
    lm_head = get_lm_head_fn(model)
    return adapter, lm_head


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def print_report(label: str, summary: dict, threshold: float):
    passed = summary["frac_pass"] >= threshold
    verdict = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n{'='*60}")
    print(f"  Oracle Report — {label}")
    print(f"{'='*60}")
    print(f"  Verdict:      {verdict}  (threshold: {threshold:.0%})")
    print(f"  Pass rate:    {summary['n_pass']}/{summary['n_total']}  ({summary['frac_pass']:.1%})")
    print(f"  Mean margin:  {summary['mean_margin']:+.3f}")
    print(f"  Min margin:   {summary['min_margin']:+.3f}")
    failures = [r for r in summary["results"] if not r["win"]]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r in sorted(failures, key=lambda x: x["margin"]):
            print(f"    {r['context']!r:40s} → {r['truth']!r}  margin={r['margin']:+.3f}")
    return passed


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True,
                        help="Path to problem .yaml or .json")
    parser.add_argument("--adapter", default=None,
                        help="Path to .npz adapter (overrides problem.yaml)")
    parser.add_argument("--repair", action="store_true",
                        help="If baseline fails, apply mixed adapter and recheck")
    parser.add_argument("--d-inner", type=int, default=64)
    args = parser.parse_args()

    problem_dir = os.path.dirname(os.path.abspath(args.problem))
    problem = load_problem(args.problem)
    facts = load_verification_set(problem, problem_dir)
    model_name = problem.get("model", "Qwen/Qwen3-4B-Base")
    threshold = float(problem.get("pass_threshold", 1.0))
    mixed_adapter_path = os.path.join(
        DESTROYER_DIR,
        "sub_experiments/exp03_correction/adapter_mixed.npz"
    )

    print(f"\nProblem:  {problem.get('name', args.problem)}")
    print(f"Model:    {model_name}")
    print(f"Facts:    {len(facts)}")
    print(f"Threshold: {threshold:.0%}")

    # Load model
    t0 = time.time()
    print(f"\nLoading {model_name}...")
    model, tokenizer = mlx_lm.load(model_name)
    model.freeze()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # --- Baseline pass ---
    print("\nRunning baseline oracle...")
    baseline = run_oracle(model, tokenizer, facts)
    baseline_pass = print_report("Baseline (no adapter)", baseline, threshold)

    # --- Repair pass (if requested and baseline failed) ---
    if not baseline_pass and args.repair:
        adapter_path = args.adapter or mixed_adapter_path
        if not os.path.exists(adapter_path):
            print(f"\n  [repair] Adapter not found: {adapter_path}")
        else:
            print(f"\nApplying adapter: {adapter_path}")
            adapter, lm_head = load_adapter_for_model(model, adapter_path, args.d_inner)
            repaired = run_oracle(model, tokenizer, facts, adapter=adapter, lm_head=lm_head)
            print_report("After adapter repair", repaired, threshold)

    # --- Explicit adapter pass ---
    elif args.adapter and args.adapter != "none":
        print(f"\nApplying adapter: {args.adapter}")
        adapter, lm_head = load_adapter_for_model(model, args.adapter, args.d_inner)
        adapted = run_oracle(model, tokenizer, facts, adapter=adapter, lm_head=lm_head)
        print_report("With adapter", adapted, threshold)

    print()


if __name__ == "__main__":
    main()
