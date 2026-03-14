#!/usr/bin/env python3
"""
Hunt for unsolved vortex conservation laws.

Dual-filter approach:
1. Numerical check: frac_var < threshold on vortex_checker ICs
2. Oracle check: margin > 0 with multi-domain adapter

Candidates that pass BOTH are discoveries.

Usage:
    python hunt_vortex.py --problem problems/vortex_unsolved.yaml
    python hunt_vortex.py --problem problems/vortex_unsolved.yaml --max-candidates 100
"""

import argparse
import json
import os
import sys
import time
import yaml
import itertools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # For vortex_checker
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))  # For experiments.*

from vortex_checker import (
    IC_MAP, integrate_vortex, parse_state, frac_var
)


# --------------------------------------------------------------------------
# Candidate generators
# --------------------------------------------------------------------------

def generate_candidates(config: dict) -> list[dict]:
    """Generate candidate expressions from templates."""
    templates = config.get("expression_templates", [])
    alpha_range = config.get("alpha_range", [0.01, 0.1, 0.5, 1.0])

    candidates = []

    # Template-based candidates
    for tmpl in templates:
        if "{alpha}" in tmpl:
            for alpha in alpha_range:
                expr = tmpl.format(alpha=alpha, beta=alpha, gamma=alpha)
                candidates.append({
                    "expr": expr,
                    "template": tmpl,
                    "params": {"alpha": alpha},
                })
        elif "{beta}" in tmpl or "{gamma}" in tmpl:
            for alpha, beta in itertools.product(alpha_range[:4], alpha_range[:4]):
                expr = tmpl.format(alpha=alpha, beta=beta, gamma=alpha)
                candidates.append({
                    "expr": expr,
                    "template": tmpl,
                    "params": {"alpha": alpha, "beta": beta},
                })
        else:
            candidates.append({
                "expr": tmpl,
                "template": tmpl,
                "params": {},
            })

    # Hard-coded interesting candidates from prior exploration
    extra = [
        # Weighted perimeter variants
        "s['r12'] + 0.01*(s['r13'] + s['r23'])",
        "s['r12'] + 0.02*(s['r13'] + s['r23'])",
        "s['r12'] + 0.005*(s['r13'] + s['r23'])",
        # Products
        "s['r12'] * (s['r13'] + s['r23'])",
        "s['r12'] * s['r13'] * s['r23']",
        # Ratios (may blow up)
        "s['r13'] / (s['r23'] + 1e-10)",
        "(s['r13'] - s['r23']) / (s['r13'] + s['r23'] + 1e-10)",
        # Symmetric polynomials
        "s['r12'] + s['r13'] + s['r23']",
        "s['r12']*s['r13'] + s['r12']*s['r23'] + s['r13']*s['r23']",
        "s['r12']**2 + s['r13']**2 + s['r23']**2",
        # Energy-like
        "s['H'] - s['Lz']",
        "s['H'] / (s['Lz'] + 1e-10)",
        # Mixed
        "s['r12']**2 * s['Lz']",
        "s['H'] * s['r12']",
    ]
    for expr in extra:
        if not any(c["expr"] == expr for c in candidates):
            candidates.append({"expr": expr, "template": "extra", "params": {}})

    return candidates


# --------------------------------------------------------------------------
# Numerical filter
# --------------------------------------------------------------------------

def check_numerical(expr: str, ic_names: list[str], threshold: float = 5e-3,
                    t_end: float = 50.0) -> dict:
    """Check if expression is approximately conserved on given ICs."""
    results = {}
    all_pass = True

    for ic_name in ic_names:
        try:
            G, pos0 = IC_MAP[ic_name]()
            t, state = integrate_vortex(G, pos0, t_end=t_end)
            s = parse_state(t, state, G)
            vals = eval(expr, {"s": s, "np": np})
            fv = frac_var(np.asarray(vals))
            passed = fv < threshold
            results[ic_name] = {"frac_var": fv, "pass": passed}
            if not passed:
                all_pass = False
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[ic_name] = {"error": str(e), "pass": False}
            all_pass = False

    return {"all_pass": all_pass, "results": results}


# --------------------------------------------------------------------------
# Oracle filter
# --------------------------------------------------------------------------

def check_oracle(expr: str, model, tokenizer, adapter, lm_head, ic_name: str = "restricted"):
    """Check if oracle assigns positive margin to this candidate."""
    from experiments.operation_destroyer.eval_mc import score_fact_mc

    # Build a fact from the expression
    context = f"Restricted 3-vortex system with Γ₃=0.01. Which quantity is approximately conserved?"
    truth = f"Q = {expr} ≈ const"
    distractors = [
        "No such quantity exists",
        "r₁₂ = const exactly",
        "r₁₂ + r₁₃ + r₂₃ = const",
    ]

    win, margin, truth_lp, best_dist_lp = score_fact_mc(
        model, tokenizer, context, truth, distractors,
        adapter=adapter, lm_head=lm_head
    )
    return {"win": bool(win), "margin": float(margin)}


# --------------------------------------------------------------------------
# Main hunt loop
# --------------------------------------------------------------------------

def run_hunt(problem_path: str, max_candidates: int = 200, skip_oracle: bool = False):
    """Run the hunt for new conserved quantities."""
    import mlx.core as mx
    import mlx_lm

    # Load problem
    with open(problem_path) as f:
        problem = yaml.safe_load(f)

    hunt_cfg = problem.get("hunt_config", {})
    numerical_threshold = float(hunt_cfg.get("numerical_threshold", 5e-3))
    margin_threshold = float(hunt_cfg.get("margin_threshold", 0.0))
    ic_priority = hunt_cfg.get("ic_priority", ["restricted", "three_random"])

    print(f"\n{'='*70}")
    print(f"  VORTEX CONSERVATION HUNT — {problem.get('name', 'unsolved')}")
    print(f"{'='*70}")
    print(f"  Objective: {problem.get('objective', 'find new invariants')[:60]}...")
    print(f"  Numerical threshold: frac_var < {numerical_threshold}")
    print(f"  Margin threshold: > {margin_threshold}")
    print(f"  ICs to check: {ic_priority}")
    print()

    # Generate candidates
    candidates = generate_candidates(hunt_cfg)[:max_candidates]
    print(f"  Generated {len(candidates)} candidate expressions")

    # Load model and adapter for oracle (unless skipping)
    model, tokenizer, adapter, lm_head = None, None, None, None
    if not skip_oracle:
        model_name = problem.get("model", "Qwen/Qwen3-4B-Base")
        adapter_path = problem.get("adapter")
        if adapter_path and not os.path.isabs(adapter_path):
            adapter_path = os.path.join(os.path.dirname(problem_path), adapter_path)

        print(f"\n  Loading model: {model_name}")
        t0 = time.time()
        model, tokenizer = mlx_lm.load(model_name)
        model.freeze()
        print(f"  Loaded in {time.time()-t0:.1f}s")

        if adapter_path and os.path.exists(adapter_path):
            print(f"  Loading adapter: {adapter_path}")
            from oracle_wrapper import load_adapter_for_model
            adapter, lm_head = load_adapter_for_model(model, adapter_path)
        else:
            print(f"  No adapter loaded (path: {adapter_path})")
            from experiments.operation_destroyer.train_v3 import get_lm_head_fn
            lm_head = get_lm_head_fn(model)

    # Hunt loop
    print(f"\n  {'─'*66}")
    print(f"  {'Candidate':<50} {'Numerical':>8} {'Oracle':>8}")
    print(f"  {'─'*66}")

    discoveries = []
    numerical_passes = []

    for i, cand in enumerate(candidates):
        expr = cand["expr"]
        short_expr = expr[:48] + ".." if len(expr) > 50 else expr

        # Numerical check
        num_result = check_numerical(expr, ic_priority, numerical_threshold)
        num_verdict = "PASS" if num_result["all_pass"] else "fail"

        if num_result["all_pass"]:
            numerical_passes.append(cand)

            # Oracle check (only if numerical passes)
            if not skip_oracle and model is not None:
                oracle_result = check_oracle(expr, model, tokenizer, adapter, lm_head)
                oracle_verdict = "PASS" if oracle_result["margin"] > margin_threshold else "fail"
                margin = oracle_result["margin"]

                if oracle_result["margin"] > margin_threshold:
                    discoveries.append({
                        **cand,
                        "numerical": num_result,
                        "oracle": oracle_result,
                    })
                    print(f"  {short_expr:<50} {num_verdict:>8} {oracle_verdict:>8}  ← DISCOVERY! margin={margin:+.2f}")
                else:
                    print(f"  {short_expr:<50} {num_verdict:>8} {oracle_verdict:>8}  margin={margin:+.2f}")
            else:
                print(f"  {short_expr:<50} {num_verdict:>8} {'skip':>8}")
        else:
            # Show first few failures for debugging
            if i < 10:
                best_ic = min(num_result["results"].items(),
                             key=lambda x: x[1].get("frac_var", 999) if "frac_var" in x[1] else 999)
                fv = best_ic[1].get("frac_var", "err")
                fv_str = f"{fv:.1e}" if isinstance(fv, float) else fv
                print(f"  {short_expr:<50} {num_verdict:>8} {'─':>8}  best={best_ic[0]}:{fv_str}")

    # Summary
    print(f"\n  {'='*66}")
    print(f"  HUNT SUMMARY")
    print(f"  {'='*66}")
    print(f"  Candidates tested:     {len(candidates)}")
    print(f"  Numerical passes:      {len(numerical_passes)}")
    print(f"  Discoveries:           {len(discoveries)}")

    if discoveries:
        print(f"\n  DISCOVERIES:")
        for d in discoveries:
            print(f"    Q = {d['expr']}")
            print(f"      margin = {d['oracle']['margin']:+.3f}")
            for ic, res in d["numerical"]["results"].items():
                if "frac_var" in res:
                    print(f"      {ic}: frac_var = {res['frac_var']:.2e}")

    # Save results
    results_path = os.path.join(HERE, "results", "hunt_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "problem": problem_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_candidates": len(candidates),
            "n_numerical_passes": len(numerical_passes),
            "n_discoveries": len(discoveries),
            "discoveries": discoveries,
            "numerical_passes": [c["expr"] for c in numerical_passes],
        }, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    return discoveries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default=os.path.join(HERE, "problems/vortex_unsolved.yaml"))
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--skip-oracle", action="store_true",
                        help="Skip oracle check (numerical only)")
    args = parser.parse_args()

    run_hunt(args.problem, args.max_candidates, args.skip_oracle)


if __name__ == "__main__":
    main()
