#!/usr/bin/env python3
"""Exp05: Targeted adapter for 9 stubborn failures.

The exp03 mixed adapter fixed 10/19 wrong facts but left 9 wrong with -0.78 avg degradation.
These 9 split into 3 failure modes:

  RECOVERABLE (moving in right direction, just not enough — 4 missing_constant facts):
    - Coulomb's law force:       -0.75 → -0.30  (+0.45)
    - gravitational force:       -0.72 → -0.32  (+0.40)
    - Stefan-Boltzmann power:    -1.53 → -0.70  (+0.83)
    - eigenvalue equation:       -1.50 → -0.79  (+0.71)

  CONFLICTED (correct answer competes with training examples — 3 truncation/linearity):
    - hybridization methane:     -0.20 → -1.37  (worse — sp3 conflicts with sp training example)
    - hybridization ethylene:    -0.46 → -1.52  (worse — sp2 conflicts with sp training example)
    - spring potential energy:   -0.20 → -6.56  (CATASTROPHIC — k*x^2 confused with v^2 training)

  BORDERLINE (near-zero, barely moved):
    - kinetic energy formula:    -0.37 → -0.39  (unchanged)
    - area under parabola:       -0.23 → -0.98  (slightly worse)

Strategy:
  1. Target the 4 RECOVERABLE facts — train a larger adapter (d_inner=256) on ONLY missing_constant
     examples, enriched with the 4 stubborn facts explicitly repeated 3x in training.
  2. Measure: do all 4 flip to positive? Does it cause regression on the other patterns?

This tests: capacity ceiling vs data density (the 4 stubborn facts need more weight).

Usage:
    python sub_experiments/exp05_stubborn/run_exp05.py
"""

import argparse, json, os, sys, time
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/snap_on")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten

from experiments.snap_on.module import SnapOnConfig, create_adapter
import experiments.operation_destroyer.train_v3 as t3

OUT_DIR = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/sub_experiments/exp05_stubborn"

# ── Training data ─────────────────────────────────────────────────────────────
# Strategy: enrich missing_constant set with the 4 stubborn facts repeated 3x
# so gradient signal focuses on exactly the patterns that nearly worked.

MISSING_CONSTANT_BASE = [
    ("Coulomb's law force",          "k*q1*q2/r^2",    ["k*q1*q2/r", "k*q1*q2*r^2", "q1*q2/r^2", "k/(q1*q2*r^2)"]),
    ("gravitational force",          "G*m1*m2/r^2",    ["m1*m2/r^2", "G*m1/r^2", "G*m1*m2/r", "G*m2/r^2"]),
    ("photon energy formula",        "E=h*f",          ["E=h*lambda", "E=h/f", "E=h*f^2", "E=f/h"]),
    ("de Broglie wavelength",        "h/(m*v)",        ["m*v/h", "h*m*v", "h/(m^2*v)", "m*v/h^2"]),
    ("Stefan-Boltzmann power",       "sigma*A*T^4",    ["A*T^4", "sigma*T^4", "A*T^4/sigma", "sigma*A/T^4"]),
    ("eigenvalue equation",          "A*v=lambda*v",   ["A*v=lambda", "A=lambda*v", "A*v=v/lambda", "A*lambda=v"]),
    ("scalar factor in inverse of 2x2 matrix [[a,b],[c,d]]", "1/(a*d-b*c)", ["1/(a*d+b*c)", "a*d-b*c", "1/(a*b-c*d)", "(a*d-b*c)"]),
    ("determinant of 2x2 matrix [[a,b],[c,d]]", "a*d-b*c", ["a*b-c*d", "a*d+b*c", "a*c-b*d", "a*b+c*d"]),
    ("Bayes theorem formula",        "P(A|B)=P(B|A)*P(A)/P(B)", ["P(A|B)=P(A)*P(B|A)", "P(A|B)=P(B|A)/P(A)", "P(A|B)=P(A and B)/P(A)", "P(A|B)=P(B|A)*P(B)/P(A)"]),
    ("standard error of the mean",  "sigma/sqrt(n)",  ["sigma/n", "sigma^2/n", "sigma*sqrt(n)", "sigma/n^2"]),
]

# The 4 stubborn ones (repeated extra times for emphasis)
STUBBORN_MC = [
    ("Coulomb's law force",          "k*q1*q2/r^2",    ["k*q1*q2/r", "k*q1*q2*r^2", "q1*q2/r^2", "k/(q1*q2*r^2)"]),
    ("gravitational force",          "G*m1*m2/r^2",    ["m1*m2/r^2", "G*m1/r^2", "G*m1*m2/r", "G*m2/r^2"]),
    ("Stefan-Boltzmann power",       "sigma*A*T^4",    ["A*T^4", "sigma*T^4", "A*T^4/sigma", "sigma*A/T^4"]),
    ("eigenvalue equation",          "A*v=lambda*v",   ["A*v=lambda", "A=lambda*v", "A*v=v/lambda", "A*lambda=v"]),
]

# Build enriched training set: base (10) + stubborn repeated 3x (12) = 22 examples
TRAIN_SET = MISSING_CONSTANT_BASE + STUBBORN_MC * 3

# Eval set: all 40 original facts (to check cross-pattern regression)
ALL_40 = {
    "positivity": [
        ("derivative of cos(x)",         "-sin(x)",        ["sin(x)", "-cos(x)", "cos(x)", "tan(x)"]),
        ("second derivative of sin(x)",  "-sin(x)",        ["sin(x)", "cos(x)", "-cos(x)", "0"]),
        ("integral of sin(x)",           "-cos(x)+C",      ["cos(x)+C", "-sin(x)+C", "sin^2(x)+C", "sin(x)+C"]),
        ("derivative of cos(2*x)",       "-2*sin(2*x)",    ["2*sin(2*x)", "-sin(2*x)", "2*cos(2*x)", "-2*cos(2*x)"]),
        ("second derivative of cos(x)",  "-cos(x)",        ["cos(x)", "-sin(x)", "sin(x)", "0"]),
        ("integral of sin(2*x)",         "-(1/2)*cos(2*x)+C", ["(1/2)*cos(2*x)+C", "-cos(2*x)+C", "cos(2*x)+C", "(1/2)*sin(2*x)+C"]),
        ("first law of thermodynamics",  "delta(U)=Q-W",   ["delta(U)=Q+W", "delta(U)=W-Q", "Q=delta(U)+W", "delta(U)=Q*W"]),
        ("Gibbs free energy formula",    "G=H-T*S",        ["G=H+T*S", "G=H*T*S", "G=T*S-H", "G=H/(T*S)"]),
        ("entropy change when heat flows out", "-Q/T",     ["Q/T", "-T/Q", "T/Q", "0"]),
        ("work done by gas in isothermal compression", "-P*delta(V)", ["P*delta(V)", "-delta(V)/P", "P/delta(V)", "delta(V)*T"]),
    ],
    "linearity": [
        ("kinetic energy formula",       "(1/2)*m*v^2",    ["m*v^2", "m*v", "(1/2)*m*v", "2*m*v^2"]),
        ("energy stored in capacitor",   "(1/2)*C*V^2",    ["C*V^2", "(1/2)*Q*V^2", "C*V^2/(2*Q)", "Q^2/(2*C)"]),
        ("centripetal acceleration",     "v^2/r",          ["v*r", "v/r", "v^2*r", "2*v/r"]),
        ("electric potential energy",    "k*q1*q2/r",      ["k*q1*q2/r^2", "k*q1*q2*r", "q1*q2/r", "k/r"]),
        ("gravitational potential energy formula", "G*m1*m2/r", ["G*m1*m2/r^2", "G*m1*m2*r", "m1*m2/r", "G*r"]),
        ("spring potential energy",      "(1/2)*k*x^2",    ["k*x^2", "k*x", "(1/2)*k*x", "2*k*x^2"]),
        ("integral of x",                "x^2/2+C",        ["x+C", "x^2+C", "2*x+C", "x^3/3+C"]),
        ("integral of x^2",              "x^3/3+C",        ["x^2/2+C", "x^2+C", "x^3+C", "3*x^2+C"]),
        ("area under parabola from 0 to a", "a^3/3",       ["a^2/2", "a^2", "a^3", "a/3"]),
        ("variance of uniform distribution from 0 to 1", "(b-a)^2/12", ["(b-a)/12", "(b-a)^2/6", "(b+a)^2/12", "(b-a)/6"]),
    ],
    "missing_constant": [
        ("Coulomb's law force",          "k*q1*q2/r^2",    ["k*q1*q2/r", "k*q1*q2*r^2", "q1*q2/r^2", "k/(q1*q2*r^2)"]),
        ("gravitational force",          "G*m1*m2/r^2",    ["m1*m2/r^2", "G*m1/r^2", "G*m1*m2/r", "G*m2/r^2"]),
        ("photon energy formula",        "E=h*f",          ["E=h*lambda", "E=h/f", "E=h*f^2", "E=f/h"]),
        ("de Broglie wavelength",        "h/(m*v)",        ["m*v/h", "h*m*v", "h/(m^2*v)", "m*v/h^2"]),
        ("Stefan-Boltzmann power",       "sigma*A*T^4",    ["A*T^4", "sigma*T^4", "A*T^4/sigma", "sigma*A/T^4"]),
        ("eigenvalue equation",          "A*v=lambda*v",   ["A*v=lambda", "A=lambda*v", "A*v=v/lambda", "A*lambda=v"]),
        ("scalar factor in inverse of 2x2 matrix [[a,b],[c,d]]", "1/(a*d-b*c)", ["1/(a*d+b*c)", "a*d-b*c", "1/(a*b-c*d)", "(a*d-b*c)"]),
        ("determinant of 2x2 matrix [[a,b],[c,d]]", "a*d-b*c", ["a*b-c*d", "a*d+b*c", "a*c-b*d", "a*b+c*d"]),
        ("Bayes theorem formula",        "P(A|B)=P(B|A)*P(A)/P(B)", ["P(A|B)=P(A)*P(B|A)", "P(A|B)=P(B|A)/P(A)", "P(A|B)=P(A and B)/P(A)", "P(A|B)=P(B|A)*P(B)/P(A)"]),
        ("standard error of the mean",  "sigma/sqrt(n)",  ["sigma/n", "sigma^2/n", "sigma*sqrt(n)", "sigma/n^2"]),
    ],
    "truncation": [
        ("hybridization of carbon in methane",   "sp3",   ["sp2", "sp", "sp3d", "sp2d"]),
        ("hybridization of carbon in ethylene",  "sp2",   ["sp3", "sp", "sp3d", "sp2d"]),
        ("hybridization of carbon in acetylene", "sp",    ["sp2", "sp3", "sp3d", "sp2d"]),
        ("Arrhenius equation for rate constant", "k=A*e^(-Ea/(R*T))", ["k=A*e^(Ea/(R*T))", "k=Ea/(R*T)", "k=A/e^(Ea/(R*T))", "k=R*T/Ea"]),
        ("Henderson-Hasselbalch equation",       "pH=pKa+log([A-]/[HA])", ["pH=pKa-log([A-]/[HA])", "pH=pKa+log([HA]/[A-])", "pH=Ka+log([A-]/[HA])", "pH=pKa*log([A-]/[HA])"]),
        ("characteristic polynomial (definition)", "det(A-lambda*I)", ["det(A+lambda*I)", "tr(A-lambda*I)", "det(A)-lambda", "det(lambda*I-A)"]),
        ("Cauchy-Schwarz inequality",            "|u*v| <= |u|*|v|", ["|u*v| >= |u|*|v|", "|u*v| = |u|*|v|", "|u*v| < |u|+|v|", "|u*v| <= |u|+|v|"]),
        ("Taylor series first two terms of e^x around 0", "1+x", ["x+x^2", "e+e*x", "1+x+x^2/2", "1-x"]),
        ("period of a simple pendulum",          "2*pi*sqrt(L/g)", ["2*pi*sqrt(g/L)", "2*pi*L/g", "pi*sqrt(L/g)", "2*pi*sqrt(m/k)"]),
        ("escape velocity formula",              "sqrt(2*G*M/r)", ["sqrt(G*M/r)", "2*G*M/r^2", "sqrt(G*M/(2*r))", "G*M/r^2"]),
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_lm_head(model):
    try:
        return lambda h: model.model.embed_tokens.as_linear(h)
    except Exception:
        return model.lm_head


def compute_lp(adapter, lm_head, model, prompt, completion, tokenizer):
    prompt_ids = tokenizer.encode(prompt)
    comp_ids   = tokenizer.encode(completion)
    full_ids   = prompt_ids + comp_ids
    if not comp_ids:
        return mx.array(-1e9)
    tokens = mx.array(full_ids)[None, :]
    h = model.model(tokens)
    base_logits = lm_head(h)
    shifts = adapter(base_logits)
    shifts = shifts - shifts.mean(axis=-1, keepdims=True)
    combined = base_logits + shifts
    logits = t3.LOGIT_SOFTCAP * mx.tanh(combined / t3.LOGIT_SOFTCAP)
    total_lp = mx.array(0.0)
    for i, tok_id in enumerate(comp_ids):
        pos = len(prompt_ids) - 1 + i
        lv  = logits[0, pos]
        lse = mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv)
        total_lp = total_lp + lv[tok_id] - lse
    return total_lp


def hinge_loss(adapter, lm_head, model, tokenizer, ctx, truth, distractors, margin=1.5):
    prompt    = f"context: {ctx}\nanswer:"
    truth_lp  = compute_lp(adapter, lm_head, model, prompt, " " + truth, tokenizer)
    dist_lps  = [compute_lp(adapter, lm_head, model, prompt, " " + d, tokenizer) for d in distractors]
    best_dist = mx.max(mx.stack(dist_lps))
    loss      = mx.maximum(mx.array(0.0), mx.array(margin) - (truth_lp - best_dist))
    return loss, float(truth_lp - best_dist)


def clip_grads(grads, max_norm=1.0):
    from mlx.utils import tree_flatten, tree_map
    leaves = tree_flatten(grads)
    norm   = sum(float(mx.sum(g**2)) for _, g in leaves) ** 0.5
    scale  = min(1.0, max_norm / (norm + 1e-6))
    return tree_map(lambda g: g * scale, grads)


def eval_all_40(adapter, lm_head, model, tokenizer):
    results = {}
    for pat, examples in ALL_40.items():
        wins, margins = [], []
        for ctx, truth, distractors in examples:
            prompt   = f"context: {ctx}\nanswer:"
            truth_lp = float(compute_lp(adapter, lm_head, model, prompt, " " + truth, tokenizer))
            dist_lps = [float(compute_lp(adapter, lm_head, model, prompt, " " + d, tokenizer)) for d in distractors]
            m = truth_lp - max(dist_lps)
            wins.append(m > 0)
            margins.append(m)
        results[pat] = {"acc": sum(wins)/len(wins), "wins": sum(wins), "n": len(wins),
                        "mean_margin": sum(margins)/len(margins)}
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_inner",       type=int,   default=256)
    parser.add_argument("--steps",         type=int,   default=3000)
    parser.add_argument("--lr",            type=float, default=5e-7)
    parser.add_argument("--margin",        type=float, default=1.5)
    parser.add_argument("--early_stop_at", type=int,   default=4,
                        help="Stop if this many of the 4 stubborn targets flip positive")
    args = parser.parse_args()

    print(f"Loading Qwen/Qwen3-4B-Base...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    lm_head = get_lm_head(model)
    mx.eval(model.parameters())
    print(f"Loaded in {time.time()-t0:.1f}s")

    cfg = SnapOnConfig(vocab_size=151936, d_inner=args.d_inner, n_heads=4, mode="logit")
    adapter = create_adapter(cfg)
    mx.eval(adapter.parameters())
    optimizer = optim.Adam(learning_rate=args.lr)

    print(f"\nConfig: d_inner={args.d_inner}, steps={args.steps}, lr={args.lr}, margin={args.margin}")
    print(f"Training set: {len(TRAIN_SET)} examples (base 10 + stubborn 4×3)")
    print(f"\nEvaluating baseline on all 40 facts...")
    base_results = eval_all_40(adapter, lm_head, model, tokenizer)
    print("Baseline:")
    for pat, r in base_results.items():
        print(f"  {pat:20s}: {r['wins']}/{r['n']} ({r['acc']:.0%}) mean_margin={r['mean_margin']:+.2f}")

    print(f"\nTraining ({args.steps} steps, early stop at {args.early_stop_at}/4 stubborn fixed)...")

    step_log = []
    stopped_early = False
    for step in range(args.steps):
        example = TRAIN_SET[step % len(TRAIN_SET)]
        ctx, truth, distractors = example

        def loss_fn(a):
            return hinge_loss(a, lm_head, model, tokenizer, ctx, truth, distractors, args.margin)[0]

        loss, grads = nn.value_and_grad(adapter, loss_fn)(adapter)
        grads = clip_grads(grads, max_norm=1.0)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if (step + 1) % 200 == 0:
            l = float(loss)
            # Check how many stubborn targets are now positive
            n_fixed = 0
            for s_ctx, s_truth, s_dist in STUBBORN_MC:
                prompt   = f"context: {s_ctx}\nanswer:"
                t_lp     = float(compute_lp(adapter, lm_head, model, prompt, " " + s_truth, tokenizer))
                d_lps    = [float(compute_lp(adapter, lm_head, model, prompt, " " + d, tokenizer)) for d in s_dist]
                if t_lp - max(d_lps) > 0:
                    n_fixed += 1
            print(f"  Step {step+1:4d} | loss={l:.4f} | stubborn fixed: {n_fixed}/4")
            step_log.append({"step": step+1, "loss": l, "stubborn_fixed": n_fixed})
            if n_fixed >= args.early_stop_at:
                print(f"  Early stop: {n_fixed}/{args.early_stop_at} stubborn targets fixed.")
                stopped_early = True
                break

    print(f"\nEvaluating after training on all 40 facts...")
    post_results = eval_all_40(adapter, lm_head, model, tokenizer)
    print("\nResults (baseline → trained):")
    for pat, r in post_results.items():
        b = base_results[pat]
        delta_acc = r['acc'] - b['acc']
        delta_m   = r['mean_margin'] - b['mean_margin']
        sign = "+" if delta_acc >= 0 else ""
        print(f"  {pat:20s}: {b['wins']}/{b['n']} → {r['wins']}/{r['n']} "
              f"({sign}{delta_acc:+.0%}) | margin {b['mean_margin']:+.2f} → {r['mean_margin']:+.2f} ({delta_m:+.2f})")

    print("\nStubborn facts specifically (the 4 targets):")
    for ctx, truth, distractors in STUBBORN_MC:
        prompt   = f"context: {ctx}\nanswer:"
        truth_lp = float(compute_lp(adapter, lm_head, model, prompt, " " + truth, tokenizer))
        dist_lps = [float(compute_lp(adapter, lm_head, model, prompt, " " + d, tokenizer)) for d in distractors]
        m = truth_lp - max(dist_lps)
        sym = "✓" if m > 0 else "✗"
        print(f"  {ctx[:45]:45s} margin={m:+.2f} {sym}")

    out = {
        "config":       vars(args),
        "n_train":      len(TRAIN_SET),
        "stopped_early": stopped_early,
        "baseline":     {pat: r for pat, r in base_results.items()},
        "post_train":   {pat: r for pat, r in post_results.items()},
        "step_log":     step_log,
    }
    mx.savez(os.path.join(OUT_DIR, "adapter_stubborn.npz"),
             **dict(tree_flatten(adapter.parameters())))
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved adapter and results to {OUT_DIR}/")


if __name__ == "__main__":
    main()
