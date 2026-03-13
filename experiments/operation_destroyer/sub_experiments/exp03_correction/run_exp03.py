#!/usr/bin/env python3
"""Exp03: Targeted adapter correction of the 4 systematic STEM bias patterns.

Design:
- 4 bias types: positivity, linearity, missing_constant, truncation
- Each has ~10 train examples + hold-out examples from other domains
- Train 5 adapters: one per bias type + one mixed (all 4)
- Evaluate each adapter on all 4 bias types to produce 4x4 transfer matrix
- Key question: does fixing calculus positivity bias transfer to physics positivity bias?
  (STEM analog of the syco->bias cross-transfer finding)

Usage:
    python sub_experiments/exp03_correction/run_exp03.py
    python sub_experiments/exp03_correction/run_exp03.py --arms positivity linearity
    python sub_experiments/exp03_correction/run_exp03.py --steps 500
"""

import argparse, json, os, sys, time
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/snap_on")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from experiments.operation_destroyer.eval_mc import get_completion_logprob, score_fact_mc
from experiments.snap_on.module import SnapOnConfig, create_adapter
import experiments.operation_destroyer.train_v3 as t3

OUT_DIR = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/sub_experiments/exp03_correction"

# ─────────────────────────────────────────────────────────────────────────────
# Training examples for each bias pattern
# Labeled with which domain they come from for cross-transfer tracking
# ─────────────────────────────────────────────────────────────────────────────

# Pattern 1: POSITIVITY BIAS — truth has a leading minus sign; model prefers unsigned form
POSITIVITY = [
    # Calculus (train split)
    ("derivative of cos(x)",         "-sin(x)",        ["sin(x)", "-cos(x)", "cos(x)", "tan(x)"],           "calculus"),
    ("second derivative of sin(x)",  "-sin(x)",        ["sin(x)", "cos(x)", "-cos(x)", "0"],                "calculus"),
    ("integral of sin(x)",           "-cos(x)+C",      ["cos(x)+C", "-sin(x)+C", "sin^2(x)+C", "sin(x)+C"], "calculus"),
    ("derivative of cos(2*x)",       "-2*sin(2*x)",    ["2*sin(2*x)", "-sin(2*x)", "2*cos(2*x)", "-2*cos(2*x)"], "calculus"),
    ("second derivative of cos(x)",  "-cos(x)",        ["cos(x)", "-sin(x)", "sin(x)", "0"],                 "calculus"),
    ("integral of sin(2*x)",         "-(1/2)*cos(2*x)+C", ["(1/2)*cos(2*x)+C", "-cos(2*x)+C", "cos(2*x)+C", "(1/2)*sin(2*x)+C"], "calculus"),
    # Physics (cross-transfer test split)
    ("first law of thermodynamics",  "delta(U)=Q-W",   ["delta(U)=Q+W", "delta(U)=W-Q", "Q=delta(U)+W", "delta(U)=Q*W"], "physics"),
    ("Gibbs free energy formula",    "G=H-T*S",        ["G=H+T*S", "G=H*T*S", "G=T*S-H", "G=H/(T*S)"],    "chemistry"),
    ("entropy change when heat flows out", "-Q/T",     ["Q/T", "-T/Q", "T/Q", "0"],                         "physics"),
    ("work done by gas in isothermal compression", "-P*delta(V)", ["P*delta(V)", "-delta(V)/P", "P/delta(V)", "delta(V)*T"], "physics"),
]

# Pattern 2: LINEARITY BIAS — truth has v^2 or r^2; model prefers v or r (linear form)
LINEARITY = [
    # Physics (train split)
    ("kinetic energy formula",       "(1/2)*m*v^2",    ["m*v^2", "m*v", "(1/2)*m*v", "2*m*v^2"],            "physics"),
    ("energy stored in capacitor",   "(1/2)*C*V^2",    ["C*V^2", "(1/2)*Q*V^2", "C*V^2/(2*Q)", "Q^2/(2*C)"], "physics"),
    ("centripetal acceleration",     "v^2/r",          ["v*r", "v/r", "v^2*r", "2*v/r"],                    "physics"),
    ("electric potential energy",    "k*q1*q2/r",      ["k*q1*q2/r^2", "k*q1*q2*r", "q1*q2/r", "k/r"],     "physics"),
    ("gravitational potential energy formula", "G*m1*m2/r", ["G*m1*m2/r^2", "G*m1*m2*r", "m1*m2/r", "G*r"], "physics"),
    ("spring potential energy",      "(1/2)*k*x^2",    ["k*x^2", "k*x", "(1/2)*k*x", "2*k*x^2"],            "physics"),
    # Calculus (cross-transfer test split)
    ("integral of x",                "x^2/2+C",        ["x+C", "x^2+C", "2*x+C", "x^3/3+C"],               "calculus"),
    ("integral of x^2",              "x^3/3+C",        ["x^2/2+C", "x^2+C", "x^3+C", "3*x^2+C"],           "calculus"),
    ("area under parabola from 0 to a", "a^3/3",       ["a^2/2", "a^2", "a^3", "a/3"],                      "calculus"),
    ("variance of uniform distribution from 0 to 1", "(b-a)^2/12", ["(b-a)/12", "(b-a)^2/6", "(b+a)^2/12", "(b-a)/6"], "statistics"),
]

# Pattern 3: MISSING CONSTANT BIAS — truth has a proportionality constant; model drops it
MISSING_CONSTANT = [
    # Physics (train split)
    ("Coulomb's law force",          "k*q1*q2/r^2",    ["k*q1*q2/r", "k*q1*q2*r^2", "q1*q2/r^2", "k/(q1*q2*r^2)"], "physics"),
    ("gravitational force",          "G*m1*m2/r^2",    ["m1*m2/r^2", "G*m1/r^2", "G*m1*m2/r", "G*m2/r^2"], "physics"),
    ("photon energy formula",        "E=h*f",          ["E=h*lambda", "E=h/f", "E=h*f^2", "E=f/h"],         "physics"),
    ("de Broglie wavelength",        "h/(m*v)",        ["m*v/h", "h*m*v", "h/(m^2*v)", "m*v/h^2"],          "physics"),
    ("Stefan-Boltzmann power",       "sigma*A*T^4",    ["A*T^4", "sigma*T^4", "A*T^4/sigma", "sigma*A/T^4"], "physics"),
    # Linear algebra (cross-transfer test split)
    ("eigenvalue equation",          "A*v=lambda*v",   ["A*v=lambda", "A=lambda*v", "A*v=v/lambda", "A*lambda=v"], "linear_algebra"),
    ("scalar factor in inverse of 2x2 matrix [[a,b],[c,d]]", "1/(a*d-b*c)", ["1/(a*d+b*c)", "a*d-b*c", "1/(a*b-c*d)", "(a*d-b*c)"], "linear_algebra"),
    ("determinant of 2x2 matrix [[a,b],[c,d]]", "a*d-b*c", ["a*b-c*d", "a*d+b*c", "a*c-b*d", "a*b+c*d"],   "linear_algebra"),
    ("Bayes theorem formula",        "P(A|B)=P(B|A)*P(A)/P(B)", ["P(A|B)=P(A)*P(B|A)", "P(A|B)=P(B|A)/P(A)", "P(A|B)=P(A and B)/P(A)", "P(A|B)=P(B|A)*P(B)/P(A)"], "statistics"),
    ("standard error of the mean",  "sigma/sqrt(n)",  ["sigma/n", "sigma^2/n", "sigma*sqrt(n)", "sigma/n^2"], "statistics"),
]

# Pattern 4: TRUNCATION BIAS — truth is a complete symbolic form; model gives shorter version
TRUNCATION = [
    # Chemistry (train split)
    ("hybridization of carbon in methane",   "sp3",   ["sp2", "sp", "sp3d", "sp2d"],                        "chemistry"),
    ("hybridization of carbon in ethylene",  "sp2",   ["sp3", "sp", "sp3d", "sp2d"],                        "chemistry"),
    ("hybridization of carbon in acetylene", "sp",    ["sp2", "sp3", "sp3d", "sp2d"],                       "chemistry"),
    ("Arrhenius equation for rate constant", "k=A*e^(-Ea/(R*T))", ["k=A*e^(Ea/(R*T))", "k=Ea/(R*T)", "k=A/e^(Ea/(R*T))", "k=R*T/Ea"], "chemistry"),
    ("Henderson-Hasselbalch equation",       "pH=pKa+log([A-]/[HA])", ["pH=pKa-log([A-]/[HA])", "pH=pKa+log([HA]/[A-])", "pH=Ka+log([A-]/[HA])", "pH=pKa*log([A-]/[HA])"], "chemistry"),
    # Linear algebra (cross-transfer test split)
    ("characteristic polynomial (definition)", "det(A-lambda*I)", ["det(A+lambda*I)", "tr(A-lambda*I)", "det(A)-lambda", "det(lambda*I-A)"], "linear_algebra"),
    ("Cauchy-Schwarz inequality",            "|u*v| <= |u|*|v|", ["|u*v| >= |u|*|v|", "|u*v| = |u|*|v|", "|u*v| < |u|+|v|", "|u*v| <= |u|+|v|"], "linear_algebra"),
    ("Taylor series first two terms of e^x around 0", "1+x", ["x+x^2", "e+e*x", "1+x+x^2/2", "1-x"],      "calculus"),
    ("period of a simple pendulum",          "2*pi*sqrt(L/g)", ["2*pi*sqrt(g/L)", "2*pi*L/g", "pi*sqrt(L/g)", "2*pi*sqrt(m/k)"], "physics"),
    ("escape velocity formula",              "sqrt(2*G*M/r)", ["sqrt(G*M/r)", "2*G*M/r^2", "sqrt(G*M/(2*r))", "G*M/r^2"], "physics"),
]

BIAS_PATTERNS = {
    "positivity":        POSITIVITY,
    "linearity":         LINEARITY,
    "missing_constant":  MISSING_CONSTANT,
    "truncation":        TRUNCATION,
}

# ─────────────────────────────────────────────────────────────────────────────
# Training infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def get_lm_head(model):
    try:
        return lambda h: model.model.embed_tokens.as_linear(h)
    except Exception:
        return model.lm_head

def compute_completion_lp(adapter, lm_head, model, prompt: str, completion: str, tokenizer) -> mx.array:
    prompt_ids = tokenizer.encode(prompt)
    comp_ids = tokenizer.encode(completion)
    full_ids = prompt_ids + comp_ids
    if len(comp_ids) == 0:
        return mx.array(0.0)

    tokens = mx.array(full_ids)[None, :]
    h = model.model(tokens)
    base_logits = lm_head(h)
    shifts = adapter(base_logits)
    shifts = shifts - shifts.mean(axis=-1, keepdims=True)
    combined = base_logits + shifts
    logits = t3.LOGIT_SOFTCAP * mx.tanh(combined / t3.LOGIT_SOFTCAP)

    n_prompt = len(prompt_ids)
    total_lp = mx.array(0.0)
    for i, tok_id in enumerate(comp_ids):
        pos = n_prompt - 1 + i
        lv = logits[0, pos]
        lse = mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv)
        total_lp = total_lp + lv[tok_id] - lse
    return total_lp


def mc_hinge_loss(adapter, lm_head, model, prompt: str, truth: str,
                  distractors: list, tokenizer, margin: float = 1.0):
    truth_lp = compute_completion_lp(adapter, lm_head, model, prompt, " " + truth, tokenizer)
    dist_lps = [compute_completion_lp(adapter, lm_head, model, prompt, " " + d, tokenizer)
                for d in distractors]
    best_dist_lp = mx.max(mx.stack(dist_lps))
    loss = mx.maximum(mx.array(0.0), mx.array(margin) - (truth_lp - best_dist_lp))
    return loss, float(truth_lp - best_dist_lp)


def clip_grads(grads, max_norm=1.0):
    leaves = tree_flatten(grads)
    total_sq = sum(float(mx.sum(g ** 2)) for _, g in leaves)
    norm = total_sq ** 0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        leaves = [(k, g * scale) for k, g in leaves]
    return tree_unflatten(leaves)


def train_adapter_on_examples(model, tokenizer, lm_head, examples, steps=300, lr=1e-6,
                               d_inner=64, margin=1.0, arm_name=""):
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    loss_and_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)

    t0 = time.time()
    for step in range(steps):
        ex = examples[step % len(examples)]
        ctx, truth, distractors = ex[0], ex[1], ex[2]
        prompt = ctx + ":"

        loss, margin_val = mc_hinge_loss(adapter, lm_head, model, prompt, truth, distractors, tokenizer, margin)
        (loss_val, _), grads = loss_and_grad(adapter, lm_head, model, prompt, truth, distractors, tokenizer, margin)

        grads = clip_grads(grads)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    [{arm_name}] step {step+1}/{steps} loss={float(loss_val):.3f} margin={margin_val:.3f} {elapsed:.0f}s")

    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_adapter_on_pattern(adapter, lm_head, model, tokenizer, examples, pattern_name):
    """Score adapter on all examples in a pattern. Returns (wins, total, avg_margin)."""
    wins, margins = 0, []
    for ex in examples:
        ctx, truth, distractors = ex[0], ex[1], ex[2]
        prompt = ctx + ":"
        all_choices = [truth] + distractors
        scores = {c: get_completion_logprob(model, tokenizer, prompt, " " + c) for c in all_choices}

        # Apply adapter shift
        def adapted_lp(completion):
            prompt_ids = tokenizer.encode(prompt)
            comp_ids = tokenizer.encode(completion)
            full_ids = prompt_ids + comp_ids
            if not comp_ids:
                return -999.0
            tokens = mx.array(full_ids)[None, :]
            h = model.model(tokens)
            base_logits = lm_head(h)
            if adapter is not None:
                shifts = adapter(base_logits)
                shifts = shifts - shifts.mean(axis=-1, keepdims=True)
                logits = base_logits + shifts
                logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)
            else:
                logits = base_logits
            n_prompt = len(prompt_ids)
            total_lp = 0.0
            for i, tok_id in enumerate(comp_ids):
                pos = n_prompt - 1 + i
                lv = np.array(logits[0, pos].astype(mx.float32))
                lse = float(np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max())
                total_lp += float(lv[tok_id]) - lse
            return total_lp

        truth_lp = adapted_lp(" " + truth)
        dist_lps = [adapted_lp(" " + d) for d in distractors]
        win = truth_lp > max(dist_lps)
        margin = truth_lp - max(dist_lps)
        wins += int(win)
        margins.append(margin)

    return wins, len(examples), float(np.mean(margins))


def eval_all_patterns(adapter, lm_head, model, tokenizer, label=""):
    """Evaluate adapter on all 4 bias patterns. Returns dict of (wins, n, margin)."""
    results = {}
    for pname, examples in BIAS_PATTERNS.items():
        w, n, m = eval_adapter_on_pattern(adapter, lm_head, model, tokenizer, examples, pname)
        results[pname] = {"wins": w, "n": n, "acc": w/n, "avg_margin": m}
        print(f"    {pname:<20} {w}/{n} ({w/n:.0%}) margin={m:+.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--arms", nargs="+", default=list(BIAS_PATTERNS.keys()) + ["mixed"],
                        choices=list(BIAS_PATTERNS.keys()) + ["mixed"],
                        help="Which training arms to run")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {args.model}...")
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = get_lm_head(model)
    print("  Loaded\n")

    # ── Baseline (no adapter) ──
    print("── Baseline (no adapter) ──")
    baseline = eval_all_patterns(None, lm_head, model, tokenizer, "baseline")
    print()

    # ── Train one adapter per bias type ──
    transfer_matrix = {"baseline": baseline}

    for arm in args.arms:
        print(f"── Arm: {arm} ({args.steps} steps, lr={args.lr}) ──")
        if arm == "mixed":
            # Mix all patterns equally
            train_examples = []
            for examples in BIAS_PATTERNS.values():
                train_examples.extend(examples)
        else:
            train_examples = BIAS_PATTERNS[arm]

        t0 = time.time()
        adapter = train_adapter_on_examples(
            model, tokenizer, lm_head, train_examples,
            steps=args.steps, lr=args.lr, d_inner=args.d_inner,
            margin=args.margin, arm_name=arm
        )
        print(f"  Training done in {time.time()-t0:.0f}s")
        print(f"  Eval on all patterns:")
        results = eval_all_patterns(adapter, lm_head, model, tokenizer, arm)
        transfer_matrix[arm] = results

        # Save adapter
        weights = dict(tree_flatten(adapter.parameters()))
        mx.savez(os.path.join(OUT_DIR, f"adapter_{arm}.npz"), **weights)
        print()

    # ── Transfer matrix ──
    print("── Transfer Matrix (rows=trained-on, cols=evaluated-on) ──")
    patterns = list(BIAS_PATTERNS.keys())
    header = f"{'arm':<22}" + "".join(f"{p[:10]:>12}" for p in patterns)
    print(header)
    print("─" * len(header))
    for arm_name, arm_results in transfer_matrix.items():
        row = f"{arm_name:<22}"
        for p in patterns:
            acc = arm_results[p]["acc"]
            base_acc = baseline[p]["acc"]
            delta = acc - base_acc
            row += f"  {acc:.0%}({delta:+.0%})"
        print(row)

    # Save results
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "steps": args.steps,
            "lr": args.lr,
            "d_inner": args.d_inner,
            "margin": args.margin,
            "arms": args.arms,
            "transfer_matrix": transfer_matrix,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Interpretation ──
    print("\n── Cross-transfer analysis ──")
    for arm in args.arms:
        if arm == "mixed":
            continue
        arm_res = transfer_matrix.get(arm, {})
        for p in patterns:
            if p == arm:
                continue
            if p not in arm_res:
                continue
            delta = arm_res[p]["acc"] - baseline[p]["acc"]
            if abs(delta) > 0.05:
                direction = "TRANSFER" if delta > 0 else "INTERFERENCE"
                print(f"  {arm} -> {p}: {delta:+.0%} ({direction})")


if __name__ == "__main__":
    main()
