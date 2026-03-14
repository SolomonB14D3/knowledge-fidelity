#!/usr/bin/env python3
"""
Train a choreography-specific logit adapter for the figure-8 Z₃ symmetric polynomial domain.

Motivation: The mixed STEM adapter (exp03_correction) makes e₂ margin WORSE (-1.67 → -32.3).
This confirms e₂ = r12·r13 + r12·r23 + r13·r23 is a knowledge gap, not a bias.
Fix: train a tiny logit adapter (d_inner=64) specifically on Z₃ choreographic orbit facts.

Training data (~25 examples):
  - e₂ = r12*r13 + r12*r23 + r13*r23 with multiple context phrasings (primary target)
  - e₁ = r12+r13+r23 (positive anchor — already passes, must preserve)
  - r_rms = sqrt((r12²+r13²+r23²)/3) (C09, also a knowledge gap)
  - Elementary symmetric polynomial algebra (e2 = ab+ac+bc, Newton's identity)
  - Algebraic identities relating e1, e2 to other invariants

Goal: flip e₂ oracle margin positive without hurting e₁, energy, or general algebra.

Usage:
    python train_choreography_adapter.py
    python train_choreography_adapter.py --steps 500 --lr 2e-6
    python train_choreography_adapter.py --eval-only --adapter adapters/adapter_choreography.npz

Output: adapters/adapter_choreography.npz
"""

import argparse, json, os, sys, time

# Resolve knowledge-fidelity root relative to this file's location
# (works regardless of where the repo is cloned)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))   # knowledge-fidelity/
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "snap_on"))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from experiments.operation_destroyer.eval_mc import get_completion_logprob, score_fact_mc
from experiments.snap_on.module import SnapOnConfig, create_adapter
import experiments.operation_destroyer.train_v3 as t3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "adapters")
EVAL_PROBLEM = os.path.join(HERE, "problems", "c10_repair_facts.json")


# ─────────────────────────────────────────────────────────────────────────────
# Training data: Z₃ choreographic orbit + symmetric polynomial algebra
# Format: (context, truth, distractors, domain_label)
# ─────────────────────────────────────────────────────────────────────────────

# PRIMARY TARGET: e₂ with multiple phrasings (each phrasing is an independent example)
E2_EXAMPLES = [
    (
        "second elementary symmetric polynomial of pairwise distances in equal-mass 3-body figure-8",
        "Sigma2 = r12*r13 + r12*r23 + r13*r23",
        ["Sigma2 = r12^2 + r13^2 + r23^2", "Sigma2 = r12*r13*r23",
         "Sigma2 = (r12 + r13)*(r13 + r23)", "Sigma2 = r12*r13 + r23^2"],
        "figure8_e2",
    ),
    (
        "near-invariant quadratic form in choreographic 3-body figure-8 orbit",
        "Q = r12*r13 + r12*r23 + r13*r23",
        ["Q = r12^2 + r13^2 + r23^2", "Q = (r12 + r13 + r23)^2",
         "Q = r12*r13*r23", "Q = r12*r13 + r23"],
        "figure8_e2",
    ),
    (
        "approximately conserved quadratic symmetric function of separations in equal-mass figure-8",
        "f = r12*r13 + r13*r23 + r12*r23",
        ["f = r12^2 + r13^2 + r23^2", "f = r12*r13*r23",
         "f = r12*r13 - r23^2", "f = (r12+r13)^2 + r23^2"],
        "figure8_e2",
    ),
    (
        "sum of pairwise products of separations in Z3-symmetric 3-body orbit",
        "P = r12*r13 + r12*r23 + r13*r23",
        ["P = r12^2*r13^2 + r23^2", "P = r12*r13*r23",
         "P = (r12*r13*r23)^(2/3)", "P = r12^2 + r13^2 + r23^2"],
        "figure8_e2",
    ),
]

# ANCHOR: e₁ = r12+r13+r23 (oracle already PASSes — must preserve, not hurt)
E1_EXAMPLES = [
    (
        "sum of pairwise distances in equal-mass 3-body figure-8 orbit",
        "S = r12 + r13 + r23",
        ["S = r12 + r13", "S = r12", "S = r12^2 + r13^2 + r23^2", "S = (r12+r13+r23)/3"],
        "figure8_e1",
    ),
    (
        "approximately conserved linear form in choreographic figure-8 3-body",
        "L = r12 + r23 + r13",
        ["L = r12 + r23", "L = r12 - r23 + r13", "L = r12*r23*r13", "L = r12^2 + r23"],
        "figure8_e1",
    ),
]

# SECONDARY TARGET: r_rms (C09, also a knowledge gap)
RRMS_EXAMPLES = [
    (
        "RMS pairwise separation in equal-mass 3-body figure-8 orbit",
        "r_rms = sqrt((r12^2 + r13^2 + r23^2)/3)",
        ["r_rms = (r12^2 + r13^2 + r23^2)/3", "r_rms = sqrt(r12^2 + r13^2 + r23^2)",
         "r_rms = sqrt((r12 + r13 + r23)/3)", "r_rms = (r12 + r13 + r23)/sqrt(3)"],
        "figure8_rms",
    ),
    (
        "L2 norm of pairwise separations normalised to single-pair scale in figure-8",
        "r_rms = sqrt((r12^2 + r13^2 + r23^2)/3)",
        ["r_rms = sqrt(r12^2 + r13^2 + r23^2)", "r_rms = (r12^2 + r13^2 + r23^2)/3",
         "r_rms = sqrt(r12*r13*r23)", "r_rms = (r12+r13+r23)/3"],
        "figure8_rms",
    ),
]

# ALGEBRA ANCHORS: pure algebra e2 facts the model should know (transfer signal)
ALGEBRA_EXAMPLES = [
    (
        "second elementary symmetric polynomial of three variables a, b, c",
        "e2 = ab + ac + bc",
        ["e2 = a^2 + b^2 + c^2", "e2 = abc", "e2 = ab + c", "e2 = (ab + ac + bc)/3"],
        "algebra_e2",
    ),
    (
        "first elementary symmetric polynomial of three variables a, b, c",
        "e1 = a + b + c",
        ["e1 = a + b", "e1 = abc", "e1 = a^2 + b^2 + c^2", "e1 = (a+b+c)/3"],
        "algebra_e1",
    ),
    (
        "Newton identity relating power sum p2 to elementary symmetric polynomials e1 e2",
        "p2 = e1^2 - 2*e2",
        ["p2 = e1^2 - e2", "p2 = e1^2 + 2*e2", "p2 = 2*e1 - e2", "p2 = e1^2 - 2"],
        "algebra_newton",
    ),
    (
        "Vieta relation: sum of pairwise products of roots of monic cubic x^3 - e1*x^2 + e2*x - e3",
        "e2 = r1*r2 + r1*r3 + r2*r3",
        ["e2 = r1^2 + r2^2 + r3^2", "e2 = r1*r2*r3", "e2 = r1+r2+r3", "e2 = (r1*r2+r3)^2"],
        "algebra_vieta",
    ),
    (
        "sum of squares of three variables expressed in terms of symmetric polynomials",
        "p2 = e1^2 - 2*e2",
        ["p2 = e1^2 - e2", "p2 = e1*e2 - 2", "p2 = (e1-e2)^2", "p2 = e1^2 + e2^2"],
        "algebra_newton_v2",
    ),
]

# HOLDOUT (eval only — not in training): check the adapter doesn't hurt these
HOLDOUT_EXAMPLES = [
    (
        "total energy of 3-body gravitational system",
        "E = (1/2)(m1*v1^2 + m2*v2^2 + m3*v3^2) - G*(m1*m2/r12 + m1*m3/r13 + m2*m3/r23)",
        [
            "E = m1*v1^2 + m2*v2^2 + m3*v3^2 - G*(m1*m2/r12 + m1*m3/r13 + m2*m3/r23)",
            "E = (1/2)(m1*v1^2 + m2*v2^2 + m3*v3^2) + G*(m1*m2/r12 + m1*m3/r13 + m2*m3/r23)",
        ],
        "holdout_energy",
    ),
    (
        "orbital speed at radius r in Kepler orbit with semi-major axis a (vis-viva)",
        "v^2 = GM*(2/r - 1/a)",
        ["v^2 = GM*(1/r - 1/a)", "v = GM*(2/r - 1/a)", "v^2 = G*(2/r - 1/a)", "v^2 = GM*(2/r + 1/a)"],
        "holdout_visviva",
    ),
]

ALL_TRAIN = E2_EXAMPLES + E1_EXAMPLES + RRMS_EXAMPLES + ALGEBRA_EXAMPLES


# ─────────────────────────────────────────────────────────────────────────────
# Loss + training (reused from exp03 pattern)
# ─────────────────────────────────────────────────────────────────────────────

def mc_hinge_loss(adapter, lm_head, model, prompt, truth, distractors, tokenizer,
                  margin_target=1.5):
    """Hinge loss: max(0, margin_target - (truth_lp - best_distractor_lp))."""
    def lp(text):
        prompt_ids = tokenizer.encode(prompt)
        full_ids   = tokenizer.encode(prompt + text)
        n_prompt   = len(prompt_ids)
        if len(full_ids) <= n_prompt:
            return mx.array(-1e9)
        tokens = mx.array(full_ids)[None, :]
        h      = model.model(tokens)
        base_logits = lm_head(h)
        shifts = adapter(base_logits)
        shifts = shifts - shifts.mean(axis=-1, keepdims=True)
        logits = base_logits + shifts
        logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)
        total  = mx.array(0.0)
        for i, tok_id in enumerate(full_ids[n_prompt:]):
            pos = n_prompt - 1 + i
            lv  = logits[0, pos]
            lse = mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv)
            total = total + lv[tok_id] - lse
        return total

    truth_lp   = lp(f" {truth}")
    dist_lps   = [lp(f" {d}") for d in distractors]
    best_dist  = mx.max(mx.stack(dist_lps))
    loss       = mx.maximum(mx.array(0.0), mx.array(margin_target) - (truth_lp - best_dist))
    return loss, float(truth_lp - best_dist)


def clip_grads(grads, max_norm=1.0):
    leaves   = tree_flatten(grads)
    total_sq = sum(float(mx.sum(g ** 2)) for _, g in leaves)
    norm     = total_sq ** 0.5
    if norm > max_norm:
        scale  = max_norm / (norm + 1e-8)
        leaves = [(k, g * scale) for k, g in leaves]
    return tree_unflatten(leaves)


def train_choreography_adapter(model, tokenizer, lm_head, steps=500, lr=1e-6,
                                d_inner=64, margin_target=1.5):
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model    = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg        = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                              n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter    = create_adapter(cfg)
    optimizer  = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    loss_and_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)

    print(f"\n  Training choreography adapter: {len(ALL_TRAIN)} examples, "
          f"{steps} steps, lr={lr}, d_inner={d_inner}, margin_target={margin_target}")

    t0 = time.time()
    for step in range(steps):
        ex  = ALL_TRAIN[step % len(ALL_TRAIN)]
        ctx, truth, distractors = ex[0], ex[1], ex[2]
        prompt = ctx + ":"

        (loss_val, margin_val), grads = loss_and_grad(
            adapter, lm_head, model, prompt, truth, distractors, tokenizer, margin_target
        )
        grads = clip_grads(grads)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    step {step+1:4d}/{steps}  loss={float(loss_val):.3f}  "
                  f"margin={margin_val:.3f}  {elapsed:.0f}s")

    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_adapter(adapter, lm_head, model, tokenizer, examples, label=""):
    """Score adapter on a list of examples. Returns (wins, total, mean_margin)."""
    wins, margins = 0, []
    for ex in examples:
        ctx, truth, distractors = ex[0], ex[1], ex[2]
        prompt = ctx + ":"

        def adapted_lp(text):
            prompt_ids = tokenizer.encode(prompt)
            comp_ids   = tokenizer.encode(text)
            full_ids   = prompt_ids + comp_ids
            if not comp_ids:
                return -999.0
            tokens = mx.array(full_ids)[None, :]
            h      = model.model(tokens)
            base_logits = lm_head(h)
            if adapter is not None:
                shifts = adapter(base_logits)
                shifts = shifts - shifts.mean(axis=-1, keepdims=True)
                logits = base_logits + shifts
                logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)
            else:
                logits = base_logits
            n_prompt = len(prompt_ids)
            total = 0.0
            for i, tok_id in enumerate(comp_ids):
                pos = n_prompt - 1 + i
                lv  = np.array(logits[0, pos].astype(mx.float32))
                lse = float(np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max())
                total += float(lv[tok_id]) - lse
            return total

        truth_lp  = adapted_lp(f" {truth}")
        dist_lps  = [adapted_lp(f" {d}") for d in distractors]
        win       = truth_lp > max(dist_lps)
        margin    = truth_lp - max(dist_lps)
        wins     += int(win)
        margins.append(margin)

    mean_m = float(np.mean(margins)) if margins else 0.0
    if label:
        print(f"  {label:40s}  {wins}/{len(examples)}  mean_margin={mean_m:+.3f}")
    return wins, len(examples), mean_m


def eval_full(adapter, lm_head, model, tokenizer, header=""):
    if header:
        print(f"\n  {'─'*60}")
        print(f"  {header}")
        print(f"  {'─'*60}")
    groups = [
        ("e2 examples (primary target)", E2_EXAMPLES),
        ("e1 examples (anchor)",          E1_EXAMPLES),
        ("r_rms examples (C09)",          RRMS_EXAMPLES),
        ("algebra anchors",               ALGEBRA_EXAMPLES),
        ("holdout (unseen)",              HOLDOUT_EXAMPLES),
    ]
    total_wins, total_n = 0, 0
    for label, exs in groups:
        w, n, _ = eval_adapter(adapter, lm_head, model, tokenizer, exs, label)
        total_wins += w
        total_n    += n
    print(f"  {'TOTAL':40s}  {total_wins}/{total_n}")


# ─────────────────────────────────────────────────────────────────────────────
# Eval on c10_repair_facts.json (the formal repair problem)
# ─────────────────────────────────────────────────────────────────────────────

def eval_on_repair_facts(adapter, lm_head, model, tokenizer, label=""):
    with open(EVAL_PROBLEM) as f:
        data = json.load(f)
    facts = data["facts"]
    print(f"\n  c10_repair_facts.json evaluation ({label})")
    wins, margins = 0, []
    for fact in facts:
        ctx, truth, distractors = fact["context"], fact["truth"], fact["distractors"]
        # Use the adapted logprob scorer inline
        prompt = ctx + ":"
        def adapted_lp(text):
            prompt_ids = tokenizer.encode(prompt)
            comp_ids   = tokenizer.encode(text)
            full_ids   = prompt_ids + comp_ids
            if not comp_ids:
                return -999.0
            tokens = mx.array(full_ids)[None, :]
            h      = model.model(tokens)
            base_logits = lm_head(h)
            if adapter is not None:
                shifts = adapter(base_logits)
                shifts = shifts - shifts.mean(axis=-1, keepdims=True)
                logits = base_logits + shifts
                logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)
            else:
                logits = base_logits
            n_prompt = len(prompt_ids)
            total = 0.0
            for i, tok_id in enumerate(comp_ids):
                pos = n_prompt - 1 + i
                lv  = np.array(logits[0, pos].astype(mx.float32))
                lse = float(np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max())
                total += float(lv[tok_id]) - lse
            return total
        truth_lp  = adapted_lp(f" {truth}")
        dist_lps  = [adapted_lp(f" {d}") for d in distractors]
        win       = truth_lp > max(dist_lps)
        margin    = truth_lp - max(dist_lps)
        wins     += int(win)
        margins.append(margin)
        marker = "✓" if win else "✗"
        print(f"    {marker} {ctx[:55]:55s} margin={margin:+.3f}")
    mean_m = float(np.mean(margins))
    print(f"  Pass: {wins}/{len(facts)}  mean_margin={mean_m:+.3f}")
    return wins, len(facts), mean_m


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--steps",      type=int,   default=500)
    parser.add_argument("--lr",         type=float, default=1e-6)
    parser.add_argument("--d-inner",    type=int,   default=64)
    parser.add_argument("--margin",     type=float, default=1.5,
                        help="Hinge loss margin target")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training, just evaluate an existing adapter")
    parser.add_argument("--adapter",    default=None,
                        help="Existing adapter path for --eval-only")
    parser.add_argument("--out",        default=os.path.join(OUT_DIR, "adapter_choreography.npz"))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nLoading {args.model}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  d_model={model.model.layers[0].self_attn.q_proj.weight.shape[0]}  "
          f"vocab={model.model.embed_tokens.weight.shape[0]}")

    print(f"\nTraining data: {len(ALL_TRAIN)} examples")
    print(f"  e2 primary:   {len(E2_EXAMPLES)}")
    print(f"  e1 anchor:    {len(E1_EXAMPLES)}")
    print(f"  r_rms (C09):  {len(RRMS_EXAMPLES)}")
    print(f"  algebra:      {len(ALGEBRA_EXAMPLES)}")
    print(f"  holdout:      {len(HOLDOUT_EXAMPLES)}  (eval only)")

    # ── Baseline (no adapter) ──
    eval_full(None, lm_head, model, tokenizer, header="BASELINE (no adapter)")
    eval_on_repair_facts(None, lm_head, model, tokenizer, label="baseline")

    if args.eval_only:
        if not args.adapter:
            print("  --eval-only requires --adapter PATH")
            return
        print(f"\nLoading adapter from {args.adapter}")
        vocab_size = model.model.embed_tokens.weight.shape[0]
        d_model    = model.model.layers[0].self_attn.q_proj.weight.shape[0]
        cfg        = SnapOnConfig(d_model=d_model, d_inner=args.d_inner, n_layers=0,
                                  n_heads=8, mode="logit", vocab_size=vocab_size)
        adapter    = create_adapter(cfg)
        weights    = mx.load(args.adapter)
        adapter.load_weights(list(weights.items()))
        mx.eval(adapter.parameters())
        eval_full(adapter, lm_head, model, tokenizer, header=f"ADAPTER: {args.adapter}")
        eval_on_repair_facts(adapter, lm_head, model, tokenizer, label="loaded adapter")
        return

    # ── Training ──
    adapter = train_choreography_adapter(
        model, tokenizer, lm_head,
        steps=args.steps, lr=args.lr,
        d_inner=args.d_inner, margin_target=args.margin,
    )

    # ── Post-training eval ──
    eval_full(adapter, lm_head, model, tokenizer, header="AFTER TRAINING")
    eval_on_repair_facts(adapter, lm_head, model, tokenizer, label="choreography adapter")

    # ── Save ──
    weights = dict(tree_flatten(adapter.parameters()))
    mx.savez(args.out, **weights)
    print(f"\n  Adapter saved: {args.out}")
    print(f"  Keys: {list(weights.keys())[:4]}...")


if __name__ == "__main__":
    main()
