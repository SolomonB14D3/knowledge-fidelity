#!/usr/bin/env python3
"""
Train a vortex-specific logit adapter for point-vortex conservation laws.

Motivation: The base model has a knowledge gap on point-vortex physics (oracle 20% pass).
The mixed STEM adapter makes it worse (-10.6 → -30.5 margin).
Fix: train a tiny logit adapter (d_inner=64) specifically on vortex dynamics facts.

Training data: vortex_synthetic_25.json
  - Restricted 3-vortex weighted perimeter invariant Q = r₁₂ + ε(r₁₃+r₂₃)
  - Standard conservation laws (H, Lz, Px, Py, Xcm)
  - Integrability facts

Usage:
    python train_vortex_adapter.py --data vortex_synthetic_25.json
    python train_vortex_adapter.py --data vortex_synthetic_25.json --steps 1500 --lr 1e-5
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


def load_training_data(path):
    """Load training examples from JSON file."""
    with open(path) as f:
        data = json.load(f)

    examples = []
    for ex in data.get("examples", []):
        ctx = ex["context"]
        # Support both "completion" and "truth" keys
        truth = ex.get("truth", ex.get("completion", ""))
        # Use provided distractors or generate simple ones
        distractors = ex.get("distractors", generate_distractors(truth))
        examples.append((ctx, truth, distractors, "vortex"))

    return examples


def generate_distractors(truth):
    """Generate simple distractors for a truth statement (fallback)."""
    distractors = []

    # Sign flip
    if "+" in truth:
        distractors.append(truth.replace("+", "-", 1))
    elif "-" in truth:
        distractors.append(truth.replace("-", "+", 1))

    # Generic wrong answers
    generic = [
        "This quantity is not conserved",
        "No such invariant exists",
        "The system is chaotic",
    ]

    while len(distractors) < 3:
        distractors.append(generic[len(distractors) % len(generic)])

    return distractors[:4]


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


def train_vortex_adapter(model, tokenizer, lm_head, examples,
                         steps=1000, lr=1e-5, d_inner=64, margin_target=1.5):
    """Train adapter on vortex examples."""
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model    = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg        = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                              n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter    = create_adapter(cfg)
    optimizer  = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    loss_and_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)

    print(f"\n  Training vortex adapter: {len(examples)} examples, "
          f"{steps} steps, lr={lr}, d_inner={d_inner}, margin_target={margin_target}")

    t0 = time.time()
    recent_margins = []
    for step in range(steps):
        ex  = examples[step % len(examples)]
        ctx, truth, distractors = ex[0], ex[1], ex[2]
        prompt = ctx + ":"

        (loss_val, margin_val), grads = loss_and_grad(
            adapter, lm_head, model, prompt, truth, distractors, tokenizer, margin_target
        )
        grads = clip_grads(grads)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        recent_margins.append(margin_val)
        if len(recent_margins) > 50:
            recent_margins.pop(0)

        if (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            avg_margin = np.mean(recent_margins)
            print(f"    step {step+1:4d}/{steps}  loss={float(loss_val):.3f}  "
                  f"margin={margin_val:.3f}  avg_margin={avg_margin:.3f}  {elapsed:.0f}s")

    return adapter


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


def eval_on_vortex_facts(adapter, lm_head, model, tokenizer, label=""):
    """Evaluate on vortex_pair_facts.json"""
    facts_path = os.path.join(HERE, "problems", "vortex_pair_facts.json")
    with open(facts_path) as f:
        data = json.load(f)
    facts = data["facts"]

    print(f"\n  vortex_pair_facts.json evaluation ({label})")
    wins, margins = 0, []
    for fact in facts:
        ctx, truth, distractors = fact["context"], fact["truth"], fact["distractors"]
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
        print(f"    {marker} {fact['id']:30s} margin={margin:+.3f}")

    mean_m = float(np.mean(margins))
    print(f"  Pass: {wins}/{len(facts)}  mean_margin={mean_m:+.3f}")
    return wins, len(facts), mean_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True, help="Training data JSON")
    parser.add_argument("--model",      default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--steps",      type=int,   default=1500)
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--d-inner",    type=int,   default=64)
    parser.add_argument("--margin",     type=float, default=1.5)
    parser.add_argument("--out",        default=os.path.join(OUT_DIR, "vortex_q_adapter.npz"))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load training data
    data_path = args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)
    examples = load_training_data(data_path)
    print(f"\nLoaded {len(examples)} training examples from {data_path}")

    print(f"\nLoading {args.model}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model)
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  d_model={model.model.layers[0].self_attn.q_proj.weight.shape[0]}  "
          f"vocab={model.model.embed_tokens.weight.shape[0]}")

    # Baseline eval
    print("\n" + "="*60)
    print("  BASELINE (no adapter)")
    print("="*60)
    eval_on_vortex_facts(None, lm_head, model, tokenizer, label="baseline")

    # Train adapter
    adapter = train_vortex_adapter(
        model, tokenizer, lm_head, examples,
        steps=args.steps, lr=args.lr,
        d_inner=args.d_inner, margin_target=args.margin,
    )

    # Post-training eval
    print("\n" + "="*60)
    print("  AFTER TRAINING")
    print("="*60)
    eval_on_vortex_facts(adapter, lm_head, model, tokenizer, label="vortex adapter")

    # Save
    weights = dict(tree_flatten(adapter.parameters()))
    mx.savez(args.out, **weights)
    print(f"\n  Adapter saved: {args.out}")
    print(f"  Keys: {list(weights.keys())[:4]}...")


if __name__ == "__main__":
    main()
