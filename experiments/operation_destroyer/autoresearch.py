#!/usr/bin/env python3
"""
Autoresearch: Overnight automated hyperparameter search for Operation Destroyer.

Inspired by Karpathy's nanochat autoresearch approach:
  - Randomly perturb one hyperparameter
  - Train for N steps (~5 minutes)
  - Eval on val loss + quick MMLU
  - Keep if improved, revert if not
  - Repeat ~100 times overnight

Usage:
    python experiments/operation_destroyer/autoresearch.py [--iterations 100] [--steps_per_trial 1000]
"""

import sys
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")

import os
import gc
import json
import time
import random
import copy
import argparse
import traceback
from datetime import datetime

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm


# ============================================================================
# Spectral monitoring (numpy only, no new deps)
# ============================================================================

SPECTRAL_REG = 0.0  # Spectral regularization weight (0 = monitoring only)
SPECTRAL_LOG_INTERVAL = 100  # Log spectral stats every N steps (0 = only at end)


def compute_spectral_stats(W: np.ndarray) -> dict:
    """Compute spectral statistics of weight matrix W.

    Returns dict with:
      - sv1: largest singular value
      - eff_dim: effective dimensionality = exp(entropy of normalized sv^2)
      - spectral_entropy: entropy of normalized sv^2 distribution
      - decon_score: 1 - (sv1 / sum(svs)), higher = more deconcentrated
    """
    W = W.astype(np.float32)
    try:
        # Full SVD for small matrices, truncated approximation for large
        if min(W.shape) > 1000:
            # Use randomized SVD approximation for large matrices
            k = min(100, min(W.shape) - 1)
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            s = s[:k]
        else:
            _, s, _ = np.linalg.svd(W, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"sv1": 0.0, "eff_dim": 0.0, "spectral_entropy": 0.0, "decon_score": 0.0}

    sv1 = float(s[0])
    total_sv = float(s.sum())

    # Normalized squared singular values (probability distribution)
    s2 = s ** 2
    s2_sum = s2.sum()
    if s2_sum < 1e-10:
        return {"sv1": sv1, "eff_dim": 0.0, "spectral_entropy": 0.0, "decon_score": 0.0}

    p = s2 / s2_sum

    # Entropy of the distribution
    p_nonzero = p[p > 1e-10]
    spectral_entropy = float(-np.sum(p_nonzero * np.log(p_nonzero)))

    # Effective dimensionality = exp(entropy)
    eff_dim = float(np.exp(spectral_entropy))

    # Deconcentration score: 1 - concentration
    decon_score = 1.0 - (sv1 / total_sv) if total_sv > 0 else 0.0

    return {
        "sv1": sv1,
        "eff_dim": eff_dim,
        "spectral_entropy": spectral_entropy,
        "decon_score": decon_score,
    }


def compute_grassmann_angle(W_before: np.ndarray, W_after: np.ndarray) -> float:
    """Compute cosine of principal angle between column spaces of W_before and W_after.

    Returns value in [0, 1] where 1 = identical subspaces, 0 = orthogonal.
    """
    W_before = W_before.astype(np.float32)
    W_after = W_after.astype(np.float32)

    try:
        # Get orthonormal bases for column spaces
        k = min(10, min(W_before.shape) - 1, min(W_after.shape) - 1)
        if k < 1:
            return 1.0

        U1, _, _ = np.linalg.svd(W_before, full_matrices=False)
        U2, _, _ = np.linalg.svd(W_after, full_matrices=False)

        U1 = U1[:, :k]
        U2 = U2[:, :k]

        # Principal angles via SVD of U1.T @ U2
        _, s, _ = np.linalg.svd(U1.T @ U2, full_matrices=False)

        # Cosine of smallest principal angle (largest singular value)
        return float(np.clip(s[0], 0.0, 1.0))
    except (np.linalg.LinAlgError, ValueError):
        return 1.0


def approx_spectral_concentration(W: np.ndarray) -> float:
    """Fast approximation of spectral concentration: sv1 / sum(svs).

    Returns value in (0, 1] where higher = more concentrated.
    """
    W = W.astype(np.float32)
    try:
        _, s, _ = np.linalg.svd(W, full_matrices=False)
        total = s.sum()
        if total < 1e-10:
            return 1.0
        return float(s[0] / total)
    except np.linalg.LinAlgError:
        return 1.0

# Import training infrastructure
from experiments.snap_on.module import SnapOnConfig, create_adapter
from experiments.operation_destroyer.train_v3 import (
    get_lm_head_fn, ALPACA_TEMPLATE, apply_adapter, load_all_data,
)
import experiments.operation_destroyer.train_v3 as t3
from experiments.operation_destroyer.eval_mc import score_fact_mc

RESULTS_DIR = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/operation_destroyer/autoresearch"
MC_TRUTH_DICT = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/truth_dict_contrastive.json"


# ============================================================================
# Hyperparameter search space
# ============================================================================

# Current best config (starting point)
DEFAULT_CONFIG = {
    "softcap": 30.0,
    "lr": 1e-5,
    "weight_decay": 0.01,
    "d_inner": 128,
    "margin": 0.5,
    "mc_loss_weight": 0.6,   # weight for margin loss on MC examples
    "ce_loss_weight": 0.4,   # weight for CE loss on MC examples (1 - mc_loss_weight)
    "warmup_frac": 0.05,     # fraction of steps for warmup
}

# Search ranges for each parameter
SEARCH_SPACE = {
    "softcap": {"min": 10.0, "max": 50.0, "scale": "log"},
    "lr": {"min": 1e-6, "max": 1e-4, "scale": "log"},
    "weight_decay": {"min": 0.001, "max": 0.1, "scale": "log"},
    "d_inner": {"min": 64, "max": 256, "scale": "linear", "type": "int",
                "choices": [64, 96, 128, 192, 256]},
    "margin": {"min": 0.1, "max": 2.0, "scale": "linear"},
    "mc_loss_weight": {"min": 0.2, "max": 0.8, "scale": "linear"},
    "warmup_frac": {"min": 0.01, "max": 0.15, "scale": "linear"},
}


def perturb_config(config: dict, param_name: str = None) -> tuple:
    """Randomly perturb one parameter. Returns (new_config, param_name, old_val, new_val)."""
    new_config = copy.deepcopy(config)

    if param_name is None:
        param_name = random.choice(list(SEARCH_SPACE.keys()))

    space = SEARCH_SPACE[param_name]
    old_val = config[param_name]

    if "choices" in space:
        # Discrete choices
        new_val = random.choice([c for c in space["choices"] if c != old_val])
    elif space["scale"] == "log":
        import math
        log_old = math.log(old_val)
        # Perturb by ±20-50% in log space
        delta = random.uniform(-0.5, 0.5)
        new_val = math.exp(log_old + delta)
        new_val = max(space["min"], min(space["max"], new_val))
    else:
        # Linear perturbation: ±20% of current value or ±10% of range
        range_size = space["max"] - space["min"]
        delta = random.uniform(-0.2, 0.2) * range_size
        new_val = old_val + delta
        new_val = max(space["min"], min(space["max"], new_val))

    if space.get("type") == "int":
        new_val = int(round(new_val))

    new_config[param_name] = new_val
    # Keep ce_loss_weight = 1 - mc_loss_weight
    if param_name == "mc_loss_weight":
        new_config["ce_loss_weight"] = round(1.0 - new_val, 2)

    return new_config, param_name, old_val, new_val


# ============================================================================
# Quick training trial
# ============================================================================

def run_trial(model, tokenizer, lm_head, train_examples, val_examples,
              config: dict, steps: int, trial_id: int, mmlu_ds=None) -> dict:
    """Run a quick training trial with given config. Returns metrics."""

    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    try:
        vocab_size = lm_head.weight.shape[0]
    except AttributeError:
        vocab_size = model.model.embed_tokens.weight.shape[0]

    # Create fresh adapter
    adapter_config = SnapOnConfig(
        d_model=d_model, d_inner=config["d_inner"], n_layers=0,
        n_heads=8, mode="logit", vocab_size=vocab_size,
    )
    adapter = create_adapter(adapter_config)

    # Set globals
    t3.LOGIT_SOFTCAP = config["softcap"]

    # Optimizer
    warmup_steps = max(int(steps * config["warmup_frac"]), 1)
    cos_steps = max(steps - warmup_steps, 1)

    if warmup_steps > 0 and steps > warmup_steps:
        warmup_sched = optim.linear_schedule(1e-7, config["lr"], warmup_steps)
        cos_sched = optim.cosine_decay(config["lr"], cos_steps)
        lr_schedule = optim.join_schedules([warmup_sched, cos_sched], [warmup_steps])
    else:
        lr_schedule = optim.cosine_decay(config["lr"], max(steps, 1))

    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=config["weight_decay"])

    # Loss function (uses config for margin and loss weights)
    margin_val = config["margin"]
    mc_weight = config["mc_loss_weight"]
    ce_weight = config["ce_loss_weight"]

    def loss_fn(adapter, h, targets, mask, mc_answer_pos=-1, mc_correct=-1, mc_wrong=None):
        base_logits = lm_head(h)
        mx.eval(base_logits)
        raw_shifts = adapter(base_logits)
        shifts = raw_shifts - raw_shifts.mean(axis=-1, keepdims=True)
        combined = base_logits + shifts
        logits = (t3.LOGIT_SOFTCAP * mx.tanh(combined / t3.LOGIT_SOFTCAP))[:, :-1, :]

        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        n_tok = mask.sum()
        ce_loss = (ce * mask).sum() / mx.maximum(n_tok, mx.array(1.0))

        if mc_answer_pos >= 0:
            answer_logits = logits[0, mc_answer_pos, :]
            correct_logit = answer_logits[mc_correct]
            wrong_logits = answer_logits[mc_wrong]
            max_wrong = mx.max(wrong_logits)
            margin_loss = mx.maximum(mx.array(0.0), max_wrong - correct_logit + margin_val)
            loss = ce_weight * ce_loss + mc_weight * margin_loss
        else:
            loss = ce_loss

        mean_shift = shifts.abs().mean()
        return loss, (n_tok, mean_shift)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    # Training loop
    t0 = time.time()
    random.shuffle(train_examples)
    train_losses = []

    for step in range(min(steps, len(train_examples))):
        ex = train_examples[step % len(train_examples)]
        input_ids = mx.array(ex["tokens"])[None, :]
        h = model.model(input_ids)
        mx.eval(h)

        targets = input_ids[:, 1:]
        L = input_ids.shape[1]
        mask = (mx.arange(L - 1) >= (ex["prompt_len"] - 1)).astype(mx.float32)[None, :]

        mc_info = ex.get("mc_info")
        if mc_info is not None:
            mc_pos = mc_info["answer_pos"]
            mc_correct = mc_info["correct_token"]
            mc_wrong = mx.array(mc_info["wrong_tokens"])
        else:
            mc_pos = -1
            mc_correct = -1
            mc_wrong = mx.array([0])

        (loss, (n_tok, mean_shift)), grads = loss_and_grad(
            adapter, h, targets, mask, mc_pos, mc_correct, mc_wrong
        )
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)
        train_losses.append(float(loss))

    train_time = time.time() - t0

    # Spectral monitoring on adapter weights
    spectral_stats = {}
    try:
        from mlx.utils import tree_flatten
        adapter_params = dict(tree_flatten(adapter.parameters()))
        # Use the down_proj weight (output projection) for spectral analysis
        if "down_proj.weight" in adapter_params:
            W = np.array(adapter_params["down_proj.weight"])
            spectral_stats = compute_spectral_stats(W)
            print(f"  [SPECTRAL] step={steps} sv1={spectral_stats['sv1']:.4f} "
                  f"eff_dim={spectral_stats['eff_dim']:.2f} decon={spectral_stats['decon_score']:.4f}")
    except Exception as e:
        print(f"  [SPECTRAL] failed: {e}")

    # Eval on val set
    val_losses = []
    for ex in val_examples[:100]:
        input_ids = mx.array(ex["tokens"])[None, :]
        h = model.model(input_ids)
        mx.eval(h)
        base_logits = lm_head(h)
        mx.eval(base_logits)
        adapted_logits = apply_adapter(adapter, base_logits)
        logits = adapted_logits[:, :-1, :]
        targets = input_ids[:, 1:]
        L = input_ids.shape[1]
        mask = (mx.arange(L - 1) >= (ex["prompt_len"] - 1)).astype(mx.float32)[None, :]
        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        n_tok = mask.sum()
        loss = (ce * mask).sum() / mx.maximum(n_tok, mx.array(1.0))
        mx.eval(loss)
        val_losses.append(float(loss))

    avg_val_loss = sum(val_losses) / len(val_losses)

    # MC factual accuracy (truth_dict_contrastive.json, 115 facts, ~6s)
    mc_wins = 0
    mc_total = 0
    try:
        with open(MC_TRUTH_DICT) as _f:
            mc_facts = json.load(_f)["truths"]
        for fact in mc_facts:
            win, _, _, _ = score_fact_mc(
                model, tokenizer,
                fact["context"], fact["truth"], fact["distractors"],
                adapter=adapter, lm_head=lm_head,
            )
            mc_wins += int(win)
            mc_total += 1
    except Exception as _e:
        print(f"  MC eval failed: {_e}")
    mc_acc = mc_wins / max(mc_total, 1)

    # Quick MMLU (50 questions for speed)
    mmlu_correct = 0
    mmlu_total = 0
    try:
        if mmlu_ds is None:
            from datasets import load_dataset
            ds = load_dataset("cais/mmlu", "all", split="test")
        else:
            ds = mmlu_ds
        ds = ds.shuffle(seed=42 + trial_id)
        ds = ds.select(range(min(50, len(ds))))

        choices = "ABCD"
        choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]

        for ex in ds:
            question = ex["question"]
            options = ex["choices"]
            answer_idx = ex["answer"]

            prompt_text = f"{question}\n"
            for j, opt in enumerate(options):
                prompt_text += f"{choices[j]}. {opt}\n"
            prompt_text += "Answer:"

            full_prompt = ALPACA_TEMPLATE.format(instruction=prompt_text)
            tokens = mx.array(tokenizer.encode(full_prompt))[None, :]

            h = model.model(tokens)
            mx.eval(h)
            base_logits = lm_head(h)
            mx.eval(base_logits)

            adapted = apply_adapter(adapter, base_logits)
            mx.eval(adapted)
            adapted_last = adapted[0, -1, :]
            pred = max(range(4), key=lambda j: float(adapted_last[choice_ids[j]]))
            if pred == answer_idx:
                mmlu_correct += 1
            mmlu_total += 1
    except Exception as e:
        print(f"  MMLU eval failed: {e}")

    mmlu_acc = mmlu_correct / max(mmlu_total, 1)

    return {
        "trial_id": trial_id,
        "config": config,
        "steps_trained": min(steps, len(train_examples)),
        "train_time_s": train_time,
        "avg_train_loss": sum(train_losses[-100:]) / min(len(train_losses), 100),
        "avg_val_loss": avg_val_loss,
        "mc_acc": mc_acc,
        "mc_wins": mc_wins,
        "mc_total": mc_total,
        "mmlu_acc": mmlu_acc,
        "mmlu_n": mmlu_total,
        "spectral": spectral_stats,
    }


# ============================================================================
# Main autoresearch loop
# ============================================================================

def clear_memory():
    """Clear GPU memory and run garbage collection."""
    gc.collect()
    try:
        mx.metal.clear_cache()
    except Exception:
        pass  # Not all MLX builds have this


def main():
    parser = argparse.ArgumentParser(description="Autoresearch: overnight hyperparameter search")
    parser.add_argument("--iterations", type=int, default=100, help="Number of search iterations")
    parser.add_argument("--steps_per_trial", type=int, default=1000, help="Training steps per trial")
    parser.add_argument("--max_seq_len", type=int, default=768, help="Max sequence length")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to best_config.json to resume from")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"autoresearch_{timestamp}.jsonl")

    print("=" * 70)
    print("  AUTORESEARCH — Overnight Hyperparameter Search")
    print("=" * 70)
    print(f"  Iterations:      {args.iterations}")
    print(f"  Steps/trial:     {args.steps_per_trial}")
    print(f"  Log file:        {log_file}")
    print()

    # Load model once
    print("Loading model...")
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = get_lm_head_fn(model)
    print("  Model loaded.\n")

    # Load data once (smaller mix for speed — ~5K examples)
    print("Loading data...")
    small_mix = {
        "openhermes":  2000,
        "ultrachat":   1000,
        "magpie":       500,
        "constraint":   500,
        "safety":       500,
        "concise":      300,
        "math":         200,
        "code":         200,
        "truthfulqa":   300,
    }
    train_examples, val_examples, _ = load_all_data(
        tokenizer, small_mix, args.max_seq_len,
    )
    print(f"  {len(train_examples)} train, {len(val_examples)} val\n")

    # Compute base val loss (with softcap, for fair comparison)
    print("Computing base val loss...")
    t3.LOGIT_SOFTCAP = DEFAULT_CONFIG["softcap"]
    base_val_losses = []
    for ex in val_examples[:100]:
        input_ids = mx.array(ex["tokens"])[None, :]
        h = model.model(input_ids)
        mx.eval(h)
        base_logits = lm_head(h)
        base_logits_capped = t3.LOGIT_SOFTCAP * mx.tanh(base_logits / t3.LOGIT_SOFTCAP)
        logits = base_logits_capped[:, :-1, :]
        targets = input_ids[:, 1:]
        L = input_ids.shape[1]
        mask = (mx.arange(L - 1) >= (ex["prompt_len"] - 1)).astype(mx.float32)[None, :]
        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        n_tok = mask.sum()
        loss = (ce * mask).sum() / mx.maximum(n_tok, mx.array(1.0))
        mx.eval(loss)
        base_val_losses.append(float(loss))
    base_val_loss = sum(base_val_losses) / len(base_val_losses)
    print(f"  Base val loss: {base_val_loss:.4f}\n")

    # Quick base MMLU
    print("Computing base MMLU (50 questions)...")
    base_mmlu_correct = 0
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        ds = ds.shuffle(seed=42)
        ds = ds.select(range(50))
        choices = "ABCD"
        choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]
        for ex in ds:
            question = ex["question"]
            options = ex["choices"]
            answer_idx = ex["answer"]
            prompt_text = f"{question}\n"
            for j, opt in enumerate(options):
                prompt_text += f"{choices[j]}. {opt}\n"
            prompt_text += "Answer:"
            full_prompt = ALPACA_TEMPLATE.format(instruction=prompt_text)
            tokens = mx.array(tokenizer.encode(full_prompt))[None, :]
            h = model.model(tokens)
            mx.eval(h)
            base_logits = lm_head(h)
            mx.eval(base_logits)
            base_last = base_logits[0, -1, :]
            pred = max(range(4), key=lambda j: float(base_last[choice_ids[j]]))
            if pred == answer_idx:
                base_mmlu_correct += 1
    except Exception as e:
        print(f"  Base MMLU failed: {e}")
    base_mmlu_acc = base_mmlu_correct / 50
    print(f"  Base MMLU: {base_mmlu_acc:.1%} ({base_mmlu_correct}/50)\n")

    # Base MC factual accuracy (no adapter)
    print("Computing base MC factual accuracy (115 facts)...")
    base_mc_wins = 0
    try:
        with open(MC_TRUTH_DICT) as _f:
            mc_facts_base = json.load(_f)["truths"]
        for fact in mc_facts_base:
            win, _, _, _ = score_fact_mc(model, tokenizer, fact["context"], fact["truth"], fact["distractors"])
            base_mc_wins += int(win)
        base_mc_acc = base_mc_wins / len(mc_facts_base)
    except Exception as _e:
        print(f"  Base MC eval failed: {_e}")
        base_mc_acc = 0.0
    print(f"  Base MC: {base_mc_acc:.1%} ({base_mc_wins}/{len(mc_facts_base)})\n")

    # Run autoresearch - initialize or resume
    start_trial = 0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        with open(args.resume) as f:
            resume_data = json.load(f)
        current_config = copy.deepcopy(resume_data["config"])
        best_config = copy.deepcopy(resume_data["config"])
        best_val_loss = resume_data["best_val_loss"]
        best_mmlu_acc = resume_data["best_mmlu_acc"]
        best_mc_acc = resume_data.get("best_mc_acc", 0.0)
        improvements = resume_data.get("improvements", 0)
        start_trial = resume_data.get("trials_completed", 0)
        print(f"  Starting from trial {start_trial + 1}")
        print(f"  Best val_loss: {best_val_loss:.4f}, MMLU: {best_mmlu_acc:.1%}, MC: {best_mc_acc:.1%}")
        print()
    else:
        current_config = copy.deepcopy(DEFAULT_CONFIG)
        best_val_loss = float("inf")
        best_mmlu_acc = 0.0
        best_mc_acc = base_mc_acc  # Start threshold at actual base model performance
        best_config = copy.deepcopy(current_config)
        improvements = 0
    total_time_start = time.time()

    # Pre-load MMLU dataset once to avoid repeated loading
    print("Pre-loading MMLU dataset...")
    mmlu_ds = None
    try:
        from datasets import load_dataset
        mmlu_ds = load_dataset("cais/mmlu", "all", split="test")
        print(f"  Loaded {len(mmlu_ds)} MMLU examples.\n")
    except Exception as e:
        print(f"  MMLU load failed: {e}\n")

    print("=" * 70)
    print("  STARTING SEARCH")
    print("=" * 70)

    total_iterations = start_trial + args.iterations
    for iteration in range(start_trial, total_iterations):
        iter_start = time.time()

        # Clear memory before each trial
        clear_memory()

        # Perturb one parameter
        trial_config, param_name, old_val, new_val = perturb_config(current_config)

        print(f"\n{'─' * 70}")
        print(f"  Trial {iteration + 1}/{total_iterations}: "
              f"{param_name} = {old_val} → {new_val}")
        print(f"{'─' * 70}")

        # Run trial with error handling
        try:
            metrics = run_trial(
                model, tokenizer, lm_head, train_examples, val_examples,
                trial_config, args.steps_per_trial, iteration, mmlu_ds
            )
        except Exception as e:
            print(f"  Trial FAILED: {e}")
            traceback.print_exc()
            # Skip this trial and continue
            clear_memory()
            continue

        # Decide: keep or revert
        # Primary: MC factual accuracy (higher is better, directly measures truth recall)
        # Secondary: val loss (lower is better, prevents capability regression)
        # Tertiary: MMLU accuracy (higher is better)
        improved = False
        reason = ""

        if metrics["mc_acc"] > best_mc_acc + 0.01 and metrics["avg_val_loss"] < best_val_loss + 0.02:
            # MC accuracy improved without significant val loss regression
            improved = True
            reason = f"mc_acc {best_mc_acc:.1%} → {metrics['mc_acc']:.1%}"
        elif metrics["avg_val_loss"] < best_val_loss - 0.001 and metrics["mc_acc"] >= best_mc_acc - 0.02:
            # Val loss improved without MC regression
            improved = True
            reason = f"val_loss {best_val_loss:.4f} → {metrics['avg_val_loss']:.4f}"
        elif (metrics["mmlu_acc"] > best_mmlu_acc + 0.02 and
              metrics["avg_val_loss"] < best_val_loss + 0.01 and
              metrics["mc_acc"] >= best_mc_acc - 0.02):
            # MMLU improvement without regressions
            improved = True
            reason = f"mmlu {best_mmlu_acc:.1%} → {metrics['mmlu_acc']:.1%}"

        if improved:
            improvements += 1
            best_val_loss = metrics["avg_val_loss"]
            best_mmlu_acc = max(best_mmlu_acc, metrics["mmlu_acc"])
            best_mc_acc = max(best_mc_acc, metrics["mc_acc"])
            current_config = copy.deepcopy(trial_config)
            best_config = copy.deepcopy(trial_config)
            status = f"✓ KEEP ({reason})"
        else:
            status = (f"✗ REVERT (val={metrics['avg_val_loss']:.4f}, "
                      f"mc={metrics['mc_acc']:.1%}, mmlu={metrics['mmlu_acc']:.1%})")

        iter_time = time.time() - iter_start
        total_time = time.time() - total_time_start

        print(f"  {status}")
        print(f"  mc={metrics['mc_acc']:.1%} ({metrics['mc_wins']}/{metrics['mc_total']}) | "
              f"val_loss={metrics['avg_val_loss']:.4f} | "
              f"mmlu={metrics['mmlu_acc']:.1%} | "
              f"train_loss={metrics['avg_train_loss']:.4f} | "
              f"time={iter_time:.0f}s | "
              f"total={total_time/60:.1f}m")
        print(f"  Improvements so far: {improvements}/{iteration + 1}")

        # Log to file
        log_entry = {
            "trial": iteration + 1,
            "param": param_name,
            "old_val": old_val,
            "new_val": new_val,
            "improved": improved,
            "reason": reason if improved else "",
            "metrics": metrics,
            "current_config": current_config,
            "best_val_loss": best_val_loss,
            "best_mmlu_acc": best_mmlu_acc,
            "iter_time_s": iter_time,
            "total_time_s": total_time,
            "timestamp": datetime.now().isoformat(),
        }
        # Remove non-serializable items
        log_entry["metrics"].pop("config", None)
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Also save current best config
        best_config_file = os.path.join(RESULTS_DIR, f"best_config_{timestamp}.json")
        with open(best_config_file, "w") as f:
            json.dump({
                "config": best_config,
                "best_val_loss": best_val_loss,
                "best_mmlu_acc": best_mmlu_acc,
                "best_mc_acc": best_mc_acc,
                "base_val_loss": base_val_loss,
                "base_mmlu_acc": base_mmlu_acc,
                "base_mc_acc": base_mc_acc,
                "improvements": improvements,
                "trials_completed": iteration + 1,
                "total_time_s": total_time,
            }, f, indent=2)

        # Clear memory after each trial
        clear_memory()

    # Final summary
    total_time = time.time() - total_time_start
    print(f"\n{'=' * 70}")
    print(f"  AUTORESEARCH COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total trials:   {total_iterations} ({args.iterations} this run)")
    print(f"  Improvements:   {improvements}")
    print(f"  Total time:     {total_time/3600:.1f}h")
    print(f"\n  Base MC acc:    {base_mc_acc:.1%} ({base_mc_wins}/{len(mc_facts_base)})")
    print(f"  Best MC acc:    {best_mc_acc:.1%}")
    print(f"  Base val loss:  {base_val_loss:.4f}")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Base MMLU:      {base_mmlu_acc:.1%}")
    print(f"  Best MMLU:      {best_mmlu_acc:.1%}")
    print(f"\n  Best config:")
    for k, v in best_config.items():
        default = DEFAULT_CONFIG.get(k)
        changed = " ← CHANGED" if v != default else ""
        print(f"    {k}: {v}{changed}")
    print(f"\n  Log: {log_file}")
    print(f"  Config: {best_config_file}")


if __name__ == "__main__":
    main()
