#!/usr/bin/env python3
"""
NoetherSolve — PyTorch/CUDA backend.

Drop-in replacement for the MLX stack when running on Linux/CUDA or any
non-Apple-Silicon machine.  No MLX dependency.  Produces .npz adapters
compatible with the MLX oracle_wrapper.py (same weight key names, same
SwiGLU architecture).

Modes
-----
  train-adapter   Train a logit-space adapter from a JSON training set.
  eval-oracle     Score a problem's verification set (oracle pass/fail).
  check-checker   Run the numerical checker (pure scipy, no model needed).

Requirements
------------
  pip install torch transformers accelerate scipy numpy pyyaml

Usage
-----
  # Train a vortex domain adapter
  python noethersolve_torch.py train-adapter \\
      --data problems/vortex_aligned_30.json \\
      --model Qwen/Qwen3-4B-Base \\
      --out adapters/vortex_q_adapter_cuda.npz

  # Run oracle on a verification set
  python noethersolve_torch.py eval-oracle \\
      --problem problems/vortex_pair_conservation.yaml \\
      --model Qwen/Qwen3-4B-Base

  # With adapter
  python noethersolve_torch.py eval-oracle \\
      --problem problems/vortex_pair_conservation.yaml \\
      --adapter adapters/vortex_q_adapter_cuda.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

LOGIT_SOFTCAP = 30.0   # Matches train_v3.py / oracle_wrapper.py


# ─────────────────────────────────────────────────────────────────────────────
# Logit-space SwiGLU adapter (matches SnapOnLogitMLP weight key names exactly)
# .npz keys: gate_proj.weight, up_proj.weight, down_proj.weight
# ─────────────────────────────────────────────────────────────────────────────

class LogitAdapter(nn.Module):
    """SwiGLU logit adapter — identical architecture to MLX SnapOnLogitMLP.

    base_logits  →  gate_proj / up_proj  →  SwiGLU  →  down_proj  →  shifts
    final_logits = base_logits + shifts (centered)

    Weight shapes (out, in convention — matches PyTorch AND MLX Linear):
      gate_proj.weight : (d_inner, vocab_size)
      up_proj.weight   : (d_inner, vocab_size)
      down_proj.weight : (vocab_size, d_inner)
    """

    def __init__(self, vocab_size: int, d_inner: int = 64):
        super().__init__()
        self.gate_proj = nn.Linear(vocab_size, d_inner, bias=False)
        self.up_proj   = nn.Linear(vocab_size, d_inner, bias=False)
        self.down_proj = nn.Linear(d_inner, vocab_size, bias=False)
        # Init matching SnapOn defaults
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight,   std=0.02)
        nn.init.zeros_(self.down_proj.weight)   # zero-init → identity at start

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (batch, seq_len, vocab_size) → shifts (same shape)."""
        gate   = F.silu(self.gate_proj(logits))
        up     = self.up_proj(logits)
        shifts = self.down_proj(gate * up)
        shifts = shifts - shifts.mean(dim=-1, keepdim=True)  # centre shifts
        return shifts


def save_adapter(adapter: LogitAdapter, path: str):
    """Save adapter weights as .npz — compatible with MLX oracle_wrapper.py."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    weights = {
        "gate_proj.weight": adapter.gate_proj.weight.detach().cpu().float().numpy(),
        "up_proj.weight":   adapter.up_proj.weight.detach().cpu().float().numpy(),
        "down_proj.weight": adapter.down_proj.weight.detach().cpu().float().numpy(),
    }
    np.savez(path, **weights)
    print(f"  Adapter saved: {path}  "
          f"({sum(v.size for v in weights.values()):,} params)")


def load_adapter(path: str, vocab_size: int, d_inner: int = 64,
                 device: torch.device = None) -> LogitAdapter:
    """Load adapter from .npz (works with both MLX-trained and torch-trained)."""
    data    = np.load(path)
    adapter = LogitAdapter(vocab_size, d_inner)
    adapter.gate_proj.weight.data = torch.from_numpy(data["gate_proj.weight"]).float()
    adapter.up_proj.weight.data   = torch.from_numpy(data["up_proj.weight"]).float()
    adapter.down_proj.weight.data = torch.from_numpy(data["down_proj.weight"]).float()
    if device:
        adapter = adapter.to(device)
    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, device: torch.device):
    """Load HuggingFace model + tokenizer. Freezes model weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {model_name} on {device}...")
    t0 = time.time()
    dtype  = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    model  = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto" if device.type == "cuda" else None
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = model.get_output_embeddings().weight.shape[0]
    d_model    = model.config.hidden_size
    print(f"  Loaded in {time.time()-t0:.1f}s  "
          f"d_model={d_model}  vocab={vocab_size}")
    return model, tokenizer, vocab_size


# ─────────────────────────────────────────────────────────────────────────────
# Log-probability scoring
# ─────────────────────────────────────────────────────────────────────────────

def get_completion_logprob(model, tokenizer, prompt: str, completion: str,
                           adapter: LogitAdapter = None,
                           device: torch.device = None) -> float:
    """
    log P(completion | prompt) under the model (+adapter).
    Sums log-probs over completion tokens only.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False,
                           return_tensors="pt").input_ids
    full_ids   = tokenizer(prompt + " " + completion.lstrip(),
                           add_special_tokens=False,
                           return_tensors="pt").input_ids
    n_prompt   = prompt_ids.shape[1]

    if full_ids.shape[1] <= n_prompt:
        return -1e9

    input_ids = full_ids.to(device or next(model.parameters()).device)

    with torch.no_grad():
        base_logits = model(input_ids).logits   # (1, seq_len, vocab)

    if adapter is not None:
        dev = base_logits.device
        adapter_dev = adapter.to(dev)
        with torch.no_grad():
            shifts      = adapter_dev(base_logits)
        logits = base_logits + shifts
        logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
    else:
        logits = base_logits

    log_probs = torch.log_softmax(logits[0], dim=-1)  # (seq_len, vocab)

    total = 0.0
    for i in range(n_prompt, full_ids.shape[1]):
        tok_id = full_ids[0, i].item()
        total += log_probs[i - 1, tok_id].item()
    return total


def score_fact(model, tokenizer, context: str, truth: str, distractors: list,
               adapter: LogitAdapter = None,
               device: torch.device = None) -> tuple:
    """
    Returns (win: bool, margin: float).
    win = True  ↔  log P(truth) > log P(best distractor)
    margin > 0  ↔  oracle PASS  (from Paper 9: zero false positives)
    """
    prompt    = context + ":"
    truth_lp  = get_completion_logprob(model, tokenizer, prompt, truth,
                                        adapter=adapter, device=device)
    dist_lps  = [get_completion_logprob(model, tokenizer, prompt, d,
                                         adapter=adapter, device=device)
                 for d in distractors]
    margin    = truth_lp - max(dist_lps)
    return margin > 0, margin


# ─────────────────────────────────────────────────────────────────────────────
# Training loss (requires gradient through adapter)
# ─────────────────────────────────────────────────────────────────────────────

def mc_hinge_loss_grad(adapter: LogitAdapter, model,
                        tokenizer, prompt: str, truth: str, distractors: list,
                        device: torch.device, margin_target: float = 1.5):
    """
    Hinge loss with gradient through adapter.
    Model is frozen; gradient flows only through adapter.
    """
    def lp_with_grad(text: str) -> torch.Tensor:
        full_text  = prompt + " " + text.lstrip()
        prompt_ids = tokenizer(prompt,     add_special_tokens=False,
                                return_tensors="pt").input_ids
        full_ids   = tokenizer(full_text,  add_special_tokens=False,
                                return_tensors="pt").input_ids.to(device)
        n_prompt   = prompt_ids.shape[1]

        if full_ids.shape[1] <= n_prompt:
            return torch.tensor(-1e9, device=device)

        with torch.no_grad():
            base_logits = model(full_ids).logits            # frozen

        shifts = adapter(base_logits)                       # gradient here
        logits = base_logits + shifts
        logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)

        total = torch.zeros(1, device=device)
        for i in range(n_prompt, full_ids.shape[1]):
            tok_id = full_ids[0, i].item()
            lv     = logits[0, i - 1]
            lse    = torch.logsumexp(lv, dim=0)
            total  = total + lv[tok_id] - lse
        return total.squeeze()

    truth_lp  = lp_with_grad(truth)
    dist_lps  = [lp_with_grad(d) for d in distractors]
    best_dist = torch.stack(dist_lps).max()
    margin    = truth_lp - best_dist
    loss      = torch.clamp(torch.tensor(margin_target, device=device) - margin, min=0.0)
    return loss, float(margin.detach())


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data(path: str) -> list:
    """Load training examples from JSON. Returns list of (context, truth, distractors)."""
    with open(path) as f:
        data = json.load(f)
    examples = []
    for ex in data.get("examples", []):
        ctx         = ex["context"]
        truth       = ex.get("truth", ex.get("completion", ""))
        distractors = ex.get("distractors", [
            "This quantity is not conserved",
            "No such invariant exists",
            "The system is chaotic",
        ])
        examples.append((ctx, truth, distractors))
    return examples


def train_adapter(model, tokenizer, vocab_size: int, examples: list,
                  steps: int = 800, lr: float = 1e-6, d_inner: int = 64,
                  margin_target: float = 1.5, device: torch.device = None) -> LogitAdapter:

    adapter   = LogitAdapter(vocab_size, d_inner).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.01)

    print(f"\n  Training adapter: {len(examples)} examples, {steps} steps, "
          f"lr={lr}, d_inner={d_inner}, margin_target={margin_target}")

    t0             = time.time()
    recent_margins = []

    for step in range(steps):
        ctx, truth, distractors = examples[step % len(examples)]
        prompt = ctx + ":"

        optimizer.zero_grad()
        loss, margin_val = mc_hinge_loss_grad(
            adapter, model, tokenizer, prompt, truth, distractors,
            device=device, margin_target=margin_target,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        optimizer.step()

        recent_margins.append(margin_val)
        if len(recent_margins) > 50:
            recent_margins.pop(0)

        if (step + 1) % 100 == 0:
            avg = np.mean(recent_margins)
            print(f"    step {step+1:4d}/{steps}  loss={float(loss):.3f}  "
                  f"margin={margin_val:.3f}  avg={avg:.3f}  {time.time()-t0:.0f}s")

    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# Oracle evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_oracle(model, tokenizer, facts: list,
                adapter: LogitAdapter = None,
                device: torch.device = None,
                label: str = "") -> tuple:
    """Score all facts. Returns (wins, total, mean_margin)."""
    wins, margins = 0, []
    label_str = f" ({label})" if label else ""
    print(f"\n  Oracle evaluation{label_str} — {len(facts)} facts")
    for fact in facts:
        win, margin = score_fact(
            model, tokenizer,
            fact["context"], fact["truth"], fact["distractors"],
            adapter=adapter, device=device,
        )
        wins    += int(win)
        margins.append(margin)
        marker = "✓" if win else "✗"
        print(f"    {marker} {fact.get('id','?'):30s}  margin={margin:+.3f}")

    mean_m = float(np.mean(margins))
    print(f"  Pass: {wins}/{len(facts)} ({100*wins/len(facts):.1f}%)  "
          f"mean_margin={mean_m:+.3f}")
    return wins, len(facts), mean_m


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cmd_train(args):
    device = detect_device()
    print(f"  Device: {device}")

    data_path = args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)
    examples  = load_training_data(data_path)
    print(f"  Loaded {len(examples)} training examples from {data_path}")

    model, tokenizer, vocab_size = load_model(args.model, device)

    # Baseline eval if requested
    if args.eval_before and os.path.exists(args.eval_facts or ""):
        with open(args.eval_facts) as f:
            facts = json.load(f)["facts"]
        eval_oracle(model, tokenizer, facts, adapter=None,
                    device=device, label="baseline")

    adapter = train_adapter(
        model, tokenizer, vocab_size, examples,
        steps=args.steps, lr=args.lr, d_inner=args.d_inner,
        margin_target=args.margin, device=device,
    )

    out_path = args.out or os.path.join(HERE, "adapters",
                                         os.path.splitext(os.path.basename(args.data))[0]
                                         + "_adapter.npz")
    save_adapter(adapter, out_path)

    # Post-training eval
    if args.eval_facts and os.path.exists(args.eval_facts):
        with open(args.eval_facts) as f:
            facts = json.load(f)["facts"]
        eval_oracle(model, tokenizer, facts, adapter=adapter,
                    device=device, label="post-training")


def cmd_eval(args):
    device = detect_device()
    print(f"  Device: {device}")

    # Load problem definition
    problem_path = args.problem if os.path.isabs(args.problem) else os.path.join(HERE, args.problem)
    if problem_path.endswith(".yaml") or problem_path.endswith(".yml"):
        with open(problem_path) as f:
            problem = yaml.safe_load(f)
    else:
        with open(problem_path) as f:
            problem = json.load(f)

    # Load verification set
    vset_name = problem.get("verification_set") or problem.get("facts_file")
    if vset_name:
        vset_path = os.path.join(os.path.dirname(problem_path), vset_name)
    elif args.facts:
        vset_path = args.facts
    else:
        print("ERROR: no verification_set in problem yaml and no --facts given")
        sys.exit(1)

    with open(vset_path) as f:
        facts = json.load(f)["facts"]

    model_name = args.model or problem.get("model", "Qwen/Qwen3-4B-Base")
    model, tokenizer, vocab_size = load_model(model_name, device)

    # Load adapter if provided
    adapter = None
    adapter_path = args.adapter or problem.get("adapter")
    if adapter_path:
        if not os.path.isabs(adapter_path):
            adapter_path = os.path.join(HERE, adapter_path)
        if os.path.exists(adapter_path):
            adapter = load_adapter(adapter_path, vocab_size, d_inner=args.d_inner,
                                    device=device)
            print(f"  Loaded adapter: {adapter_path}")
        else:
            print(f"  WARNING: adapter not found at {adapter_path}, running baseline")

    label = "baseline" if adapter is None else "with adapter"
    wins, total, mean_m = eval_oracle(model, tokenizer, facts,
                                       adapter=adapter, device=device, label=label)

    # Quadrant diagnosis if both baseline and adapter results available
    if adapter is not None and args.diagnose:
        base_wins, _, base_mean = eval_oracle(
            model, tokenizer, facts, adapter=None, device=device, label="baseline"
        )
        delta = mean_m - base_mean
        print(f"\n  Quadrant diagnosis:")
        print(f"    baseline mean_margin:  {base_mean:+.3f}")
        print(f"    adapter  mean_margin:  {mean_m:+.3f}")
        print(f"    delta:                 {delta:+.3f}")
        if delta < -5.0:
            print("    → KNOWLEDGE_GAP (adapter makes it worse > 5 margin points)")
            print("    → Action: generate domain-specific training data, train domain adapter")
        elif mean_m > base_mean:
            print("    → FIXABLE_BIAS (adapter improves margin)")
            print("    → Action: apply this adapter, re-verify")
        else:
            print("    → ORACLE_FAIL_UNCHECKED (adapter neither helps nor hurts significantly)")


def main():
    parser = argparse.ArgumentParser(
        description="NoetherSolve PyTorch/CUDA backend"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── train-adapter ──────────────────────────────────────────────────────
    p_train = sub.add_parser("train-adapter", help="Train a logit adapter")
    p_train.add_argument("--data",         required=True,
                          help="Training data JSON (problems/*.json)")
    p_train.add_argument("--model",        default="Qwen/Qwen3-4B-Base")
    p_train.add_argument("--steps",        type=int,   default=800)
    p_train.add_argument("--lr",           type=float, default=1e-6)
    p_train.add_argument("--d-inner",      type=int,   default=64)
    p_train.add_argument("--margin",       type=float, default=1.5)
    p_train.add_argument("--out",          default=None,
                          help="Output .npz path (default: adapters/<data_stem>_adapter.npz)")
    p_train.add_argument("--eval-facts",   default=None,
                          help="Facts JSON to evaluate on before/after training")
    p_train.add_argument("--eval-before",  action="store_true",
                          help="Run baseline eval before training")

    # ── eval-oracle ────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval-oracle", help="Run oracle on a problem")
    p_eval.add_argument("--problem",  required=True,
                         help="Problem YAML or JSON (problems/*.yaml)")
    p_eval.add_argument("--model",    default=None,
                         help="Override model from problem yaml")
    p_eval.add_argument("--adapter",  default=None,
                         help="Adapter .npz (overrides problem yaml)")
    p_eval.add_argument("--facts",    default=None,
                         help="Facts JSON (overrides problem yaml)")
    p_eval.add_argument("--d-inner",  type=int, default=64)
    p_eval.add_argument("--diagnose", action="store_true",
                         help="Also run baseline and print quadrant diagnosis")

    args = parser.parse_args()

    if   args.cmd == "train-adapter":  cmd_train(args)
    elif args.cmd == "eval-oracle":    cmd_eval(args)


if __name__ == "__main__":
    main()
