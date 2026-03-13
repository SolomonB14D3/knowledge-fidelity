#!/usr/bin/env python3
"""Snap-On Communication Module — Phase 1 Proof of Concept.

Train a tiny adapter on Qwen2.5-7B base to produce instruction-following
output without modifying base model weights.

The adapter learns to adjust the base model's final hidden states so that
the (frozen) unembedding matrix produces instruction-quality logits.

Usage:
    # Quick smoke test (100 examples, 1 epoch)
    python experiments/snap_on/train.py --n_train 100 --epochs 1

    # Full Phase 1 (10K examples, 3 epochs, ~1-2h on M3 Ultra)
    python experiments/snap_on/train.py --n_train 10000 --epochs 3

    # Transformer variant
    python experiments/snap_on/train.py --arch transformer --d_inner 512 --n_layers 2
"""

import argparse
import json
import os
import sys
import time
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
from mlx.utils import tree_flatten

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from module import SnapOnConfig, create_adapter


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_alpaca(example: dict) -> tuple:
    """Format Alpaca example → (prompt_str, response_str)."""
    instruction = example["instruction"]
    if example.get("input"):
        instruction += f"\n\n{example['input']}"
    prompt = ALPACA_TEMPLATE.format(instruction=instruction)
    response = example["output"]
    return prompt, response


def load_and_tokenize(tokenizer, n_train: int, n_val: int, max_seq_len: int, seed: int = 42):
    """Load Alpaca dataset, tokenize, return train/val splits."""
    from datasets import load_dataset

    print("Loading Alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=seed)

    def tokenize_split(subset):
        examples = []
        skipped = 0
        for ex in subset:
            prompt, response = format_alpaca(ex)
            prompt_tokens = tokenizer.encode(prompt)
            # Encode response without adding BOS
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            full_tokens = prompt_tokens + response_tokens

            if len(full_tokens) < 10:
                skipped += 1
                continue
            if len(full_tokens) > max_seq_len:
                full_tokens = full_tokens[:max_seq_len]

            examples.append({
                "tokens": full_tokens,
                "prompt_len": len(prompt_tokens),
            })
        if skipped:
            print(f"  Skipped {skipped} examples (too short)")
        return examples

    train_examples = tokenize_split(ds.select(range(n_train)))
    val_examples = tokenize_split(ds.select(range(n_train, n_train + n_val)))

    # Sort by length for consistent memory usage
    train_examples.sort(key=lambda x: len(x["tokens"]))

    avg_len = sum(len(e["tokens"]) for e in train_examples) / len(train_examples)
    avg_resp = avg_len - sum(e["prompt_len"] for e in train_examples) / len(train_examples)
    print(f"  Train: {len(train_examples)} examples, avg length {avg_len:.0f} tokens "
          f"(prompt + {avg_resp:.0f} response)")
    print(f"  Val:   {len(val_examples)} examples")
    return train_examples, val_examples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    base_model,
    tokenizer,
    adapter,
    train_examples,
    val_examples,
    *,
    epochs: int = 3,
    lr: float = 1e-4,
    warmup_steps: int = 100,
    log_every: int = 50,
    eval_every: int = 500,
    save_dir: str = "results/snap_on",
    grad_accum: int = 1,
    mode: str = "hidden",
):
    """Train the adapter."""
    # Optimizer with warmup + cosine decay
    total_steps = len(train_examples) * epochs
    warmup_steps = min(warmup_steps, total_steps // 2)  # Don't warm up more than half
    cos_steps = max(total_steps - warmup_steps, 1)      # Avoid zero division

    if warmup_steps > 0 and total_steps > warmup_steps:
        warmup_sched = optim.linear_schedule(1e-7, lr, warmup_steps)
        cos_sched = optim.cosine_decay(lr, cos_steps)
        lr_schedule = optim.join_schedules(
            [warmup_sched, cos_sched], [warmup_steps]
        )
    else:
        lr_schedule = optim.cosine_decay(lr, max(total_steps, 1))
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

    # Reference to lm_head (frozen, used in loss computation)
    lm_head = base_model.lm_head
    logit_mode = (mode == "logit")

    def loss_fn(adapter, h, targets, mask):
        """CE loss on response tokens. h is pre-computed & materialized."""
        if logit_mode:
            # Logit mode: adapter operates on logits, knowledge path untouched
            base_logits = lm_head(h)
            mx.eval(base_logits)
            logits = base_logits + adapter(base_logits)
        else:
            # Hidden mode: adapter adjusts hidden states before lm_head
            adjustment = adapter(h)
            logits = lm_head(h + adjustment)
        # Shift: logits[:, :-1] predicts targets = input_ids[:, 1:]
        logits = logits[:, :-1, :]
        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        n_tok = mask.sum()
        loss = (ce * mask).sum() / mx.maximum(n_tok, mx.array(1.0))
        return loss, n_tok

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    # Also compute base model loss (no adapter) for comparison
    def base_loss(h, targets, mask):
        logits = lm_head(h)[:, :-1, :]
        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        n_tok = mask.sum()
        return (ce * mask).sum() / mx.maximum(n_tok, mx.array(1.0))

    # Compute initial base model loss on a few val examples
    print("\nComputing base model loss (no adapter)...")
    base_losses = []
    for ex in val_examples[:50]:
        input_ids = mx.array(ex["tokens"])[None, :]
        h = base_model.model(input_ids)
        mx.eval(h)
        targets = input_ids[:, 1:]
        L = input_ids.shape[1]
        mask = (mx.arange(L - 1) >= (ex["prompt_len"] - 1)).astype(mx.float32)[None, :]
        bl = base_loss(h, targets, mask)
        mx.eval(bl)
        base_losses.append(float(bl))
    avg_base = sum(base_losses) / len(base_losses)
    print(f"  Base model avg CE on val: {avg_base:.4f} (ppl {2**avg_base:.1f})")

    # Training loop
    total_steps = len(train_examples) * epochs
    global_step = 0
    best_val_loss = float("inf")

    print(f"\nTraining: {epochs} epochs × {len(train_examples)} examples = "
          f"{total_steps} steps")
    print(f"  LR: {lr}, warmup: {warmup_steps}, grad_accum: {grad_accum}")
    print()

    for epoch in range(epochs):
        # Shuffle each epoch (but keep sorted within each shuffle group
        # for memory efficiency — skip for now, just shuffle)
        indices = list(range(len(train_examples)))
        random.seed(42 + epoch)
        random.shuffle(indices)

        epoch_loss = 0.0
        epoch_tokens = 0
        t_epoch = time.time()
        accum_grads = None
        accum_count = 0

        for step_in_epoch, idx in enumerate(indices):
            ex = train_examples[idx]
            input_ids = mx.array(ex["tokens"])[None, :]

            # Base model forward (frozen, materialized)
            h = base_model.model(input_ids)
            mx.eval(h)

            # Targets and mask
            targets = input_ids[:, 1:]
            L = input_ids.shape[1]
            mask = (mx.arange(L - 1) >= (ex["prompt_len"] - 1)).astype(mx.float32)[None, :]

            # Forward + backward through adapter only
            (loss, n_tok), grads = loss_and_grad(adapter, h, targets, mask)

            # Gradient accumulation
            if grad_accum > 1:
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = dict_add(accum_grads, grads)
                accum_count += 1

                if accum_count >= grad_accum:
                    avg_grads = dict_scale(accum_grads, 1.0 / grad_accum)
                    optimizer.update(adapter, avg_grads)
                    accum_grads = None
                    accum_count = 0
                else:
                    mx.eval(loss)
                    epoch_loss += float(loss)
                    epoch_tokens += int(n_tok)
                    global_step += 1
                    continue
            else:
                optimizer.update(adapter, grads)

            mx.eval(adapter.parameters(), optimizer.state, loss)

            epoch_loss += float(loss)
            epoch_tokens += int(n_tok)
            global_step += 1

            # Log
            if global_step % log_every == 0:
                avg = epoch_loss / (step_in_epoch + 1)
                elapsed = time.time() - t_epoch
                tps = epoch_tokens / max(elapsed, 1e-6)
                current_lr = float(lr_schedule(global_step))
                print(f"  step {global_step:>6d} | epoch {epoch+1} "
                      f"[{step_in_epoch+1}/{len(indices)}] | "
                      f"loss {avg:.4f} | lr {current_lr:.2e} | "
                      f"{tps:.0f} tok/s", flush=True)

            # Eval
            if global_step % eval_every == 0:
                val_loss = evaluate_loss(
                    adapter, base_model, val_examples, loss_fn, mode=mode
                )
                print(f"  >>> val loss: {val_loss:.4f} "
                      f"(base: {avg_base:.4f}, Δ={avg_base - val_loss:+.4f})",
                      flush=True)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_adapter(adapter, save_dir, "best")
                    print(f"  >>> saved best model (val_loss={val_loss:.4f})")

        # End of epoch
        avg_epoch = epoch_loss / len(indices)
        elapsed = time.time() - t_epoch
        print(f"\nEpoch {epoch+1}: avg_loss={avg_epoch:.4f}, "
              f"time={elapsed:.1f}s, tok/s={epoch_tokens/elapsed:.0f}")

        # Epoch-end eval
        val_loss = evaluate_loss(adapter, base_model, val_examples, loss_fn, mode=mode)
        print(f"  Val loss: {val_loss:.4f} (base: {avg_base:.4f}, "
              f"Δ={avg_base - val_loss:+.4f})")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_adapter(adapter, save_dir, "best")
            print(f"  Saved best model (val_loss={val_loss:.4f})")
        print()

    # Save final
    save_adapter(adapter, save_dir, "final")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return best_val_loss


def evaluate_loss(adapter, base_model, examples, loss_fn, mode="hidden"):
    """Compute average loss on examples."""
    lm_head = base_model.lm_head
    logit_mode = (mode == "logit")
    total_loss = 0.0
    n = 0
    for ex in examples:
        input_ids = mx.array(ex["tokens"])[None, :]
        h = base_model.model(input_ids)
        mx.eval(h)
        targets = input_ids[:, 1:]
        L = input_ids.shape[1]
        mask = (mx.arange(L - 1) >= (ex["prompt_len"] - 1)).astype(mx.float32)[None, :]
        # Forward only (no grad)
        if logit_mode:
            base_logits = lm_head(h)
            mx.eval(base_logits)
            logits = (base_logits + adapter(base_logits))[:, :-1, :]
        else:
            adjustment = adapter(h)
            logits = lm_head(h + adjustment)[:, :-1, :]
        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        n_tok = mask.sum()
        loss = (ce * mask).sum() / mx.maximum(n_tok, mx.array(1.0))
        mx.eval(loss)
        total_loss += float(loss)
        n += 1
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_with_adapter(base_model, adapter, tokenizer, prompt: str,
                          max_tokens: int = 150, temperature: float = 0.0,
                          mode: str = "hidden"):
    """Autoregressive generation using base model + adapter."""
    tokens = tokenizer.encode(prompt)
    generated = []
    logit_mode = (mode == "logit")

    for _ in range(max_tokens):
        input_ids = mx.array(tokens)[None, :]
        h = base_model.model(input_ids)
        mx.eval(h)
        if logit_mode:
            base_logits = base_model.lm_head(h)
            mx.eval(base_logits)
            logits = base_logits + adapter(base_logits)
        else:
            adjustment = adapter(h)
            logits = base_model.lm_head(h + adjustment)
        mx.eval(logits)

        last_logits = logits[0, -1, :]
        if temperature > 0:
            probs = mx.softmax(last_logits / temperature)
            next_token = int(mx.random.categorical(mx.log(probs)))
        else:
            next_token = int(mx.argmax(last_logits))

        eos_id = getattr(tokenizer, "eos_token_id", None)
        if next_token == eos_id:
            break
        tokens.append(next_token)
        generated.append(next_token)

    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_base_only(base_model, tokenizer, prompt: str,
                       max_tokens: int = 150, temperature: float = 0.0):
    """Autoregressive generation with base model only (no adapter)."""
    tokens = tokenizer.encode(prompt)
    generated = []

    for _ in range(max_tokens):
        input_ids = mx.array(tokens)[None, :]
        logits = base_model(input_ids)
        mx.eval(logits)

        last_logits = logits[0, -1, :]
        if temperature > 0:
            probs = mx.softmax(last_logits / temperature)
            next_token = int(mx.random.categorical(mx.log(probs)))
        else:
            next_token = int(mx.argmax(last_logits))

        eos_id = getattr(tokenizer, "eos_token_id", None)
        if next_token == eos_id:
            break
        tokens.append(next_token)
        generated.append(next_token)

    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# MMLU Evaluation
# ---------------------------------------------------------------------------

def evaluate_mmlu(base_model, adapter, tokenizer, n_questions: int = 200,
                  mode: str = "hidden"):
    """Quick MMLU logit accuracy with and without adapter."""
    try:
        from rho_eval.unlock.expression_gap import _load_mmlu, _format_mmlu_prompt
        from rho_eval.unlock.contrastive import get_answer_token_ids
    except ImportError:
        print("  (skipping MMLU eval — rho_eval.unlock not available)")
        return None, None

    logit_mode = (mode == "logit")
    print(f"\nEvaluating MMLU ({n_questions} questions, mode={mode})...")
    questions = _load_mmlu(n=n_questions, seed=42)
    answer_ids_dict = get_answer_token_ids(tokenizer, n_choices=4)
    letters = "ABCD"
    # Convert dict {"A": id, "B": id, ...} to list [id_A, id_B, id_C, id_D]
    answer_id_list = [answer_ids_dict[l] for l in letters]

    correct_base = 0
    correct_adapter = 0

    for i, q in enumerate(questions):
        prompt = _format_mmlu_prompt(tokenizer, q)
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]

        # Base model hidden states
        h = base_model.model(input_ids)
        mx.eval(h)

        # Base model logits (same for both modes — this is the knowledge pathway)
        base_logits = base_model.lm_head(h)
        mx.eval(base_logits)
        base_last = base_logits[0, -1, :]

        # Adapter logits
        if logit_mode:
            adapter_logits = base_logits + adapter(base_logits)
        else:
            adjustment = adapter(h)
            adapter_logits = base_model.lm_head(h + adjustment)
        mx.eval(adapter_logits)
        adapter_last = adapter_logits[0, -1, :]

        # Check predictions
        answer_idx = q["answer_idx"]
        correct_letter = letters[answer_idx]

        base_pred = max(range(4), key=lambda j: float(base_last[answer_id_list[j]]))
        adapter_pred = max(range(4), key=lambda j: float(adapter_last[answer_id_list[j]]))

        if letters[base_pred] == correct_letter:
            correct_base += 1
        if letters[adapter_pred] == correct_letter:
            correct_adapter += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_questions}: base={correct_base/(i+1):.1%}, "
                  f"adapter={correct_adapter/(i+1):.1%}", flush=True)

    base_acc = correct_base / n_questions
    adapter_acc = correct_adapter / n_questions
    print(f"\nMMLU Results ({n_questions} questions):")
    print(f"  Base model logit acc:    {base_acc:.1%}")
    print(f"  With adapter logit acc:  {adapter_acc:.1%}")
    print(f"  Delta:                   {adapter_acc - base_acc:+.1%}")
    return base_acc, adapter_acc


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def dict_add(a, b):
    """Recursively add two gradient dicts."""
    if isinstance(a, dict):
        return {k: dict_add(a[k], b[k]) for k in a}
    if isinstance(a, list):
        return [dict_add(ai, bi) for ai, bi in zip(a, b)]
    return a + b


def dict_scale(d, s):
    """Recursively scale a gradient dict."""
    if isinstance(d, dict):
        return {k: dict_scale(v, s) for k, v in d.items()}
    if isinstance(d, list):
        return [dict_scale(v, s) for v in d]
    return d * s


def save_adapter(adapter, save_dir: str, name: str = "adapter"):
    """Save adapter weights and config."""
    os.makedirs(save_dir, exist_ok=True)
    weights = dict(tree_flatten(adapter.parameters()))
    mx.savez(os.path.join(save_dir, f"{name}.npz"), **weights)
    adapter.config.save(os.path.join(save_dir, f"{name}_config.json"))


def load_adapter(save_dir: str, name: str = "adapter"):
    """Load adapter from saved weights."""
    config = SnapOnConfig.load(os.path.join(save_dir, f"{name}_config.json"))
    adapter = create_adapter(config)
    weights = mx.load(os.path.join(save_dir, f"{name}.npz"))
    # Unflatten and load
    adapter.load_weights(list(weights.items()))
    return adapter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train snap-on communication module"
    )
    # Model
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B",
                        help="Base model name/path")
    # Data
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=512)
    # Architecture
    parser.add_argument("--arch", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--d_inner", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Transformer layers (only for --arch transformer)")
    parser.add_argument("--n_heads", type=int, default=8)
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    # Mode
    parser.add_argument("--mode", choices=["hidden", "logit"], default="hidden",
                        help="hidden: adapter on h before lm_head; "
                             "logit: adapter on logits after lm_head")
    # Output
    parser.add_argument("--save_dir", default="results/snap_on")
    # Eval
    parser.add_argument("--skip_mmlu", action="store_true")
    parser.add_argument("--mmlu_n", type=int, default=200)

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Load base model (frozen)
    # -----------------------------------------------------------------------
    print(f"Loading base model: {args.model}")
    base_model, tokenizer = mlx_lm.load(args.model)
    base_model.freeze()

    # Get d_model and vocab_size from the model
    d_model = base_model.model.layers[0].self_attn.q_proj.weight.shape[0]
    vocab_size = base_model.lm_head.weight.shape[0]
    print(f"  d_model = {d_model}")
    print(f"  vocab_size = {vocab_size}")
    print(f"  mode = {args.mode}")

    # -----------------------------------------------------------------------
    # 2. Load & tokenize data
    # -----------------------------------------------------------------------
    train_examples, val_examples = load_and_tokenize(
        tokenizer, args.n_train, args.n_val, args.max_seq_len
    )

    # -----------------------------------------------------------------------
    # 3. Create adapter
    # -----------------------------------------------------------------------
    # For logit mode, default d_inner to 64 (vs 1024 for hidden mode)
    # unless explicitly set, to keep params reasonable with vocab_size I/O
    d_inner = args.d_inner
    if args.mode == "logit" and d_inner == 1024:
        d_inner = 64  # ~29M params with vocab_size=152064
        print(f"  (logit mode: auto d_inner={d_inner} for reasonable param count)")

    config = SnapOnConfig(
        d_model=d_model,
        d_inner=d_inner,
        n_layers=args.n_layers if args.arch == "transformer" else 0,
        n_heads=args.n_heads,
        mode=args.mode,
        vocab_size=vocab_size,
    )
    adapter = create_adapter(config)

    n_params = sum(p.size for _, p in tree_flatten(adapter.parameters()))
    n_trainable = sum(p.size for _, p in tree_flatten(adapter.trainable_parameters()))
    print(f"\nAdapter: {args.arch} (mode={args.mode})")
    print(f"  Config: d_inner={config.d_inner}, n_layers={config.n_layers}")
    print(f"  Total params:     {n_params:>12,}")
    print(f"  Trainable params: {n_trainable:>12,}")
    print(f"  Size:             {n_params * 4 / 1e6:.1f} MB (float32)")

    # -----------------------------------------------------------------------
    # 4. Train
    # -----------------------------------------------------------------------
    best_val = train(
        base_model, tokenizer, adapter,
        train_examples, val_examples,
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
        grad_accum=args.grad_accum,
        mode=args.mode,
    )

    # -----------------------------------------------------------------------
    # 5. Sample generations
    # -----------------------------------------------------------------------
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function that checks if a number is prime.",
        "What are the main causes of climate change?",
    ]

    print("\n" + "=" * 70)
    print("SAMPLE GENERATIONS")
    print("=" * 70)

    for prompt_text in test_prompts:
        full_prompt = ALPACA_TEMPLATE.format(instruction=prompt_text)
        print(f"\n{'─' * 70}")
        print(f"INSTRUCTION: {prompt_text}")

        print(f"\n[BASE MODEL (no adapter)]:")
        base_out = generate_base_only(
            base_model, tokenizer, full_prompt, max_tokens=100
        )
        print(f"  {base_out[:200]}")

        print(f"\n[WITH ADAPTER ({args.mode} mode)]:")
        adapter_out = generate_with_adapter(
            base_model, adapter, tokenizer, full_prompt, max_tokens=150,
            mode=args.mode
        )
        print(f"  {adapter_out[:300]}")

    # -----------------------------------------------------------------------
    # 6. MMLU evaluation
    # -----------------------------------------------------------------------
    if not args.skip_mmlu:
        base_acc, adapter_acc = evaluate_mmlu(
            base_model, adapter, tokenizer, n_questions=args.mmlu_n,
            mode=args.mode
        )

    # -----------------------------------------------------------------------
    # 7. Save results summary
    # -----------------------------------------------------------------------
    results = {
        "model": args.model,
        "arch": args.arch,
        "mode": args.mode,
        "config": vars(config) if hasattr(config, '__dict__') else str(config),
        "n_params": n_params,
        "n_train": args.n_train,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_val_loss": best_val,
    }
    if not args.skip_mmlu and base_acc is not None:
        results["mmlu_base_acc"] = base_acc
        results["mmlu_adapter_acc"] = adapter_acc
        results["mmlu_delta"] = adapter_acc - base_acc

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.save_dir}/results.json")


if __name__ == "__main__":
    main()
