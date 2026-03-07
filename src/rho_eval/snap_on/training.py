"""Snap-On Communication Module — Training.

Training loop, data loading, and utilities for training snap-on adapters
on frozen base models.
"""

import json
import os
import time
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .module import SnapOnConfig, create_adapter


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
    """Format Alpaca example -> (prompt_str, response_str)."""
    instruction = example["instruction"]
    if example.get("input"):
        instruction += f"\n\n{example['input']}"
    prompt = ALPACA_TEMPLATE.format(instruction=instruction)
    response = example["output"]
    return prompt, response


def load_alpaca_data(tokenizer, n_train: int = 10000, n_val: int = 500,
                     max_seq_len: int = 512, seed: int = 42):
    """Load Alpaca dataset, tokenize, return train/val splits.

    Returns:
        (train_examples, val_examples) where each example is a dict with
        'tokens' (list[int]) and 'prompt_len' (int).
    """
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
    adapter.load_weights(list(weights.items()))
    return adapter


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
    """Train the adapter on pre-tokenized examples.

    Args:
        base_model: Frozen MLX language model.
        tokenizer: Tokenizer (for logging only).
        adapter: SnapOnMLP or SnapOnLogitMLP adapter module.
        train_examples: List of dicts with 'tokens' and 'prompt_len'.
        val_examples: List of dicts with 'tokens' and 'prompt_len'.
        epochs: Number of training epochs.
        lr: Peak learning rate.
        warmup_steps: Linear warmup steps.
        log_every: Print training loss every N steps.
        eval_every: Evaluate on val set every N steps.
        save_dir: Directory for checkpoints.
        grad_accum: Gradient accumulation steps.
        mode: "hidden" or "logit".

    Returns:
        Best validation loss achieved.
    """
    # Optimizer with warmup + cosine decay
    total_steps = len(train_examples) * epochs
    warmup_steps = min(warmup_steps, total_steps // 2)
    cos_steps = max(total_steps - warmup_steps, 1)

    if warmup_steps > 0 and total_steps > warmup_steps:
        warmup_sched = optim.linear_schedule(1e-7, lr, warmup_steps)
        cos_sched = optim.cosine_decay(lr, cos_steps)
        lr_schedule = optim.join_schedules(
            [warmup_sched, cos_sched], [warmup_steps]
        )
    else:
        lr_schedule = optim.cosine_decay(lr, max(total_steps, 1))
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

    lm_head = base_model.lm_head
    logit_mode = (mode == "logit")

    def loss_fn(adapter, h, targets, mask):
        if logit_mode:
            base_logits = lm_head(h)
            mx.eval(base_logits)
            logits = base_logits + adapter(base_logits)
        else:
            adjustment = adapter(h)
            logits = lm_head(h + adjustment)
        logits = logits[:, :-1, :]
        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        n_tok = mask.sum()
        loss = (ce * mask).sum() / mx.maximum(n_tok, mx.array(1.0))
        return loss, n_tok

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    def base_loss(h, targets, mask):
        logits = lm_head(h)[:, :-1, :]
        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        n_tok = mask.sum()
        return (ce * mask).sum() / mx.maximum(n_tok, mx.array(1.0))

    # Compute initial base model loss
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

    print(f"\nTraining: {epochs} epochs x {len(train_examples)} examples = "
          f"{total_steps} steps")
    print(f"  LR: {lr}, warmup: {warmup_steps}, grad_accum: {grad_accum}")
    print()

    for epoch in range(epochs):
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

            h = base_model.model(input_ids)
            mx.eval(h)

            targets = input_ids[:, 1:]
            L = input_ids.shape[1]
            mask = (mx.arange(L - 1) >= (ex["prompt_len"] - 1)).astype(mx.float32)[None, :]

            (loss, n_tok), grads = loss_and_grad(adapter, h, targets, mask)

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

            if global_step % log_every == 0:
                avg = epoch_loss / (step_in_epoch + 1)
                elapsed = time.time() - t_epoch
                tps = epoch_tokens / max(elapsed, 1e-6)
                current_lr = float(lr_schedule(global_step))
                print(f"  step {global_step:>6d} | epoch {epoch+1} "
                      f"[{step_in_epoch+1}/{len(indices)}] | "
                      f"loss {avg:.4f} | lr {current_lr:.2e} | "
                      f"{tps:.0f} tok/s", flush=True)

            if global_step % eval_every == 0:
                val_loss = evaluate_loss(
                    adapter, base_model, val_examples, loss_fn, mode=mode
                )
                print(f"  >>> val loss: {val_loss:.4f} "
                      f"(base: {avg_base:.4f}, D={avg_base - val_loss:+.4f})",
                      flush=True)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_adapter(adapter, save_dir, "best")
                    print(f"  >>> saved best model (val_loss={val_loss:.4f})")

        avg_epoch = epoch_loss / len(indices)
        elapsed = time.time() - t_epoch
        print(f"\nEpoch {epoch+1}: avg_loss={avg_epoch:.4f}, "
              f"time={elapsed:.1f}s, tok/s={epoch_tokens/elapsed:.0f}")

        val_loss = evaluate_loss(adapter, base_model, val_examples, loss_fn, mode=mode)
        print(f"  Val loss: {val_loss:.4f} (base: {avg_base:.4f}, "
              f"D={avg_base - val_loss:+.4f})")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_adapter(adapter, save_dir, "best")
            print(f"  Saved best model (val_loss={val_loss:.4f})")
        print()

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
