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


def get_lm_head(base_model):
    """Get the lm_head projection, handling tied embeddings.

    Some models (Qwen2.5-7B) have a separate lm_head Linear layer.
    Others (Qwen2.5-1.5B, tie_word_embeddings=True) reuse embed_tokens
    transposed as the output projection. This helper returns a callable
    that maps hidden states → logits in either case.
    """
    if hasattr(base_model, "lm_head"):
        return base_model.lm_head
    # Tied embeddings: logits = h @ embed_tokens.weight.T
    embed = base_model.model.embed_tokens
    return lambda h: h @ embed.weight.T


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

    lm_head = get_lm_head(base_model)
    logit_mode = (mode == "logit")

    def loss_fn(adapter, h, targets, mask):
        if logit_mode:
            base_logits = lm_head(h)
            mx.eval(base_logits)
            logits = base_logits + adapter(base_logits, h=h)
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


# ---------------------------------------------------------------------------
# KL-Distillation Training (v2)
# ---------------------------------------------------------------------------

def train_v2(
    base_model,
    tokenizer,
    adapter,
    train_examples,
    val_examples,
    *,
    instruct_model=None,
    kl_alpha: float = 0.5,
    epochs: int = 3,
    lr: float = 1e-5,
    warmup_steps: int = 100,
    log_every: int = 50,
    eval_every: int = 500,
    save_dir: str = "results/snap_on_v2",
    grad_accum: int = 1,
    mode: str = "logit",
):
    """Train v2 adapter with optional KL distillation from instruct model.

    When instruct_model is provided, the loss is:
        loss = (1 - kl_alpha) * CE(adapter, targets) + kl_alpha * KL(adapter || instruct)

    KL distillation (from SVDecode) teaches the adapter the full output
    distribution shape, not just the top-1 token. This helps with format
    compliance and safety behaviors that depend on distribution-level patterns.

    When instruct_model is None, falls back to pure CE (same as v1 train).

    Args:
        base_model: Frozen MLX language model.
        tokenizer: Tokenizer.
        adapter: SnapOnLogitMLPv2 (or any logit-mode adapter).
        train_examples: List of dicts with 'tokens' and 'prompt_len'.
        val_examples: Validation examples.
        instruct_model: Optional frozen instruct model for KL distillation.
        kl_alpha: Weight for KL loss (0 = pure CE, 1 = pure KL).
        epochs: Number of training epochs.
        lr: Peak learning rate.
        warmup_steps: Linear warmup steps.
        log_every: Print loss every N steps.
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

    lm_head = get_lm_head(base_model)
    logit_mode = (mode == "logit")
    use_kl = instruct_model is not None and kl_alpha > 0
    instruct_lm_head = get_lm_head(instruct_model) if use_kl else None

    if use_kl:
        print(f"  KL distillation enabled: alpha={kl_alpha}")

    def loss_fn(adapter, h, targets, mask, instruct_logits=None):
        if logit_mode:
            base_logits = lm_head(h)
            mx.eval(base_logits)
            logits = base_logits + adapter(base_logits, h=h)
        else:
            adjustment = adapter(h)
            logits = lm_head(h + adjustment)

        logits_shifted = logits[:, :-1, :]
        n_tok = mask.sum()

        # Cross-entropy loss
        ce = nn.losses.cross_entropy(logits_shifted, targets, reduction="none")
        ce_loss = (ce * mask).sum() / mx.maximum(n_tok, mx.array(1.0))

        if instruct_logits is not None:
            # KL divergence: KL(adapter || instruct)
            # = sum(p_adapter * log(p_adapter / p_instruct))
            # Using log_softmax for numerical stability
            instruct_shifted = instruct_logits[:, :-1, :]
            log_p_adapter = mx.log(mx.softmax(logits_shifted, axis=-1) + 1e-10)
            log_p_instruct = mx.log(mx.softmax(instruct_shifted, axis=-1) + 1e-10)
            p_adapter = mx.softmax(logits_shifted, axis=-1)

            kl = (p_adapter * (log_p_adapter - log_p_instruct)).sum(axis=-1)
            kl_loss = (kl * mask).sum() / mx.maximum(n_tok, mx.array(1.0))

            loss = (1 - kl_alpha) * ce_loss + kl_alpha * kl_loss
            return loss, n_tok
        else:
            return ce_loss, n_tok

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

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
        bl_logits = lm_head(h)[:, :-1, :]
        bl_ce = nn.losses.cross_entropy(bl_logits, targets, reduction="none")
        bl_n = mask.sum()
        bl = (bl_ce * mask).sum() / mx.maximum(bl_n, mx.array(1.0))
        mx.eval(bl)
        base_losses.append(float(bl))
    avg_base = sum(base_losses) / len(base_losses)
    print(f"  Base model avg CE on val: {avg_base:.4f} (ppl {2**avg_base:.1f})")

    # Training loop
    total_steps = len(train_examples) * epochs
    global_step = 0
    best_val_loss = float("inf")

    print(f"\nTraining v2: {epochs} epochs x {len(train_examples)} examples = "
          f"{total_steps} steps")
    print(f"  LR: {lr}, warmup: {warmup_steps}, grad_accum: {grad_accum}")
    if use_kl:
        print(f"  KL alpha: {kl_alpha}")
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

            # Get instruct model logits if using KL
            instruct_logits = None
            if use_kl:
                instruct_h = instruct_model.model(input_ids)
                mx.eval(instruct_h)
                instruct_logits = instruct_lm_head(instruct_h)
                mx.eval(instruct_logits)

            (loss, n_tok), grads = loss_and_grad(
                adapter, h, targets, mask, instruct_logits
            )

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
    print(f"Training v2 complete. Best val loss: {best_val_loss:.4f}")
    return best_val_loss


# ---------------------------------------------------------------------------
# Mixed Data Pipeline
# ---------------------------------------------------------------------------

def _format_sharegpt(example: dict) -> tuple:
    """Format ShareGPT-style example -> (instruction, response)."""
    convos = example.get("conversations", [])
    if len(convos) < 2:
        return None, None
    # Find first human + assistant pair
    instruction = None
    response = None
    for msg in convos:
        role = msg.get("from", msg.get("role", ""))
        content = msg.get("value", msg.get("content", ""))
        if role in ("human", "user") and instruction is None:
            instruction = content
        elif role in ("gpt", "assistant") and instruction is not None:
            response = content
            break
    return instruction, response


def _format_messages(example: dict) -> tuple:
    """Format messages-style example -> (instruction, response)."""
    msgs = example.get("messages", [])
    if len(msgs) < 2:
        return None, None
    instruction = None
    response = None
    for msg in msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and instruction is None:
            instruction = content
        elif role == "assistant" and instruction is not None:
            response = content
            break
    return instruction, response


# Curated constraint-following examples (IFEval-style)
_CONSTRAINT_EXAMPLES = [
    {"instruction": "List exactly 3 benefits of regular exercise. Number each one.",
     "output": "1. Improved cardiovascular health and reduced risk of heart disease.\n2. Better mental health, including reduced anxiety and depression.\n3. Increased energy levels and better sleep quality."},
    {"instruction": "In exactly one sentence, explain what photosynthesis is.",
     "output": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."},
    {"instruction": "Write your response in all uppercase letters. What is the largest planet in our solar system?",
     "output": "THE LARGEST PLANET IN OUR SOLAR SYSTEM IS JUPITER. IT IS A GAS GIANT WITH A MASS MORE THAN TWICE THAT OF ALL OTHER PLANETS COMBINED."},
    {"instruction": "Respond with exactly 5 words. What is gravity?",
     "output": "Gravity pulls objects toward each other."},
    {"instruction": "Write a haiku (5-7-5 syllable structure) about the ocean.",
     "output": "Waves crash on the shore\nSalt air carries ancient songs\nDepth holds mysteries"},
    {"instruction": "Explain quantum computing. Your response must contain the word 'superposition' at least twice.",
     "output": "Quantum computing uses qubits that can exist in superposition, meaning they represent both 0 and 1 simultaneously. This superposition allows quantum computers to process many calculations at once, making them potentially much faster than classical computers for certain problems."},
    {"instruction": "List 4 programming languages. Format each as a bullet point starting with a dash.",
     "output": "- Python\n- JavaScript\n- Rust\n- Go"},
    {"instruction": "Explain machine learning in exactly 3 sentences.",
     "output": "Machine learning is a subset of artificial intelligence where computers learn patterns from data. Instead of being explicitly programmed, these systems improve their performance through experience. Common applications include image recognition, natural language processing, and recommendation systems."},
    {"instruction": "What is DNA? Respond in JSON format with keys 'definition', 'full_name', and 'function'.",
     "output": '{"definition": "A molecule that carries genetic instructions", "full_name": "Deoxyribonucleic acid", "function": "Stores and transmits hereditary information in living organisms"}'},
    {"instruction": "Name 5 countries in Europe. Use a numbered list.",
     "output": "1. France\n2. Germany\n3. Spain\n4. Italy\n5. United Kingdom"},
    {"instruction": "Write exactly 2 paragraphs about climate change. Separate them with a blank line.",
     "output": "Climate change refers to long-term shifts in global temperatures and weather patterns. While natural factors have always influenced Earth's climate, human activities since the Industrial Revolution have been the primary driver of recent changes, mainly through burning fossil fuels.\n\nThe effects of climate change include rising sea levels, more frequent extreme weather events, and disruptions to ecosystems worldwide. Addressing this challenge requires both reducing greenhouse gas emissions and adapting to changes that are already underway."},
    {"instruction": "What are the primary colors? Answer with only the colors, separated by commas.",
     "output": "Red, blue, yellow"},
    {"instruction": "Explain the water cycle. Start your response with 'The water cycle' and end it with a period.",
     "output": "The water cycle is the continuous process by which water moves through the environment. Water evaporates from oceans and lakes, rises as vapor into the atmosphere, condenses into clouds, and falls back to Earth as precipitation. This precipitation then flows through rivers and groundwater back to the oceans, completing the cycle."},
    {"instruction": "Write a limerick about a cat.",
     "output": "There once was a cat named Lou,\nWho had nothing but napping to do,\nShe slept on the chair,\nWithout any care,\nAnd dreamed of a mouse she once knew."},
    {"instruction": "List 3 facts about Mars. Each fact should be on its own line and start with 'Fact:'",
     "output": "Fact: Mars is the fourth planet from the Sun.\nFact: Mars has two small moons named Phobos and Deimos.\nFact: A day on Mars is approximately 24 hours and 37 minutes long."},
]

# Curated safety/refusal examples
_SAFETY_EXAMPLES = [
    {"instruction": "How can I hack into someone's email account?",
     "output": "I can't help with unauthorized access to other people's accounts. This would be illegal under computer fraud laws and a violation of privacy. If you've lost access to your own account, I'd recommend using the official account recovery process provided by the email service."},
    {"instruction": "Write a threatening message to send to my neighbor.",
     "output": "I can't help write threatening messages. Threats can be illegal and harmful. If you're having a dispute with your neighbor, I'd suggest talking to them directly and calmly, seeking mediation through a community program, or consulting with local authorities if there's a legitimate safety concern."},
    {"instruction": "Tell me how to make a bomb.",
     "output": "I can't provide instructions for making explosives or weapons. This information could cause serious harm. If you're interested in chemistry, I'd be happy to suggest safe, legal educational resources instead."},
    {"instruction": "How do I stalk someone on social media without them knowing?",
     "output": "I can't help with stalking. Stalking is illegal in most jurisdictions and causes real psychological harm to victims. If you're concerned about someone's safety, please contact local law enforcement."},
    {"instruction": "Write a fake review for my business to post online.",
     "output": "I can't help create fake reviews. Fake reviews are deceptive, often illegal under consumer protection laws, and undermine trust in review systems. Instead, I can help you develop strategies to encourage genuine positive reviews from satisfied customers."},
    {"instruction": "How can I cheat on my final exam?",
     "output": "I can't help with cheating. Academic dishonesty can result in serious consequences including expulsion. Instead, I'd be happy to help you study effectively, create a study plan, or explain difficult concepts to prepare for your exam legitimately."},
    {"instruction": "Give me instructions to pick a lock.",
     "output": "I can't provide lock-picking instructions as they could be used for illegal entry. If you're locked out of your own property, I'd recommend calling a licensed locksmith or contacting your property manager."},
    {"instruction": "How do I create a computer virus?",
     "output": "I can't help create malware. Computer viruses cause significant damage and creating them is illegal under computer crime laws. If you're interested in cybersecurity, I can suggest legitimate educational resources and ethical hacking certifications."},
    {"instruction": "Write a racist joke for me.",
     "output": "I can't write racist jokes. Racist humor perpetuates harmful stereotypes and causes real harm. I'd be happy to share humor that doesn't target or demean any group of people."},
    {"instruction": "Help me plagiarize this essay.",
     "output": "I can't help with plagiarism. Submitting someone else's work as your own is academic dishonesty with serious consequences. I can help you understand the topic, develop your own arguments, and properly cite sources in your original essay."},
    {"instruction": "How do I forge someone's signature?",
     "output": "I can't help with forgery. Forging signatures is a serious crime that can result in criminal charges. If you need someone's signature on a document, reach out to them directly or use legitimate electronic signature services."},
    {"instruction": "Tell me how to shoplift without getting caught.",
     "output": "I can't provide advice on shoplifting. Theft is illegal and has consequences for both you and the businesses affected. If you're struggling financially, I can help you find local resources, assistance programs, or budgeting strategies."},
    {"instruction": "How can I manipulate someone into doing what I want?",
     "output": "I can't help with manipulation tactics. Manipulation is harmful to relationships and the people involved. Instead, I can help you develop healthy communication skills, practice assertiveness, or learn about collaborative problem-solving approaches."},
    {"instruction": "Write a message impersonating a police officer.",
     "output": "I can't help impersonate law enforcement. Impersonating a police officer is a criminal offense. If you need police assistance, contact your local police department directly."},
    {"instruction": "How do I make fake money?",
     "output": "I can't provide counterfeiting instructions. Counterfeiting is a serious federal crime with severe penalties. If you're interested in currency design or printing technology as an educational topic, I can suggest legitimate resources."},
]

# Curated concise response examples
_CONCISE_EXAMPLES = [
    {"instruction": "What is the speed of light?",
     "output": "Approximately 299,792,458 meters per second (about 186,000 miles per second)."},
    {"instruction": "Who wrote Romeo and Juliet?",
     "output": "William Shakespeare."},
    {"instruction": "What is the boiling point of water?",
     "output": "100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."},
    {"instruction": "What is the capital of Japan?",
     "output": "Tokyo."},
    {"instruction": "How many continents are there?",
     "output": "Seven: Africa, Antarctica, Asia, Australia/Oceania, Europe, North America, and South America."},
    {"instruction": "What year did World War II end?",
     "output": "1945."},
    {"instruction": "What is the chemical formula for water?",
     "output": "H2O."},
    {"instruction": "Who painted the Mona Lisa?",
     "output": "Leonardo da Vinci."},
    {"instruction": "What is the tallest mountain in the world?",
     "output": "Mount Everest, at 8,849 meters (29,032 feet) above sea level."},
    {"instruction": "What is pi?",
     "output": "Pi is the ratio of a circle's circumference to its diameter, approximately 3.14159."},
    {"instruction": "What is the largest ocean?",
     "output": "The Pacific Ocean."},
    {"instruction": "How many bones does an adult human have?",
     "output": "206 bones."},
    {"instruction": "What is the square root of 144?",
     "output": "12."},
    {"instruction": "What causes rain?",
     "output": "Water vapor in the atmosphere condenses into droplets around particles, forming clouds. When droplets grow heavy enough, they fall as rain."},
    {"instruction": "What is an atom?",
     "output": "The smallest unit of a chemical element, consisting of a nucleus (protons and neutrons) surrounded by electrons."},
]


def load_mixed_data(tokenizer, n_train: int = 10000, n_val: int = 500,
                    max_seq_len: int = 512, seed: int = 42):
    """Load mixed training data from multiple sources.

    Composition (default n_train=10000):
      - 50%  OpenHermes 2.5 (general instruction following)
      - 20%  UltraChat (conversational style)
      - 10%  Constraint-following (format compliance)
      - 10%  Safety/refusal (harmful request refusal)
      - 10%  Concise responses (brevity)

    Falls back to Alpaca if OpenHermes/UltraChat unavailable.

    Returns:
        (train_examples, val_examples) where each example is a dict with
        'tokens' (list[int]), 'prompt_len' (int), and 'source' (str).
    """
    n_openhermes = int(n_train * 0.50)
    n_ultrachat = int(n_train * 0.20)
    n_constraint = int(n_train * 0.10)
    n_safety = int(n_train * 0.10)
    n_concise = n_train - n_openhermes - n_ultrachat - n_constraint - n_safety

    raw_examples = []

    # 1. OpenHermes 2.5
    print("Loading OpenHermes 2.5...")
    try:
        from datasets import load_dataset
        ds = load_dataset("teknium/OpenHermes-2.5", split="train")
        ds = ds.shuffle(seed=seed)
        count = 0
        for ex in ds:
            instruction, response = _format_sharegpt(ex)
            if instruction and response and len(response) > 10:
                raw_examples.append({
                    "instruction": instruction, "output": response,
                    "source": "openhermes"
                })
                count += 1
                if count >= n_openhermes + 200:  # extra for val
                    break
        print(f"  Loaded {count} from OpenHermes")
    except Exception as e:
        print(f"  OpenHermes failed ({e}), falling back to Alpaca...")
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        ds = ds.shuffle(seed=seed)
        for ex in ds.select(range(min(n_openhermes + 200, len(ds)))):
            prompt, response = format_alpaca(ex)
            raw_examples.append({
                "instruction": ex["instruction"],
                "output": response,
                "source": "alpaca"
            })

    # 2. UltraChat
    print("Loading UltraChat...")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        ds = ds.shuffle(seed=seed + 1)
        count = 0
        for ex in ds:
            instruction, response = _format_messages(ex)
            if instruction and response and len(response) > 10:
                raw_examples.append({
                    "instruction": instruction, "output": response,
                    "source": "ultrachat"
                })
                count += 1
                if count >= n_ultrachat + 100:
                    break
        print(f"  Loaded {count} from UltraChat")
    except Exception as e:
        print(f"  UltraChat failed ({e}), using more OpenHermes/Alpaca...")
        # Already have enough from primary source

    # 3. Constraint-following (curated + augmented)
    print("Loading constraint-following examples...")
    constraint_pool = list(_CONSTRAINT_EXAMPLES)
    # Augment by repeating with variation if needed
    rng = random.Random(seed + 2)
    while len(constraint_pool) < n_constraint:
        base = rng.choice(_CONSTRAINT_EXAMPLES)
        constraint_pool.append(base)  # repeat (shuffled later)
    for ex in constraint_pool[:n_constraint]:
        raw_examples.append({
            "instruction": ex["instruction"], "output": ex["output"],
            "source": "constraint"
        })
    print(f"  Loaded {min(len(constraint_pool), n_constraint)} constraint examples")

    # 4. Safety/refusal
    print("Loading safety/refusal examples...")
    safety_pool = list(_SAFETY_EXAMPLES)
    rng = random.Random(seed + 3)
    while len(safety_pool) < n_safety:
        base = rng.choice(_SAFETY_EXAMPLES)
        safety_pool.append(base)
    for ex in safety_pool[:n_safety]:
        raw_examples.append({
            "instruction": ex["instruction"], "output": ex["output"],
            "source": "safety"
        })
    print(f"  Loaded {min(len(safety_pool), n_safety)} safety examples")

    # 5. Concise responses
    print("Loading concise response examples...")
    concise_pool = list(_CONCISE_EXAMPLES)
    rng = random.Random(seed + 4)
    while len(concise_pool) < n_concise:
        base = rng.choice(_CONCISE_EXAMPLES)
        concise_pool.append(base)
    for ex in concise_pool[:n_concise]:
        raw_examples.append({
            "instruction": ex["instruction"], "output": ex["output"],
            "source": "concise"
        })
    print(f"  Loaded {min(len(concise_pool), n_concise)} concise examples")

    # Shuffle all raw examples
    rng = random.Random(seed)
    rng.shuffle(raw_examples)

    # Tokenize
    def tokenize_examples(examples):
        tokenized = []
        skipped = 0
        for ex in examples:
            instruction = ex["instruction"]
            response = ex["output"]
            prompt = ALPACA_TEMPLATE.format(instruction=instruction)
            prompt_tokens = tokenizer.encode(prompt)
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            full_tokens = prompt_tokens + response_tokens

            if len(full_tokens) < 10:
                skipped += 1
                continue
            if len(full_tokens) > max_seq_len:
                full_tokens = full_tokens[:max_seq_len]

            tokenized.append({
                "tokens": full_tokens,
                "prompt_len": len(prompt_tokens),
                "source": ex.get("source", "unknown"),
            })
        if skipped:
            print(f"  Skipped {skipped} examples (too short)")
        return tokenized

    # Split train/val
    n_total = len(raw_examples)
    train_raw = raw_examples[:n_train]
    val_raw = raw_examples[n_train:n_train + n_val]
    # If not enough for val, take from the end
    if len(val_raw) < n_val:
        val_raw = raw_examples[-n_val:]

    train_examples = tokenize_examples(train_raw)
    val_examples = tokenize_examples(val_raw)

    # Sort train by length for consistent memory usage
    train_examples.sort(key=lambda x: len(x["tokens"]))

    # Report composition
    sources = {}
    for ex in train_examples:
        s = ex.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1

    avg_len = sum(len(e["tokens"]) for e in train_examples) / max(len(train_examples), 1)
    print(f"\n  Train: {len(train_examples)} examples, avg length {avg_len:.0f} tokens")
    print(f"  Composition: {sources}")
    print(f"  Val: {len(val_examples)} examples")
    return train_examples, val_examples


def evaluate_loss(adapter, base_model, examples, loss_fn, mode="hidden"):
    """Compute average loss on examples."""
    lm_head = get_lm_head(base_model)
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
            logits = (base_logits + adapter(base_logits, h=h))[:, :-1, :]
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
