"""Calibration data loading and gentle fine-tuning for post-compression recovery.

After SVD compression, unfrozen layers need a brief fine-tuning pass to adapt
to the modified weight matrices. This module provides:

  1. A mixed calibration dataset (instruction data + your own probes)
  2. A gentle_finetune() function using LoRA for memory-efficient recovery

The freeze ratio determines which layers get LoRA adapters:
  - Low freeze (0-50%): LoRA on many layers → better recovery of distributed traits
  - High freeze (75-100%): LoRA on few layers → denoising gains preserved

Motivation: Jaiswal et al. (2023, "Compressing LLMs: The Truth is Rarely Pure
and Never Simple", arXiv:2310.01382) showed that standard benchmarks miss
knowledge-intensive failures after compression. Fu et al. (2025, "TPLO",
arXiv:2509.00096) showed pruning degrades truthfulness detection. Our
calibration step uses factual probes in the recovery mix to protect against
exactly these failure modes.

Why LoRA instead of full FT:
  Full Adam on a 7B model needs ~56GB RAM (model + 2x Adam states + gradients).
  LoRA only trains ~0.1% of parameters → optimizer states are ~50MB, not 28GB.
  This makes CPU training feasible without swap thrashing.

Usage:
    from rho_eval.calibration import load_calibration_data, gentle_finetune

    calib = load_calibration_data(n=1000)
    gentle_finetune(model, tokenizer, calib, epochs=1, lr=2e-4)
"""

import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from .utils import DATA_DIR


CACHE_DIR = DATA_DIR / "calibration_cache"


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION DATASET
# ═══════════════════════════════════════════════════════════════════════════

class TextDataset(Dataset):
    """Simple text dataset for causal LM training."""

    def __init__(self, texts, tokenizer, max_length=256):
        self.encodings = []
        for text in texts:
            enc = tokenizer(
                text, truncation=True, max_length=max_length,
                padding="max_length", return_tensors="pt",
            )
            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": enc["input_ids"].squeeze(0).clone(),
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def load_calibration_data(
    n: int = 1000,
    seed: int = 42,
    include_probes: bool = True,
    include_instruct: bool = True,
    tokenizer=None,
    max_length: int = 256,
) -> "TextDataset":
    """Load a mixed calibration dataset for post-compression recovery FT.

    Sources:
      - Your own factual probes (true/false pairs) — ~112 texts
      - Alpaca instruct data (diverse, clean) — remaining budget

    Args:
        n: total number of examples
        seed: random seed
        include_probes: include your factual probes in the mix
        include_instruct: include instruction-following data
        tokenizer: tokenizer for encoding (required)
        max_length: max token length per example

    Returns:
        TextDataset ready for training
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for load_calibration_data")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    texts = []

    # ── Source 1: Your own probes (factual true/false pairs) ──────────
    if include_probes:
        from .probes import get_all_probes
        probes = get_all_probes()
        for p in probes:
            texts.append(p["text"])   # true version
            texts.append(p["false"])  # false version
        print(f"  [calib] {len(probes)*2} probe texts loaded")

    # ── Source 2: Alpaca instruction data ─────────────────────────────
    if include_instruct:
        remaining = n - len(texts)
        if remaining > 0:
            try:
                from datasets import load_dataset
                ds = load_dataset(
                    "tatsu-lab/alpaca", split="train",
                    cache_dir=str(CACHE_DIR),
                )
                # Format as instruction-response pairs
                instruct_texts = []
                for ex in ds:
                    if ex.get("input"):
                        text = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
                    else:
                        text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
                    instruct_texts.append(text)

                rng.shuffle(instruct_texts)
                texts.extend(instruct_texts[:remaining])
                print(f"  [calib] {min(remaining, len(instruct_texts))} Alpaca texts loaded")
            except Exception as e:
                print(f"  [calib] Alpaca load failed ({e}), using probes only")

    # Shuffle and trim
    rng.shuffle(texts)
    texts = texts[:n]
    print(f"  [calib] Total: {len(texts)} calibration texts")

    return TextDataset(texts, tokenizer, max_length=max_length)


# ═══════════════════════════════════════════════════════════════════════════
# GENTLE FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════════

def _get_trainable_layer_indices(model):
    """Identify which transformer layer indices have trainable (unfrozen) params.

    Returns list of integer layer indices where at least one param has requires_grad=True.
    This respects the prior freeze_layers() call.
    """
    from .utils import get_layers
    layers = get_layers(model)
    trainable_indices = []
    for i, layer in enumerate(layers):
        if any(p.requires_grad for p in layer.parameters()):
            trainable_indices.append(i)
    return trainable_indices


def _get_target_modules_for_lora(model):
    """Detect attention projection names in the model for LoRA targeting."""
    # Common attention projection patterns across architectures
    candidates = ["q_proj", "k_proj", "o_proj", "v_proj",
                   "qkv_proj", "out_proj", "query", "key", "value"]
    found = set()
    for name, _ in model.named_modules():
        for c in candidates:
            if name.endswith(c):
                # Only target Q, K, O (not V — V is sensitive per CF90 safety rules)
                if c not in ("v_proj", "value"):
                    found.add(c)
    # Fallback: at least target q_proj and k_proj
    if not found:
        found = {"q_proj", "k_proj", "o_proj"}
    return list(found)


def gentle_finetune(
    model,
    tokenizer,
    dataset: "TextDataset",
    epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 2,
    max_steps: Optional[int] = None,
    device: str = "cpu",
    gradient_accumulation_steps: int = 4,
    logging_steps: int = 50,
    lora_rank: int = 8,
    lora_alpha: int = 16,
) -> dict:
    """Run gentle LoRA fine-tuning on a compressed model.

    Uses LoRA (Low-Rank Adaptation) instead of full fine-tuning to keep
    memory feasible on CPU. Only applies LoRA adapters to layers that are
    NOT frozen (respects prior freeze_layers() call).

    Memory comparison (7B model):
      Full Adam FT:  ~56GB (model 14GB + Adam 28GB + grads 14GB)
      LoRA (rank 8): ~15GB (model 14GB + LoRA params ~10MB + Adam ~20MB)

    Args:
        model: compressed + frozen model
        tokenizer: tokenizer
        dataset: TextDataset from load_calibration_data()
        epochs: number of epochs (default: 1)
        lr: learning rate (default: 2e-4, typical for LoRA)
        batch_size: per-device batch size
        max_steps: if set, stop after this many steps (overrides epochs)
        device: device string for eval (training always on CPU)
        gradient_accumulation_steps: accumulate gradients over N steps
        logging_steps: log every N steps
        lora_rank: LoRA rank (default: 8)
        lora_alpha: LoRA alpha scaling (default: 16)

    Returns:
        dict with training stats (loss, steps, time)
    """
    import time
    import gc
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, TaskType

    total = sum(p.numel() for p in model.parameters())

    # Identify which layers are trainable (not frozen)
    trainable_indices = _get_trainable_layer_indices(model)
    if not trainable_indices:
        print(f"  [ft] WARNING: No trainable layers! Skipping FT.")
        return {"loss": 0.0, "steps": 0, "time": 0.0, "skipped": True}

    print(f"  [ft] Trainable layers: {trainable_indices} "
          f"({len(trainable_indices)} of {total/1e9:.1f}B params)")

    # Detect target modules
    target_modules = _get_target_modules_for_lora(model)
    print(f"  [ft] LoRA targets: {target_modules}, rank={lora_rank}, alpha={lora_alpha}")

    # Build layers_to_transform list for LoRA — only unfrozen layers
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        layers_to_transform=trainable_indices,
        bias="none",
    )

    t0 = time.time()

    # Move to CPU for training
    print(f"  [ft] Moving model to CPU for LoRA training...")
    model.cpu()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    gc.collect()

    # Unfreeze all before applying LoRA (LoRA manages its own trainability)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all base params

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    lora_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_total = sum(p.numel() for p in model.parameters())
    print(f"  [ft] LoRA trainable: {lora_trainable/1e6:.2f}M / {lora_total/1e6:.1f}M "
          f"({lora_trainable/lora_total:.4%})")
    print(f"  [ft] Config: epochs={epochs}, lr={lr}, batch={batch_size}, "
          f"grad_accum={gradient_accumulation_steps}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # Training arguments — force CPU
    training_args = TrainingArguments(
        output_dir="/tmp/kf_gentle_ft",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=logging_steps,
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,
        use_cpu=True,
        fp16=False,
        bf16=False,
        max_steps=max_steps if max_steps else -1,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    result = trainer.train()
    elapsed = time.time() - t0

    train_loss = result.training_loss if hasattr(result, 'training_loss') else 0.0
    steps = result.global_step if hasattr(result, 'global_step') else 0

    print(f"  [ft] LoRA done: loss={train_loss:.4f}, steps={steps}, time={elapsed:.1f}s")

    # Merge LoRA weights back into base model
    print(f"  [ft] Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # Clean up any residual peft metadata to avoid warnings on next LoRA application
    for attr in ("peft_config", "_hf_peft_config_loaded"):
        try:
            if hasattr(model, attr):
                object.__delattr__(model, attr)
        except (AttributeError, Exception):
            pass  # Some attrs may not be deletable on nn.Module subclasses

    # Free optimizer/trainer memory
    del trainer
    gc.collect()
    if device != "cpu":
        print(f"  [ft] Moving model back to {device}...")
        model.to(device)

    model.eval()

    print(f"  [ft] Done: loss={train_loss:.4f}, steps={steps}, time={elapsed:.1f}s")

    return {
        "loss": float(train_loss),
        "steps": int(steps),
        "time": float(elapsed),
        "trainable_params": int(lora_trainable),
        "trainable_pct": lora_trainable / lora_total,
        "lora_rank": lora_rank,
        "lora_layers": trainable_indices,
        "skipped": False,
        "method": "lora",
        "merged_model": model,  # IMPORTANT: caller must use this reference
    }


# ═══════════════════════════════════════════════════════════════════════════
# MLX GENTLE FINE-TUNING (Apple Silicon)
# ═══════════════════════════════════════════════════════════════════════════

def mlx_gentle_finetune(
    model,
    tokenizer,
    texts: list[str],
    epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 2,
    max_steps: Optional[int] = None,
    gradient_accumulation_steps: int = 4,
    logging_steps: int = 50,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    max_length: int = 256,
    verbose: bool = True,
) -> dict:
    """Run gentle LoRA fine-tuning on a compressed model — MLX backend.

    MLX equivalent of gentle_finetune(). Runs on Apple Silicon unified
    memory, avoiding PyTorch MPS NaN gradient bugs.

    Args:
        model: mlx-lm model (loaded via mlx_lm.load()).
        tokenizer: mlx-lm TokenizerWrapper.
        texts: Raw text strings for training (NOT a PyTorch Dataset).
        epochs: Number of epochs (default: 1).
        lr: Learning rate (default: 2e-4, typical for LoRA).
        batch_size: Per-step batch size.
        max_steps: Stop after this many steps (overrides epochs).
        gradient_accumulation_steps: Accumulate gradients over N steps.
        logging_steps: Log every N steps.
        lora_rank: LoRA rank (default: 8).
        lora_alpha: LoRA alpha scaling (default: 16).
        max_length: Maximum token length per example.
        verbose: Print progress messages.

    Returns:
        Dict with training stats (loss, steps, time, merged_model).
    """
    import math
    import time
    import gc

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as opt
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.lora import LoRALinear

    t0 = time.time()

    # ── LoRA Setup ────────────────────────────────────────────────────
    lora_config = {
        "rank": lora_rank,
        "scale": lora_alpha / lora_rank,
        "dropout": 0.05,
        "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj"],
    }

    num_layers = len(model.model.layers) if hasattr(model, 'model') else len(model.layers)
    linear_to_lora_layers(model, num_layers, lora_config)

    model.freeze()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.unfreeze(keys=["lora_a", "lora_b"])

    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))

    if verbose:
        print(f"  [mlx-ft] LoRA rank={lora_rank}, targets=Q,K,O")
        print(f"  [mlx-ft] Trainable: {trainable_params/1e6:.2f}M / "
              f"{total_params/1e6:.1f}M ({trainable_params/total_params:.4%})")

    # ── Tokenize ──────────────────────────────────────────────────────
    encoded = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        if len(tokens) >= 2:
            encoded.append(tokens)

    if verbose:
        print(f"  [mlx-ft] {len(encoded)} texts, epochs={epochs}, "
              f"lr={lr}, batch={batch_size}")

    pad_id = getattr(tokenizer, 'pad_token_id', None) or getattr(tokenizer, 'eos_token_id', 0)

    # ── Optimizer + Scheduler ─────────────────────────────────────────
    n_batches = math.ceil(len(encoded) / batch_size)
    steps_per_epoch = math.ceil(n_batches / gradient_accumulation_steps)
    total_steps = max_steps or (steps_per_epoch * epochs)
    warmup_steps = int(total_steps * 0.1)

    if warmup_steps > 0:
        warmup_fn = opt.linear_schedule(init=1e-8, end=lr, steps=warmup_steps)
        decay_fn = opt.linear_schedule(init=lr, end=0.0, steps=max(total_steps - warmup_steps, 1))
        schedule = opt.join_schedules([warmup_fn, decay_fn], [warmup_steps])
    else:
        schedule = opt.linear_schedule(init=lr, end=0.0, steps=total_steps)

    optimizer = opt.AdamW(learning_rate=schedule, weight_decay=0.01)

    # ── Loss function ─────────────────────────────────────────────────
    def ce_loss(model, batch_ids, batch_lengths):
        inputs = batch_ids[:, :-1]
        targets = batch_ids[:, 1:]
        logits = model(inputs)
        ce_per_token = nn.losses.cross_entropy(logits, targets)
        seq_pos = mx.arange(targets.shape[1])[None, :]
        mask = seq_pos < (batch_lengths[:, None] - 1)
        return (ce_per_token * mask).sum() / mx.maximum(mask.sum(), mx.array(1.0))

    loss_and_grad_fn = nn.value_and_grad(model, ce_loss)

    # ── Training Loop ─────────────────────────────────────────────────
    model.train()
    rng = __import__('random').Random(42)
    running_loss = 0.0
    global_step = 0
    log_count = 0
    accumulated_grad = None
    micro_step = 0
    done = False

    for epoch in range(epochs):
        indices = list(range(len(encoded)))
        rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_tokens = [encoded[i] for i in batch_indices]
            lengths = [len(t) for t in batch_tokens]
            max_len = max(lengths)
            padded = [t + [pad_id] * (max_len - len(t)) for t in batch_tokens]

            batch_ids = mx.array(padded)
            batch_lengths = mx.array(lengths)

            loss_val, grad = loss_and_grad_fn(model, batch_ids, batch_lengths)
            scaled_grad = tree_map(lambda g: g / gradient_accumulation_steps, grad)

            if accumulated_grad is None:
                accumulated_grad = scaled_grad
            else:
                accumulated_grad = tree_map(lambda a, b: a + b, accumulated_grad, scaled_grad)

            micro_step += 1
            running_loss += loss_val.item()

            if micro_step % gradient_accumulation_steps == 0:
                accumulated_grad, _ = opt.clip_grad_norm(accumulated_grad, max_norm=1.0)
                optimizer.update(model, accumulated_grad)
                mx.eval(model.parameters(), optimizer.state)
                accumulated_grad = None
                global_step += 1
                log_count += 1

                if verbose and global_step % logging_steps == 0:
                    avg_loss = running_loss / log_count
                    print(f"  [mlx-ft] step {global_step}/{total_steps}: "
                          f"loss={avg_loss:.4f}")
                    running_loss = 0.0
                    log_count = 0

                if max_steps and global_step >= max_steps:
                    done = True
                    break

        if done:
            break

    elapsed = time.time() - t0

    # ── Fuse LoRA ─────────────────────────────────────────────────────
    if verbose:
        print(f"  [mlx-ft] Fusing LoRA weights...")

    fused_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            fused_layers.append((name, module.fuse()))
    if fused_layers:
        model.update_modules(tree_unflatten(fused_layers))

    model.eval()

    avg_loss = running_loss / max(log_count, 1) if log_count > 0 else 0.0
    if verbose:
        print(f"  [mlx-ft] Done: loss={avg_loss:.4f}, steps={global_step}, "
              f"time={elapsed:.1f}s")

    return {
        "loss": float(avg_loss),
        "steps": int(global_step),
        "time": float(elapsed),
        "trainable_params": int(trainable_params),
        "trainable_pct": trainable_params / total_params if total_params > 0 else 0.0,
        "lora_rank": lora_rank,
        "skipped": False,
        "method": "mlx_lora",
        "merged_model": model,
    }
