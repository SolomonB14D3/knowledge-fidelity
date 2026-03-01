"""Custom training loop for rho-guided SFT — MLX backend.

MLX equivalent of trainer.py. Runs on Apple Silicon unified memory,
avoiding the MPS NaN gradient bugs that force the PyTorch version to CPU.

Which backend should I use?
  ┌─────────────────────────┬──────────────────────────────────────┐
  │ Use MLX (this file)     │ Use PyTorch (trainer.py)             │
  ├─────────────────────────┼──────────────────────────────────────┤
  │ Apple Silicon Mac       │ Linux / Windows / CUDA GPUs          │
  │ M1/M2/M3/M4 chips      │ Non-Apple hardware                   │
  │ ~10× faster than CPU PT │ Supports MPS, CUDA, CPU              │
  │ Unified memory (no OOM) │ Full audit() integration (PyTorch)   │
  │ pip install mlx mlx-lm  │ pip install peft transformers        │
  └─────────────────────────┴──────────────────────────────────────┘

  Both backends produce the same training signal (CE + contrastive loss)
  with the same LoRA targets (Q, K, O projections). Results should be
  statistically equivalent within normal training variance.

Key differences from PyTorch version:
  - Uses mlx-lm's LoRA infrastructure instead of peft
  - Gradients via nn.value_and_grad() instead of .backward()
  - No device management (MLX uses unified memory)
  - Manual gradient accumulation via tree_map
  - SFT data passed as raw text strings, tokenized on-the-fly

Usage:
    from rho_eval.alignment.mlx_trainer import mlx_rho_guided_sft
    from rho_eval.alignment.dataset import BehavioralContrastDataset

    # model, tokenizer = mlx_lm.load("Qwen/Qwen2.5-7B-Instruct")
    sft_texts = ["### Instruction:\\n...\\n### Response:\\n...", ...]
    contrast_data = BehavioralContrastDataset()
    result = mlx_rho_guided_sft(model, tokenizer, sft_texts, contrast_data)
"""

from __future__ import annotations

import gc
import math
import random
import time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .mlx_losses import mlx_rho_auxiliary_loss


# ── Batching ────────────────────────────────────────────────────────────

def _tokenize_and_pad(
    texts: list[str],
    tokenizer,
    max_length: int = 256,
) -> list[list[int]]:
    """Tokenize texts, truncating to max_length."""
    encoded = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        if len(tokens) >= 2:  # need at least 2 tokens for CE
            encoded.append(tokens)
    return encoded


def _iterate_batches(
    encoded: list[list[int]],
    batch_size: int,
    pad_token_id: int,
    shuffle: bool = True,
    rng: random.Random | None = None,
):
    """Yield (input_ids, lengths) batches from pre-tokenized texts.

    Pads each batch to the longest sequence in that batch.

    Args:
        encoded: List of token ID lists.
        batch_size: Batch size.
        pad_token_id: Token ID for padding.
        shuffle: Whether to shuffle.
        rng: Random instance for reproducibility.

    Yields:
        Tuple of (batch_ids: mx.array, lengths: mx.array)
        batch_ids shape: (batch_size, max_seq_len_in_batch)
        lengths shape: (batch_size,)
    """
    indices = list(range(len(encoded)))
    if shuffle and rng:
        rng.shuffle(indices)
    elif shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_tokens = [encoded[i] for i in batch_indices]
        lengths = [len(t) for t in batch_tokens]
        max_len = max(lengths)

        # Pad to max length in this batch
        padded = []
        for tokens in batch_tokens:
            padded.append(tokens + [pad_token_id] * (max_len - len(tokens)))

        batch_ids = mx.array(padded)       # (B, max_len)
        batch_lengths = mx.array(lengths)   # (B,)

        yield batch_ids, batch_lengths


# ── LoRA Utilities ──────────────────────────────────────────────────────

def _apply_lora(model, lora_rank: int, lora_alpha: int, dropout: float = 0.05):
    """Apply LoRA adapters to Q, K, O projections (not V — CF90 safety).

    Uses mlx-lm's built-in linear_to_lora_layers infrastructure.

    Returns:
        Tuple of (trainable_params, total_params).
    """
    from mlx_lm.tuner.utils import linear_to_lora_layers

    # Target Q, K, O attention projections (not V — per CF90 safety rules)
    lora_config = {
        "rank": lora_rank,
        "scale": lora_alpha / lora_rank,  # mlx-lm uses scale = alpha/rank
        "dropout": dropout,
        "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj"],
    }

    # Apply to all transformer layers
    num_layers = len(model.model.layers) if hasattr(model, 'model') else len(model.layers)
    linear_to_lora_layers(model, num_layers, lora_config)

    # Freeze everything, then unfreeze only LoRA params
    model.freeze()

    # Unfreeze LoRA parameters (lora_a, lora_b)
    from mlx_lm.tuner.lora import LoRALinear
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.unfreeze(keys=["lora_a", "lora_b"])

    # Count parameters
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    trainable_params = sum(
        p.size for _, p in tree_flatten(model.trainable_parameters())
    )

    return trainable_params, total_params


def _fuse_lora(model):
    """Fuse LoRA weights into base model (merge and unload equivalent).

    After fusing, LoRALinear layers become regular nn.Linear layers.
    """
    from mlx_lm.tuner.lora import LoRALinear

    fused_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            fused_layers.append((name, module.fuse()))

    if fused_layers:
        model.update_modules(tree_unflatten(fused_layers))


# ── Training Loop ───────────────────────────────────────────────────────

def mlx_rho_guided_sft(
    model,
    tokenizer,
    sft_texts: list[str],
    contrast_dataset,
    rho_weight: float = 0.2,
    gamma_weight: float = 0.0,
    protection_dataset=None,
    epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    margin: float = 0.1,
    contrast_pairs_per_step: int = 4,
    protection_pairs_per_step: int = 4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    logging_steps: int = 50,
    max_steps: Optional[int] = None,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_length: int = 256,
    verbose: bool = True,
) -> dict:
    """Run rho-guided SFT with combined CE + contrastive + protection loss on MLX.

    Training loop:
      For each step:
        1. Get SFT batch → ce_loss (teacher-forced CE)
        2. Sample contrast pairs → rho_loss (contrastive confidence loss)
        3. Sample protection pairs → gamma_loss (protection confidence loss)
        4. total_loss = ce_loss + rho_weight * rho_loss + gamma_weight * gamma_loss
        5. Gradients via value_and_grad + accumulate
        6. Every N steps: optimizer.update(), mx.eval()

    The γ (gamma) protection loss prevents collateral damage on protected
    behaviors (e.g., bias) during target behavior SFT (e.g., sycophancy).
    It uses the same contrastive structure as the ρ loss but with different
    contrast data — bias pairs instead of sycophancy pairs.

    When rho_weight=0, this is equivalent to standard SFT (CE only).
    When gamma_weight=0, this is equivalent to the original rho-guided SFT.

    Args:
        model: mlx-lm model (loaded via mlx_lm.load()).
        tokenizer: mlx-lm TokenizerWrapper.
        sft_texts: List of raw text strings for SFT training.
        contrast_dataset: BehavioralContrastDataset for target behavior (sycophancy).
        rho_weight: Weight of the auxiliary rho loss (0 = CE only).
        gamma_weight: Weight of the γ protection loss (0 = no protection).
        protection_dataset: BehavioralContrastDataset for protected behaviors (bias).
            Required when gamma_weight > 0.
        epochs: Number of training epochs.
        lr: Learning rate (default: 2e-4, typical for LoRA).
        batch_size: Per-step batch size for SFT data.
        gradient_accumulation_steps: Accumulate gradients over N steps.
        margin: Contrastive margin (in CE loss units).
        contrast_pairs_per_step: Number of contrast pairs per step (for rho).
        protection_pairs_per_step: Number of protection pairs per step (for gamma).
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        logging_steps: Print progress every N steps.
        max_steps: Stop after this many steps (overrides epochs).
        warmup_ratio: Fraction of steps for LR warmup.
        weight_decay: AdamW weight decay.
        max_length: Max token length for encoding.
        verbose: Print progress messages.

    Returns:
        Dict with training stats and the fused model:
          {ce_loss, rho_loss, gamma_loss, total_loss, steps, time,
           trainable_params, trainable_pct, lora_rank,
           merged_model, method}
    """
    t0 = time.time()

    # ── LoRA Setup ────────────────────────────────────────────────────
    if verbose:
        print(f"  [mlx-rho-sft] Applying LoRA (rank={lora_rank}, "
              f"alpha={lora_alpha})...")

    trainable_params, total_params = _apply_lora(
        model, lora_rank, lora_alpha,
    )

    if verbose:
        print(f"  [mlx-rho-sft] Trainable: {trainable_params/1e6:.2f}M / "
              f"{total_params/1e6:.1f}M ({trainable_params/total_params:.4%})")
        print(f"  [mlx-rho-sft] rho_weight={rho_weight}, gamma_weight={gamma_weight}, "
              f"margin={margin}, contrast_pairs={contrast_pairs_per_step}")
        if gamma_weight > 0 and protection_dataset is not None:
            print(f"  [mlx-rho-sft] γ protection: {len(protection_dataset)} pairs, "
                  f"{protection_pairs_per_step}/step")

    # ── Tokenize SFT data ─────────────────────────────────────────────
    if verbose:
        print(f"  [mlx-rho-sft] Tokenizing {len(sft_texts)} SFT texts...")

    encoded = _tokenize_and_pad(sft_texts, tokenizer, max_length)

    if verbose:
        print(f"  [mlx-rho-sft] {len(encoded)} texts after filtering")

    # Determine pad token
    pad_token_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, 'eos_token_id', 0)

    # ── Optimizer + Scheduler ─────────────────────────────────────────
    n_batches = math.ceil(len(encoded) / batch_size)
    steps_per_epoch = math.ceil(n_batches / gradient_accumulation_steps)
    total_steps = max_steps or (steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    # Build schedule: linear warmup then linear decay
    if warmup_steps > 0:
        warmup_fn = opt.linear_schedule(
            init=1e-8, end=lr, steps=warmup_steps,
        )
        decay_fn = opt.linear_schedule(
            init=lr, end=0.0,
            steps=max(total_steps - warmup_steps, 1),
        )
        schedule = opt.join_schedules(
            [warmup_fn, decay_fn],
            [warmup_steps],
        )
    else:
        schedule = opt.linear_schedule(init=lr, end=0.0, steps=total_steps)

    optimizer = opt.AdamW(
        learning_rate=schedule,
        weight_decay=weight_decay,
    )

    if verbose:
        print(f"  [mlx-rho-sft] epochs={epochs}, lr={lr}, batch={batch_size}, "
              f"grad_accum={gradient_accumulation_steps}")
        print(f"  [mlx-rho-sft] total_steps={total_steps}, warmup={warmup_steps}")

    # ── Loss function for value_and_grad ──────────────────────────────
    def sft_loss_fn(model, batch_ids, batch_lengths):
        """CE loss on a padded SFT batch, with length masking."""
        inputs = batch_ids[:, :-1]    # (B, L-1)
        targets = batch_ids[:, 1:]    # (B, L-1)

        logits = model(inputs)        # (B, L-1, V)

        # Per-token CE
        ce_per_token = nn.losses.cross_entropy(logits, targets)  # (B, L-1)

        # Mask padding tokens: only count positions < length-1
        seq_positions = mx.arange(targets.shape[1])[None, :]  # (1, L-1)
        mask = seq_positions < (batch_lengths[:, None] - 1)    # (B, L-1)

        # Masked mean
        ce_sum = (ce_per_token * mask).sum()
        n_tokens = mask.sum()
        ce_loss = ce_sum / mx.maximum(n_tokens, mx.array(1.0))

        return ce_loss, n_tokens

    loss_and_grad_fn = nn.value_and_grad(model, sft_loss_fn)

    # ── Training Loop ─────────────────────────────────────────────────
    model.train()
    rng = random.Random(42)

    running_ce = 0.0
    running_rho = 0.0
    running_gamma = 0.0
    running_total = 0.0
    global_step = 0
    log_count = 0
    accumulated_grad = None
    micro_step = 0
    done = False

    for epoch in range(epochs):
        for batch_ids, batch_lengths in _iterate_batches(
            encoded, batch_size, pad_token_id, shuffle=True, rng=rng,
        ):
            # ── CE loss on SFT batch ───────────────────────────────
            (ce_loss_val, n_tokens), grad = loss_and_grad_fn(
                model, batch_ids, batch_lengths,
            )

            # Scale for gradient accumulation
            scaled_grad = tree_map(
                lambda g: g / gradient_accumulation_steps, grad,
            )

            # ── Rho auxiliary loss on contrast pairs ──────────────
            if rho_weight > 0 and len(contrast_dataset) > 0:
                pairs = contrast_dataset.sample(contrast_pairs_per_step, rng)

                # Compute rho loss and its gradient separately
                def rho_loss_fn(model):
                    return mlx_rho_auxiliary_loss(
                        model, tokenizer, pairs,
                        margin=margin, max_length=max_length,
                    )

                rho_grad_fn = nn.value_and_grad(model, rho_loss_fn)
                rho_loss_val, rho_grad = rho_grad_fn(model)

                # Scale rho gradient by weight and accumulation
                rho_scaled = tree_map(
                    lambda g: g * (rho_weight / gradient_accumulation_steps),
                    rho_grad,
                )

                # Add rho gradient to CE gradient
                scaled_grad = tree_map(
                    lambda a, b: a + b, scaled_grad, rho_scaled,
                )
            else:
                rho_loss_val = mx.array(0.0)

            # ── Gamma protection loss on protected pairs ──────────
            if gamma_weight > 0 and protection_dataset is not None and len(protection_dataset) > 0:
                prot_pairs = protection_dataset.sample(protection_pairs_per_step, rng)

                # Compute gamma loss and its gradient separately
                def gamma_loss_fn(model):
                    return mlx_rho_auxiliary_loss(
                        model, tokenizer, prot_pairs,
                        margin=margin, max_length=max_length,
                    )

                gamma_grad_fn = nn.value_and_grad(model, gamma_loss_fn)
                gamma_loss_val, gamma_grad = gamma_grad_fn(model)

                # Scale gamma gradient by weight and accumulation
                gamma_scaled = tree_map(
                    lambda g: g * (gamma_weight / gradient_accumulation_steps),
                    gamma_grad,
                )

                # Add gamma gradient to combined gradient
                scaled_grad = tree_map(
                    lambda a, b: a + b, scaled_grad, gamma_scaled,
                )
            else:
                gamma_loss_val = mx.array(0.0)

            # ── Accumulate ────────────────────────────────────────
            if accumulated_grad is None:
                accumulated_grad = scaled_grad
            else:
                accumulated_grad = tree_map(
                    lambda a, b: a + b, accumulated_grad, scaled_grad,
                )

            micro_step += 1

            # Track losses (evaluate lazily)
            ce_val = ce_loss_val.item()
            rho_val = rho_loss_val.item()
            gamma_val = gamma_loss_val.item()
            running_ce += ce_val
            running_rho += rho_val
            running_gamma += gamma_val
            running_total += ce_val + rho_weight * rho_val + gamma_weight * gamma_val

            # ── Optimizer step every N micro-steps ────────────────
            if micro_step % gradient_accumulation_steps == 0:
                # Clip gradients
                accumulated_grad, _ = opt.clip_grad_norm(
                    accumulated_grad, max_norm=1.0,
                )

                # Update model
                optimizer.update(model, accumulated_grad)

                # Evaluate (materialize lazy computation)
                mx.eval(model.parameters(), optimizer.state)

                accumulated_grad = None
                global_step += 1
                log_count += 1

                if verbose and global_step % logging_steps == 0:
                    avg_ce = running_ce / log_count
                    avg_rho = running_rho / log_count
                    avg_gamma = running_gamma / log_count
                    avg_total = running_total / log_count
                    elapsed = time.time() - t0
                    gamma_str = f", gamma={avg_gamma:.4f}" if gamma_weight > 0 else ""
                    print(f"  [mlx-rho-sft] step {global_step}/{total_steps}: "
                          f"ce={avg_ce:.4f}, rho={avg_rho:.4f}{gamma_str}, "
                          f"total={avg_total:.4f}, "
                          f"elapsed={elapsed:.0f}s", flush=True)
                    running_ce = 0.0
                    running_rho = 0.0
                    running_gamma = 0.0
                    running_total = 0.0
                    log_count = 0

                if max_steps and global_step >= max_steps:
                    done = True
                    break

        if done:
            break

    # Final stats
    elapsed = time.time() - t0

    if verbose:
        print(f"  [mlx-rho-sft] Training done: {global_step} steps, "
              f"{elapsed:.1f}s")

    # ── Fuse LoRA + Cleanup ───────────────────────────────────────────
    if verbose:
        print(f"  [mlx-rho-sft] Fusing LoRA weights into base model...")

    _fuse_lora(model)

    model.eval()

    if verbose:
        print(f"  [mlx-rho-sft] Done: {global_step} steps, {elapsed:.1f}s")

    return {
        "ce_loss": running_ce / max(log_count, 1) if log_count > 0 else 0.0,
        "rho_loss": running_rho / max(log_count, 1) if log_count > 0 else 0.0,
        "gamma_loss": running_gamma / max(log_count, 1) if log_count > 0 else 0.0,
        "total_loss": running_total / max(log_count, 1) if log_count > 0 else 0.0,
        "steps": global_step,
        "time": elapsed,
        "trainable_params": trainable_params,
        "trainable_pct": trainable_params / total_params if total_params > 0 else 0.0,
        "lora_rank": lora_rank,
        "rho_weight": rho_weight,
        "gamma_weight": gamma_weight,
        "margin": margin,
        "method": "mlx_rho_guided_sft",
        "merged_model": model,
    }
