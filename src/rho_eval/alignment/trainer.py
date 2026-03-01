"""Custom training loop for rho-guided SFT.

Combines standard cross-entropy loss on SFT data with an auxiliary
contrastive loss on behavioral contrast pairs. Uses LoRA for
memory-efficient fine-tuning.

Why a custom loop instead of HF Trainer subclass:
  The HF Trainer.compute_loss() receives a single batch from a single
  dataset. We need to interleave batches from two datasets (SFT data
  and contrast pairs). A custom loop is cleaner and easier to debug
  for this research setting.

The training loop alternates:
  1. CE loss on an SFT batch (standard instruction-following)
  2. Contrastive loss on behavioral pairs (rho alignment signal)
  3. Combined gradient update

Usage:
    from rho_eval.alignment import rho_guided_sft, BehavioralContrastDataset
    from rho_eval.alignment.dataset import load_sft_dataset

    sft_data = load_sft_dataset(tokenizer, n=2000)
    contrast_data = BehavioralContrastDataset()
    result = rho_guided_sft(model, tokenizer, sft_data, contrast_data)
"""

from __future__ import annotations

import gc
import math
import random
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .losses import rho_auxiliary_loss


def rho_guided_sft(
    model,
    tokenizer,
    sft_dataset,
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
    lora_rank: int = 8,
    lora_alpha: int = 16,
    device: str = "cpu",
    logging_steps: int = 50,
    max_steps: Optional[int] = None,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_length: int = 256,
    verbose: bool = True,
) -> dict:
    """Run rho-guided SFT with combined CE + contrastive + protection loss.

    Training loop (Rho-Surgery loss):
      For each step:
        1. Get SFT batch → ce_loss = model(**batch).loss
        2. Sample contrast pairs → rho_loss = rho_auxiliary_loss(target_pairs)
        3. Sample protection pairs → gamma_loss = rho_auxiliary_loss(protected_pairs)
        4. total_loss = ce_loss + ρ * rho_loss + γ * gamma_loss
        5. Backward + accumulate gradients
        6. Every N steps: optimizer.step(), scheduler.step()

    When rho_weight=0, this is equivalent to standard SFT (CE only).
    When gamma_weight=0, this is equivalent to original rho-guided SFT.
    When gamma_weight>0, the γ term penalizes confidence drops on a
    protected behavior (e.g., bias) while training on the target (e.g.,
    sycophancy), preventing cross-behavior collateral damage.

    Args:
        model: HuggingFace CausalLM.
        tokenizer: Corresponding tokenizer.
        sft_dataset: TextDataset from load_sft_dataset() (for CE loss).
        contrast_dataset: BehavioralContrastDataset (for rho loss on target).
        rho_weight: Weight of the auxiliary rho loss (0 = CE only).
        gamma_weight: Weight of the γ protection loss (0 = no protection).
            The protection loss penalizes confidence drops on the protected
            behavior's contrast pairs during training.
        protection_dataset: BehavioralContrastDataset for the protected
            behavior (e.g., bias pairs when targeting sycophancy).
            Required when gamma_weight > 0.
        epochs: Number of training epochs.
        lr: Learning rate (default: 2e-4, typical for LoRA).
        batch_size: Per-step batch size for SFT data.
        gradient_accumulation_steps: Accumulate gradients over N steps.
        margin: Contrastive margin (in CE loss units).
        contrast_pairs_per_step: Number of contrast pairs per step.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        device: Torch device (training forced to CPU for LoRA).
        logging_steps: Print progress every N steps.
        max_steps: Stop after this many steps (overrides epochs).
        warmup_ratio: Fraction of steps for LR warmup.
        weight_decay: AdamW weight decay.
        max_length: Max token length for contrast pair encoding.
        verbose: Print progress messages.

    Returns:
        Dict with training stats and the merged model:
          {ce_loss, rho_loss, gamma_loss, total_loss, steps, time,
           trainable_params, trainable_pct, lora_rank,
           merged_model, method}
    """
    from peft import LoraConfig, get_peft_model, TaskType
    from ..calibration import _get_target_modules_for_lora

    t0 = time.time()

    # ── LoRA Setup ────────────────────────────────────────────────────
    target_modules = _get_target_modules_for_lora(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )

    # Determine training device.
    # MPS LoRA training deadlocks on Apple Silicon with large models (7B+)
    # during backward passes. Force CPU for MPS. The caller (hybrid pipeline)
    # handles moving the model back to MPS for inference after training.
    # CUDA training works fine.
    train_device = device if device else "cpu"
    if train_device == "mps":
        train_device = "cpu"

    if train_device != "cpu":
        if verbose:
            print(f"  [rho-sft] Training on {train_device}")
        model.to(train_device)
    else:
        if verbose:
            print(f"  [rho-sft] Moving model to CPU for LoRA training...")
        model.cpu()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        gc.collect()

    # Freeze all base params, then apply LoRA
    for param in model.parameters():
        param.requires_grad = False
    model = get_peft_model(model, lora_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"  [rho-sft] LoRA targets: {target_modules}, rank={lora_rank}")
        print(f"  [rho-sft] Trainable: {trainable_params/1e6:.2f}M / "
              f"{total_params/1e6:.1f}M ({trainable_params/total_params:.4%})")
        print(f"  [rho-sft] rho_weight={rho_weight}, margin={margin}, "
              f"contrast_pairs={contrast_pairs_per_step}")
        if gamma_weight > 0:
            n_prot = len(protection_dataset) if protection_dataset else 0
            print(f"  [rho-sft] gamma_weight={gamma_weight}, "
                  f"protection_pairs={n_prot}")

    # ── DataLoader ────────────────────────────────────────────────────
    from transformers import DataCollatorForLanguageModeling

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        sft_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    # ── Optimizer + Scheduler ─────────────────────────────────────────
    steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    total_steps = (max_steps or steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    # Linear warmup then linear decay
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if verbose:
        print(f"  [rho-sft] epochs={epochs}, lr={lr}, batch={batch_size}, "
              f"grad_accum={gradient_accumulation_steps}")
        print(f"  [rho-sft] total_steps={total_steps}, warmup={warmup_steps}")

    # ── Training Loop ─────────────────────────────────────────────────
    model.train()
    rng = random.Random(42)

    running_ce = 0.0
    running_rho = 0.0
    running_gamma = 0.0
    running_total = 0.0
    global_step = 0
    log_count = 0

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to training device
            batch = {k: v.to(train_device) for k, v in batch.items()}

            # ── CE loss on SFT data ───────────────────────────────
            outputs = model(**batch)
            ce_loss = outputs.loss / gradient_accumulation_steps

            # ── Rho auxiliary loss on contrast pairs ──────────────
            if rho_weight > 0 and len(contrast_dataset) > 0:
                pairs = contrast_dataset.sample(contrast_pairs_per_step, rng)
                rho_loss = rho_auxiliary_loss(
                    model, tokenizer, pairs,
                    margin=margin, device=train_device, max_length=max_length,
                )
                rho_loss_scaled = (rho_weight * rho_loss) / gradient_accumulation_steps
            else:
                rho_loss = torch.tensor(0.0)
                rho_loss_scaled = torch.tensor(0.0)

            # ── Gamma protection loss on protected pairs ───────────
            if gamma_weight > 0 and protection_dataset is not None and len(protection_dataset) > 0:
                prot_pairs = protection_dataset.sample(contrast_pairs_per_step, rng)
                gamma_loss = rho_auxiliary_loss(
                    model, tokenizer, prot_pairs,
                    margin=margin, device=train_device, max_length=max_length,
                )
                gamma_loss_scaled = (gamma_weight * gamma_loss) / gradient_accumulation_steps
            else:
                gamma_loss = torch.tensor(0.0)
                gamma_loss_scaled = torch.tensor(0.0)

            # ── Combined backward ─────────────────────────────────
            total_loss = ce_loss + rho_loss_scaled + gamma_loss_scaled
            total_loss.backward()

            # Track raw (unscaled) losses for logging
            running_ce += ce_loss.item() * gradient_accumulation_steps
            running_rho += rho_loss.item()
            running_gamma += gamma_loss.item()
            running_total += total_loss.item() * gradient_accumulation_steps

            # ── Optimizer step every N batches ────────────────────
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                log_count += 1

                if verbose and global_step % logging_steps == 0:
                    avg_ce = running_ce / log_count
                    avg_rho = running_rho / log_count
                    avg_gamma = running_gamma / log_count
                    avg_total = running_total / log_count
                    current_lr = scheduler.get_last_lr()[0]
                    gamma_str = f", γ={avg_gamma:.4f}" if gamma_weight > 0 else ""
                    print(f"  [rho-sft] step {global_step}/{total_steps}: "
                          f"ce={avg_ce:.4f}, ρ={avg_rho:.4f}{gamma_str}, "
                          f"total={avg_total:.4f}, lr={current_lr:.2e}")
                    running_ce = 0.0
                    running_rho = 0.0
                    running_gamma = 0.0
                    running_total = 0.0
                    log_count = 0

                if max_steps and global_step >= max_steps:
                    break

        if max_steps and global_step >= max_steps:
            break

    # Final stats
    elapsed = time.time() - t0

    if verbose:
        print(f"  [rho-sft] Training done: {global_step} steps, {elapsed:.1f}s")

    # ── Merge LoRA + Cleanup ──────────────────────────────────────────
    if verbose:
        print(f"  [rho-sft] Merging LoRA weights into base model...")

    model = model.merge_and_unload()

    # Clean up peft metadata (same pattern as gentle_finetune)
    for attr in ("peft_config", "_hf_peft_config_loaded"):
        try:
            if hasattr(model, attr):
                object.__delattr__(model, attr)
        except (AttributeError, Exception):
            pass

    del optimizer, scheduler
    gc.collect()

    if device != "cpu":
        if verbose:
            print(f"  [rho-sft] Moving model back to {device}...")
        model.to(device)

    model.eval()

    if verbose:
        print(f"  [rho-sft] Done: {global_step} steps, {elapsed:.1f}s")

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
        "method": "rho_guided_sft",
        "merged_model": model,
    }
