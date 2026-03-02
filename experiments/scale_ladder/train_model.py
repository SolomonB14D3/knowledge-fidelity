#!/usr/bin/env python3
"""Train a small causal LM from scratch at a given scale.

Phase 1 of "Designed Geometry": builds the scale ladder by training
GPT-2 style models from 7M to 210M parameters on FineWeb-Edu.

Usage:
    python experiments/scale_ladder/train_model.py --size 7M --seed 42 --device cpu
    python experiments/scale_ladder/train_model.py --size 64M --seed 42 --device mps
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale_ladder.configs import SCALE_CONFIGS, SCALE_ORDER


def build_model(cfg):
    """Create a randomly-initialized GPT-2 model from config."""
    from transformers import GPT2Config, GPT2LMHeadModel

    model_config = GPT2Config(
        vocab_size=cfg["vocab_size"],
        n_positions=cfg["n_positions"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        # Standard GPT-2 defaults
        n_inner=4 * cfg["n_embd"],
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bos_token_id=50256,
        eos_token_id=50256,
    )
    model = GPT2LMHeadModel(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} params ({n_trainable:,} trainable)")
    return model, n_params


def build_tokenizer():
    """Load GPT-2 BPE tokenizer."""
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def packed_token_iterator(tokenizer, block_size, seed=42, dataset="openwebtext"):
    """Stream training data, tokenize, and yield packed token blocks.

    Concatenates documents with EOS separator, then chunks into
    fixed-size blocks. No padding waste.

    Supported datasets:
      - "openwebtext": OpenWebText (default, reliable streaming)
      - "fineweb-edu": FineWeb-Edu sample-10BT
      - "wikitext": WikiText-103 (small, fast, good for testing)
    """
    from datasets import load_dataset

    if dataset == "fineweb-edu":
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            split="train",
            streaming=True,
        )
        text_key = "text"
    elif dataset == "wikitext":
        ds = load_dataset(
            "wikitext",
            "wikitext-103-v1",
            split="train",
            streaming=True,
        )
        text_key = "text"
    else:  # openwebtext
        ds = load_dataset(
            "Skylion007/openwebtext",
            split="train",
            streaming=True,
        )
        text_key = "text"

    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    buffer = []
    eos_id = tokenizer.eos_token_id

    for example in ds:
        text = example[text_key]
        if not text or len(text.strip()) < 10:
            continue
        tokens = tokenizer.encode(text)
        buffer.extend(tokens)
        buffer.append(eos_id)

        while len(buffer) >= block_size:
            yield buffer[:block_size]
            buffer = buffer[block_size:]


def train(
    size: str,
    seed: int = 42,
    device_str: str = "cpu",
    output_dir: str | None = None,
    dataset: str = "openwebtext",
):
    """Train a model at the given scale."""
    if size not in SCALE_CONFIGS:
        print(f"  ERROR: Unknown size '{size}'. Choose from: {SCALE_ORDER}")
        return 1

    cfg = SCALE_CONFIGS[size]
    if output_dir is None:
        output_dir = f"results/scale_ladder/{size}_seed{seed}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Scale Ladder — Training {size}")
    print(f"  Config: {cfg['n_layer']}L, {cfg['n_head']}H, d={cfg['n_embd']}")
    print(f"  Target: {cfg['train_tokens']:,} tokens")
    print(f"  Device: {device_str}")
    print(f"  Seed: {seed}")
    print(f"{'='*60}\n")

    torch.manual_seed(seed)

    # Build model
    print("  Building model...", flush=True)
    model, n_params = build_model(cfg)
    device = torch.device(device_str)
    model = model.to(device)
    model.train()

    # Build tokenizer
    tokenizer = build_tokenizer()

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # LR schedule: linear warmup then cosine decay
    block_size = cfg["n_positions"]
    batch_size = cfg["batch_size"]
    tokens_per_step = block_size * batch_size
    total_steps = cfg["train_tokens"] // tokens_per_step
    warmup_steps = min(500, total_steps // 10)
    min_lr = cfg["lr"] * 0.1

    def get_lr(step):
        if step < warmup_steps:
            return cfg["lr"] * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr + 0.5 * (cfg["lr"] - min_lr) * (1 + math.cos(math.pi * progress))

    print(f"  Total steps: {total_steps:,} ({warmup_steps} warmup)")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Batch size: {batch_size}, Block size: {block_size}")
    print(f"  LR: {cfg['lr']} → {min_lr} (cosine)\n", flush=True)

    # Training loop
    data_iter = packed_token_iterator(tokenizer, block_size, seed=seed, dataset=dataset)
    loss_history = []
    tokens_seen = 0
    best_loss = float("inf")
    log_interval = 100
    save_interval = max(total_steps // 5, 1000)  # Save ~5 checkpoints

    for step in range(total_steps):
        # Collect batch
        batch_tokens = []
        for _ in range(batch_size):
            try:
                batch_tokens.append(next(data_iter))
            except StopIteration:
                # Restart data stream
                data_iter = packed_token_iterator(tokenizer, block_size, seed=seed + step, dataset=dataset)
                batch_tokens.append(next(data_iter))

        input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        labels = input_ids.clone()

        # Update LR
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + backward
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()

        tokens_seen += tokens_per_step
        loss_val = loss.item()
        loss_history.append({"step": step, "loss": loss_val, "lr": lr, "tokens": tokens_seen})

        if loss_val < best_loss:
            best_loss = loss_val

        # Log
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t_start
            tok_per_sec = tokens_seen / elapsed
            print(
                f"  step {step+1:>6d}/{total_steps} | "
                f"loss={loss_val:.4f} | lr={lr:.2e} | "
                f"{tok_per_sec:,.0f} tok/s | "
                f"{elapsed/60:.1f}min",
                flush=True,
            )

        # Save checkpoint
        if (step + 1) % save_interval == 0:
            ckpt_path = output_path / f"checkpoint_{step+1}"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

    # Save final model
    print(f"\n  Saving final model to {output_path / 'model'}/...", flush=True)
    model_path = output_path / "model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Save training metrics
    elapsed = time.time() - t_start
    metrics = {
        "size": size,
        "seed": seed,
        "device": device_str,
        "n_params": n_params,
        "config": cfg,
        "total_steps": total_steps,
        "tokens_seen": tokens_seen,
        "final_loss": loss_history[-1]["loss"] if loss_history else None,
        "best_loss": best_loss,
        "elapsed_seconds": round(elapsed, 1),
        "tokens_per_second": round(tokens_seen / elapsed, 1),
        "loss_history": loss_history[::max(len(loss_history) // 200, 1)],  # Subsample to ~200 points
    }
    metrics_path = output_path / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved metrics: {metrics_path}")

    print(f"\n  Training complete!")
    print(f"  Final loss: {metrics['final_loss']:.4f}")
    print(f"  Tokens: {tokens_seen:,}")
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f}h)")
    print(f"  Throughput: {metrics['tokens_per_second']:,.0f} tok/s")
    print(f"{'='*60}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="train-scale-ladder",
        description="Train a small causal LM at a given scale",
    )
    parser.add_argument("--size", required=True, choices=SCALE_ORDER,
                        help="Model size (e.g., 7M, 64M, 210M)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps", "cuda"],
                        help="Training device (default: cpu)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory (default: results/scale_ladder/{size}_seed{seed})")
    parser.add_argument("--dataset", type=str, default="openwebtext",
                        choices=["openwebtext", "fineweb-edu", "wikitext"],
                        help="Training dataset (default: openwebtext)")
    args = parser.parse_args()

    return train(
        size=args.size,
        seed=args.seed,
        device_str=args.device,
        output_dir=args.output,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
