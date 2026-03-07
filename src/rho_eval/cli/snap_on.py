#!/usr/bin/env python3
"""snap-on: Train, generate, and evaluate Snap-On Communication Modules.

A tiny adapter that teaches instruction-following to a frozen base model
with zero knowledge damage. The logit-mode adapter operates purely on the
output distribution, never perturbing the knowledge pathway.

Subcommands:
    train     — Train a snap-on adapter on Alpaca data
    generate  — Generate text with a trained adapter
    eval      — Evaluate adapter MMLU accuracy vs base model

Usage:
    snap-on train    --model Qwen/Qwen2.5-7B --mode logit --save_dir results/snap_on/my_adapter
    snap-on generate --model Qwen/Qwen2.5-7B --adapter results/snap_on/my_adapter --prompt "What is the capital of France?"
    snap-on eval     --model Qwen/Qwen2.5-7B --adapter results/snap_on/my_adapter --mmlu_n 200
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def _json_serializer(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _load_base_model(model_id):
    """Load and freeze base model via MLX."""
    import mlx_lm

    print(f"Loading base model: {model_id}")
    base_model, tokenizer = mlx_lm.load(model_id)
    base_model.freeze()

    d_model = base_model.model.layers[0].self_attn.q_proj.weight.shape[0]
    # vocab_size: try lm_head first, fall back to model args or embed_tokens
    if hasattr(base_model, "lm_head"):
        vocab_size = base_model.lm_head.weight.shape[0]
    elif hasattr(base_model, "args") and hasattr(base_model.args, "vocab_size"):
        vocab_size = base_model.args.vocab_size
    else:
        vocab_size = base_model.model.embed_tokens.weight.shape[0]
    print(f"  d_model = {d_model}, vocab_size = {vocab_size}")

    return base_model, tokenizer, d_model, vocab_size


# ── Train subcommand ──────────────────────────────────────────────────

def _run_train(args):
    """Train a snap-on adapter."""
    from mlx.utils import tree_flatten

    from rho_eval.snap_on import (
        SnapOnConfig, create_adapter,
        load_alpaca_data, train, save_adapter,
        ALPACA_TEMPLATE,
        generate_with_adapter, generate_base_only,
        evaluate_mmlu,
    )

    base_model, tokenizer, d_model, vocab_size = _load_base_model(args.model)

    # Smart defaults based on mode
    d_inner = args.d_inner
    if d_inner is None:
        d_inner = 64 if args.mode == "logit" else 1024
        print(f"  (auto d_inner={d_inner} for {args.mode} mode)")

    lr = args.lr
    if lr is None:
        lr = 1e-5 if args.mode == "logit" else 1e-4
        print(f"  (auto lr={lr} for {args.mode} mode)")

    # Load data
    train_examples, val_examples = load_alpaca_data(
        tokenizer,
        n_train=args.n_train,
        n_val=args.n_val,
        max_seq_len=args.max_seq_len,
    )

    # Create adapter
    config = SnapOnConfig(
        d_model=d_model,
        d_inner=d_inner,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mode=args.mode,
        vocab_size=vocab_size,
    )
    adapter = create_adapter(config)

    n_params = sum(p.size for _, p in tree_flatten(adapter.parameters()))
    print(f"\nAdapter: mode={args.mode}")
    print(f"  Config: d_inner={config.d_inner}, n_layers={config.n_layers}")
    print(f"  Total params:     {n_params:>12,}")
    print(f"  Size:             {n_params * 4 / 1e6:.1f} MB (float32)")

    # Train
    best_val = train(
        base_model, tokenizer, adapter,
        train_examples, val_examples,
        epochs=args.epochs,
        lr=lr,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
        grad_accum=args.grad_accum,
        mode=args.mode,
    )

    # Sample generations
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function that checks if a number is prime.",
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

    # MMLU evaluation
    base_acc, adapter_acc = None, None
    if not args.skip_mmlu:
        base_acc, adapter_acc = evaluate_mmlu(
            base_model, adapter, tokenizer,
            n_questions=args.mmlu_n,
            mode=args.mode,
        )

    # Save results
    results = {
        "model": args.model,
        "mode": args.mode,
        "config": {
            "d_model": config.d_model,
            "d_inner": config.d_inner,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "vocab_size": config.vocab_size,
        },
        "n_params": n_params,
        "n_train": args.n_train,
        "epochs": args.epochs,
        "lr": lr,
        "best_val_loss": best_val,
    }
    if base_acc is not None:
        results["mmlu_base_acc"] = base_acc
        results["mmlu_adapter_acc"] = adapter_acc
        results["mmlu_delta"] = adapter_acc - base_acc

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=_json_serializer)
    print(f"\nResults saved to {args.save_dir}/results.json")


# ── Generate subcommand ───────────────────────────────────────────────

def _run_generate(args):
    """Generate text with a trained adapter."""
    from rho_eval.snap_on import (
        load_adapter, ALPACA_TEMPLATE,
        generate_with_adapter, generate_base_only,
    )

    base_model, tokenizer, _, _ = _load_base_model(args.model)

    # Load adapter
    print(f"Loading adapter from {args.adapter}...")
    adapter = load_adapter(args.adapter, "best")
    mode = adapter.config.mode
    print(f"  mode = {mode}")

    # Format prompt
    full_prompt = ALPACA_TEMPLATE.format(instruction=args.prompt)

    # Generate
    if args.compare:
        print(f"\n[BASE MODEL (no adapter)]:")
        base_out = generate_base_only(
            base_model, tokenizer, full_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(base_out)

    print(f"\n[WITH ADAPTER ({mode} mode)]:")
    adapter_out = generate_with_adapter(
        base_model, adapter, tokenizer, full_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        mode=mode,
    )
    print(adapter_out)


# ── Eval subcommand ───────────────────────────────────────────────────

def _run_eval(args):
    """Evaluate adapter MMLU accuracy vs base model."""
    from rho_eval.snap_on import load_adapter, evaluate_mmlu

    base_model, tokenizer, _, _ = _load_base_model(args.model)

    # Load adapter
    print(f"Loading adapter from {args.adapter}...")
    adapter = load_adapter(args.adapter, "best")
    mode = adapter.config.mode
    print(f"  mode = {mode}")

    # Evaluate
    base_acc, adapter_acc = evaluate_mmlu(
        base_model, adapter, tokenizer,
        n_questions=args.mmlu_n,
        mode=mode,
    )

    if base_acc is None:
        print("MMLU evaluation not available (missing rho_eval.unlock).",
              file=sys.stderr)
        sys.exit(1)

    # Save results
    if args.output:
        results = {
            "model": args.model,
            "adapter": args.adapter,
            "mode": mode,
            "mmlu_n": args.mmlu_n,
            "base_acc": base_acc,
            "adapter_acc": adapter_acc,
            "delta": adapter_acc - base_acc,
        }
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=_json_serializer)
        print(f"\nResults saved to {args.output}")


# ── CLI entry point ──────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="snap-on",
        description=(
            "Snap-On Communication Module — train, generate, and evaluate "
            "tiny adapters on frozen base models"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a logit-mode adapter (recommended — zero knowledge damage)
  snap-on train --model Qwen/Qwen2.5-7B --mode logit --save_dir results/snap_on/my_adapter

  # Train with custom hyperparameters
  snap-on train --model Qwen/Qwen2.5-7B --mode logit --d_inner 128 --lr 1e-5 --epochs 5 --save_dir results/snap_on/big

  # Generate with a trained adapter
  snap-on generate --model Qwen/Qwen2.5-7B --adapter results/snap_on/my_adapter --prompt "What is 2+2?"

  # Compare base vs adapter output
  snap-on generate --model Qwen/Qwen2.5-7B --adapter results/snap_on/my_adapter --prompt "Explain gravity" --compare

  # Evaluate MMLU accuracy (knowledge preservation check)
  snap-on eval --model Qwen/Qwen2.5-7B --adapter results/snap_on/my_adapter --mmlu_n 500
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # ── train ─────────────────────────────────────────────────────────
    train_p = subparsers.add_parser(
        "train",
        help="Train a snap-on adapter on Alpaca data",
        description="Train a tiny adapter to teach instruction-following to a frozen base model.",
    )
    # Model
    train_p.add_argument("--model", default="Qwen/Qwen2.5-7B",
                         help="Base model (HuggingFace ID or path, default: Qwen/Qwen2.5-7B)")
    # Mode
    train_p.add_argument("--mode", choices=["hidden", "logit"], default="logit",
                         help="hidden: adapter on h before lm_head; "
                              "logit: adapter on logits after lm_head (default: logit)")
    # Architecture
    train_p.add_argument("--d_inner", type=int, default=None,
                         help="Adapter internal dimension (auto: 64 for logit, 1024 for hidden)")
    train_p.add_argument("--n_layers", type=int, default=0,
                         help="Transformer layers (0 = MLP only, default: 0)")
    train_p.add_argument("--n_heads", type=int, default=8,
                         help="Attention heads for transformer variant (default: 8)")
    # Data
    train_p.add_argument("--n_train", type=int, default=10000,
                         help="Number of training examples (default: 10000)")
    train_p.add_argument("--n_val", type=int, default=500,
                         help="Number of validation examples (default: 500)")
    train_p.add_argument("--max_seq_len", type=int, default=512,
                         help="Maximum sequence length (default: 512)")
    # Training
    train_p.add_argument("--epochs", type=int, default=3,
                         help="Training epochs (default: 3)")
    train_p.add_argument("--lr", type=float, default=None,
                         help="Peak learning rate (auto: 1e-5 for logit, 1e-4 for hidden)")
    train_p.add_argument("--warmup_steps", type=int, default=100,
                         help="Linear warmup steps (default: 100)")
    train_p.add_argument("--grad_accum", type=int, default=1,
                         help="Gradient accumulation steps (default: 1)")
    train_p.add_argument("--log_every", type=int, default=50,
                         help="Print loss every N steps (default: 50)")
    train_p.add_argument("--eval_every", type=int, default=500,
                         help="Evaluate on val set every N steps (default: 500)")
    # Output
    train_p.add_argument("--save_dir", required=True,
                         help="Directory for checkpoints and results")
    # Eval
    train_p.add_argument("--skip_mmlu", action="store_true",
                         help="Skip MMLU evaluation after training")
    train_p.add_argument("--mmlu_n", type=int, default=200,
                         help="Number of MMLU questions (default: 200)")
    train_p.set_defaults(func=_run_train)

    # ── generate ──────────────────────────────────────────────────────
    gen_p = subparsers.add_parser(
        "generate",
        help="Generate text with a trained adapter",
        description="Generate instruction-following text using base model + adapter.",
    )
    gen_p.add_argument("--model", default="Qwen/Qwen2.5-7B",
                       help="Base model (default: Qwen/Qwen2.5-7B)")
    gen_p.add_argument("--adapter", required=True,
                       help="Path to adapter directory")
    gen_p.add_argument("--prompt", required=True,
                       help="Instruction prompt")
    gen_p.add_argument("--max_tokens", type=int, default=200,
                       help="Maximum tokens to generate (default: 200)")
    gen_p.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0 = greedy, default: 0.0)")
    gen_p.add_argument("--compare", action="store_true",
                       help="Also show base model output for comparison")
    gen_p.set_defaults(func=_run_generate)

    # ── eval ──────────────────────────────────────────────────────────
    eval_p = subparsers.add_parser(
        "eval",
        help="Evaluate MMLU accuracy with and without adapter",
        description="Measure knowledge preservation by comparing base vs adapter MMLU accuracy.",
    )
    eval_p.add_argument("--model", default="Qwen/Qwen2.5-7B",
                        help="Base model (default: Qwen/Qwen2.5-7B)")
    eval_p.add_argument("--adapter", required=True,
                        help="Path to adapter directory")
    eval_p.add_argument("--mmlu_n", type=int, default=200,
                        help="Number of MMLU questions (default: 200)")
    eval_p.add_argument("--output", "-o",
                        help="Save JSON results to this path")
    eval_p.set_defaults(func=_run_eval)

    # Parse
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
