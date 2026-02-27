"""CLI entry point for hybrid weight + activation control.

Usage:
    rho-hybrid Qwen/Qwen2.5-7B-Instruct
    rho-hybrid Qwen/Qwen2.5-7B-Instruct --compress 0.7 --freeze 0.75
    rho-hybrid Qwen/Qwen2.5-7B-Instruct --sae-layer 17 --target sycophancy
    rho-hybrid Qwen/Qwen2.5-7B-Instruct --rho-weight 0.2 --sft-epochs 1
    rho-hybrid Qwen/Qwen2.5-7B-Instruct --all   # full pipeline
"""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="rho-hybrid",
        description="Hybrid Weight + Activation Control for LLM behavioral repair",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Weight-space only (SVD + freeze)
  rho-hybrid Qwen/Qwen2.5-7B --compress 0.7 --freeze 0.75

  # Activation-space only (SAE steering on sycophancy)
  rho-hybrid Qwen/Qwen2.5-7B --sae-layer 17 --target sycophancy --scale 4.0

  # Training-time only (rho-guided SFT)
  rho-hybrid Qwen/Qwen2.5-7B --rho-weight 0.2

  # Full pipeline (all three surfaces)
  rho-hybrid Qwen/Qwen2.5-7B --compress 0.7 --freeze 0.75 \\
      --sae-layer 17 --target sycophancy --rho-weight 0.2

  # From config file
  rho-hybrid Qwen/Qwen2.5-7B --config hybrid_config.json
""",
    )

    # ── Positional ────────────────────────────────────────────────────────
    parser.add_argument(
        "model",
        help="HuggingFace model ID or local path",
    )

    # ── Config file (overrides all other flags) ───────────────────────────
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to HybridConfig JSON file (overrides all flags)",
    )

    # ── Weight-space control ──────────────────────────────────────────────
    weight_group = parser.add_argument_group("Weight-space control (SVD + freeze)")
    weight_group.add_argument(
        "--compress", type=float, default=0.7,
        help="SVD rank retention ratio (default: 0.7, 1.0 = skip)",
    )
    weight_group.add_argument(
        "--freeze", type=float, default=0.75,
        help="Fraction of layers to freeze from bottom (default: 0.75, 0.0 = skip)",
    )
    weight_group.add_argument(
        "--targets", type=str, default="q,k,o",
        help="Comma-separated projection targets (default: q,k,o)",
    )

    # ── Activation-space control ──────────────────────────────────────────
    act_group = parser.add_argument_group("Activation-space control (SAE steering)")
    act_group.add_argument(
        "--sae-layer", type=int, default=None,
        help="Layer index for SAE training (default: auto-detect)",
    )
    act_group.add_argument(
        "--sae-expansion", type=int, default=8,
        help="SAE hidden dimension multiplier (default: 8)",
    )
    act_group.add_argument(
        "--target", type=str, default="sycophancy",
        help="Comma-separated target behaviors for SAE steering (default: sycophancy)",
    )
    act_group.add_argument(
        "--scale", type=float, default=4.0,
        help="Steering vector magnitude (default: 4.0)",
    )

    # ── Training-time control ─────────────────────────────────────────────
    train_group = parser.add_argument_group("Training-time control (Rho-guided SFT)")
    train_group.add_argument(
        "--rho-weight", type=float, default=0.2,
        help="Contrastive loss weight (default: 0.2, 0.0 = skip)",
    )
    train_group.add_argument(
        "--sft-epochs", type=int, default=1,
        help="SFT epochs (default: 1)",
    )
    train_group.add_argument(
        "--sft-lr", type=float, default=2e-4,
        help="SFT learning rate (default: 2e-4)",
    )
    train_group.add_argument(
        "--margin", type=float, default=0.1,
        help="Contrastive loss margin (default: 0.1)",
    )

    # ── General ───────────────────────────────────────────────────────────
    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--eval-behaviors", type=str, default="all",
        help="Comma-separated behaviors to evaluate (default: all)",
    )
    general_group.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda, mps, cpu). Default: auto-detect",
    )
    general_group.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for results",
    )
    general_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    general_group.add_argument(
        "--trust-remote-code", action="store_true",
        help="Trust remote code when loading model",
    )

    args = parser.parse_args(argv)

    # ── Build config ──────────────────────────────────────────────────────
    from rho_eval.hybrid import HybridConfig

    if args.config:
        config = HybridConfig.from_json(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = HybridConfig(
            compress_ratio=args.compress,
            freeze_fraction=args.freeze,
            compress_targets=tuple(t.strip() for t in args.targets.split(",")),
            sae_layer=args.sae_layer,
            sae_expansion=args.sae_expansion,
            target_behaviors=tuple(b.strip() for b in args.target.split(",")),
            scale_factor=args.scale,
            rho_weight=args.rho_weight,
            sft_epochs=args.sft_epochs,
            sft_lr=args.sft_lr,
            margin=args.margin,
            eval_behaviors=tuple(b.strip() for b in args.eval_behaviors.split(",")),
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            seed=args.seed,
        )

    # ── Print banner ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  rho-hybrid: Hybrid Weight + Activation Control")
    print(f"  Model:      {args.model}")
    print(f"  Phases:     {', '.join(config.enabled_phases) or 'none'}")
    if config.weight_space_enabled:
        print(f"  Compress:   {config.compress_ratio} ({','.join(config.compress_targets)})")
        print(f"  Freeze:     {config.freeze_fraction}")
    if config.activation_space_enabled:
        print(f"  SAE layer:  {config.sae_layer} (expansion={config.sae_expansion})")
        print(f"  Targets:    {', '.join(config.target_behaviors)}")
        print(f"  Scale:      {config.scale_factor}")
    if config.training_time_enabled:
        print(f"  Rho weight: {config.rho_weight}")
        print(f"  SFT:        {config.sft_epochs} epoch(s), lr={config.sft_lr}")
    print(f"{'='*60}\n")

    # ── Run pipeline ──────────────────────────────────────────────────────
    from rho_eval.hybrid import apply_hybrid_control

    try:
        result = apply_hybrid_control(
            args.model, config,
            output_dir=args.output,
        )
    except NotImplementedError as e:
        print(f"\n⚠  {e}")
        print("\nThe hybrid pipeline is scaffolded but not yet implemented.")
        print("See src/rho_eval/hybrid/pipeline.py for the implementation plan.")
        sys.exit(1)

    # ── Output ────────────────────────────────────────────────────────────
    print(f"\n{result.to_table()}")
    print(f"\n{result.summary()}")

    if args.output:
        print(f"\nResults saved to {args.output}/hybrid_result.json")

    return result


if __name__ == "__main__":
    main()
