#!/usr/bin/env python3
"""rho-surgery: End-to-end behavioral repair pipeline for LLMs.

Runs the full Rho-Surgery pipeline: diagnose → SVD compress → LoRA SFT
with γ protection → verify → save repaired model.

Works on any platform:
  - Apple Silicon: uses MLX (fast, native GPU)
  - CUDA: uses PyTorch + CUDA
  - CPU: uses PyTorch on CPU (slow but portable)

Usage:
    rho-surgery Qwen/Qwen2.5-7B-Instruct -o ./repaired-7b/
    rho-surgery Qwen/Qwen2.5-7B-Instruct --strategy conservative -o ./repaired-7b/
    rho-surgery Qwen/Qwen2.5-7B-Instruct --plan plan.json -o ./repaired-7b/
    rho-surgery Qwen/Qwen2.5-7B-Instruct --no-save
    rho-surgery Qwen/Qwen2.5-7B-Instruct --device cuda -o ./repaired-7b/
"""

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="rho-surgery",
        description="Rho-Surgery: End-to-end behavioral repair pipeline for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run surgery, save repaired model
  rho-surgery Qwen/Qwen2.5-7B-Instruct -o ./repaired-7b/

  # Conservative strategy (stronger bias protection)
  rho-surgery Qwen/Qwen2.5-7B-Instruct --strategy conservative -o ./repaired-7b/

  # Override gamma weight
  rho-surgery Qwen/Qwen2.5-7B-Instruct --gamma 0.10 -o ./repaired-7b/

  # From a saved surgical plan
  rho-surgery Qwen/Qwen2.5-7B-Instruct --plan surgical_plan.json -o ./repaired-7b/

  # Metrics only (no model save)
  rho-surgery Qwen/Qwen2.5-7B-Instruct --no-save

  # Force CUDA backend (for cloud GPUs)
  rho-surgery Qwen/Qwen2.5-7B-Instruct --device cuda -o ./repaired-7b/
""",
    )

    # ── Positional ──────────────────────────────────────────────────
    parser.add_argument(
        "model",
        help="HuggingFace model ID or local path",
    )

    # ── Surgery config ──────────────────────────────────────────────
    surgery_group = parser.add_argument_group("Surgery configuration")
    surgery_group.add_argument(
        "--strategy", choices=["balanced", "conservative", "aggressive"],
        default="balanced",
        help="Protection strategy (default: balanced, gamma=0.10)",
    )
    surgery_group.add_argument(
        "--gamma", type=float, default=None,
        help="Override gamma protection weight (default: from strategy)",
    )
    surgery_group.add_argument(
        "--rho-weight", type=float, default=0.2,
        help="Contrastive loss weight (default: 0.2)",
    )
    surgery_group.add_argument(
        "--compress", type=float, default=0.7,
        help="SVD rank retention ratio (default: 0.7)",
    )
    surgery_group.add_argument(
        "--plan", type=str, default=None,
        help="Path to saved SurgicalPlan JSON (overrides strategy)",
    )

    # ── Output ──────────────────────────────────────────────────────
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output directory for saved model + results",
    )
    output_group.add_argument(
        "--no-save", action="store_true",
        help="Don't save model, only print metrics",
    )
    output_group.add_argument(
        "--format", choices=["table", "json", "markdown"],
        default="table",
        help="Output format (default: table)",
    )
    output_group.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "mlx", "cuda", "cpu"],
        help="Backend: auto (MLX on Apple Silicon, CUDA, CPU), mlx, cuda, cpu (default: auto)",
    )

    args = parser.parse_args(argv)

    # Validate: need output unless --no-save
    if not args.no_save and not args.output:
        parser.error("-o/--output is required (or use --no-save for metrics only)")

    # ── Run pipeline ────────────────────────────────────────────────
    t_global = time.time()

    print(f"\n{'='*60}")
    print(f"  rho-surgery: Behavioral Repair Pipeline")
    print(f"  Model:    {args.model}")
    print(f"  Strategy: {args.strategy}")
    if args.gamma is not None:
        print(f"  Gamma:    {args.gamma} (override)")
    print(f"  Compress: {args.compress}")
    print(f"  Device:   {args.device}")
    print(f"  Save:     {args.output or 'disabled'}")
    print(f"{'='*60}\n")

    # ── Step 1: Load model ──────────────────────────────────────────
    print("  Step 1: Loading model...", flush=True)
    from rho_eval.utils import load_model
    model, tokenizer, backend = load_model(args.model, device=args.device)
    print(f"  Model loaded (backend={backend}).", flush=True)

    # ── Step 2: Baseline audit ──────────────────────────────────────
    print("\n  Step 2: Baseline audit...", flush=True)
    from rho_eval import audit
    baseline_report = audit(model=model, tokenizer=tokenizer)
    print(f"  Baseline: mean_rho={baseline_report.mean_rho:.4f}, "
          f"status={baseline_report.overall_status}", flush=True)

    # ── Step 3: Generate surgical plan ──────────────────────────────
    print("\n  Step 3: Generating surgical plan...", flush=True)
    from rho_eval.surgical_planner import generate_surgical_plan, SurgicalPlan

    if args.plan:
        plan = SurgicalPlan.from_json(args.plan)
        print(f"  Loaded plan from {args.plan}", flush=True)
    else:
        plan = generate_surgical_plan(
            args.model,
            audit_report=baseline_report,
            strategy=args.strategy,
        )

    # Apply overrides
    if args.gamma is not None:
        plan.gamma_weight = args.gamma
    plan.rho_weight = args.rho_weight
    plan.compress_ratio = args.compress

    print(f"  Plan: compress={plan.compress_ratio}, rho={plan.rho_weight}, "
          f"gamma={plan.gamma_weight}", flush=True)
    print(f"  Protecting: {', '.join(plan.protection_categories)}", flush=True)

    # ── Step 4: SVD compression ─────────────────────────────────────
    print("\n  Step 4: SVD compression...", flush=True)
    if backend == "mlx":
        from rho_eval.alignment.mlx_trainer import mlx_svd_compress
        n_compressed = mlx_svd_compress(model, keep_ratio=plan.compress_ratio)
    else:
        from rho_eval.svd.compress import compress_qko
        n_compressed = compress_qko(model, ratio=plan.compress_ratio)
    print(f"  Compressed {n_compressed} matrices.", flush=True)

    # ── Step 5: LoRA SFT ────────────────────────────────────────────
    print("\n  Step 5: LoRA SFT with rho + gamma protection...", flush=True)
    from rho_eval.alignment.dataset import BehavioralContrastDataset

    # Build contrast dataset (sycophancy = target behavior)
    contrast_dataset = BehavioralContrastDataset(
        behaviors=["sycophancy"], seed=42,
    )

    # Build protection dataset (bias = protected behavior)
    protection_dataset = None
    if plan.gamma_weight > 0 and plan.protection_categories:
        protection_dataset = BehavioralContrastDataset(
            behaviors=["bias"],
            categories=list(plan.protection_categories),
            seed=42,
        )

    # Determine save path
    save_path = None
    if not args.no_save:
        save_path = str(Path(args.output) / "model")

    if backend == "mlx":
        # MLX SFT (Apple Silicon GPU)
        from rho_eval.alignment.mlx_trainer import mlx_rho_guided_sft

        sft_texts = [p for p, _ in contrast_dataset.pairs()]
        sft_result = mlx_rho_guided_sft(
            model, tokenizer,
            sft_texts,
            contrast_dataset=contrast_dataset,
            rho_weight=plan.rho_weight,
            gamma_weight=plan.gamma_weight,
            protection_dataset=protection_dataset,
            epochs=1,
            lr=2e-4,
            margin=0.1,
            save_path=save_path,
        )
    else:
        # PyTorch SFT (CUDA or CPU)
        from rho_eval.alignment.trainer import rho_guided_sft
        from rho_eval.calibration import TextDataset

        sft_texts = [p for p, _ in contrast_dataset.pairs()]
        sft_dataset = TextDataset(sft_texts, tokenizer)
        sft_result = rho_guided_sft(
            model, tokenizer,
            sft_dataset,
            contrast_dataset=contrast_dataset,
            rho_weight=plan.rho_weight,
            gamma_weight=plan.gamma_weight,
            protection_dataset=protection_dataset,
            epochs=1,
            lr=2e-4,
            margin=0.1,
            device=args.device if args.device != "auto" else "cpu",
        )

        # PyTorch: save model using HuggingFace save_pretrained
        if save_path is not None:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            # Get merged model from sft_result
            merged_model = sft_result.get("merged_model", model)
            merged_model.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            print(f"  Model saved to {save_dir}/", flush=True)

    print(f"  SFT complete: {sft_result['steps']} steps, "
          f"{sft_result['time']:.0f}s", flush=True)

    # ── Step 6: Post-surgery audit ──────────────────────────────────
    print("\n  Step 6: Post-surgery audit...", flush=True)

    # Audit the in-memory model (already fused)
    final_report = audit(model=model, tokenizer=tokenizer)
    print(f"  Final: mean_rho={final_report.mean_rho:.4f}, "
          f"status={final_report.overall_status}", flush=True)

    # ── Step 7: Compare & output ────────────────────────────────────
    print("\n  Step 7: Results", flush=True)

    # Build comparison
    print(f"\n  {'Behavior':<14s} {'Baseline':>9s} {'Final':>9s} {'Delta':>9s}")
    print(f"  {'─'*14}  {'─'*9}  {'─'*9}  {'─'*9}")
    for name in sorted(baseline_report.behaviors.keys()):
        base_rho = baseline_report.behaviors[name].rho
        final_rho = final_report.behaviors[name].rho
        delta = final_rho - base_rho
        marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else " ")
        print(f"  {name:<14s} {base_rho:>+9.4f} {final_rho:>+9.4f} "
              f"{delta:>+9.4f} {marker}")

    base_mean = baseline_report.mean_rho
    final_mean = final_report.mean_rho
    delta_mean = final_mean - base_mean
    print(f"  {'─'*14}  {'─'*9}  {'─'*9}  {'─'*9}")
    print(f"  {'MEAN':<14s} {base_mean:>+9.4f} {final_mean:>+9.4f} "
          f"{delta_mean:>+9.4f}")

    elapsed = time.time() - t_global

    # ── Save results ────────────────────────────────────────────────
    if not args.no_save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save surgical plan
        plan_path = output_dir / "surgical_plan.json"
        plan.to_json(str(plan_path))

        # Save full results
        result = {
            "model": args.model,
            "strategy": args.strategy,
            "backend": backend,
            "config": {
                "compress_ratio": plan.compress_ratio,
                "rho_weight": plan.rho_weight,
                "gamma_weight": plan.gamma_weight,
                "protection_categories": list(plan.protection_categories),
            },
            "baseline": baseline_report.to_dict(),
            "final": final_report.to_dict(),
            "sft": {
                "steps": sft_result["steps"],
                "time": sft_result["time"],
                "ce_loss": sft_result["ce_loss"],
                "rho_loss": sft_result["rho_loss"],
                "gamma_loss": sft_result["gamma_loss"],
            },
            "elapsed_seconds": round(elapsed, 1),
        }
        result_path = output_dir / "surgery_result.json"
        result_path.write_text(json.dumps(result, indent=2, default=str))

        print(f"\n  Saved:")
        print(f"    Model:   {save_path}/")
        print(f"    Plan:    {plan_path}")
        print(f"    Results: {result_path}")

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
