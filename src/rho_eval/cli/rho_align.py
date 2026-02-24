"""CLI entry point for rho-guided alignment (SFT with auxiliary behavioral loss).

Usage:
    rho-align Qwen/Qwen2.5-0.5B
    rho-align Qwen/Qwen2.5-7B-Instruct --rho-weight 0.2 --epochs 1
    rho-align model/ --behaviors factual,toxicity --baseline-only
    rho-align model/ --rho-weight 0 --epochs 1   # standard SFT (CE only)
"""

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="rho-align",
        description="Rho-guided SFT: fine-tune with auxiliary behavioral loss.",
    )

    parser.add_argument(
        "model",
        nargs="?",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--rho-weight", type=float, default=0.2,
        help="Weight of auxiliary rho loss (0 = CE only, default: 0.2)",
    )
    parser.add_argument(
        "--margin", type=float, default=0.1,
        help="Contrastive margin in CE loss units (default: 0.1)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Training epochs (default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--sft-size", type=int, default=2000,
        help="Number of SFT training examples (default: 2000)",
    )
    parser.add_argument(
        "--behaviors", "-b", default="factual,toxicity,sycophancy,bias",
        help="Comma-separated behaviors for contrast pairs (default: all 4)",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only evaluate baseline (no SFT)",
    )
    parser.add_argument(
        "--format", "-f", choices=["json", "table"], default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (default: auto-detect)",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Show version and exit",
    )

    args = parser.parse_args()

    if args.version:
        try:
            from rho_eval import __version__
            print(f"rho-align (rho-eval {__version__})")
        except ImportError:
            print("rho-align (rho-eval)")
        return

    if args.model is None:
        parser.error("Model name or path is required.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rho_eval.audit import audit
    from rho_eval.utils import get_device
    from rho_eval.alignment.dataset import (
        load_sft_dataset,
        BehavioralContrastDataset,
    )
    from rho_eval.alignment.trainer import rho_guided_sft

    # ── Setup ─────────────────────────────────────────────────────────
    device = args.device or str(get_device())
    behaviors = [b.strip() for b in args.behaviors.split(",")]
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  rho-align: Rho-Guided SFT")
    print(f"  Model:      {args.model}")
    print(f"  rho_weight: {args.rho_weight}")
    print(f"  Behaviors:  {behaviors}")
    print(f"  Device:     {device}")
    print(f"{'='*60}\n")

    # ── Load Model ────────────────────────────────────────────────────
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    # ── Baseline Evaluation ───────────────────────────────────────────
    print("\n[1/3] Evaluating baseline (pre-SFT)...")
    baseline_report = audit(
        model=model, tokenizer=tokenizer,
        behaviors="all", device=device,
    )
    baseline_scores = {
        bname: r.rho for bname, r in baseline_report.behaviors.items()
    }
    print(f"  Baseline rho scores:")
    for bname, rho in baseline_scores.items():
        print(f"    {bname:12s}: {rho:.4f}")

    if args.baseline_only:
        _print_results({"baseline": baseline_scores}, args)
        return

    # Save original state for restore between conditions
    print("\n  Saving model state for restore...")
    original_state = copy.deepcopy(model.state_dict())

    # ── Prepare Data ──────────────────────────────────────────────────
    print("\nPreparing datasets...")
    sft_data = load_sft_dataset(
        tokenizer, n=args.sft_size,
        include_traps=True, seed=42,
    )
    contrast_data = BehavioralContrastDataset(
        behaviors=behaviors, seed=42,
    )

    results = {"baseline": baseline_scores}

    # ── Standard SFT (CE only) ────────────────────────────────────────
    print("\n[2/3] Running standard SFT (rho_weight=0)...")
    model.load_state_dict(original_state)
    model.to(device)

    std_result = rho_guided_sft(
        model, tokenizer, sft_data, contrast_data,
        rho_weight=0.0,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        margin=args.margin,
        device=device,
    )
    model = std_result["merged_model"]

    print("\n  Evaluating standard SFT...")
    std_report = audit(
        model=model, tokenizer=tokenizer,
        behaviors="all", device=device,
    )
    std_scores = {bname: r.rho for bname, r in std_report.behaviors.items()}
    results["standard_sft"] = std_scores
    print(f"  Standard SFT rho scores:")
    for bname, rho in std_scores.items():
        delta = rho - baseline_scores.get(bname, 0)
        print(f"    {bname:12s}: {rho:.4f} ({delta:+.4f})")

    # ── Rho-Guided SFT ───────────────────────────────────────────────
    print(f"\n[3/3] Running rho-guided SFT (rho_weight={args.rho_weight})...")
    model.load_state_dict(original_state)
    model.to(device)

    rho_result = rho_guided_sft(
        model, tokenizer, sft_data, contrast_data,
        rho_weight=args.rho_weight,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        margin=args.margin,
        device=device,
    )
    model = rho_result["merged_model"]

    print("\n  Evaluating rho-guided SFT...")
    rho_report = audit(
        model=model, tokenizer=tokenizer,
        behaviors="all", device=device,
    )
    rho_scores = {bname: r.rho for bname, r in rho_report.behaviors.items()}
    results["rho_guided_sft"] = rho_scores
    print(f"  Rho-guided SFT rho scores:")
    for bname, rho in rho_scores.items():
        delta = rho - baseline_scores.get(bname, 0)
        print(f"    {bname:12s}: {rho:.4f} ({delta:+.4f})")

    # ── Results ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    results["config"] = {
        "model": args.model,
        "rho_weight": args.rho_weight,
        "margin": args.margin,
        "epochs": args.epochs,
        "lr": args.lr,
        "sft_size": args.sft_size,
        "lora_rank": args.lora_rank,
        "behaviors": behaviors,
        "elapsed_seconds": elapsed,
    }

    _print_results(results, args)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out_path}")

    print(f"\nTotal time: {elapsed:.1f}s")


def _print_results(results: dict, args):
    """Print comparison table."""
    if args.format == "json":
        print(json.dumps(results, indent=2))
        return

    # Table format
    conditions = [k for k in results if k != "config"]
    if not conditions:
        return

    # Collect all behaviors
    all_behaviors = set()
    for cond in conditions:
        all_behaviors.update(results[cond].keys())
    all_behaviors = sorted(all_behaviors)

    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")

    # Header
    header = f"  {'Behavior':12s}"
    for cond in conditions:
        header += f" | {cond:>16s}"
    print(header)
    print(f"  {'-'*12}" + "---+-----------------" * len(conditions))

    # Rows
    baseline = results.get("baseline", {})
    for beh in all_behaviors:
        row = f"  {beh:12s}"
        for cond in conditions:
            val = results[cond].get(beh, float("nan"))
            if cond != "baseline" and beh in baseline:
                delta = val - baseline[beh]
                row += f" | {val:7.4f} ({delta:+.3f})"
            else:
                row += f" | {val:>16.4f}"
        print(row)

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
