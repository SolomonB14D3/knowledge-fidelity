#!/usr/bin/env python3
"""rho-interpret: Mechanistic interpretability of behavioral subspaces.

Extract, analyze, and visualize the internal directions within transformer
layers that encode specific behavioral traits.

Usage:
    rho-interpret Qwen/Qwen2.5-7B-Instruct
    rho-interpret Qwen/Qwen2.5-7B-Instruct --behaviors factual,sycophancy,bias
    rho-interpret Qwen/Qwen2.5-7B-Instruct --layers 7,14,17,24
    rho-interpret Qwen/Qwen2.5-7B-Instruct --surgical
    rho-interpret Qwen/Qwen2.5-7B-Instruct --heads
    rho-interpret Qwen/Qwen2.5-7B-Instruct --format json --output interp.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        prog="rho-interpret",
        description="Mechanistic interpretability of behavioral subspaces in LLMs.",
    )

    parser.add_argument(
        "model",
        nargs="?",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--behaviors", "-b",
        default="factual,sycophancy,bias,toxicity",
        help="Comma-separated behaviors (default: factual,sycophancy,bias,toxicity)",
    )
    parser.add_argument(
        "--layers", "-l",
        default=None,
        help="Comma-separated layer indices (default: auto-select 6 layers)",
    )
    parser.add_argument(
        "--surgical",
        action="store_true",
        help="Run surgical intervention experiments (rank-1, orthogonal projection)",
    )
    parser.add_argument(
        "--heads",
        action="store_true",
        help="Run per-head attribution analysis",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=50,
        help="Maximum number of principal directions (default: 50)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "table"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (JSON format)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: auto-detect)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    args = parser.parse_args()

    if args.version:
        from .. import __version__
        print(f"rho-interpret {__version__}")
        return

    if args.model is None:
        parser.error("model is required (e.g., rho-interpret Qwen/Qwen2.5-7B-Instruct)")

    # ── Import heavy modules only when needed ──
    import torch
    from ..utils import get_device, get_layers
    from ..interpretability import (
        extract_subspaces,
        compute_overlap,
        head_attribution,
        evaluate_surgical,
        evaluate_baseline,
        InterpretabilityReport,
    )

    # ── Setup ──
    device = args.device or str(get_device())
    behaviors = [b.strip() for b in args.behaviors.split(",")]
    layers = None
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]

    print(f"rho-interpret — Mechanistic Interpretability Analysis")
    print(f"{'=' * 60}")
    print(f"  Model:     {args.model}")
    print(f"  Behaviors: {behaviors}")
    print(f"  Layers:    {layers or 'auto'}")
    print(f"  Device:    {device}")
    print(f"  Surgical:  {args.surgical}")
    print(f"  Heads:     {args.heads}")
    print()

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    model.eval()

    n_layers = len(get_layers(model))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {time.time() - t0:.1f}s ({n_params / 1e9:.2f}B params, {n_layers} layers)")

    report = InterpretabilityReport(
        model=args.model,
        metadata={
            "n_layers": n_layers,
            "hidden_size": model.config.hidden_size,
            "n_heads": model.config.num_attention_heads,
            "device": device,
        },
    )

    # ── Phase 1: Subspace extraction ──
    print(f"\n{'─' * 60}")
    print("Phase 1: Subspace Extraction")
    print(f"{'─' * 60}")

    subspaces = extract_subspaces(
        model, tokenizer, behaviors,
        layers=layers,
        device=device,
        max_rank=args.max_rank,
    )
    report.subspaces = subspaces

    # ── Phase 2: Overlap analysis ──
    if len(behaviors) >= 2:
        print(f"\n{'─' * 60}")
        print("Phase 2: Overlap Analysis")
        print(f"{'─' * 60}")

        overlaps = compute_overlap(subspaces)
        report.overlaps = overlaps

    # ── Phase 3: Head attribution (optional) ──
    if args.heads:
        print(f"\n{'─' * 60}")
        print("Phase 3: Head Attribution")
        print(f"{'─' * 60}")

        heads = head_attribution(
            model, tokenizer, behaviors,
            layers=layers, device=device,
        )
        report.head_importance = heads

    # ── Phase 4: Surgical interventions (optional) ──
    if args.surgical:
        print(f"\n{'─' * 60}")
        print("Phase 4: Surgical Interventions")
        print(f"{'─' * 60}")

        baselines = evaluate_baseline(model, tokenizer, behaviors, device)

        # For each behavior, try rank-1 steering at each layer
        used_layers = sorted(list(subspaces.values())[0].keys()) if subspaces else []

        for target in behaviors:
            if target not in subspaces:
                continue

            for layer_idx in used_layers:
                if layer_idx not in subspaces[target]:
                    continue

                # Rank-1 steering
                print(f"\n  Rank-1: {target} @ L{layer_idx}")
                result = evaluate_surgical(
                    model, tokenizer, subspaces,
                    target_behavior=target,
                    layer_idx=layer_idx,
                    eval_behaviors=behaviors,
                    alpha=4.0,
                    device=device,
                    rank=1,
                )
                result.baseline_rho_scores = baselines
                report.surgical_results.append(result)

            # Orthogonal projection at layers where we have overlap data
            other_behaviors = [b for b in behaviors if b != target]
            for remove_beh in other_behaviors:
                for layer_idx in used_layers:
                    if (layer_idx not in subspaces.get(target, {}) or
                            layer_idx not in subspaces.get(remove_beh, {})):
                        continue

                    print(f"\n  Ortho: {target} (remove {remove_beh}) @ L{layer_idx}")
                    result = evaluate_surgical(
                        model, tokenizer, subspaces,
                        target_behavior=target,
                        layer_idx=layer_idx,
                        eval_behaviors=behaviors,
                        alpha=4.0,
                        device=device,
                        orthogonal_to=[remove_beh],
                    )
                    result.baseline_rho_scores = baselines
                    report.surgical_results.append(result)

    # ── Finalize ──
    report.elapsed_seconds = time.time() - t0

    # ── Output ──
    if args.format == "json" or args.output:
        output = report.to_json()
        if args.output:
            out_path = report.save(args.output)
            print(f"\nSaved to: {out_path}")
        else:
            print(output)
    else:
        # Table format
        _print_table(report)

    return report


def _print_table(report):
    """Print a human-readable summary table."""
    print(f"\n{'=' * 60}")
    print(f"  Interpretability Report: {report.model}")
    print(f"{'=' * 60}")

    # Subspace summary
    if report.subspaces:
        print(f"\n  Subspaces (effective dimensionality at 90% variance):")
        behaviors = sorted(report.subspaces.keys())
        layers = sorted(list(report.subspaces.values())[0].keys())

        header = f"  {'':12s}" + "".join(f"  L{l:<4d}" for l in layers)
        print(header)

        for beh in behaviors:
            row = f"  {beh:12s}"
            for l in layers:
                sr = report.subspaces[beh].get(l)
                if sr:
                    row += f"  {sr.effective_dim:5d}"
                else:
                    row += "    - "
            print(row)

    # Overlap summary
    if report.overlaps:
        # Show the layer with highest off-diagonal cosine
        print(f"\n  Highest pairwise overlaps (|cosine| of top-1 direction):")
        for layer_idx, om in sorted(report.overlaps.items()):
            n = len(om.behaviors)
            max_cos = 0.0
            max_pair = ("", "")
            for i in range(n):
                for j in range(i + 1, n):
                    val = abs(om.cosine_matrix[i][j])
                    if val > max_cos:
                        max_cos = val
                        max_pair = (om.behaviors[i], om.behaviors[j])
            if max_cos > 0.1:
                print(f"    L{layer_idx}: {max_pair[0]}-{max_pair[1]} = {max_cos:.3f}")

    # Surgical summary
    if report.surgical_results:
        print(f"\n  Surgical interventions ({len(report.surgical_results)} total):")
        for sr in report.surgical_results[:10]:  # Show first 10
            scores_str = ", ".join(
                f"{k}={v:.3f}" for k, v in sorted(sr.rho_scores.items())
            )
            print(f"    {sr.intervention:30s} → {scores_str}")

    print(f"\n  Elapsed: {report.elapsed_seconds:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
