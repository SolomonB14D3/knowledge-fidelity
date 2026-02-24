#!/usr/bin/env python3
"""Phase 6: Mechanistic interpretability of behavioral subspaces.

Extracts and analyzes the internal directions within transformer layers
that encode specific behavioral traits. Builds on Phase 3 (steering vectors)
and Phase 5 (multi-vector steering) findings.

Pipeline:
  1. Load model, extract subspaces at all candidate layers
  2. Compute overlap matrices (behavior × behavior × layer)
  3. Per-head attribution
  4. Surgical interventions (rank-1, rank-k, orthogonalized)
  5. Layer 17 deep dive (sycophancy-bias entanglement)
  6. Generate all figures

Usage:
    python experiments/subspace_analysis.py
    python experiments/subspace_analysis.py --model Qwen/Qwen2.5-7B-Instruct
    python experiments/subspace_analysis.py --quick
    python experiments/subspace_analysis.py --phase subspaces
    python experiments/subspace_analysis.py --phase surgical
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rho_eval.utils import get_layers, get_device, free_memory
from rho_eval.interpretability import (
    extract_subspaces,
    compute_overlap,
    head_attribution,
    evaluate_surgical,
    evaluate_baseline,
    InterpretabilityReport,
)
from rho_eval.interpretability.surgical import rank_k_steer, orthogonal_project

RESULTS_DIR = Path(__file__).parent.parent / "results" / "interpretability"
FIGURES_DIR = Path(__file__).parent.parent / "figures" / "interpretability"
DEVICE = str(get_device())


def _json_default(obj):
    """JSON serializer for numpy/torch types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return str(obj)


def _save_json(data, path):
    """Atomic JSON save."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    tmp.rename(path)


def load_model(model_id, device=DEVICE):
    """Load model + tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_id}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = len(get_layers(model))
    print(f"  Loaded in {time.time()-t0:.1f}s ({n_params/1e9:.2f}B params, {n_layers} layers)")
    return model, tokenizer


def run_subspace_extraction(model, tokenizer, behaviors, layers, device, quick=False):
    """Phase 6A: Extract subspaces."""
    print(f"\n{'='*60}")
    print(f"  PHASE 6A: Subspace Extraction")
    print(f"{'='*60}")

    max_probes = 30 if quick else None
    subspaces = extract_subspaces(
        model, tokenizer, behaviors,
        layers=layers, device=device,
        max_rank=50 if not quick else 10,
        max_probes=max_probes,
    )
    return subspaces


def run_overlap_analysis(subspaces, top_k=10):
    """Phase 6B: Overlap analysis."""
    print(f"\n{'='*60}")
    print(f"  PHASE 6B: Overlap Analysis")
    print(f"{'='*60}")

    overlaps = compute_overlap(subspaces, top_k=top_k)
    return overlaps


def run_head_attribution(model, tokenizer, behaviors, layers, device, quick=False):
    """Phase 6C: Head attribution."""
    print(f"\n{'='*60}")
    print(f"  PHASE 6C: Head Attribution")
    print(f"{'='*60}")

    max_probes = 20 if quick else None
    heads = head_attribution(
        model, tokenizer, behaviors,
        layers=layers, device=device,
        max_probes=max_probes,
    )
    return heads


def run_surgical_experiments(model, tokenizer, subspaces, behaviors, device, quick=False):
    """Phase 6D: Surgical interventions."""
    print(f"\n{'='*60}")
    print(f"  PHASE 6D: Surgical Interventions")
    print(f"{'='*60}")

    baselines = evaluate_baseline(model, tokenizer, behaviors, device)
    results = []
    used_layers = sorted(list(subspaces.values())[0].keys()) if subspaces else []

    # Rank-1 steering for each behavior at each layer
    for target in behaviors:
        if target not in subspaces:
            continue
        for layer_idx in used_layers:
            if layer_idx not in subspaces[target]:
                continue

            print(f"\n  Rank-1: {target} @ L{layer_idx}")
            result = evaluate_surgical(
                model, tokenizer, subspaces,
                target_behavior=target,
                layer_idx=layer_idx,
                eval_behaviors=behaviors,
                alpha=4.0, device=device, rank=1,
            )
            result.baseline_rho_scores = baselines
            results.append(result)

    if quick:
        return baselines, results

    # Rank-k sweep at the most interesting layer per behavior
    for target in behaviors:
        if target not in subspaces:
            continue
        # Find layer with highest steering vector norm
        best_layer = max(
            subspaces[target].keys(),
            key=lambda l: subspaces[target][l].steering_vector.norm().item()
        )
        for k in [2, 3, 5, 10, 20]:
            if k >= subspaces[target][best_layer].directions.shape[0]:
                continue
            print(f"\n  Rank-{k}: {target} @ L{best_layer}")
            result = evaluate_surgical(
                model, tokenizer, subspaces,
                target_behavior=target,
                layer_idx=best_layer,
                eval_behaviors=behaviors,
                alpha=4.0, device=device, rank=k,
            )
            result.baseline_rho_scores = baselines
            results.append(result)

    # Orthogonal projection — test all cross-behavior pairs
    for target in behaviors:
        if target not in subspaces:
            continue
        other_behaviors = [b for b in behaviors if b != target and b in subspaces]
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
                    alpha=4.0, device=device,
                    orthogonal_to=[remove_beh],
                )
                result.baseline_rho_scores = baselines
                results.append(result)

    return baselines, results


def run_visualization(report, model_short):
    """Phase 6E: Generate figures."""
    print(f"\n{'='*60}")
    print(f"  PHASE 6E: Visualization")
    print(f"{'='*60}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    from rho_eval.interpretability.visualize import (
        plot_overlap_heatmap,
        plot_head_importance,
        plot_dimensionality,
        plot_surgical_comparison,
    )

    if report.overlaps:
        for metric in ["cosine", "shared_variance", "subspace_angles"]:
            path = FIGURES_DIR / f"overlap_{metric}_{model_short}.png"
            plot_overlap_heatmap(report.overlaps, metric=metric, save_path=path)
            print(f"  Saved: {path}")

    if report.subspaces:
        path = FIGURES_DIR / f"dimensionality_{model_short}.png"
        plot_dimensionality(report.subspaces, save_path=path)
        print(f"  Saved: {path}")

    if report.head_importance:
        path = FIGURES_DIR / f"head_importance_{model_short}.png"
        plot_head_importance(report.head_importance, save_path=path)
        print(f"  Saved: {path}")

    if report.surgical_results:
        baselines = {}
        if report.surgical_results[0].baseline_rho_scores:
            baselines = report.surgical_results[0].baseline_rho_scores
        path = FIGURES_DIR / f"surgical_{model_short}.png"
        plot_surgical_comparison(
            report.surgical_results[:8],  # First 8 for readability
            baselines,
            save_path=path,
        )
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Mechanistic interpretability of behavioral subspaces"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--behaviors", default="factual,sycophancy,bias,toxicity",
        help="Comma-separated behaviors",
    )
    parser.add_argument(
        "--layers", default=None,
        help="Comma-separated layer indices (default: auto)",
    )
    parser.add_argument(
        "--phase", default="all",
        choices=["all", "subspaces", "overlap", "heads", "surgical", "viz"],
        help="Run only a specific phase",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer probes/layers for fast testing",
    )
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    behaviors = [b.strip() for b in args.behaviors.split(",")]
    layers = None
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]

    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Model:     {args.model}")
    print(f"  Behaviors: {behaviors}")
    print(f"  Layers:    {layers or 'auto'}")
    print(f"  Phase:     {args.phase}")
    print(f"  Quick:     {args.quick}")
    print(f"  Device:    {args.device}")

    t0 = time.time()
    model, tokenizer = load_model(args.model, args.device)

    report = InterpretabilityReport(
        model=args.model,
        metadata={
            "n_layers": len(get_layers(model)),
            "hidden_size": model.config.hidden_size,
            "n_heads": model.config.num_attention_heads,
            "device": args.device,
            "quick": args.quick,
        },
    )

    # Phase 6A: Subspaces
    if args.phase in ("all", "subspaces"):
        report.subspaces = run_subspace_extraction(
            model, tokenizer, behaviors, layers, args.device, args.quick
        )
        # Save checkpoint
        report.save(RESULTS_DIR / f"subspaces_{model_short}.json")
        report.save_tensors(RESULTS_DIR / f"subspaces_{model_short}.pt")
        print(f"\n  Saved subspaces checkpoint")

    # Phase 6B: Overlap
    if args.phase in ("all", "overlap"):
        if not report.subspaces:
            # Try to load from checkpoint
            json_path = RESULTS_DIR / f"subspaces_{model_short}.json"
            pt_path = RESULTS_DIR / f"subspaces_{model_short}.pt"
            if json_path.exists():
                report = InterpretabilityReport.load(json_path, pt_path if pt_path.exists() else None)
                print(f"  Loaded subspaces from checkpoint")

        if report.subspaces and len(report.subspaces) >= 2:
            report.overlaps = run_overlap_analysis(report.subspaces)
            report.save(RESULTS_DIR / f"overlaps_{model_short}.json")
            print(f"\n  Saved overlaps checkpoint")

    # Phase 6C: Heads
    if args.phase in ("all", "heads"):
        report.head_importance = run_head_attribution(
            model, tokenizer, behaviors, layers, args.device, args.quick
        )
        report.save(RESULTS_DIR / f"heads_{model_short}.json")
        print(f"\n  Saved heads checkpoint")

    # Phase 6D: Surgical
    if args.phase in ("all", "surgical"):
        if not report.subspaces:
            json_path = RESULTS_DIR / f"subspaces_{model_short}.json"
            pt_path = RESULTS_DIR / f"subspaces_{model_short}.pt"
            if json_path.exists():
                report = InterpretabilityReport.load(json_path, pt_path if pt_path.exists() else None)

        if report.subspaces:
            baselines, surgical_results = run_surgical_experiments(
                model, tokenizer, report.subspaces, behaviors, args.device, args.quick
            )
            report.surgical_results = surgical_results
            report.save(RESULTS_DIR / f"surgical_{model_short}.json")
            print(f"\n  Saved surgical checkpoint")

    # Phase 6E: Visualization
    if args.phase in ("all", "viz"):
        run_visualization(report, model_short)

    # Final save
    report.elapsed_seconds = time.time() - t0
    final_path = RESULTS_DIR / f"full_report_{model_short}.json"
    report.save(final_path)
    report.save_tensors(RESULTS_DIR / f"full_report_{model_short}.pt")

    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"{'='*60}")
    print(f"  Elapsed: {report.elapsed_seconds:.1f}s")
    print(f"  Report:  {final_path}")
    print(f"  Tensors: {final_path.with_suffix('.pt')}")


if __name__ == "__main__":
    main()
