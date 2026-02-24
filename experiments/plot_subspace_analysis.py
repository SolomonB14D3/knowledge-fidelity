#!/usr/bin/env python3
"""Generate figures from subspace analysis results.

Loads JSON results from experiments/subspace_analysis.py and produces
publication-quality figures.

Usage:
    python experiments/plot_subspace_analysis.py
    python experiments/plot_subspace_analysis.py --model qwen2.5_7b_instruct
    python experiments/plot_subspace_analysis.py --results-dir results/interpretability/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rho_eval.interpretability.schema import InterpretabilityReport
from rho_eval.interpretability.visualize import (
    plot_overlap_heatmap,
    plot_head_importance,
    plot_dimensionality,
    plot_surgical_comparison,
)


RESULTS_DIR = Path(__file__).parent.parent / "results" / "interpretability"
FIGURES_DIR = Path(__file__).parent.parent / "figures" / "interpretability"


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures from subspace analysis results"
    )
    parser.add_argument(
        "--model", default=None,
        help="Model short name (e.g., qwen2.5_7b_instruct). Auto-detects if not set.",
    )
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR),
        help="Results directory",
    )
    parser.add_argument(
        "--figures-dir", default=str(FIGURES_DIR),
        help="Output figures directory",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Find report files
    if args.model:
        json_path = results_dir / f"full_report_{args.model}.json"
        pt_path = results_dir / f"full_report_{args.model}.pt"
    else:
        # Auto-detect: find the most recent full_report_*.json
        reports = sorted(results_dir.glob("full_report_*.json"), key=lambda p: p.stat().st_mtime)
        if not reports:
            print(f"No report files found in {results_dir}")
            print("Run experiments/subspace_analysis.py first.")
            return
        json_path = reports[-1]
        pt_path = json_path.with_suffix(".pt")
        print(f"Auto-detected: {json_path.name}")

    if not json_path.exists():
        print(f"Report not found: {json_path}")
        return

    # Load report
    report = InterpretabilityReport.load(
        json_path,
        tensor_path=pt_path if pt_path.exists() else None,
    )
    model_short = json_path.stem.replace("full_report_", "")

    print(f"Loaded: {report}")
    print(f"  Subspaces: {len(report.subspaces)} behaviors")
    print(f"  Overlaps:  {len(report.overlaps)} layers")
    print(f"  Heads:     {len(report.head_importance)} behaviors")
    print(f"  Surgical:  {len(report.surgical_results)} experiments")

    # Generate figures
    if report.overlaps:
        for metric in ["cosine", "shared_variance", "subspace_angles"]:
            path = figures_dir / f"overlap_{metric}_{model_short}.png"
            plot_overlap_heatmap(report.overlaps, metric=metric, save_path=path)
            print(f"  Saved: {path.name}")

    if report.subspaces:
        path = figures_dir / f"dimensionality_{model_short}.png"
        plot_dimensionality(report.subspaces, save_path=path)
        print(f"  Saved: {path.name}")

    if report.head_importance:
        path = figures_dir / f"head_importance_{model_short}.png"
        plot_head_importance(report.head_importance, save_path=path)
        print(f"  Saved: {path.name}")

    if report.surgical_results:
        baselines = {}
        for sr in report.surgical_results:
            if sr.baseline_rho_scores:
                baselines = sr.baseline_rho_scores
                break

        path = figures_dir / f"surgical_{model_short}.png"
        plot_surgical_comparison(
            report.surgical_results[:8],
            baselines,
            save_path=path,
        )
        print(f"  Saved: {path.name}")

    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
