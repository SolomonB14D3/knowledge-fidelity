#!/usr/bin/env python3
"""Merge multiple ablation result files into a single combined file.

The ablation_sft_mlx.py script overwrites its output file on each run.
When running additional seeds, we save to a new file. This script merges
the original (seeds 42, 123) with new seeds into one file for analysis.

Usage:
    python experiments/merge_ablation_results.py
    python experiments/merge_ablation_results.py --files file1.json file2.json --output merged.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RESULTS_DIR = Path(__file__).parent.parent / "results" / "alignment"

DEFAULT_FILES = [
    RESULTS_DIR / "ablation_Qwen_Qwen2.5-7B-Instruct.json",
    RESULTS_DIR / "ablation_Qwen_Qwen2.5-7B-Instruct_s456_789_1337.json",
]

DEFAULT_OUTPUT = RESULTS_DIR / "ablation_Qwen_Qwen2.5-7B-Instruct_merged.json"


def merge_ablation_files(paths: list[Path], output: Path):
    """Merge ablation JSON files by combining their runs lists."""

    merged_runs = []
    baseline_quick = None
    baseline_calibration = None
    model = None
    config = None
    all_seeds = set()

    for path in paths:
        if not path.exists():
            print(f"  Warning: {path.name} not found, skipping")
            continue

        print(f"  Loading: {path.name}")
        with open(path) as f:
            data = json.load(f)

        # Use first file's baseline (should be identical across files)
        if baseline_quick is None:
            baseline_quick = data.get("baseline_quick")
            baseline_calibration = data.get("baseline_calibration")
            model = data.get("model")
            config = data.get("config", {})

        runs = data.get("runs", [])
        for run in runs:
            if "error" not in run:
                # Deduplicate: skip if we already have this (condition, seed)
                key = (run.get("condition"), run.get("seed"))
                if key not in all_seeds:
                    all_seeds.add(key)
                    merged_runs.append(run)
                else:
                    print(f"    Skipping duplicate: condition={key[0]}, seed={key[1]}")

    # Sort by condition then seed for readability
    condition_order = {"sft-only": 0, "rho-guided": 1, "contrastive-only": 2, "shuffled-pairs": 3}
    merged_runs.sort(key=lambda r: (
        condition_order.get(r.get("condition", ""), 99),
        r.get("seed", 0)
    ))

    # Update config with merged seed list
    all_seed_values = sorted(set(r["seed"] for r in merged_runs))
    all_conditions = sorted(set(r["condition"] for r in merged_runs),
                           key=lambda c: condition_order.get(c, 99))

    merged = {
        "model": model,
        "backend": "mlx",
        "experiment": "ablation",
        "baseline_quick": baseline_quick,
        "baseline_calibration": baseline_calibration,
        "config": {
            **config,
            "seeds": all_seed_values,
            "conditions": all_conditions,
            "n_seeds": len(all_seed_values),
            "merged_from": [str(p.name) for p in paths if p.exists()],
        },
        "runs": merged_runs,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(merged, f, indent=2)

    # Summary
    from collections import Counter
    cond_counts = Counter(r["condition"] for r in merged_runs)

    print(f"\n  Merged {len(merged_runs)} runs:")
    for cond, count in sorted(cond_counts.items(), key=lambda x: condition_order.get(x[0], 99)):
        seeds = sorted(set(r["seed"] for r in merged_runs if r["condition"] == cond))
        print(f"    {cond:20s}: n={count} (seeds: {seeds})")
    print(f"\n  Output: {output}")


def main():
    parser = argparse.ArgumentParser(description="Merge ablation result files")
    parser.add_argument("--files", nargs="+", type=Path, default=None,
                        help="Input files (default: original + s456_789_1337)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output merged file")
    args = parser.parse_args()

    files = args.files or DEFAULT_FILES
    output = args.output or DEFAULT_OUTPUT

    print(f"Merging ablation results...")
    merge_ablation_files(files, output)


if __name__ == "__main__":
    main()
