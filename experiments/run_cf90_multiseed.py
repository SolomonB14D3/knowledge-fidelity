#!/usr/bin/env python3
"""
Multi-seed CF90 validation with confidence auditing.

Runs CF90 compression + confidence audit across multiple seeds for
statistical significance testing. This is the main experiment for the paper.

Usage:
    python experiments/run_cf90_multiseed.py
    python experiments/run_cf90_multiseed.py --model meta-llama/Llama-3.1-8B-Instruct --seeds 3
    python experiments/run_cf90_multiseed.py --model Qwen/Qwen2.5-7B --seeds 5
"""

import sys
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_fidelity import compress_and_audit, get_default_probes


def main():
    parser = argparse.ArgumentParser(description="Multi-seed CF90 + Audit")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--ratio", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print(f"Multi-Seed CF90 + Confidence Audit")
    print(f"Model: {args.model} | Seeds: {args.seeds} | Ratio: {args.ratio}")
    print("=" * 70)

    all_retentions = []
    all_rho_before = []
    all_rho_after = []
    seed_results = []

    for seed in range(args.seeds):
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        report = compress_and_audit(
            model_name_or_path=args.model,
            ratio=args.ratio,
            device=args.device,
        )

        all_retentions.append(report["retention"])
        all_rho_before.append(report["rho_before"])
        all_rho_after.append(report["rho_after"])

        seed_results.append({
            "seed": seed,
            "retention": report["retention"],
            "rho_before": report["rho_before"],
            "rho_after": report["rho_after"],
            "rho_drop": report["rho_before"] - report["rho_after"],
            "n_compressed": report["compression"]["n_compressed"],
            "elapsed": report["elapsed_seconds"],
        })

        # Free memory between seeds
        del report
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Statistics
    ret_arr = np.array(all_retentions)
    rho_before_arr = np.array(all_rho_before)
    rho_after_arr = np.array(all_rho_after)
    rho_drop_arr = rho_before_arr - rho_after_arr

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"  Retention:  {ret_arr.mean():.1%} +/- {ret_arr.std():.1%}")
    print(f"  rho before: {rho_before_arr.mean():.3f} +/- {rho_before_arr.std():.3f}")
    print(f"  rho after:  {rho_after_arr.mean():.3f} +/- {rho_after_arr.std():.3f}")
    print(f"  rho drop:   {rho_drop_arr.mean():.3f} +/- {rho_drop_arr.std():.3f}")

    if args.seeds >= 3:
        # Paired t-test: is rho drop significantly > 0?
        t_stat, p_val = sp_stats.ttest_1samp(rho_drop_arr, 0)
        print(f"\n  t-test (rho drop > 0): t={t_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.05:
            print(f"  -> Significant rho drop at p<0.05")
        else:
            print(f"  -> No significant rho drop (confidence preserved!)")

    # Save
    output_path = args.output or f"results/cf90_multiseed_{Path(args.model).name}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "ratio": args.ratio,
        "n_seeds": args.seeds,
        "retention_mean": float(ret_arr.mean()),
        "retention_std": float(ret_arr.std()),
        "rho_before_mean": float(rho_before_arr.mean()),
        "rho_after_mean": float(rho_after_arr.mean()),
        "rho_drop_mean": float(rho_drop_arr.mean()),
        "rho_drop_std": float(rho_drop_arr.std()),
        "seeds": seed_results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
