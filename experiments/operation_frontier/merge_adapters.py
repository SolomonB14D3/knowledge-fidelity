#!/usr/bin/env python3
"""
Merge multiple domain adapters into a single multi-domain adapter.

Strategy: Simple averaging of weights (effective for non-overlapping domains).
For overlapping domains, could use task vectors or TIES merging.

Usage:
    python merge_adapters.py --out adapters/multi_domain_v1.npz \
        adapters/adapter_choreography.npz \
        adapters/vortex_q_adapter_v4.npz
"""

import argparse
import os
import mlx.core as mx
import numpy as np


def merge_adapters(adapter_paths: list[str], output_path: str, weights: list[float] = None):
    """Merge multiple adapters by weighted averaging."""
    if weights is None:
        weights = [1.0 / len(adapter_paths)] * len(adapter_paths)

    assert len(weights) == len(adapter_paths), "weights must match number of adapters"
    assert abs(sum(weights) - 1.0) < 1e-6, "weights must sum to 1"

    print(f"\nMerging {len(adapter_paths)} adapters:")
    for p, w in zip(adapter_paths, weights):
        print(f"  {w:.2f} × {os.path.basename(p)}")

    # Load all adapters
    all_weights = []
    for path in adapter_paths:
        w = mx.load(path)
        all_weights.append(w)
        print(f"  Loaded {path}: {len(w)} keys")

    # Check all have same keys
    keys = set(all_weights[0].keys())
    for i, w in enumerate(all_weights[1:], 1):
        if set(w.keys()) != keys:
            raise ValueError(f"Adapter {i} has different keys: {set(w.keys())} vs {keys}")

    # Merge by weighted average
    merged = {}
    for key in keys:
        merged_val = sum(float(wt) * all_weights[i][key] for i, wt in enumerate(weights))
        merged[key] = merged_val

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mx.savez(output_path, **merged)
    print(f"\nSaved merged adapter: {output_path}")
    print(f"  Keys: {list(merged.keys())[:4]}...")

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("adapters", nargs="+", help="Adapter .npz files to merge")
    parser.add_argument("--out", required=True, help="Output path")
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Weights for each adapter (must sum to 1)")
    args = parser.parse_args()

    merge_adapters(args.adapters, args.out, args.weights)


if __name__ == "__main__":
    main()
