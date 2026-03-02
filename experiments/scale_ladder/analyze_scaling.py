#!/usr/bin/env python3
"""Aggregate scale ladder results and identify the focal point.

Computes three scaling curves:
  1. Mean behavioral d-prime vs parameter count
  2. Grassmann angle variance vs parameter count
  3. Effective dimensionality spread vs parameter count

The focal point is the smallest scale where all three cross a
significance threshold — this is the experimental scale for Phase 2.

Usage:
    python experiments/scale_ladder/analyze_scaling.py
    python experiments/scale_ladder/analyze_scaling.py --results-dir results/scale_ladder
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale_ladder.configs import SCALE_ORDER, SUBSPACE_BEHAVIORS


def load_all_results(results_dir):
    """Load audit, subspace, and d-prime results for all completed scales."""
    results_dir = Path(results_dir)
    all_results = {}

    for size in SCALE_ORDER:
        # Find matching directories (any seed)
        matches = sorted(results_dir.glob(f"{size}_seed*"))
        if not matches:
            continue

        for ckpt_dir in matches:
            audit_path = ckpt_dir / "audit_report.json"
            dprime_path = ckpt_dir / "dprime_bootstrap.json"
            subspace_path = ckpt_dir / "subspace_report.json"
            metrics_path = ckpt_dir / "training_metrics.json"

            if not audit_path.exists():
                continue

            entry = {
                "size": size,
                "checkpoint": str(ckpt_dir),
                "audit": json.loads(audit_path.read_text()),
            }

            if dprime_path.exists():
                entry["dprime"] = json.loads(dprime_path.read_text())

            if subspace_path.exists():
                entry["subspace"] = json.loads(subspace_path.read_text())

            if metrics_path.exists():
                entry["metrics"] = json.loads(metrics_path.read_text())

            key = ckpt_dir.name  # e.g. "7M_seed42"
            all_results[key] = entry

    return all_results


def compute_behavioral_dprime(results):
    """Compute mean behavioral d-prime across behavior pairs for each scale.

    d-prime = |ρ_i - ρ_j| / sqrt(σ²_i + σ²_j)

    Uses bootstrap d-prime std as σ estimate. Falls back to rho differences
    if bootstrap data isn't available.
    """
    dprime_by_scale = {}

    for key, entry in results.items():
        size = entry["size"]
        audit = entry["audit"]

        # Extract rho values per behavior
        rho_map = {r["behavior"]: r["rho"] for r in audit["results"]}

        if "dprime" in entry:
            # Use bootstrap std for σ
            dprime_data = entry["dprime"]
            std_map = {
                beh: dprime_data[beh].get("dprime_std", 0.1)
                for beh in dprime_data
                if "dprime_std" in dprime_data[beh]
            }
        else:
            std_map = {}

        # Compute pairwise d-prime
        behaviors = list(rho_map.keys())
        dprimes = []
        for i in range(len(behaviors)):
            for j in range(i + 1, len(behaviors)):
                b_i, b_j = behaviors[i], behaviors[j]
                rho_i, rho_j = rho_map[b_i], rho_map[b_j]

                # Use bootstrap std if available, else use a default
                sigma_i = std_map.get(b_i, 0.1)
                sigma_j = std_map.get(b_j, 0.1)

                pooled_sigma = np.sqrt(sigma_i**2 + sigma_j**2)
                if pooled_sigma < 1e-6:
                    pooled_sigma = 0.1

                d = abs(rho_i - rho_j) / pooled_sigma
                dprimes.append(d)

        dprime_by_scale[key] = {
            "size": size,
            "n_params": audit.get("n_params", 0),
            "mean_dprime": float(np.mean(dprimes)) if dprimes else 0.0,
            "max_dprime": float(np.max(dprimes)) if dprimes else 0.0,
            "n_pairs": len(dprimes),
            "individual_rhos": rho_map,
        }

    return dprime_by_scale


def compute_grassmann_variance(results):
    """Compute variance of Grassmann angles across behavior pairs per scale.

    High variance = differentiated geometry (different behaviors occupy
    different relative orientations). Low variance = random/uniform geometry.
    """
    grass_by_scale = {}

    for key, entry in results.items():
        if "subspace" not in entry:
            continue

        subspace = entry["subspace"]
        overlap = subspace.get("overlap", {})

        # Extract all pairwise angles
        all_angles = []
        for pair_key, data in overlap.items():
            if isinstance(data, list):
                all_angles.extend([float(a) for a in data])
            elif isinstance(data, dict):
                for layer_key, angles in data.items():
                    if isinstance(angles, list):
                        all_angles.extend([float(a) for a in angles])
                    elif isinstance(angles, (int, float)):
                        all_angles.append(float(angles))

        if not all_angles:
            continue

        grass_by_scale[key] = {
            "size": entry["size"],
            "n_params": subspace.get("n_layers", 0),
            "angle_mean": float(np.mean(all_angles)),
            "angle_std": float(np.std(all_angles)),
            "angle_variance": float(np.var(all_angles)),
            "n_angles": len(all_angles),
            "angle_min": float(np.min(all_angles)),
            "angle_max": float(np.max(all_angles)),
        }

    return grass_by_scale


def compute_eff_dim_spread(results):
    """Compute spread of effective dimensionality across behaviors per scale.

    High spread = different behaviors occupy different-shaped subspaces.
    """
    dim_by_scale = {}

    for key, entry in results.items():
        if "subspace" not in entry:
            continue

        subspace = entry["subspace"]
        eff_dim = subspace.get("effective_dimensionality", {})

        if not eff_dim:
            continue

        # Collect effective dims per behavior, averaged across layers
        beh_dims = {}
        for beh_name, layer_dict in eff_dim.items():
            dims = []
            for layer_key, layer_data in layer_dict.items():
                if isinstance(layer_data, dict) and "effective_dim" in layer_data:
                    dims.append(layer_data["effective_dim"])
            if dims:
                beh_dims[beh_name] = {
                    "mean_dim": float(np.mean(dims)),
                    "max_dim": float(np.max(dims)),
                    "min_dim": float(np.min(dims)),
                }

        if len(beh_dims) < 2:
            continue

        mean_dims = [v["mean_dim"] for v in beh_dims.values()]

        dim_by_scale[key] = {
            "size": entry["size"],
            "dim_spread": float(np.std(mean_dims)),
            "dim_variance": float(np.var(mean_dims)),
            "dim_mean": float(np.mean(mean_dims)),
            "per_behavior": beh_dims,
        }

    return dim_by_scale


def identify_focal_point(dprime_data, grass_data, dim_data, thresholds=None):
    """Identify the smallest scale where all three metrics cross significance.

    Default thresholds:
      - d-prime > 1.0 (behavioral discriminability)
      - Grassmann angle std > 5° (geometric differentiation)
      - Effective dim spread > 1.0 (different-shaped subspaces)
    """
    if thresholds is None:
        thresholds = {
            "dprime": 1.0,
            "grassmann_std": 5.0,
            "dim_spread": 1.0,
        }

    # Align by size order
    focal = None
    for size in SCALE_ORDER:
        # Find entries for this size
        dp_entries = [v for v in dprime_data.values() if v["size"] == size]
        gr_entries = [v for v in grass_data.values() if v["size"] == size]
        dm_entries = [v for v in dim_data.values() if v["size"] == size]

        if not dp_entries:
            continue

        dp_ok = any(e["mean_dprime"] > thresholds["dprime"] for e in dp_entries)
        gr_ok = any(e["angle_std"] > thresholds["grassmann_std"] for e in gr_entries) if gr_entries else False
        dm_ok = any(e["dim_spread"] > thresholds["dim_spread"] for e in dm_entries) if dm_entries else False

        status = {
            "size": size,
            "dprime_pass": dp_ok,
            "grassmann_pass": gr_ok,
            "dim_spread_pass": dm_ok,
            "all_pass": dp_ok and gr_ok and dm_ok,
            "dprime_val": dp_entries[0]["mean_dprime"] if dp_entries else None,
            "grassmann_val": gr_entries[0]["angle_std"] if gr_entries else None,
            "dim_spread_val": dm_entries[0]["dim_spread"] if dm_entries else None,
        }

        if status["all_pass"] and focal is None:
            focal = size

        print(f"  {size:>5s}: d'={status['dprime_val'] or 0:.2f} "
              f"{'PASS' if dp_ok else 'FAIL':>4s} | "
              f"grass={status['grassmann_val'] or 0:.1f}° "
              f"{'PASS' if gr_ok else 'FAIL':>4s} | "
              f"dim={status['dim_spread_val'] or 0:.2f} "
              f"{'PASS' if dm_ok else 'FAIL':>4s}"
              f"{' ← FOCAL POINT' if status['all_pass'] and size == focal else ''}")

    return focal


def generate_plots(dprime_data, grass_data, dim_data, focal_point, output_dir):
    """Generate scaling curve plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data points ordered by parameter count
    sizes = []
    params = []
    dprimes = []
    grass_stds = []
    dim_spreads = []

    for size in SCALE_ORDER:
        dp_entries = [v for v in dprime_data.values() if v["size"] == size]
        gr_entries = [v for v in grass_data.values() if v["size"] == size]
        dm_entries = [v for v in dim_data.values() if v["size"] == size]

        if dp_entries:
            sizes.append(size)
            params.append(dp_entries[0]["n_params"])
            dprimes.append(dp_entries[0]["mean_dprime"])
            grass_stds.append(gr_entries[0]["angle_std"] if gr_entries else 0)
            dim_spreads.append(dm_entries[0]["dim_spread"] if dm_entries else 0)

    if len(sizes) < 2:
        print(f"  Only {len(sizes)} data points, need ≥2 for plots")
        return

    params = np.array(params)

    # Fig 1: Three-panel scaling curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].semilogx(params, dprimes, "o-", color="tab:blue", linewidth=2)
    axes[0].axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="threshold")
    axes[0].set_xlabel("Parameters")
    axes[0].set_ylabel("Mean Behavioral d-prime")
    axes[0].set_title("Behavioral Discriminability")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for i, s in enumerate(sizes):
        axes[0].annotate(s, (params[i], dprimes[i]), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)

    axes[1].semilogx(params, grass_stds, "s-", color="tab:green", linewidth=2)
    axes[1].axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label="threshold")
    axes[1].set_xlabel("Parameters")
    axes[1].set_ylabel("Grassmann Angle Std (degrees)")
    axes[1].set_title("Geometric Differentiation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    for i, s in enumerate(sizes):
        axes[1].annotate(s, (params[i], grass_stds[i]), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)

    axes[2].semilogx(params, dim_spreads, "^-", color="tab:orange", linewidth=2)
    axes[2].axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="threshold")
    axes[2].set_xlabel("Parameters")
    axes[2].set_ylabel("Effective Dim Spread (std)")
    axes[2].set_title("Subspace Shape Diversity")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    for i, s in enumerate(sizes):
        axes[2].annotate(s, (params[i], dim_spreads[i]), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)

    if focal_point:
        for ax in axes:
            fp_idx = sizes.index(focal_point) if focal_point in sizes else None
            if fp_idx is not None:
                ax.axvline(x=params[fp_idx], color="purple", linestyle=":",
                          alpha=0.5, label=f"focal: {focal_point}")
                ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "scaling_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'scaling_curves.png'}")

    # Fig 2: Rho per behavior across scales
    if len(sizes) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        behaviors = set()
        for v in dprime_data.values():
            behaviors.update(v.get("individual_rhos", {}).keys())

        for beh in sorted(behaviors):
            beh_rhos = []
            beh_params = []
            for size in SCALE_ORDER:
                entries = [v for v in dprime_data.values() if v["size"] == size]
                if entries and beh in entries[0].get("individual_rhos", {}):
                    beh_rhos.append(entries[0]["individual_rhos"][beh])
                    beh_params.append(entries[0]["n_params"])

            if beh_rhos:
                ax.semilogx(beh_params, beh_rhos, "o-", label=beh, linewidth=1.5)

        ax.set_xlabel("Parameters")
        ax.set_ylabel("ρ (Spearman)")
        ax.set_title("Per-Behavior ρ Across Scales")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "rho_per_behavior.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_dir / 'rho_per_behavior.png'}")


def main():
    parser = argparse.ArgumentParser(
        prog="analyze-scaling",
        description="Aggregate scale ladder results and find the focal point",
    )
    parser.add_argument("--results-dir", type=str,
                        default="results/scale_ladder",
                        help="Directory containing scale ladder results")
    parser.add_argument("--figures-dir", type=str,
                        default="figures/scale_ladder",
                        help="Directory for output plots")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Scale Ladder Analysis")
    print(f"  Results: {args.results_dir}")
    print(f"{'='*60}\n")

    # Load all results
    results = load_all_results(args.results_dir)
    if not results:
        print("  ERROR: No results found. Train and audit some models first.")
        return 1

    print(f"  Found {len(results)} completed model(s): {sorted(results.keys())}\n")

    # Compute three scaling metrics
    print("  --- Behavioral d-prime ---")
    dprime_data = compute_behavioral_dprime(results)
    for key, val in sorted(dprime_data.items()):
        print(f"    {key}: mean d'={val['mean_dprime']:.3f}, "
              f"max d'={val['max_dprime']:.3f} ({val['n_pairs']} pairs)")

    print("\n  --- Grassmann Angle Variance ---")
    grass_data = compute_grassmann_variance(results)
    for key, val in sorted(grass_data.items()):
        print(f"    {key}: mean={val['angle_mean']:.1f}°, "
              f"std={val['angle_std']:.1f}°, range=[{val['angle_min']:.1f}°, {val['angle_max']:.1f}°]")

    print("\n  --- Effective Dimensionality Spread ---")
    dim_data = compute_eff_dim_spread(results)
    for key, val in sorted(dim_data.items()):
        print(f"    {key}: spread={val['dim_spread']:.2f}, "
              f"mean dim={val['dim_mean']:.1f}")

    # Identify focal point
    print(f"\n  --- Focal Point Identification ---")
    focal = identify_focal_point(dprime_data, grass_data, dim_data)

    if focal:
        print(f"\n  FOCAL POINT: {focal}")
    else:
        print(f"\n  No focal point found yet — need more scales or lower thresholds")

    # Generate plots
    print(f"\n  --- Generating Plots ---")
    generate_plots(dprime_data, grass_data, dim_data, focal, args.figures_dir)

    # Save summary
    summary = {
        "n_models": len(results),
        "models": sorted(results.keys()),
        "focal_point": focal,
        "dprime": {k: {kk: vv for kk, vv in v.items() if kk != "individual_rhos"}
                   for k, v in dprime_data.items()},
        "grassmann": grass_data,
        "eff_dim": dim_data,
    }

    summary_path = Path(args.results_dir) / "scaling_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  Saved: {summary_path}")

    print(f"\n{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
