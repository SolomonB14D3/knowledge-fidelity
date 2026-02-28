"""Hybrid Control Sweep — Grid search over control surface parameters.

Sweeps compress_ratio × sae_layer × rho_weight to find the Pareto frontier
of target improvement vs collateral damage.

Usage:
    python experiments/hybrid_control_sweep.py Qwen/Qwen2.5-7B-Instruct
    python experiments/hybrid_control_sweep.py Qwen/Qwen2.5-7B-Instruct --quick
    python experiments/hybrid_control_sweep.py Qwen/Qwen2.5-7B-Instruct --target bias,sycophancy
"""

import argparse
import json
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path

# ── Default sweep grid ────────────────────────────────────────────────────

COMPRESS_RATIOS = [0.5, 0.7, 0.85, 1.0]      # 1.0 = skip SVD
SAE_LAYERS = [None, 12, 17, 22]               # None = skip SAE
RHO_WEIGHTS = [0.0, 0.1, 0.2, 0.5]            # 0.0 = skip SFT

# Quick sweep (for testing the pipeline)
QUICK_COMPRESS = [0.7, 1.0]
QUICK_SAE = [None, 17]
QUICK_RHO = [0.0, 0.2]


def build_sweep_configs(
    target_behaviors: tuple[str, ...] = ("sycophancy",),
    quick: bool = False,
) -> list[dict]:
    """Generate all sweep configurations."""
    from rho_eval.hybrid import HybridConfig

    compress = QUICK_COMPRESS if quick else COMPRESS_RATIOS
    sae = QUICK_SAE if quick else SAE_LAYERS
    rho = QUICK_RHO if quick else RHO_WEIGHTS

    configs = []
    for cr, sl, rw in product(compress, sae, rho):
        # Skip no-op config (no compression, no SAE, no SFT)
        if cr == 1.0 and sl is None and rw == 0.0:
            continue

        config = HybridConfig(
            compress_ratio=cr,
            sae_layer=sl,
            rho_weight=rw,
            target_behaviors=target_behaviors,
        )
        configs.append(config)

    return configs


def run_sweep(
    model_name: str,
    configs: list,
    output_dir: Path,
) -> list[dict]:
    """Run all sweep configurations and collect results."""
    from rho_eval.hybrid import apply_hybrid_control

    results = []
    n = len(configs)

    for i, config in enumerate(configs):
        tag = (
            f"cr{config.compress_ratio}_"
            f"sae{config.sae_layer}_"
            f"rho{config.rho_weight}"
        )
        run_dir = output_dir / tag
        print(f"\n{'='*60}")
        print(f"  Sweep {i+1}/{n}: {tag}")
        print(f"  Phases: {', '.join(config.enabled_phases)}")
        print(f"{'='*60}")

        try:
            result = apply_hybrid_control(
                model_name, config,
                output_dir=str(run_dir),
            )
            entry = {
                "tag": tag,
                "config": config.to_dict(),
                "target_improvement": result.target_improvement,
                "non_target_regression": result.non_target_regression,
                "collateral_damage": result.collateral_damage,
                "audit_before": result.audit_before,
                "audit_after": result.audit_after,
                "elapsed_sec": result.total_elapsed_sec,
                "error": None,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            entry = {
                "tag": tag,
                "config": config.to_dict() if hasattr(config, 'to_dict') else {},
                "error": str(e),
            }

        results.append(entry)

    return results


def print_summary_table(results: list[dict]) -> str:
    """Print a summary table of sweep results."""
    lines = [
        f"\n{'Tag':<30} {'Target Δ':>10} {'Collateral':>10} {'Time':>8}",
        "─" * 62,
    ]
    for r in results:
        if r.get("error"):
            lines.append(f"{r['tag']:<30} {'ERROR':>10} {r['error'][:20]:>10}")
            continue
        lines.append(
            f"{r['tag']:<30} "
            f"{r.get('target_improvement', 0):>+10.4f} "
            f"{r.get('non_target_regression', 0):>+10.4f} "
            f"{r.get('elapsed_sec', 0):>7.0f}s"
        )
    table = "\n".join(lines)
    print(table)
    return table


def print_collateral_matrix(results: list[dict]) -> str:
    """Print full collateral damage matrix (behavior × config)."""
    # Collect all behaviors
    behaviors = set()
    for r in results:
        if r.get("collateral_damage"):
            behaviors.update(r["collateral_damage"].keys())
    behaviors = sorted(behaviors)

    if not behaviors:
        return "No results with collateral data."

    # Header
    bw = 12
    header = f"{'Config':<25} " + " ".join(f"{b[:bw]:<{bw}}" for b in behaviors)
    lines = [f"\nCollateral Damage Matrix:", header, "─" * len(header)]

    for r in results:
        if r.get("error") or not r.get("collateral_damage"):
            continue
        cols = []
        for b in behaviors:
            delta = r["collateral_damage"].get(b, 0.0)
            marker = "✓" if delta >= 0 else "⚠"
            cols.append(f"{delta:>+{bw-2}.4f}{marker} ")
        lines.append(f"{r['tag']:<25} " + " ".join(cols))

    matrix = "\n".join(lines)
    print(matrix)
    return matrix


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid control parameter sweep",
    )
    parser.add_argument("model", help="HuggingFace model ID")
    parser.add_argument(
        "--target", type=str, default="sycophancy",
        help="Comma-separated target behaviors (default: sycophancy)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick sweep (reduced grid for testing)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/hybrid_sweep",
        help="Output directory (default: results/hybrid_sweep)",
    )
    args = parser.parse_args()

    target = tuple(b.strip() for b in args.target.split(","))
    model_short = args.model.split("/")[-1]  # e.g. "Qwen2.5-7B-Instruct"
    output_dir = Path(args.output) / model_short
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nHybrid Control Sweep")
    print(f"  Model:   {args.model}")
    print(f"  Targets: {target}")
    print(f"  Mode:    {'quick' if args.quick else 'full'}")
    print(f"  Output:  {output_dir}")

    configs = build_sweep_configs(target_behaviors=target, quick=args.quick)
    print(f"  Configs: {len(configs)} combinations")

    results = run_sweep(args.model, configs, output_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary_table(results)
    print_collateral_matrix(results)

    # ── Save ──────────────────────────────────────────────────────────────
    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSweep summary saved to {summary_path}")


if __name__ == "__main__":
    main()
