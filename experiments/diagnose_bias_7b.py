"""Diagnostic: Re-run bias evaluation on 7B with new diversified probes.

This script re-runs the full hybrid pipeline for specific configs,
using the updated bias evaluation that includes:
  - Biology-grounded probes (sexual orientation, gender)
  - Bridge probes (native MC + converted pairs)
  - Per-category disaggregation
  - Per-source disaggregation

Usage:
    python experiments/diagnose_bias_7b.py Qwen/Qwen2.5-7B-Instruct --configs star control
    python experiments/diagnose_bias_7b.py Qwen/Qwen2.5-7B-Instruct --configs star --bias-only

Configs:
    star    = cr0.7_saeNone_rho0.2 (compression + rho SFT, no SAE)
    control = cr0.7_sae17_rho0.0   (compression + SAE, no SFT)
    sae_rho = cr0.7_sae17_rho0.2   (compression + SAE + rho SFT)
    rho_only = cr1.0_saeNone_rho0.2 (rho SFT only, no compression)
"""

import argparse
import json
import time
from pathlib import Path


# ── Named configs matching the sweep ─────────────────────────────────────

NAMED_CONFIGS = {
    "star": {
        "compress_ratio": 0.7,
        "freeze_fraction": 0.75,
        "sae_layer": None,
        "rho_weight": 0.2,
        "tag": "cr0.7_saeNone_rho0.2",
    },
    "control": {
        "compress_ratio": 0.7,
        "freeze_fraction": 0.75,
        "sae_layer": 17,
        "rho_weight": 0.0,
        "tag": "cr0.7_sae17_rho0.0",
    },
    "sae_rho": {
        "compress_ratio": 0.7,
        "freeze_fraction": 0.75,
        "sae_layer": 17,
        "rho_weight": 0.2,
        "tag": "cr0.7_sae17_rho0.2",
    },
    "rho_only": {
        "compress_ratio": 1.0,
        "freeze_fraction": 0.75,
        "sae_layer": None,
        "rho_weight": 0.2,
        "tag": "cr1.0_saeNone_rho0.2",
    },
}


def run_diagnostic(
    model_name: str,
    config_name: str,
    output_dir: Path,
    bias_only: bool = False,
):
    """Run a single diagnostic config with diversified bias probes."""
    from rho_eval.hybrid import HybridConfig, apply_hybrid_control

    params = NAMED_CONFIGS[config_name]
    config = HybridConfig(
        compress_ratio=params["compress_ratio"],
        freeze_fraction=params["freeze_fraction"],
        sae_layer=params["sae_layer"],
        rho_weight=params["rho_weight"],
        target_behaviors=("sycophancy",),
        eval_behaviors=("bias",) if bias_only else ("all",),
    )

    tag = params["tag"]
    run_dir = output_dir / f"{tag}_diag"

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC: {config_name} ({tag})")
    print(f"  Phases: {', '.join(config.enabled_phases)}")
    print(f"  Eval: {'bias-only' if bias_only else 'all behaviors'}")
    print(f"  Output: {run_dir}")
    print(f"{'='*70}")

    t0 = time.time()
    result = apply_hybrid_control(
        model_name, config,
        output_dir=str(run_dir),
    )
    elapsed = time.time() - t0

    # ── Extract and print disaggregated bias results ─────────────────
    print(f"\n  Completed in {elapsed/3600:.1f}h")
    print(f"  {result.summary()}")

    # Detailed bias disaggregation
    for phase_result in result.phases:
        report = phase_result.details.get("report", {})
        behaviors = report.get("behaviors", {})
        if "bias" in behaviors:
            bias = behaviors["bias"]
            phase_name = phase_result.phase
            print(f"\n  ── {phase_name} bias: rho={bias['rho']:.4f} ({bias['positive_count']}/{bias['total']}) ──")

            # Category metrics (from our new evaluation code)
            cat_metrics = bias.get("metadata", {}).get("category_metrics", {})
            if cat_metrics:
                print(f"\n  {'Category':<30s} {'Accuracy':>10s} {'N':>5s} {'Biased %':>10s}")
                print(f"  {'-'*60}")
                for cat, data in sorted(cat_metrics.items(), key=lambda x: -x[1]["accuracy"]):
                    print(f"  {cat:<30s} {data['accuracy']:>9.1%} {data['n']:>5d} {data.get('biased_rate',0):>9.1%}")

            # Source metrics
            src_metrics = bias.get("metadata", {}).get("source_metrics", {})
            if src_metrics:
                print(f"\n  {'Source':<20s} {'Accuracy':>10s} {'N':>5s}")
                print(f"  {'-'*40}")
                for src, data in sorted(src_metrics.items(), key=lambda x: -x[1]["accuracy"]):
                    print(f"  {src:<20s} {data['accuracy']:>9.1%} {data['n']:>5d}")

    return result


def compare_results(output_dir: Path, config_names: list[str]):
    """Compare disaggregated bias results across configs."""
    results = {}
    for name in config_names:
        tag = NAMED_CONFIGS[name]["tag"]
        path = output_dir / f"{tag}_diag" / "hybrid_result.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)

    if len(results) < 2:
        print("Need at least 2 completed configs to compare.")
        return

    print(f"\n{'='*80}")
    print(f"  COMPARISON: {' vs '.join(results.keys())}")
    print(f"{'='*80}")

    # Extract final bias details for each config
    for name, r in results.items():
        final = r["phases"][-1]["details"]["report"]["behaviors"].get("bias", {})
        baseline = r["phases"][0]["details"]["report"]["behaviors"].get("bias", {})
        print(f"\n  {name}: bias {baseline.get('rho',0):.3f} → {final.get('rho',0):.3f}")

        cat_metrics = final.get("metadata", {}).get("category_metrics", {})
        base_cat_metrics = baseline.get("metadata", {}).get("category_metrics", {})

        if cat_metrics and base_cat_metrics:
            print(f"  {'Category':<30s} {'Before':>10s} {'After':>10s} {'Delta':>8s}")
            print(f"  {'-'*60}")
            for cat in sorted(cat_metrics.keys()):
                before = base_cat_metrics.get(cat, {}).get("accuracy", 0)
                after = cat_metrics[cat]["accuracy"]
                delta = after - before
                marker = '▼' if delta < -0.05 else ('▲' if delta > 0.05 else ' ')
                print(f"  {cat:<30s} {before:>9.1%} {after:>9.1%} {delta:>+7.1%} {marker}")


def main():
    parser = argparse.ArgumentParser(
        description="Bias diagnostic for 7B hybrid sweep configs",
    )
    parser.add_argument("model", help="HuggingFace model ID")
    parser.add_argument(
        "--configs", nargs="+", default=["star", "control"],
        choices=list(NAMED_CONFIGS.keys()),
        help="Which configs to diagnose (default: star control)",
    )
    parser.add_argument(
        "--bias-only", action="store_true",
        help="Only evaluate bias behavior (faster)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/hybrid_sweep/diagnostics",
        help="Output directory (default: results/hybrid_sweep/diagnostics)",
    )
    parser.add_argument(
        "--compare-only", action="store_true",
        help="Skip runs, just compare existing results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.compare_only:
        for config_name in args.configs:
            run_diagnostic(
                args.model, config_name, output_dir,
                bias_only=args.bias_only,
            )

    if len(args.configs) >= 2:
        compare_results(output_dir, args.configs)


if __name__ == "__main__":
    main()
