#!/usr/bin/env python3
"""Fidelity-Bench 2.0 — Run standardized adversarial benchmark.

Measures the Truth-Gap (ΔF = ρ_baseline − ρ_pressured) across model
families to quantify how much truth models sacrifice under social pressure.

Usage:
    # Quick validation with Qwen-0.5B (3 probes per domain, 3 levels)
    python experiments/fidelity_bench_2.py --validate

    # Full benchmark on a single model
    python experiments/fidelity_bench_2.py --models Qwen/Qwen2.5-0.5B

    # Multi-model comparison
    python experiments/fidelity_bench_2.py --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-7B-Instruct

    # Show benchmark dataset info
    python experiments/fidelity_bench_2.py --info
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B",
]

VALIDATE_MODELS = ["Qwen/Qwen2.5-0.5B"]


def free_memory():
    """Free GPU/MPS memory between model evaluations."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except ImportError:
        pass


def run_experiment(args):
    """Run the Fidelity-Bench experiment."""
    from rho_eval.benchmarking import (
        generate_certificate,
        BenchmarkConfig,
        FidelityCertificate,
    )
    from rho_eval.benchmarking.loader import get_bench_metadata

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine configuration
    if args.validate:
        models = VALIDATE_MODELS
        config = BenchmarkConfig(
            pressure_levels=3,
            n_probes_per_domain=3,
            seed=42,
            n_bootstrap=100,
            device=args.device,
        )
        print("\n[VALIDATE MODE] Quick test with reduced probes/levels\n")
    else:
        models = args.models or DEFAULT_MODELS
        config = BenchmarkConfig(
            pressure_levels=args.pressure_levels,
            n_probes_per_domain=args.n_probes,
            seed=42,
            n_bootstrap=args.n_bootstrap,
            device=args.device,
        )

    # Show benchmark info
    meta = get_bench_metadata()
    print(f"Fidelity-Bench {meta['version']}")
    print(f"  Probes: {meta['n_probes']} across {len(meta['domains'])} domains")
    print(f"  Hash: {meta['probe_hash'][:16]}...")
    print(f"  Models: {len(models)}")
    print()

    # Run benchmarks
    certificates = []
    t0_total = time.time()

    for i, model_name in enumerate(models):
        print(f"\n{'='*60}")
        print(f"  Model {i + 1}/{len(models)}: {model_name}")
        print(f"{'='*60}")

        try:
            cert = generate_certificate(
                model_name,
                config=config,
                verbose=True,
            )
            certificates.append(cert)

            # Save individual certificate
            slug = model_name.replace("/", "__")
            cert_path = output_dir / f"{slug}.json"
            cert.save(cert_path)
            print(f"  Saved: {cert_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        finally:
            free_memory()

    total_time = time.time() - t0_total

    # Summary comparison table
    if len(certificates) >= 1:
        print(f"\n\n{'='*70}")
        print(f"  FIDELITY-BENCH 2.0 SUMMARY")
        print(f"{'='*70}\n")

        header = (
            f"  {'Model':<35s}  {'Grade':>5s}  {'Composite':>9s}  "
            f"{'ΔF(logic)':>9s}  {'ΔF(social)':>10s}  {'ΔF(clin)':>9s}"
        )
        print(header)
        print(f"  {'─'*35}  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*9}")

        for cert in certificates:
            model_short = cert.model
            if len(model_short) > 35:
                model_short = "..." + model_short[-32:]

            composite = cert.fidelity_score.composite if cert.fidelity_score else 0
            df_logic = cert.truth_gaps.get("logic", None)
            df_social = cert.truth_gaps.get("social", None)
            df_clin = cert.truth_gaps.get("clinical", None)

            print(
                f"  {model_short:<35s}  {cert.grade:>5s}  {composite:>9.3f}  "
                f"{df_logic.delta_f if df_logic else 0:>+9.3f}  "
                f"{df_social.delta_f if df_social else 0:>+10.3f}  "
                f"{df_clin.delta_f if df_clin else 0:>+9.3f}"
            )

        print(f"\n  Total time: {total_time:.1f}s")

    # Save combined results
    combined = {
        "benchmark_version": meta["version"],
        "probe_hash": meta["probe_hash"],
        "n_models": len(certificates),
        "total_elapsed_seconds": total_time,
        "certificates": [c.to_dict() for c in certificates],
    }
    combined_path = output_dir / "fidelity_bench_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Combined results: {combined_path}")


def show_info():
    """Show benchmark dataset information."""
    from rho_eval.benchmarking.loader import get_bench_metadata
    from rho_eval.benchmarking.adversarial import get_all_template_counts, LEVEL_NAMES

    meta = get_bench_metadata()

    print(f"\n  Fidelity-Bench {meta['version']}")
    print(f"  {'─'*40}")
    print(f"  Probe hash:   {meta['probe_hash'][:16]}...")
    print(f"  Total probes: {meta['n_probes']}")
    for d, c in meta['domain_counts'].items():
        print(f"    {d:<10s}  {c} probes")

    print(f"\n  Pressure Levels:")
    for level, count in sorted(get_all_template_counts().items()):
        print(f"    {level}: {LEVEL_NAMES.get(level, '?')} ({count} templates)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Fidelity-Bench 2.0: Adversarial behavioral benchmark experiment.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model names to benchmark",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Quick validation with Qwen-0.5B, reduced probes",
    )
    parser.add_argument(
        "--output-dir", default="results/fidelity_bench",
        help="Directory for saving results",
    )
    parser.add_argument(
        "--pressure-levels", type=int, default=5,
        help="Pressure levels 1-5 (default: 5)",
    )
    parser.add_argument(
        "-n", "--n-probes", type=int, default=None,
        help="Probes per domain (default: all)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Show benchmark dataset info and exit",
    )
    args = parser.parse_args()

    if args.info:
        show_info()
        return

    run_experiment(args)


if __name__ == "__main__":
    main()
