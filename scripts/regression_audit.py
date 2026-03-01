#!/usr/bin/env python3
"""Behavioral regression testing for LLM audits.

Generates a baseline behavioral profile for a model and checks subsequent
runs against it. Any behavior that drops by more than a threshold triggers
a non-zero exit code — suitable for CI gating.

Usage:
    # Generate baseline (run once per model)
    python scripts/regression_audit.py Qwen/Qwen2.5-0.5B --generate-baseline

    # Check regression (run in CI or before release)
    python scripts/regression_audit.py Qwen/Qwen2.5-0.5B

    # Custom threshold
    python scripts/regression_audit.py Qwen/Qwen2.5-0.5B --threshold 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINES_DIR = ROOT / "baselines"


def safe_model_name(model_name: str) -> str:
    """Convert model name to filesystem-safe string."""
    return model_name.replace("/", "_").replace("\\", "_")


def generate_baseline(
    model_name: str,
    device: str = "cpu",
    n: int = 150,
    behaviors: str = "all",
) -> dict:
    """Run rho-eval audit and save as baseline.

    Args:
        model_name: HuggingFace model name.
        device: Torch device.
        n: Number of probes per behavior.
        behaviors: Comma-separated behaviors or "all".

    Returns:
        Baseline dict with rho scores per behavior.
    """
    import rho_eval

    print(f"  Generating baseline for {model_name}")
    print(f"  Device: {device}, Probes: {n}, Behaviors: {behaviors}")

    report = rho_eval.audit(
        model_name,
        behaviors=behaviors,
        n=n,
        device=device,
    )

    baseline = {
        "model_name": model_name,
        "rho_eval_version": rho_eval.__version__,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_probes": n,
        "behaviors": behaviors,
        "device": device,
        "scores": {},
    }

    for result in report.results:
        baseline["scores"][result.behavior] = {
            "rho": round(result.rho, 6),
            "p_value": round(result.p_value, 6) if result.p_value is not None else None,
            "n_probes": result.n_probes,
            "status": result.status,
        }

    # Save
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINES_DIR / f"{safe_model_name(model_name)}.json"
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\n  Saved baseline: {path}")
    print(f"  Behaviors: {list(baseline['scores'].keys())}")
    for beh, score in baseline["scores"].items():
        print(f"    {beh}: ρ={score['rho']:.4f} ({score['status']})")

    return baseline


def check_regression(
    model_name: str,
    threshold: float = 0.005,
    device: str = "cpu",
    n: int = 150,
    behaviors: str = "all",
) -> bool:
    """Run audit and compare against saved baseline.

    Args:
        model_name: HuggingFace model name.
        threshold: Maximum allowed ρ drop before failing.
        device: Torch device.
        n: Number of probes per behavior.
        behaviors: Comma-separated behaviors or "all".

    Returns:
        True if all behaviors pass, False if any regression detected.
    """
    import rho_eval

    # Load baseline
    path = BASELINES_DIR / f"{safe_model_name(model_name)}.json"
    if not path.exists():
        print(f"  ERROR: No baseline found at {path}")
        print(f"  Run with --generate-baseline first")
        return False

    with open(path) as f:
        baseline = json.load(f)

    print(f"  Checking regression for {model_name}")
    print(f"  Baseline from: {baseline['generated_at']}")
    print(f"  Threshold: {threshold}")

    # Run fresh audit
    report = rho_eval.audit(
        model_name,
        behaviors=behaviors,
        n=n,
        device=device,
    )

    # Compare
    all_pass = True
    print(f"\n  {'Behavior':<15} {'Baseline':>10} {'Current':>10} {'Delta':>10} {'Status':>8}")
    print(f"  {'─' * 55}")

    for result in report.results:
        beh = result.behavior
        current_rho = result.rho

        if beh not in baseline["scores"]:
            print(f"  {beh:<15} {'N/A':>10} {current_rho:>10.4f} {'N/A':>10} {'NEW':>8}")
            continue

        baseline_rho = baseline["scores"][beh]["rho"]
        delta = current_rho - baseline_rho

        if delta < -threshold:
            status = "FAIL ✗"
            all_pass = False
        elif delta > threshold:
            status = "BETTER"
        else:
            status = "PASS ✓"

        print(f"  {beh:<15} {baseline_rho:>10.4f} {current_rho:>10.4f} {delta:>+10.4f} {status:>8}")

    print()
    if all_pass:
        print("  RESULT: PASS — no behavioral regressions detected")
    else:
        print(f"  RESULT: FAIL — one or more behaviors dropped by >{threshold}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral regression testing for LLM audits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", help="HuggingFace model name (e.g., Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--generate-baseline", action="store_true",
                        help="Generate a new baseline (overwrites existing)")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="Max allowed ρ drop (default: 0.005)")
    parser.add_argument("--device", default="cpu",
                        help="Torch device (default: cpu)")
    parser.add_argument("-n", type=int, default=150,
                        help="Number of probes per behavior (default: 150)")
    parser.add_argument("--behaviors", default="all",
                        help="Behaviors to test (comma-separated or 'all')")

    args = parser.parse_args()

    if args.generate_baseline:
        generate_baseline(
            args.model,
            device=args.device,
            n=args.n,
            behaviors=args.behaviors,
        )
        sys.exit(0)
    else:
        passed = check_regression(
            args.model,
            threshold=args.threshold,
            device=args.device,
            n=args.n,
            behaviors=args.behaviors,
        )
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
