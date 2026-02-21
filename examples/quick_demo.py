#!/usr/bin/env python3
"""
Quick demo: Compress an LLM and audit its knowledge fidelity in one call.

Usage:
    python examples/quick_demo.py
    python examples/quick_demo.py --model Qwen/Qwen2.5-0.5B
    python examples/quick_demo.py --model meta-llama/Llama-3.1-8B-Instruct

Runtime: ~5 min for 0.5B on CPU, ~30 min for 7-8B
"""

import sys
import argparse
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_fidelity import compress_and_audit


def main():
    parser = argparse.ArgumentParser(description="Knowledge Fidelity Quick Demo")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name (default: Qwen-0.5B for speed)",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.7,
        help="SVD compression ratio (default: 0.7 = keep 70%%)",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"],
        help="Device (use CPU for training stability)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Knowledge Fidelity — Compress & Audit Demo")
    print("=" * 70)

    report = compress_and_audit(
        model_name_or_path=args.model,
        ratio=args.ratio,
        device=args.device,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Compression: {args.ratio:.0%} rank ({report['compression']['n_compressed']} matrices)")
    print(f"  Frozen:      {report['freeze']['n_frozen']}/{report['freeze']['n_layers']} layers")
    print(f"  Retention:   {report['retention']:.0%}")
    print(f"  rho before:  {report['rho_before']:.3f}")
    print(f"  rho after:   {report['rho_after']:.3f}")
    print(f"  rho drop:    {report['rho_before'] - report['rho_after']:.3f}")
    print(f"  Time:        {report['elapsed_seconds']:.1f}s")

    # Show per-probe details
    print("\n  Per-probe confidence deltas (true - false):")
    audit = report["audit_after"]
    for i, (tc, fc, delta) in enumerate(zip(
        audit["true_confs"], audit["false_confs"], audit["deltas"]
    )):
        marker = "+" if delta > 0 else "-"
        print(f"    [{marker}] delta={delta:+.3f}  (true={tc:.3f}, false={fc:.3f})")

    print(f"\n  {audit['n_positive_delta']}/{audit['n_probes']} probes still "
          f"show higher confidence on truth after compression")


if __name__ == "__main__":
    main()
