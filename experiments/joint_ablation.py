#!/usr/bin/env python3
"""
Joint ablation: Compress at multiple ratios, measure confidence preservation.

Tests: How much false-belief sensor signal survives compression?

For each compression ratio (50%, 60%, 70%, 80%, 90%, 100%):
  1. Compress model with CF90
  2. Measure Mandela + medical + default confidence ratios
  3. Compare rho before vs after

Output: Table + JSON results for paper figures.

Usage:
    python experiments/joint_ablation.py
    python experiments/joint_ablation.py --model meta-llama/Llama-3.1-8B-Instruct
    python experiments/joint_ablation.py --ratios 0.5,0.7,0.9
"""

import sys
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from knowledge_fidelity.svd import compress_qko, freeze_layers
from knowledge_fidelity.core import _run_audit
from knowledge_fidelity.probes import (
    get_default_probes,
    get_mandela_probes,
    get_medical_probes,
)


def run_ablation(model_name: str, ratios: list[float], device: str = "cpu"):
    """Run compression ablation across multiple ratios."""

    all_probes = {
        "default": get_default_probes(),
        "mandela": get_mandela_probes(),
        "medical": get_medical_probes(),
    }

    results = []

    for ratio in ratios:
        print(f"\n{'='*60}")
        print(f"Compression ratio: {ratio:.0%}")
        print(f"{'='*60}")

        # Fresh model load for each ratio
        print(f"  Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True,
        ).to(device)
        model.eval()

        # Audit BEFORE
        print("  Auditing BEFORE compression...")
        audits_before = {}
        for probe_set, probes in all_probes.items():
            audits_before[probe_set] = _run_audit(model, tokenizer, probes, device)

        # Compress (skip if ratio == 1.0 = no compression)
        if ratio < 1.0:
            n_compressed = compress_qko(model, ratio=ratio)
            freeze_stats = freeze_layers(model, ratio=0.75)
            print(f"  Compressed {n_compressed} matrices, "
                  f"frozen {freeze_stats['n_frozen']}/{freeze_stats['n_layers']} layers")
        else:
            n_compressed = 0

        # Audit AFTER
        print("  Auditing AFTER compression...")
        model.eval()
        audits_after = {}
        for probe_set, probes in all_probes.items():
            audits_after[probe_set] = _run_audit(model, tokenizer, probes, device)

        # Summarize
        entry = {"ratio": ratio, "n_compressed": n_compressed}
        for probe_set in all_probes:
            entry[f"{probe_set}_rho_before"] = audits_before[probe_set]["rho"]
            entry[f"{probe_set}_rho_after"] = audits_after[probe_set]["rho"]
            entry[f"{probe_set}_mean_delta_before"] = audits_before[probe_set]["mean_delta"]
            entry[f"{probe_set}_mean_delta_after"] = audits_after[probe_set]["mean_delta"]
            entry[f"{probe_set}_n_positive_before"] = audits_before[probe_set]["n_positive_delta"]
            entry[f"{probe_set}_n_positive_after"] = audits_after[probe_set]["n_positive_delta"]

        results.append(entry)

        # Print summary for this ratio
        for probe_set in all_probes:
            rho_b = entry[f"{probe_set}_rho_before"]
            rho_a = entry[f"{probe_set}_rho_after"]
            pos_b = entry[f"{probe_set}_n_positive_before"]
            pos_a = entry[f"{probe_set}_n_positive_after"]
            n_probes = len(all_probes[probe_set])
            print(f"  {probe_set:>10}: rho {rho_b:.3f} -> {rho_a:.3f} | "
                  f"positive {pos_b}/{n_probes} -> {pos_a}/{n_probes}")

        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def main():
    parser = argparse.ArgumentParser(description="Joint Ablation: Compression vs Confidence")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--ratios", default="0.5,0.6,0.7,0.8,0.9,1.0",
                       help="Comma-separated compression ratios")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    ratios = [float(r) for r in args.ratios.split(",")]
    results = run_ablation(args.model, ratios, args.device)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Ratio':>6} | {'Default rho':>15} | {'Mandela rho':>15} | {'Medical rho':>15}")
    print("-" * 60)
    for r in results:
        dr = f"{r['default_rho_before']:.3f}->{r['default_rho_after']:.3f}"
        mr = f"{r['mandela_rho_before']:.3f}->{r['mandela_rho_after']:.3f}"
        med = f"{r['medical_rho_before']:.3f}->{r['medical_rho_after']:.3f}"
        print(f"{r['ratio']:>5.0%} | {dr:>15} | {mr:>15} | {med:>15}")

    # Save results
    output_path = args.output or f"results/joint_ablation_{Path(args.model).name}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
