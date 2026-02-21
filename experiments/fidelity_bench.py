"""Fidelity-Bench: benchmark any model across all probe categories.

Run a full audit across all probe categories and output a per-category
score table. Designed so any compression paper can add one line:

    python experiments/fidelity_bench.py --model my-compressed-model

Usage:
    python experiments/fidelity_bench.py --model Qwen/Qwen2.5-0.5B
    python experiments/fidelity_bench.py --model Qwen/Qwen2.5-7B-Instruct --json
    python experiments/fidelity_bench.py --model ./compressed_output --device cuda
"""

import argparse
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from knowledge_fidelity.core import _run_audit
from knowledge_fidelity.probes import (
    get_default_probes,
    get_mandela_probes,
    get_medical_probes,
)


def get_all_categories():
    """Return all probe categories with names and probe lists."""
    categories = [
        ("default", get_default_probes()),
        ("mandela", get_mandela_probes()),
        ("medical", get_medical_probes()),
    ]
    # Try loading file-based probes
    try:
        from knowledge_fidelity.probes import get_commonsense_probes
        categories.append(("commonsense", get_commonsense_probes()))
    except (ImportError, FileNotFoundError):
        pass
    try:
        from knowledge_fidelity.probes import get_truthfulqa_probes
        categories.append(("truthfulqa", get_truthfulqa_probes()))
    except (ImportError, FileNotFoundError):
        pass
    return categories


def main():
    parser = argparse.ArgumentParser(description="Fidelity-Bench: per-category audit")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", dest="json_output", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True,
    ).to(args.device)
    model.eval()

    categories = get_all_categories()
    results = []
    start = time.time()

    for name, probes in categories:
        print(f"\nAuditing {name} ({len(probes)} probes)...")
        audit = _run_audit(model, tokenizer, probes, args.device)
        result = {
            "category": name,
            "n_probes": len(probes),
            "rho": audit["rho"],
            "rho_p": audit["rho_p"],
            "mean_delta": audit["mean_delta"],
            "n_positive": audit["n_positive_delta"],
            "accuracy": audit["n_positive_delta"] / len(probes) if probes else 0,
        }
        results.append(result)
        print(f"  rho={audit['rho']:.3f} | "
              f"{audit['n_positive_delta']}/{len(probes)} correct | "
              f"mean_delta={audit['mean_delta']:.4f}")

    elapsed = time.time() - start

    full_output = {
        "model": args.model,
        "categories": results,
        "elapsed_seconds": elapsed,
    }

    if args.json_output:
        print(json.dumps(full_output, indent=2))
    else:
        # Markdown table
        print(f"\n{'='*70}")
        print(f"Fidelity-Bench: {args.model}")
        print(f"{'='*70}")
        print(f"\n| Category | Probes | rho | Correct | Accuracy | Mean Delta |")
        print(f"|----------|--------|-----|---------|----------|------------|")
        for r in results:
            print(f"| {r['category']:<8} | {r['n_probes']:>6} | "
                  f"{r['rho']:>5.3f} | {r['n_positive']:>3}/{r['n_probes']:<3} | "
                  f"{r['accuracy']:>7.0%} | {r['mean_delta']:>+10.4f} |")
        print(f"\nTotal time: {elapsed:.1f}s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(full_output, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
