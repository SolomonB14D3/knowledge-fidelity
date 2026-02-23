#!/usr/bin/env python3
"""Audit merged models for the rho leaderboard.

Runs rho-audit across multiple merged model variants and saves
results for comparison. Designed to demonstrate the rho-audit CLI
on real-world mergekit outputs.

Usage:
    python experiments/audit_merged_models.py
    python experiments/audit_merged_models.py --behaviors factual,bias
    python experiments/audit_merged_models.py --quick  # factual only, mandela probes
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_fidelity.behavioral import load_behavioral_probes, evaluate_behavior
from knowledge_fidelity.probes import get_all_probes

RESULTS_DIR = Path(__file__).parent.parent / "results" / "leaderboard"

# Models to audit: baseline + merge variants
# All Yuuta208 models merge Qwen2.5-7B-Instruct + Qwen2.5-Coder-7B
# with different merge strategies — perfect controlled comparison.
MODELS = {
    "qwen2.5-7b-instruct": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "type": "baseline",
        "description": "Base instruct model (no merge)",
    },
    "qwen2.5-7b-slerp": {
        "id": "Yuuta208/Qwen2.5-7B-Instruct-Qwen2.5-Coder-7B-Merged-slerp-29",
        "type": "merge-slerp",
        "description": "SLERP merge: Instruct + Coder",
    },
    "qwen2.5-7b-ties": {
        "id": "Yuuta208/Qwen2.5-7B-Instruct-Qwen2.5-Coder-7B-Merged-ties-29",
        "type": "merge-ties",
        "description": "TIES merge: Instruct + Coder",
    },
    "qwen2.5-7b-dare-ties": {
        "id": "Yuuta208/Qwen2.5-7B-Instruct-Qwen2.5-Coder-7B-Merged-dare_ties-29",
        "type": "merge-dare-ties",
        "description": "DARE-TIES merge: Instruct + Coder",
    },
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(model_id, device=DEVICE):
    """Load model + tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_id}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {time.time()-t0:.1f}s ({n_params/1e9:.2f}B params)")
    return model, tokenizer


def audit_one_model(name, model_info, behaviors, probes_dict, device, results, output_path):
    """Audit a single model across specified behaviors."""
    if name in results:
        print(f"\n  [{name}] Already done — skipping")
        return

    model_id = model_info["id"]
    print(f"\n{'=' * 60}")
    print(f"  AUDITING: {name}")
    print(f"  Model:    {model_id}")
    print(f"  Type:     {model_info['type']}")
    print(f"{'=' * 60}")

    model, tokenizer = load_model(model_id, device)

    t0 = time.time()
    behavior_results = {}
    for behavior in behaviors:
        probes = probes_dict[behavior]
        bt0 = time.time()
        r = evaluate_behavior(behavior, model, tokenizer, probes, device)
        dt = time.time() - bt0
        print(f"    [{behavior}] rho={r['rho']:.4f} ({dt:.1f}s)")
        behavior_results[behavior] = {k: v for k, v in r.items() if k != "details"}

    elapsed = time.time() - t0

    results[name] = {
        "model_id": model_id,
        "model_type": model_info["type"],
        "description": model_info["description"],
        "behaviors": behavior_results,
        "elapsed_seconds": round(elapsed, 1),
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }

    # Save after each model (crash-safe)
    _save(results, output_path)
    print(f"  Saved ({elapsed:.0f}s total)")

    # Free memory
    del model, tokenizer
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _save(results, path):
    """Atomic JSON save."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=float)
    tmp.rename(path)


def print_leaderboard(results, behaviors):
    """Print a comparison table."""
    print(f"\n{'=' * 80}")
    print(f"  LEADERBOARD")
    print(f"{'=' * 80}")

    # Header
    header = f"  {'Model':<28}"
    for b in behaviors:
        header += f" {b:>10}"
    print(header)
    print("  " + "-" * (28 + 11 * len(behaviors)))

    # Rows sorted by factual rho (if available)
    sorted_names = sorted(results.keys(),
                          key=lambda n: results[n]["behaviors"].get("factual", {}).get("rho", 0),
                          reverse=True)

    for name in sorted_names:
        data = results[name]
        row = f"  {name:<28}"
        for b in behaviors:
            rho = data["behaviors"].get(b, {}).get("rho", None)
            if rho is not None:
                row += f" {rho:>10.3f}"
            else:
                row += f" {'—':>10}"
        row += f"  ({data['model_type']})"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Audit merged models for rho leaderboard")
    parser.add_argument("--behaviors", type=str, default="factual,bias,sycophancy",
                        help="Comma-separated behaviors (default: factual,bias,sycophancy)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: factual only")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific model keys to audit")
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    if args.quick:
        behaviors = ["factual"]
    else:
        behaviors = [b.strip() for b in args.behaviors.split(",")]

    models = MODELS
    if args.models:
        models = {k: v for k, v in MODELS.items() if k in args.models}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "merged_audit.json"

    # Load or resume results
    results = {}
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resumed {len(results)} entries from {output_path}")

    # Load probes once
    print("Loading probes...")
    probes_dict = {}
    for b in behaviors:
        if b == "factual":
            probes_dict[b] = get_all_probes()
        else:
            probes_dict[b] = load_behavioral_probes(b, seed=42)
        print(f"  {b}: {len(probes_dict[b])} probes")

    # Audit each model
    for name, model_info in models.items():
        audit_one_model(name, model_info, behaviors, probes_dict,
                        args.device, results, output_path)

    # Print leaderboard
    print_leaderboard(results, behaviors)

    print(f"\nResults: {output_path}")


if __name__ == "__main__":
    main()
