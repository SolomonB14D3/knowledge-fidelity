#!/usr/bin/env python3
"""
Cross-Behavioral CF90 Denoising Sweep

Tests whether the denoising effect (rho improvement at 60-70% CF90)
generalizes beyond factual/myth probes to toxicity, bias, sycophancy,
and reasoning collapse.

For each (model, ratio, seed, behavior):
  1. Load fresh model
  2. Evaluate baseline (pre-compression)
  3. Apply CF90 at the given ratio
  4. Evaluate post-compression
  5. Compute denoising delta = post_rho - pre_rho

Usage:
    # Full sweep (both models, all ratios, 3 seeds)
    python experiments/cross_behavioral_sweep.py

    # Single model, key ratios
    python experiments/cross_behavioral_sweep.py --models qwen2.5-7b --ratios 0.6,0.7,0.8

    # Quick validation on small model
    python experiments/cross_behavioral_sweep.py --validate

    # Resume interrupted sweep
    python experiments/cross_behavioral_sweep.py --resume results/cross_behavioral/sweep.json
"""

import argparse
import copy
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from knowledge_fidelity.svd import compress_qko, freeze_layers
from knowledge_fidelity.svd.freeze import unfreeze_all
from knowledge_fidelity.probes import get_all_probes
from knowledge_fidelity.behavioral import load_behavioral_probes, evaluate_behavior

# ── Configuration ─────────────────────────────────────────────────────────

MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

RATIOS = [0.50, 0.60, 0.70, 0.80, 0.90]
SEEDS = [0, 1, 2]
BEHAVIORS = ["factual", "toxicity", "bias", "sycophancy", "reasoning"]
DEVICE = "cpu"

RESULTS_DIR = Path(__file__).parent.parent / "results" / "cross_behavioral"


# ── Helpers ───────────────────────────────────────────────────────────────

def load_model(model_id, device=DEVICE):
    """Load a fresh model and tokenizer."""
    print(f"  Loading {model_id}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {time.time()-t0:.1f}s ({n_params/1e9:.2f}B params)")
    return model, tokenizer


def load_all_behavioral_probes(behaviors, seed=42):
    """Load probes for all behaviors."""
    probes = {}
    for behavior in behaviors:
        print(f"  Loading {behavior} probes...")
        if behavior == "factual":
            probes[behavior] = get_all_probes()
        else:
            probes[behavior] = load_behavioral_probes(behavior, seed=seed)
        print(f"    {len(probes[behavior])} probes loaded")
    return probes


# ── Single condition ──────────────────────────────────────────────────────

def run_condition(model_id, ratio, seed, behaviors, probes_dict, device=DEVICE):
    """Run one (model, ratio, seed) condition across all behaviors.

    Returns (results_dict, compress_stats_dict)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, tokenizer = load_model(model_id, device)
    results = {}

    # Baseline evaluation
    for behavior in behaviors:
        t0 = time.time()
        baseline = evaluate_behavior(behavior, model, tokenizer, probes_dict[behavior], device)
        dt = time.time() - t0
        print(f"    [{behavior}] baseline rho={baseline['rho']:.4f} ({dt:.1f}s)")
        results[behavior] = {
            "baseline": {k: v for k, v in baseline.items() if k != "details"},
        }

    # Apply CF90 compression
    print(f"    Applying CF90 at ratio={ratio:.0%}...")
    t0 = time.time()
    n_compressed = compress_qko(model, ratio=ratio)
    freeze_stats = freeze_layers(model, ratio=0.75)
    model.eval()
    compress_time = time.time() - t0
    print(f"    Compressed {n_compressed} matrices, "
          f"froze {freeze_stats['n_frozen']}/{freeze_stats['n_layers']} layers ({compress_time:.1f}s)")

    # Post-compression evaluation
    for behavior in behaviors:
        t0 = time.time()
        post = evaluate_behavior(behavior, model, tokenizer, probes_dict[behavior], device)
        dt = time.time() - t0
        delta = post["rho"] - results[behavior]["baseline"]["rho"]
        arrow = "+" if delta > 0 else ""
        print(f"    [{behavior}] post rho={post['rho']:.4f} (delta={arrow}{delta:.4f}, {dt:.1f}s)")
        results[behavior]["compressed"] = {k: v for k, v in post.items() if k != "details"}
        results[behavior]["delta"] = delta

    del model, tokenizer
    gc.collect()

    return results, {"n_compressed": n_compressed, **freeze_stats, "compress_time": compress_time}


# ── Full sweep ────────────────────────────────────────────────────────────

def run_sweep(models, ratios, seeds, behaviors, device, resume_path=None):
    """Run the full cross-behavioral denoising sweep."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LOADING PROBES")
    print("=" * 70)
    probes_dict = load_all_behavioral_probes(behaviors, seed=42)

    # Load or init results
    all_results = {}
    if resume_path and Path(resume_path).exists():
        with open(resume_path) as f:
            all_results = json.load(f)
        print(f"Resumed {len(all_results)} conditions from {resume_path}")

    total = len(models) * len(ratios) * len(seeds)
    done = 0
    output_path = RESULTS_DIR / "sweep.json"

    for model_name, model_id in models.items():
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 70}")

        for ratio in ratios:
            for seed in seeds:
                key = f"{model_name}_r{ratio:.2f}_s{seed}"
                done += 1

                if key in all_results:
                    print(f"\n  [{done}/{total}] {key} — SKIPPED (done)")
                    continue

                print(f"\n  [{done}/{total}] {key}")
                t0 = time.time()
                try:
                    cond_results, compress_stats = run_condition(
                        model_id, ratio, seed, behaviors, probes_dict, device)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    cond_results = {"error": str(e)}
                    compress_stats = {}

                all_results[key] = {
                    "model": model_name,
                    "model_id": model_id,
                    "ratio": ratio,
                    "seed": seed,
                    "behaviors": cond_results,
                    "compress_stats": compress_stats,
                    "elapsed_seconds": time.time() - t0,
                    "timestamp": datetime.now().isoformat(),
                }

                # Save after each condition
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2, default=float)

    print(f"\n{'=' * 70}")
    print(f"SWEEP COMPLETE — {len(all_results)} conditions")
    print(f"Results: {output_path}")
    print(f"{'=' * 70}")
    return all_results


# ── Analysis ──────────────────────────────────────────────────────────────

def analyze(results):
    """Analyze sweep results and print summary."""
    from scipy import stats
    from collections import defaultdict

    table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for key, data in results.items():
        if "error" in data.get("behaviors", {}):
            continue
        model = data["model"]
        ratio = data["ratio"]
        for behavior, bdata in data.get("behaviors", {}).items():
            if isinstance(bdata, str) or "delta" not in bdata:
                continue
            table[model][behavior][ratio].append(bdata["delta"])

    for model in sorted(table):
        print(f"\n{'=' * 90}")
        print(f"MODEL: {model}")
        print(f"{'=' * 90}")

        behaviors = sorted(table[model])
        ratios = sorted(set(r for b in behaviors for r in table[model][b]))

        header = f"{'Behavior':<14}"
        for r in ratios:
            header += f" | {r:>5.0%} delta  p"
        print(header)
        print("-" * len(header))

        denoising_at_60_70 = []
        for behavior in behaviors:
            row = f"{behavior:<14}"
            for r in ratios:
                deltas = table[model][behavior].get(r, [])
                if len(deltas) >= 2:
                    mean_d = np.mean(deltas)
                    _, p = stats.ttest_1samp(deltas, 0)
                    sig = "*" if p < 0.05 else " "
                    row += f" | {mean_d:>+6.4f}{sig} {p:.2f}"
                elif deltas:
                    row += f" | {deltas[0]:>+6.4f}  n=1"
                else:
                    row += f" |     —      "
            print(row)

            # Check 60-70% denoising
            for r in [0.60, 0.70]:
                deltas = table[model][behavior].get(r, [])
                if deltas and np.mean(deltas) > 0:
                    if len(deltas) >= 2:
                        _, p = stats.ttest_1samp(deltas, 0)
                        if p < 0.10:
                            denoising_at_60_70.append(behavior)
                            break
                    else:
                        denoising_at_60_70.append(behavior)
                        break

        n = len(set(denoising_at_60_70))
        total_b = len(behaviors)
        print(f"\nDenoising at 60-70%: {n}/{total_b} behaviors")
        if n >= 3:
            print(f"  BEST CASE: Universal behavioral regularizer ({', '.join(set(denoising_at_60_70))})")
        elif n >= 2:
            print(f"  GOOD CASE: Multi-behavior denoising ({', '.join(set(denoising_at_60_70))})")
        elif n == 1:
            print(f"  NEUTRAL: Effect is {denoising_at_60_70[0]}-specific")
        else:
            print(f"  WORST CASE: No generalization")


# ── Validation ────────────────────────────────────────────────────────────

def validate():
    """Quick pipeline validation on Qwen-0.5B with small probes."""
    print("=" * 70)
    print("VALIDATION (Qwen-0.5B, reduced probes)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load tiny probe sets
    probes = {"factual": get_all_probes()[:10]}
    for behavior in ["toxicity", "bias", "sycophancy", "reasoning"]:
        try:
            n = 10 if behavior == "reasoning" else 20
            probes[behavior] = load_behavioral_probes(behavior, n=n, seed=42)
            print(f"  {behavior}: {len(probes[behavior])} probes")
        except Exception as e:
            print(f"  {behavior}: FAILED — {e}")

    model, tokenizer = load_model("Qwen/Qwen2.5-0.5B", DEVICE)
    available = list(probes.keys())

    # Baseline
    print("\n--- Baseline ---")
    baseline = {}
    for b in available:
        r = evaluate_behavior(b, model, tokenizer, probes[b], DEVICE)
        baseline[b] = r
        print(f"  {b}: rho={r['rho']:.4f}")

    # Compress
    print("\n--- CF90 at 70% ---")
    compress_qko(model, ratio=0.70)
    freeze_layers(model, ratio=0.75)
    model.eval()

    # Post
    print("\n--- Post-compression ---")
    deltas = {}
    for b in available:
        r = evaluate_behavior(b, model, tokenizer, probes[b], DEVICE)
        d = r["rho"] - baseline[b]["rho"]
        deltas[b] = d
        print(f"  {b}: rho={r['rho']:.4f} (delta={d:+.4f})")

    print(f"\n{'Behavior':<14} {'Baseline':>10} {'Post':>10} {'Delta':>10}")
    print("-" * 50)
    for b in available:
        print(f"{b:<14} {baseline[b]['rho']:>10.4f} {baseline[b]['rho']+deltas[b]:>10.4f} {deltas[b]:>+10.4f}")

    validation = {"model": "Qwen-0.5B", "ratio": 0.70, "deltas": deltas}
    with open(RESULTS_DIR / "validation.json", "w") as f:
        json.dump(validation, f, indent=2, default=float)

    print(f"\nValidation saved to {RESULTS_DIR / 'validation.json'}")
    del model, tokenizer
    gc.collect()


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-behavioral CF90 denoising sweep")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model short names (default: all)")
    parser.add_argument("--ratios", default=None,
                        help="Comma-separated compression ratios")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--behaviors", nargs="+", default=None)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--resume", default=None,
                        help="Resume from partial results JSON")
    parser.add_argument("--validate", action="store_true",
                        help="Quick validation on Qwen-0.5B")
    parser.add_argument("--analyze", default=None,
                        help="Analyze existing results JSON")
    args = parser.parse_args()

    if args.validate:
        validate()
        return

    if args.analyze:
        with open(args.analyze) as f:
            results = json.load(f)
        analyze(results)
        return

    models = MODELS
    if args.models:
        models = {k: v for k, v in MODELS.items() if k in args.models}
    ratios = [float(r) for r in args.ratios.split(",")] if args.ratios else RATIOS

    results = run_sweep(
        models=models,
        ratios=ratios,
        seeds=args.seeds or SEEDS,
        behaviors=args.behaviors or BEHAVIORS,
        device=args.device,
        resume_path=args.resume,
    )
    analyze(results)


if __name__ == "__main__":
    main()
