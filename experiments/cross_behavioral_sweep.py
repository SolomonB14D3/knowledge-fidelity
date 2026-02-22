#!/usr/bin/env python3
"""
Cross-Behavioral CF90 Denoising Sweep

Tests whether the denoising effect (rho improvement at 60-70% CF90)
generalizes beyond factual/myth probes to toxicity, bias, sycophancy,
and reasoning collapse.

Architecture (optimized):
  - Load each model ONCE, deepcopy state_dict, restore between ratios
  - Baseline evaluated ONCE per model (doesn't depend on ratio)
  - SVD is deterministic → seeds only matter for generation evals
  - Generation uses greedy decoding → seeds are redundant → run 1 seed

Usage:
    python experiments/cross_behavioral_sweep.py
    python experiments/cross_behavioral_sweep.py --models qwen2.5-7b --ratios 0.6,0.7,0.8
    python experiments/cross_behavioral_sweep.py --validate
    python experiments/cross_behavioral_sweep.py --analyze results/cross_behavioral/sweep.json
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
BEHAVIORS = ["factual", "toxicity", "bias", "sycophancy", "reasoning"]
DEVICE = "cpu"

RESULTS_DIR = Path(__file__).parent.parent / "results" / "cross_behavioral"


# ── Helpers ───────────────────────────────────────────────────────────────

def load_model(model_id, device=DEVICE):
    """Load model and tokenizer."""
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


def eval_all_behaviors(model, tokenizer, behaviors, probes_dict, device):
    """Evaluate all behaviors, return dict of results (without details)."""
    results = {}
    for behavior in behaviors:
        t0 = time.time()
        r = evaluate_behavior(behavior, model, tokenizer, probes_dict[behavior], device)
        dt = time.time() - t0
        print(f"    [{behavior}] rho={r['rho']:.4f} ({dt:.1f}s)")
        results[behavior] = {k: v for k, v in r.items() if k != "details"}
    return results


# ── Per-model sweep (load once, restore between ratios) ──────────────────

def sweep_one_model(model_name, model_id, ratios, behaviors, probes_dict,
                    device, all_results, output_path):
    """Run all ratios for one model. Single model load."""

    # Check which ratios still need running
    needed = [r for r in ratios
              if f"{model_name}_r{r:.2f}" not in all_results]
    if not needed:
        print(f"  All ratios already done for {model_name}, skipping.")
        return

    model, tokenizer = load_model(model_id, device)

    # ── Baseline (once per model) ─────────────────────────────────────
    baseline_key = f"{model_name}_baseline"
    if baseline_key in all_results:
        baseline = all_results[baseline_key]["behaviors"]
        print(f"  Baseline loaded from cache")
    else:
        print(f"\n  Evaluating baseline (no compression)...")
        baseline = eval_all_behaviors(model, tokenizer, behaviors, probes_dict, device)
        all_results[baseline_key] = {
            "model": model_name,
            "model_id": model_id,
            "ratio": 1.0,
            "behaviors": baseline,
            "timestamp": datetime.now().isoformat(),
        }
        _save(all_results, output_path)

    # ── Save original weights ─────────────────────────────────────────
    print(f"  Saving original state dict...")
    t0 = time.time()
    original_state = copy.deepcopy(model.state_dict())
    print(f"  Saved in {time.time()-t0:.1f}s")

    # ── Sweep ratios ──────────────────────────────────────────────────
    for i, ratio in enumerate(ratios):
        key = f"{model_name}_r{ratio:.2f}"
        if key in all_results:
            print(f"\n  [{i+1}/{len(ratios)}] ratio={ratio:.0%} — SKIPPED (done)")
            continue

        print(f"\n  [{i+1}/{len(ratios)}] ratio={ratio:.0%}")

        # Restore original weights
        model.load_state_dict(original_state)
        model.eval()

        # Compress
        t0 = time.time()
        n_compressed = compress_qko(model, ratio=ratio)
        freeze_stats = freeze_layers(model, ratio=0.75)
        model.eval()
        compress_time = time.time() - t0
        print(f"    Compressed {n_compressed} matrices, "
              f"froze {freeze_stats['n_frozen']}/{freeze_stats['n_layers']} layers "
              f"({compress_time:.1f}s)")

        # Evaluate
        post = eval_all_behaviors(model, tokenizer, behaviors, probes_dict, device)

        # Compute deltas
        behavior_results = {}
        for behavior in behaviors:
            delta = post[behavior]["rho"] - baseline[behavior]["rho"]
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"    [{behavior}] delta={delta:+.4f} {arrow}")
            behavior_results[behavior] = {
                "baseline": baseline[behavior],
                "compressed": post[behavior],
                "delta": delta,
            }

        all_results[key] = {
            "model": model_name,
            "model_id": model_id,
            "ratio": ratio,
            "behaviors": behavior_results,
            "compress_stats": {
                "n_compressed": n_compressed,
                **freeze_stats,
                "compress_time": compress_time,
            },
            "timestamp": datetime.now().isoformat(),
        }
        _save(all_results, output_path)

    del model, tokenizer, original_state
    gc.collect()


def _save(results, path):
    """Save results JSON (atomic-ish)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=float)
    tmp.rename(path)


# ── Full sweep ────────────────────────────────────────────────────────────

def run_sweep(models, ratios, behaviors, device, resume_path=None):
    """Run the full cross-behavioral denoising sweep."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LOADING PROBES")
    print("=" * 70)
    probes_dict = load_all_behavioral_probes(behaviors, seed=42)

    # Load or init results
    output_path = RESULTS_DIR / "sweep.json"
    all_results = {}
    if resume_path and Path(resume_path).exists():
        with open(resume_path) as f:
            all_results = json.load(f)
        print(f"Resumed {len(all_results)} entries from {resume_path}")
    elif output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)
        print(f"Auto-resumed {len(all_results)} entries from {output_path}")

    for model_name, model_id in models.items():
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name} ({model_id})")
        print(f"{'=' * 70}")
        sweep_one_model(model_name, model_id, ratios, behaviors,
                        probes_dict, device, all_results, output_path)

    print(f"\n{'=' * 70}")
    print(f"SWEEP COMPLETE — {len(all_results)} entries")
    print(f"Results: {output_path}")
    print(f"{'=' * 70}")
    return all_results


# ── Analysis ──────────────────────────────────────────────────────────────

def analyze(results):
    """Analyze sweep results and print summary table."""
    from scipy import stats
    from collections import defaultdict

    # Collect deltas: table[model][behavior][ratio] = delta
    table = defaultdict(lambda: defaultdict(dict))

    for key, data in results.items():
        if "_baseline" in key or "error" in data.get("behaviors", {}):
            continue
        model = data["model"]
        ratio = data["ratio"]
        for behavior, bdata in data.get("behaviors", {}).items():
            if isinstance(bdata, str) or "delta" not in bdata:
                continue
            table[model][behavior][ratio] = bdata["delta"]

    for model in sorted(table):
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model}")
        print(f"{'=' * 80}")

        behaviors = sorted(table[model])
        ratios = sorted(set(r for b in behaviors for r in table[model][b]))

        # Print header
        header = f"{'Behavior':<14}"
        for r in ratios:
            header += f" | {r:>6.0%}"
        print(header)
        print("-" * len(header))

        # Print rows
        denoising_60_70 = []
        for behavior in behaviors:
            row = f"{behavior:<14}"
            for r in ratios:
                d = table[model][behavior].get(r)
                if d is not None:
                    row += f" | {d:>+6.3f}"
                    if r in (0.60, 0.70) and d > 0:
                        denoising_60_70.append(behavior)
                else:
                    row += f" |     —"
            print(row)

        # Best ratio per behavior
        print(f"\nBest denoising ratio per behavior:")
        for behavior in behaviors:
            if not table[model][behavior]:
                continue
            best_r = max(table[model][behavior],
                         key=lambda r: table[model][behavior][r])
            best_d = table[model][behavior][best_r]
            marker = " **" if best_d > 0 else ""
            print(f"  {behavior:<14}: {best_r:.0%} (delta={best_d:+.4f}){marker}")

        # Verdict
        unique = list(set(denoising_60_70))
        n = len(unique)
        total_b = len(behaviors)
        print(f"\nDenoising at 60-70%: {n}/{total_b} behaviors")
        if n >= 3:
            print(f"  → BEST CASE: Universal behavioral regularizer ({', '.join(unique)})")
        elif n >= 2:
            print(f"  → GOOD CASE: Multi-behavior denoising ({', '.join(unique)})")
        elif n == 1:
            print(f"  → NEUTRAL: Effect is {unique[0]}-specific")
        else:
            print(f"  → No denoising observed at 60-70%")


# ── Validation ────────────────────────────────────────────────────────────

def validate():
    """Quick pipeline validation on Qwen-0.5B."""
    print("=" * 70)
    print("VALIDATION (Qwen-0.5B, reduced probes)")
    print("=" * 70)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    probes = {"factual": get_all_probes()[:10]}
    for b in ["toxicity", "bias", "sycophancy", "reasoning"]:
        try:
            n = 10 if b == "reasoning" else 20
            probes[b] = load_behavioral_probes(b, n=n, seed=42)
            print(f"  {b}: {len(probes[b])} probes")
        except Exception as e:
            print(f"  {b}: FAILED — {e}")

    model, tokenizer = load_model("Qwen/Qwen2.5-0.5B", DEVICE)
    available = list(probes.keys())

    print("\n--- Baseline ---")
    baseline = {}
    for b in available:
        r = evaluate_behavior(b, model, tokenizer, probes[b], DEVICE)
        baseline[b] = r
        print(f"  {b}: rho={r['rho']:.4f}")

    original_state = copy.deepcopy(model.state_dict())

    print("\n--- Ratio sweep ---")
    for ratio in [0.60, 0.70, 0.80]:
        model.load_state_dict(original_state)
        model.eval()
        compress_qko(model, ratio=ratio)
        freeze_layers(model, ratio=0.75)
        model.eval()
        print(f"\n  ratio={ratio:.0%}:")
        for b in available:
            r = evaluate_behavior(b, model, tokenizer, probes[b], DEVICE)
            d = r["rho"] - baseline[b]["rho"]
            print(f"    {b}: rho={r['rho']:.4f} (delta={d:+.4f})")

    del model, tokenizer, original_state
    gc.collect()


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-behavioral CF90 denoising sweep")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--ratios", default=None, help="Comma-separated ratios")
    parser.add_argument("--behaviors", nargs="+", default=None)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--analyze", default=None)
    args = parser.parse_args()

    if args.validate:
        validate()
        return

    if args.analyze:
        with open(args.analyze) as f:
            analyze(json.load(f))
        return

    models = MODELS
    if args.models:
        models = {k: v for k, v in MODELS.items() if k in args.models}
    ratios = [float(r) for r in args.ratios.split(",")] if args.ratios else RATIOS

    results = run_sweep(
        models=models,
        ratios=ratios,
        behaviors=args.behaviors or BEHAVIORS,
        device=args.device,
        resume_path=args.resume,
    )
    analyze(results)


if __name__ == "__main__":
    main()
