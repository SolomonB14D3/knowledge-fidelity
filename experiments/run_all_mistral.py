#!/usr/bin/env python3
"""Combined Mistral-7B experiment runner.

Reloads model from disk cache (~3s) instead of deepcopy/load_state_dict.
Runs all experiments sequentially, saves results incrementally.
"""

import json
import time
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_fidelity.core import _run_audit
from knowledge_fidelity.svd.compress import compress_qko
from knowledge_fidelity.svd.freeze import freeze_layers
from knowledge_fidelity.probes import (
    get_default_probes, get_mandela_probes, get_medical_probes,
    get_commonsense_probes, get_truthfulqa_probes,
)

MODEL = "mistralai/Mistral-7B-v0.1"
DEVICE = "cpu"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_PATH = RESULTS_DIR / "mistral_7b_combined.json"


def fresh_model():
    """Load a fresh model from disk cache. ~3s on cached model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, trust_remote_code=True,
    ).to(DEVICE)
    model.eval()
    print(f"  [reload {time.time()-t0:.1f}s]", flush=True)
    return model, tokenizer


def save_incremental(results):
    """Save results after each experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)


def run_fidelity_bench():
    """Experiment 1: Audit across all probe categories (one model load)."""
    print(f"\n{'='*60}", flush=True)
    print("EXPERIMENT 1: Fidelity-Bench", flush=True)
    print(f"{'='*60}", flush=True)

    model, tokenizer = fresh_model()

    categories = {
        "default": get_default_probes,
        "mandela": get_mandela_probes,
        "medical": get_medical_probes,
        "commonsense": get_commonsense_probes,
        "truthfulqa": get_truthfulqa_probes,
    }

    results = {}
    for name, getter in categories.items():
        try:
            probes = getter()
        except FileNotFoundError:
            print(f"  Skipping {name} (probes not found)", flush=True)
            continue

        print(f"  {name} ({len(probes)} probes)...", end=" ", flush=True)
        audit = _run_audit(model, tokenizer, probes, DEVICE)
        n_correct = audit["n_positive_delta"]
        results[name] = {
            "n_probes": len(probes),
            "rho": audit["rho"],
            "rho_p": audit["rho_p"],
            "n_correct": n_correct,
            "accuracy": n_correct / len(probes),
            "mean_delta": audit["mean_delta"],
        }
        print(f"rho={audit['rho']:.3f}, {n_correct}/{len(probes)} ({n_correct/len(probes):.0%})", flush=True)

    del model
    return results


def run_denoise():
    """Experiment 2: Find optimal denoise ratio. Reloads model per ratio."""
    print(f"\n{'='*60}", flush=True)
    print("EXPERIMENT 2: Denoise (Mandela probes)", flush=True)
    print(f"{'='*60}", flush=True)

    probes = get_mandela_probes()

    # Baseline
    model, tokenizer = fresh_model()
    baseline = _run_audit(model, tokenizer, probes, DEVICE)
    baseline_rho = baseline["rho"]
    print(f"  Baseline rho: {baseline_rho:.3f}", flush=True)
    del model

    best_rho = baseline_rho
    best_ratio = 1.0
    all_results = []

    for ratio in [0.5, 0.6, 0.7, 0.8, 0.9]:
        model, tokenizer = fresh_model()
        compress_qko(model, ratio=ratio)
        freeze_layers(model, ratio=0.75)
        model.eval()

        audit = _run_audit(model, tokenizer, probes, DEVICE)
        rho = audit["rho"]
        improved = rho > baseline_rho
        marker = " ** IMPROVED **" if improved else ""
        print(f"  ratio={ratio:.0%}: rho={rho:.3f}{marker}", flush=True)

        all_results.append({"ratio": ratio, "rho": rho, "improved": improved})
        if rho > best_rho:
            best_rho = rho
            best_ratio = ratio
        del model

    denoising = best_rho > baseline_rho
    improvement = best_rho - baseline_rho
    tag = f"DENOISING DETECTED: {baseline_rho:.3f}->{best_rho:.3f} (+{improvement:.3f}) at {best_ratio:.0%}" if denoising else "No denoising effect"
    print(f"  {tag}", flush=True)

    return {
        "baseline_rho": baseline_rho, "optimal_ratio": best_ratio,
        "optimal_rho": best_rho, "improvement": improvement,
        "denoising_detected": denoising, "all_results": all_results,
    }


def run_cf90_multiseed():
    """Experiment 3: CF90 at 70% rank, 3 seeds."""
    print(f"\n{'='*60}", flush=True)
    print("EXPERIMENT 3: CF90 Multi-Seed (70% rank, 3 seeds)", flush=True)
    print(f"{'='*60}", flush=True)

    probes = get_default_probes()

    # Baseline
    model, tokenizer = fresh_model()
    baseline = _run_audit(model, tokenizer, probes, DEVICE)
    baseline_rho = baseline["rho"]
    print(f"  Baseline rho: {baseline_rho:.3f}", flush=True)

    seed_results = []
    for seed in [42, 123, 789]:
        # Reload fresh for each seed (no state_dict copy needed)
        del model
        torch.manual_seed(seed)
        model, tokenizer = fresh_model()

        n_compressed = compress_qko(model, ratio=0.7)
        freeze_stats = freeze_layers(model, ratio=0.75)
        model.eval()

        audit = _run_audit(model, tokenizer, probes, DEVICE)
        n_retained = sum(
            1 for i in range(len(probes))
            if (baseline["true_confs"][i] - baseline["false_confs"][i] > 0
                and audit["true_confs"][i] - audit["false_confs"][i] > 0)
            or baseline["true_confs"][i] - baseline["false_confs"][i] <= 0
        )
        retention = n_retained / len(probes)
        rho_drop = baseline_rho - audit["rho"]

        print(f"  Seed {seed}: rho={audit['rho']:.3f}, ret={retention:.0%}, drop={rho_drop:.3f}", flush=True)
        seed_results.append({
            "seed": seed, "rho_after": audit["rho"], "retention": retention,
            "rho_drop": rho_drop, "n_compressed": n_compressed,
            "n_frozen": freeze_stats["n_frozen"], "n_layers": freeze_stats["n_layers"],
        })

    del model
    rhos = [r["rho_after"] for r in seed_results]
    drops = [r["rho_drop"] for r in seed_results]
    rets = [r["retention"] for r in seed_results]
    print(f"  Mean: rho={np.mean(rhos):.3f}±{np.std(rhos):.3f}, ret={np.mean(rets):.0%}, drop={np.mean(drops):.3f}±{np.std(drops):.3f}", flush=True)

    return {
        "baseline_rho": baseline_rho, "seeds": seed_results,
        "mean_rho_after": float(np.mean(rhos)), "std_rho_after": float(np.std(rhos)),
        "mean_retention": float(np.mean(rets)),
        "mean_rho_drop": float(np.mean(drops)), "std_rho_drop": float(np.std(drops)),
        "n_compressed": seed_results[0]["n_compressed"],
        "n_frozen": seed_results[0]["n_frozen"], "n_layers": seed_results[0]["n_layers"],
    }


def run_joint_ablation():
    """Experiment 4: Ratio ablation across default/mandela/medical probes."""
    print(f"\n{'='*60}", flush=True)
    print("EXPERIMENT 4: Joint Ablation", flush=True)
    print(f"{'='*60}", flush=True)

    probe_sets = {
        "default": get_default_probes(),
        "mandela": get_mandela_probes(),
        "medical": get_medical_probes(),
    }

    # Baselines (one load)
    model, tokenizer = fresh_model()
    baselines = {}
    for name, probes in probe_sets.items():
        audit = _run_audit(model, tokenizer, probes, DEVICE)
        baselines[name] = audit["rho"]
    print(f"  Baselines: " + ", ".join(f"{k}={v:.3f}" for k, v in baselines.items()), flush=True)
    del model

    results = []
    for ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        model, tokenizer = fresh_model()
        if ratio < 1.0:
            compress_qko(model, ratio=ratio)
            freeze_layers(model, ratio=0.75)
            model.eval()

        row = {"ratio": ratio}
        for name, probes in probe_sets.items():
            audit = _run_audit(model, tokenizer, probes, DEVICE)
            row[f"{name}_rho"] = audit["rho"]

        print(f"  {ratio:.0%}: " + ", ".join(f"{k}={row[f'{k}_rho']:.3f}" for k in probe_sets), flush=True)
        results.append(row)
        del model

    return {"baselines": baselines, "ratios": results}


def main():
    start = time.time()
    all_results = {"model": MODEL, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    all_results["fidelity_bench"] = run_fidelity_bench()
    save_incremental(all_results)

    all_results["denoise"] = run_denoise()
    save_incremental(all_results)

    all_results["cf90_multiseed"] = run_cf90_multiseed()
    save_incremental(all_results)

    all_results["joint_ablation"] = run_joint_ablation()

    elapsed = time.time() - start
    all_results["total_elapsed_seconds"] = elapsed
    save_incremental(all_results)

    print(f"\n{'='*60}", flush=True)
    print(f"ALL DONE in {elapsed/60:.1f} min. Results: {OUT_PATH}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
