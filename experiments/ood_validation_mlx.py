#!/usr/bin/env python3
"""Phase 4: Out-of-Distribution (OOD) validation for rho-guided SFT (MLX).

Tests whether rho-guided SFT generalizes beyond in-distribution probes.
The training uses factual/toxicity/sycophancy/bias probes — this script
evaluates on Fidelity-Bench 2.0 probes that are NEVER seen during training:

  1. Clinical/Scientific: Medical safety, engineering hazards, physics (42 probes)
  2. Social Pressure: Myths under opinion/authority/peer pressure (40 probes)
  3. Logic/Reasoning: Mathematical and logical facts (42 probes)

Additionally tests "conflicting user opinion" scenarios — does the model
maintain factual confidence when presented with popular misconceptions?

Key questions:
  - Does rho-guided SFT transfer to unseen domains (positive transfer)?
  - Does rho-guided SFT avoid degrading OOD performance (no harm)?
  - Is there differential transfer by pressure type (opinion vs authority vs peer)?

Usage:
    # Full OOD validation with sweep
    python experiments/ood_validation_mlx.py --model qwen2.5-7b \\
        --rho-weights 0.0,0.2 --seeds 42,123

    # Baseline-only OOD evaluation (no training)
    python experiments/ood_validation_mlx.py --model qwen2.5-7b --baseline-only

    # Analyze saved results
    python experiments/ood_validation_mlx.py --analyze results/alignment/ood_*.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results" / "alignment"
BENCH_DIR = Path(__file__).parent.parent / "src" / "rho_eval" / "probes" / "data" / "bench"

MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
}

# In-distribution behaviors (used during training)
TRAIN_BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias"]

# OOD domains (NEVER seen during training)
OOD_DOMAINS = ["clinical", "social", "logic"]


# ── OOD Probe Loading ────────────────────────────────────────────────

def load_ood_probes(domains: list[str] | None = None) -> dict[str, list[dict]]:
    """Load Fidelity-Bench 2.0 probes for OOD domains.

    Returns:
        Dict mapping domain name -> list of probe dicts.
        Each probe has: text, false, domain, id, and domain-specific fields.
    """
    domains = domains or OOD_DOMAINS
    probes = {}

    for domain in domains:
        path = BENCH_DIR / f"{domain}.json"
        if not path.exists():
            print(f"  [ood] Warning: {path} not found, skipping {domain}")
            continue

        with open(path) as f:
            data = json.load(f)

        probes[domain] = data
        print(f"  [ood] Loaded {len(data)} {domain} probes")

    return probes


def build_ood_pairs(probes: list[dict]) -> list[dict]:
    """Build contrast pairs from OOD probes (same format as factual probes)."""
    pairs = []
    for p in probes:
        if "text" in p and "false" in p:
            pairs.append({
                "positive": p["text"],
                "negative": p["false"],
                "id": p.get("id", ""),
            })
    return pairs


# ── OOD Scoring ──────────────────────────────────────────────────────

def score_ood(
    model,
    tokenizer,
    ood_probes: dict[str, list[dict]],
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Score OOD probes: confidence gaps, ECE, Brier, per-domain and per-pressure.

    Returns:
        Dict with per-domain metrics, per-pressure-type breakdown,
        and aggregate OOD metrics.
    """
    from rho_eval.behaviors.metrics import (
        get_mean_logprob, score_probe_pairs,
        compute_ece, compute_brier,
    )

    results = {}
    all_confs = []
    all_accs = []
    all_gaps = []

    for domain, probes in ood_probes.items():
        pairs = build_ood_pairs(probes)
        if not pairs:
            results[domain] = {"error": "no pairs"}
            continue

        # Score all pairs
        scored = score_probe_pairs(model, tokenizer, pairs)

        confs = np.array([s["confidence"] for s in scored])
        accs = np.array([s["correct"] for s in scored])
        gaps = np.array([s["gap"] for s in scored])

        # ECE + Brier
        ece_result = compute_ece(confs, accs)
        brier = compute_brier(confs, accs)

        domain_result = {
            "mean_gap": float(gaps.mean()),
            "accuracy": float(accs.mean()),
            "mean_confidence": float(confs.mean()),
            "ece": ece_result["ece"],
            "brier": brier,
            "n_probes": len(scored),
            "details": scored,
        }

        # Per-pressure-type breakdown (for social domain)
        if domain == "social":
            by_pressure = defaultdict(list)
            for probe, score in zip(probes, scored):
                ptype = probe.get("pressure_type", "unknown")
                by_pressure[ptype].append(score)

            pressure_results = {}
            for ptype, pscores in sorted(by_pressure.items()):
                p_confs = np.array([s["confidence"] for s in pscores])
                p_accs = np.array([s["correct"] for s in pscores])
                p_gaps = np.array([s["gap"] for s in pscores])
                pressure_results[ptype] = {
                    "mean_gap": float(p_gaps.mean()),
                    "accuracy": float(p_accs.mean()),
                    "mean_confidence": float(p_confs.mean()),
                    "n_probes": len(pscores),
                }
            domain_result["by_pressure_type"] = pressure_results

        # Per-subdomain breakdown (for clinical domain)
        if domain == "clinical":
            by_sub = defaultdict(list)
            for probe, score in zip(probes, scored):
                sub = probe.get("subdomain", "unknown")
                by_sub[sub].append(score)

            subdomain_results = {}
            for sub, sscores in sorted(by_sub.items()):
                s_confs = np.array([s["confidence"] for s in sscores])
                s_accs = np.array([s["correct"] for s in sscores])
                s_gaps = np.array([s["gap"] for s in sscores])
                subdomain_results[sub] = {
                    "mean_gap": float(s_gaps.mean()),
                    "accuracy": float(s_accs.mean()),
                    "n_probes": len(sscores),
                }
            domain_result["by_subdomain"] = subdomain_results

        # Per-difficulty breakdown (for logic domain)
        if domain == "logic":
            by_diff = defaultdict(list)
            for probe, score in zip(probes, scored):
                diff = probe.get("difficulty", 0)
                by_diff[diff].append(score)

            difficulty_results = {}
            for diff, dscores in sorted(by_diff.items()):
                d_confs = np.array([s["confidence"] for s in dscores])
                d_accs = np.array([s["correct"] for s in dscores])
                d_gaps = np.array([s["gap"] for s in dscores])
                difficulty_results[str(diff)] = {
                    "mean_gap": float(d_gaps.mean()),
                    "accuracy": float(d_accs.mean()),
                    "n_probes": len(dscores),
                }
            domain_result["by_difficulty"] = difficulty_results

        results[domain] = domain_result

        # Accumulate for aggregate
        all_confs.extend(confs.tolist())
        all_accs.extend(accs.tolist())
        all_gaps.extend(gaps.tolist())

    # Aggregate OOD metrics
    if all_confs:
        all_confs = np.array(all_confs)
        all_accs = np.array(all_accs)
        all_gaps = np.array(all_gaps)

        agg_ece = compute_ece(all_confs, all_accs)
        agg_brier = compute_brier(all_confs, all_accs)

        results["_aggregate"] = {
            "mean_gap": float(all_gaps.mean()),
            "accuracy": float(all_accs.mean()),
            "mean_confidence": float(all_confs.mean()),
            "ece": agg_ece["ece"],
            "brier": agg_brier,
            "n_probes": len(all_confs),
        }

    if verbose:
        _print_ood_summary(results)

    return results


def _print_ood_summary(results: dict):
    """Print formatted OOD evaluation summary."""
    print(f"\n  {'Domain':12s}  {'Gap':>8s}  {'Acc':>6s}  {'ECE':>8s}  {'Brier':>8s}  {'N':>4s}")
    print(f"  {'─'*52}")

    for domain in OOD_DOMAINS:
        if domain in results and "error" not in results[domain]:
            d = results[domain]
            print(f"  {domain:12s}  {d['mean_gap']:+8.4f}  {d['accuracy']:5.1%}  "
                  f"{d['ece']:8.4f}  {d['brier']:8.4f}  {d['n_probes']:4d}")

            # Social pressure breakdown
            if "by_pressure_type" in d:
                for ptype, pdata in d["by_pressure_type"].items():
                    print(f"    └ {ptype:10s}  {pdata['mean_gap']:+8.4f}  "
                          f"{pdata['accuracy']:5.1%}  {'':8s}  {'':8s}  {pdata['n_probes']:4d}")

            # Clinical subdomain breakdown
            if "by_subdomain" in d:
                for sub, sdata in d["by_subdomain"].items():
                    print(f"    └ {sub:10s}  {sdata['mean_gap']:+8.4f}  "
                          f"{sdata['accuracy']:5.1%}  {'':8s}  {'':8s}  {sdata['n_probes']:4d}")

    if "_aggregate" in results:
        agg = results["_aggregate"]
        print(f"  {'─'*52}")
        print(f"  {'AGGREGATE':12s}  {agg['mean_gap']:+8.4f}  {agg['accuracy']:5.1%}  "
              f"{agg['ece']:8.4f}  {agg['brier']:8.4f}  {agg['n_probes']:4d}")


# ── Full Sweep with OOD Validation ───────────────────────────────────

def run_ood_sweep(
    model_name: str,
    rho_weights: list[float],
    seeds: list[int],
    sft_size: int = 1000,
    epochs: int = 1,
    lr: float = 2e-4,
    lora_rank: int = 8,
    margin: float = 0.1,
    ood_domains: list[str] | None = None,
    results_path: Path | None = None,
    verbose: bool = True,
):
    """Train with each rho_weight × seed, then evaluate OOD transfer."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load as mlx_load

    from rho_eval.alignment.dataset import (
        _load_alpaca_texts, _build_trap_texts,
        BehavioralContrastDataset, CONTRAST_BEHAVIORS,
    )
    from rho_eval.alignment.mlx_trainer import mlx_rho_guided_sft

    ood_domains = ood_domains or OOD_DOMAINS
    results_path = results_path or (
        RESULTS_DIR / f"ood_{model_name.replace('/', '_')}.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model & OOD probes ──────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Phase 4: OOD Validation (MLX): {model_name}")
    print(f"  rho_weights={rho_weights}, seeds={seeds}")
    print(f"  OOD domains={ood_domains}")
    print(f"{'='*70}\n")

    model, tokenizer = mlx_load(model_name)
    model.eval()

    print("Loading OOD probes...")
    ood_probes = load_ood_probes(ood_domains)

    # ── Baseline OOD eval ────────────────────────────────────────
    print("\nEvaluating baseline OOD performance...")
    baseline_ood = score_ood(model, tokenizer, ood_probes, verbose=verbose)
    baseline_summary = _strip_details_ood(baseline_ood)

    # Also score in-distribution for comparison
    print("\nEvaluating baseline in-distribution...")
    from rho_eval.behaviors.metrics import calibration_metrics
    try:
        baseline_id = calibration_metrics(model, tokenizer, behaviors=TRAIN_BEHAVIORS, seed=42)
        baseline_id_summary = {
            bname: {k: v for k, v in bdata.items() if k not in ("details", "ece_bins")}
            for bname, bdata in baseline_id.items() if isinstance(bdata, dict)
        }
    except Exception as e:
        print(f"  [id] Error: {e}")
        baseline_id_summary = {}

    # Save initial weights
    print("\nSaving initial weights...")
    initial_path = results_path.parent / "ood_initial_weights.safetensors"
    initial_weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(initial_path), initial_weights)
    del initial_weights
    gc.collect()

    # Prepare SFT texts
    print("\nPreparing SFT texts...")
    trap_ratio = 0.2
    n_traps = int(sft_size * trap_ratio)
    remaining = sft_size - n_traps

    trap_texts = _build_trap_texts(list(CONTRAST_BEHAVIORS), seed=42)
    random.Random(42).shuffle(trap_texts)
    trap_texts = trap_texts[:n_traps]

    alpaca_texts = _load_alpaca_texts(remaining, seed=42)
    sft_texts = trap_texts + alpaca_texts
    random.Random(42).shuffle(sft_texts)
    sft_texts = sft_texts[:sft_size]
    print(f"  {len(sft_texts)} SFT texts")

    # Results container
    all_results = {
        "model": model_name,
        "backend": "mlx",
        "experiment": "ood_validation",
        "baseline_ood": baseline_summary,
        "baseline_in_distribution": baseline_id_summary,
        "config": {
            "rho_weights": rho_weights,
            "sft_size": sft_size,
            "epochs": epochs,
            "lr": lr,
            "lora_rank": lora_rank,
            "margin": margin,
            "seeds": seeds,
            "ood_domains": ood_domains,
            "train_behaviors": TRAIN_BEHAVIORS,
        },
        "runs": [],
        "timestamp": datetime.now().isoformat(),
    }

    total_runs = len(rho_weights) * len(seeds)
    run_idx = 0

    for rho_weight in rho_weights:
        for seed in seeds:
            run_idx += 1
            label = f"λ={rho_weight}/s={seed}"
            print(f"\n{'─'*60}")
            print(f"  Run {run_idx}/{total_runs}: rho_weight={rho_weight}, seed={seed}")
            print(f"{'─'*60}")

            # Restore original weights
            model.load_weights(str(initial_path), strict=False)
            mx.eval(model.parameters())

            # Build contrast dataset
            contrast_ds = BehavioralContrastDataset(
                behaviors=TRAIN_BEHAVIORS, seed=seed,
            )

            t_start = time.time()

            try:
                train_result = mlx_rho_guided_sft(
                    model, tokenizer,
                    sft_texts, contrast_ds,
                    rho_weight=rho_weight,
                    epochs=epochs,
                    lr=lr,
                    lora_rank=lora_rank,
                    margin=margin,
                    verbose=verbose,
                )
                model = train_result["merged_model"]
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                all_results["runs"].append({
                    "rho_weight": rho_weight,
                    "seed": seed,
                    "error": str(e),
                })
                _save_checkpoint(all_results, results_path)
                continue

            # OOD eval
            print(f"\n  OOD eval {label}...")
            ood_result = score_ood(model, tokenizer, ood_probes, verbose=verbose)

            # In-distribution eval for comparison
            print(f"  ID eval {label}...")
            try:
                id_result = calibration_metrics(model, tokenizer, behaviors=TRAIN_BEHAVIORS, seed=seed)
                id_summary = {
                    bname: {k: v for k, v in bdata.items() if k not in ("details", "ece_bins")}
                    for bname, bdata in id_result.items() if isinstance(bdata, dict)
                }
            except Exception as e:
                id_summary = {"error": str(e)}

            elapsed = time.time() - t_start

            # Record
            run_record = {
                "rho_weight": rho_weight,
                "seed": seed,
                "ood": _strip_details_ood(ood_result),
                "in_distribution": id_summary,
                "train_ce_loss": train_result.get("ce_loss", 0),
                "train_rho_loss": train_result.get("rho_loss", 0),
                "elapsed_seconds": elapsed,
            }

            # Compute deltas from baseline
            deltas = {}
            for domain in ood_domains:
                base_d = baseline_ood.get(domain, {})
                post_d = ood_result.get(domain, {})
                if "error" not in base_d and "error" not in post_d and base_d and post_d:
                    deltas[domain] = {
                        "gap_delta": post_d["mean_gap"] - base_d["mean_gap"],
                        "acc_delta": post_d["accuracy"] - base_d["accuracy"],
                        "ece_delta": post_d["ece"] - base_d["ece"],
                    }
            if "_aggregate" in baseline_ood and "_aggregate" in ood_result:
                deltas["_aggregate"] = {
                    "gap_delta": ood_result["_aggregate"]["mean_gap"] - baseline_ood["_aggregate"]["mean_gap"],
                    "acc_delta": ood_result["_aggregate"]["accuracy"] - baseline_ood["_aggregate"]["accuracy"],
                    "ece_delta": ood_result["_aggregate"]["ece"] - baseline_ood["_aggregate"]["ece"],
                }
            run_record["deltas"] = deltas

            all_results["runs"].append(run_record)

            # Print delta summary
            print(f"\n  OOD Transfer for {label}:")
            for domain in ood_domains + ["_aggregate"]:
                if domain in deltas:
                    d = deltas[domain]
                    gap_dir = "↑" if d["gap_delta"] > 0.01 else ("↓" if d["gap_delta"] < -0.01 else "=")
                    acc_dir = "↑" if d["acc_delta"] > 0.01 else ("↓" if d["acc_delta"] < -0.01 else "=")
                    ece_dir = "↓" if d["ece_delta"] < -0.001 else ("↑" if d["ece_delta"] > 0.001 else "=")
                    name = "AGGREGATE" if domain == "_aggregate" else domain
                    print(f"    {name:12s}: gap {d['gap_delta']:+.4f}{gap_dir}  "
                          f"acc {d['acc_delta']:+.1%}{acc_dir}  "
                          f"ECE {d['ece_delta']:+.4f}{ece_dir}")

            _save_checkpoint(all_results, results_path)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  OOD VALIDATION COMPLETE: {total_runs} runs")
    print(f"  Results: {results_path}")
    print(f"{'='*70}")

    # Print transfer table
    _print_transfer_table(all_results)

    # Cleanup
    initial_path.unlink(missing_ok=True)

    return all_results


# ── Analysis & Printing ──────────────────────────────────────────────

def _print_transfer_table(results: dict):
    """Print formatted transfer table from results."""
    runs = results.get("runs", [])
    if not runs:
        return

    # Group by rho_weight
    by_rho = defaultdict(list)
    for run in runs:
        if "error" not in run:
            by_rho[run["rho_weight"]].append(run)

    ood_domains = results.get("config", {}).get("ood_domains", OOD_DOMAINS)

    print(f"\n{'─'*70}")
    print(f"  OOD Transfer Summary: ID Accuracy vs OOD Accuracy")
    print(f"{'─'*70}")
    print(f"  {'λ_ρ':>6s}  {'ID Acc':>8s}  ", end="")
    for d in ood_domains:
        print(f"  {d[:8]:>8s}", end="")
    print(f"  {'OOD Agg':>8s}")
    print(f"  {'─'*62}")

    # Baseline
    baseline_ood = results.get("baseline_ood", {})
    baseline_id = results.get("baseline_in_distribution", {})
    if baseline_id:
        id_accs = [v.get("accuracy", 0) for v in baseline_id.values()
                   if isinstance(v, dict) and "accuracy" in v]
        id_avg = np.mean(id_accs) if id_accs else 0
    else:
        id_avg = 0

    print(f"  {'base':>6s}  {id_avg:7.1%}  ", end="")
    for d in ood_domains:
        if d in baseline_ood and "error" not in baseline_ood[d]:
            print(f"  {baseline_ood[d]['accuracy']:7.1%}", end="")
        else:
            print(f"  {'N/A':>8s}", end="")
    if "_aggregate" in baseline_ood:
        print(f"  {baseline_ood['_aggregate']['accuracy']:7.1%}")
    else:
        print()

    for rho_w in sorted(by_rho.keys()):
        runs_at_rho = by_rho[rho_w]

        # Average metrics
        id_accs = []
        ood_accs = {d: [] for d in ood_domains}
        agg_accs = []

        for run in runs_at_rho:
            # ID accuracy
            id_data = run.get("in_distribution", {})
            if id_data and "error" not in id_data:
                accs = [v.get("accuracy", 0) for v in id_data.values()
                        if isinstance(v, dict) and "accuracy" in v]
                if accs:
                    id_accs.append(np.mean(accs))

            # OOD accuracy
            ood_data = run.get("ood", {})
            for d in ood_domains:
                if d in ood_data and "error" not in ood_data.get(d, {}):
                    ood_accs[d].append(ood_data[d].get("accuracy", 0))
            if "_aggregate" in ood_data:
                agg_accs.append(ood_data["_aggregate"].get("accuracy", 0))

        id_mean = np.mean(id_accs) if id_accs else 0
        print(f"  {rho_w:6.2f}  {id_mean:7.1%}  ", end="")
        for d in ood_domains:
            if ood_accs[d]:
                print(f"  {np.mean(ood_accs[d]):7.1%}", end="")
            else:
                print(f"  {'N/A':>8s}", end="")
        if agg_accs:
            print(f"  {np.mean(agg_accs):7.1%}")
        else:
            print()


def analyze_ood(results_path: str | Path):
    """Analyze and print OOD validation results from saved JSON."""
    path = Path(results_path)
    with open(path) as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"  OOD Validation Analysis: {data['model']}")
    print(f"{'='*70}")

    _print_transfer_table(data)

    # Pressure type analysis
    runs = data.get("runs", [])
    baseline_ood = data.get("baseline_ood", {})

    if "social" in baseline_ood and "by_pressure_type" in baseline_ood.get("social", {}):
        print(f"\n{'─'*70}")
        print(f"  Social Pressure Breakdown")
        print(f"{'─'*70}")

        base_pressure = baseline_ood["social"]["by_pressure_type"]
        print(f"  Baseline:")
        for ptype, pdata in sorted(base_pressure.items()):
            print(f"    {ptype:12s}: gap={pdata['mean_gap']:+.4f}  acc={pdata['accuracy']:.1%}")

        by_rho = defaultdict(list)
        for run in runs:
            if "error" not in run:
                by_rho[run["rho_weight"]].append(run)

        for rho_w in sorted(by_rho.keys()):
            runs_at_rho = by_rho[rho_w]
            print(f"\n  λ_ρ = {rho_w}:")
            for ptype in sorted(base_pressure.keys()):
                gaps = []
                accs = []
                for run in runs_at_rho:
                    social = run.get("ood", {}).get("social", {})
                    if "by_pressure_type" in social and ptype in social["by_pressure_type"]:
                        gaps.append(social["by_pressure_type"][ptype]["mean_gap"])
                        accs.append(social["by_pressure_type"][ptype]["accuracy"])

                if gaps:
                    base_gap = base_pressure[ptype]["mean_gap"]
                    delta = np.mean(gaps) - base_gap
                    direction = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "=")
                    print(f"    {ptype:12s}: gap={np.mean(gaps):+.4f} "
                          f"(Δ={delta:+.4f}{direction})  "
                          f"acc={np.mean(accs):.1%}")

    # Statistical tests
    print(f"\n{'─'*70}")
    print(f"  Transfer Significance Tests")
    print(f"{'─'*70}")

    ood_domains = data.get("config", {}).get("ood_domains", OOD_DOMAINS)

    for domain in ood_domains + ["_aggregate"]:
        base_data = baseline_ood.get(domain, {})
        if "error" in base_data or not base_data:
            continue
        base_gap = base_data.get("mean_gap", 0)

        by_rho = defaultdict(list)
        for run in runs:
            if "error" not in run and "deltas" in run and domain in run["deltas"]:
                by_rho[run["rho_weight"]].append(run["deltas"][domain]["gap_delta"])

        for rho_w in sorted(by_rho.keys()):
            deltas = by_rho[rho_w]
            if len(deltas) >= 2:
                from scipy import stats
                t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
                direction = "positive transfer" if np.mean(deltas) > 0 else "negative transfer"
                name = "AGGREGATE" if domain == "_aggregate" else domain
                print(f"  {name:12s} λ={rho_w}: mean Δ={np.mean(deltas):+.4f} "
                      f"({direction}) p={p_val:.4f} {sig}")


def _strip_details_ood(ood_result: dict) -> dict:
    """Strip per-probe details from OOD results for JSON storage."""
    stripped = {}
    for domain, ddata in ood_result.items():
        if isinstance(ddata, dict):
            stripped[domain] = {k: v for k, v in ddata.items() if k != "details"}
        else:
            stripped[domain] = ddata
    return stripped


def _save_checkpoint(results: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.rename(path)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4: OOD Validation")
    parser.add_argument("--model", default="qwen2.5-7b",
                        help="Model key or HF name")
    parser.add_argument("--rho-weights", default="0.0,0.2",
                        help="Comma-separated rho weights")
    parser.add_argument("--seeds", default="42,123",
                        help="Comma-separated seeds")
    parser.add_argument("--sft-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--ood-domains", default=",".join(OOD_DOMAINS),
                        help="Comma-separated OOD domains")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only evaluate baseline (no training)")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Path to results JSON for analysis")

    args = parser.parse_args()

    if args.analyze:
        analyze_ood(args.analyze)
        return

    model_name = MODELS.get(args.model, args.model)
    rho_weights = [float(w) for w in args.rho_weights.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    ood_domains = [d.strip() for d in args.ood_domains.split(",")]

    if args.baseline_only:
        from mlx_lm import load as mlx_load
        model, tokenizer = mlx_load(model_name)
        model.eval()
        print("Loading OOD probes...")
        ood_probes = load_ood_probes(ood_domains)
        print("\nEvaluating baseline OOD performance...")
        score_ood(model, tokenizer, ood_probes, verbose=True)
    else:
        run_ood_sweep(
            model_name=model_name,
            rho_weights=rho_weights,
            seeds=seeds,
            sft_size=args.sft_size,
            epochs=args.epochs,
            lr=args.lr,
            lora_rank=args.lora_rank,
            margin=args.margin,
            ood_domains=ood_domains,
        )


if __name__ == "__main__":
    main()
