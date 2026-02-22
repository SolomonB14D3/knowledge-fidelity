#!/usr/bin/env python3
"""
Freeze-Ratio Sweep v2: Layer Localization with Post-Compression Fine-Tuning

Tests which layer regions carry which behavioral signals by varying the
freeze ratio at a fixed compression level, WITH a gentle fine-tuning step
after compression to let unfrozen layers adapt.

v1 bug: freeze without FT is a no-op (SVD output is deterministic regardless
of which params are marked trainable). The freeze only matters when there's
a training step that updates the unfrozen parameters.

Pipeline per condition:
  1. Restore original weights
  2. SVD compress Q/K/O at fixed ratio (0.70)
  3. Unfreeze all → apply freeze at target ratio (bottom-up)
  4. **Gentle fine-tune** (1 epoch, 2e-5 LR, 1000 calibration examples)
  5. Move to eval device → evaluate all 5 behaviors

Key questions:
  1. Does low freeze (0-25%) let the model recover from compression?
  2. Does high freeze (85-100%) preserve denoising gains?
  3. Is there a behavior-specific optimal freeze point?

Expected effects:
  - Low freeze: more adaptation → better recovery of distributed traits
  - High freeze: stronger regularization → bigger denoising gains

Usage:
    python experiments/freeze_ratio_sweep.py
    python experiments/freeze_ratio_sweep.py --models qwen2.5-7b
    python experiments/freeze_ratio_sweep.py --freeze-ratios 0.25,0.75,0.90
    python experiments/freeze_ratio_sweep.py --no-ft  # skip FT (v1 behavior)
    python experiments/freeze_ratio_sweep.py --calib-size 500
    python experiments/freeze_ratio_sweep.py --validate
    python experiments/freeze_ratio_sweep.py --analyze results/freeze_sweep/sweep_v2.json
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
from knowledge_fidelity.svd import compress_qko
from knowledge_fidelity.svd.freeze import freeze_layers, unfreeze_all
from knowledge_fidelity.probes import get_all_probes
from knowledge_fidelity.behavioral import load_behavioral_probes, evaluate_behavior
from knowledge_fidelity.calibration import load_calibration_data, gentle_finetune

# ── Configuration ─────────────────────────────────────────────────────────

MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

DEFAULT_COMPRESS_RATIO = 0.70
FREEZE_RATIOS = [0.0, 0.25, 0.50, 0.75, 0.90]  # 5 ratios (trimmed from 8)
BEHAVIORS = ["factual", "toxicity", "bias", "sycophancy", "reasoning"]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# FT defaults
DEFAULT_CALIB_SIZE = 1000
DEFAULT_LR = 2e-4  # LoRA typical LR (higher than full FT)
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_MAX_STEPS = None  # None = full epoch

RESULTS_DIR = Path(__file__).parent.parent / "results" / "freeze_sweep"


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


def get_layer_info(model, freeze_ratio):
    """Get human-readable info about which layers are frozen/trainable."""
    from knowledge_fidelity.utils import get_layers
    layers = get_layers(model)
    n_layers = len(layers)
    n_frozen = int(n_layers * freeze_ratio)
    n_trainable = n_layers - n_frozen
    return {
        "n_layers": n_layers,
        "n_frozen": n_frozen,
        "n_trainable": n_trainable,
        "frozen_range": f"0-{n_frozen-1}" if n_frozen > 0 else "none",
        "trainable_range": f"{n_frozen}-{n_layers-1}" if n_trainable > 0 else "none",
    }


# ── Per-model sweep ──────────────────────────────────────────────────────

def sweep_one_model(model_name, model_id, compress_ratio, freeze_ratios,
                    behaviors, probes_dict, device, all_results, output_path,
                    do_ft=True, calib_size=DEFAULT_CALIB_SIZE, lr=DEFAULT_LR,
                    epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                    grad_accum=DEFAULT_GRAD_ACCUM, max_steps=DEFAULT_MAX_STEPS):
    """Run all freeze ratios for one model at a fixed compression ratio."""

    suffix = "_ft" if do_ft else "_noft"

    # Check which freeze ratios still need running
    needed = [fr for fr in freeze_ratios
              if f"{model_name}_c{compress_ratio:.2f}_f{fr:.2f}{suffix}" not in all_results]
    if not needed:
        print(f"  All freeze ratios already done for {model_name}, skipping.")
        return

    model, tokenizer = load_model(model_id, device)

    # ── Load calibration data (once per model) ────────────────────────
    calib_dataset = None
    if do_ft:
        print(f"\n  Loading calibration data ({calib_size} examples)...")
        calib_dataset = load_calibration_data(
            n=calib_size, seed=42, tokenizer=tokenizer, max_length=256,
        )

    # ── Baseline (once per model): no compression, no freeze, no FT ──
    baseline_key = f"{model_name}_baseline"
    if baseline_key in all_results:
        baseline = all_results[baseline_key]["behaviors"]
        print(f"  Baseline loaded from cache")
    else:
        print(f"\n  Evaluating baseline (no compression, no freeze, no FT)...")
        baseline = eval_all_behaviors(model, tokenizer, behaviors, probes_dict, device)
        all_results[baseline_key] = {
            "model": model_name,
            "model_id": model_id,
            "compress_ratio": 1.0,
            "freeze_ratio": 0.0,
            "fine_tuned": False,
            "behaviors": baseline,
            "timestamp": datetime.now().isoformat(),
        }
        _save(all_results, output_path)

    # ── Save original weights ─────────────────────────────────────────
    print(f"  Saving original state dict...")
    t0 = time.time()
    original_state = copy.deepcopy(model.state_dict())
    print(f"  Saved in {time.time()-t0:.1f}s")

    # ── Sweep freeze ratios ───────────────────────────────────────────
    for i, freeze_ratio in enumerate(freeze_ratios):
        key = f"{model_name}_c{compress_ratio:.2f}_f{freeze_ratio:.2f}{suffix}"
        if key in all_results:
            print(f"\n  [{i+1}/{len(freeze_ratios)}] freeze={freeze_ratio:.0%} — SKIPPED (done)")
            continue

        ft_label = "+FT" if do_ft else "no-FT"
        print(f"\n  [{i+1}/{len(freeze_ratios)}] freeze={freeze_ratio:.0%} "
              f"(compress={compress_ratio:.0%}, {ft_label})")

        # ── Step 1: Restore original weights ──────────────────────────
        model.load_state_dict(original_state)
        model.eval()

        # ── Step 2: SVD compress Q/K/O ────────────────────────────────
        t0 = time.time()
        n_compressed = compress_qko(model, ratio=compress_ratio)
        compress_time = time.time() - t0

        # ── Step 3: Unfreeze all → apply freeze ──────────────────────
        unfreeze_all(model)
        if freeze_ratio > 0:
            freeze_stats = freeze_layers(model, ratio=freeze_ratio)
        else:
            from knowledge_fidelity.utils import get_layers
            layers = get_layers(model)
            freeze_stats = {
                "n_layers": len(layers),
                "n_frozen": 0,
                "freeze_ratio": 0.0,
                "trainable_params": sum(p.numel() for p in model.parameters()),
                "total_params": sum(p.numel() for p in model.parameters()),
                "trainable_pct": 1.0,
            }

        layer_info = get_layer_info(model, freeze_ratio)
        print(f"    Compressed {n_compressed} matrices, "
              f"froze {layer_info['n_frozen']}/{layer_info['n_layers']} layers "
              f"(trainable: {layer_info['trainable_range']}) "
              f"({compress_time:.1f}s)")

        # ── Step 4: Gentle LoRA fine-tune (the key fix!) ──────────────
        ft_stats = {"skipped": True}
        if do_ft and calib_dataset is not None:
            # Check if there are trainable params
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if n_trainable > 0:
                ft_stats = gentle_finetune(
                    model, tokenizer, calib_dataset,
                    epochs=epochs, lr=lr, batch_size=batch_size,
                    device=device,
                    gradient_accumulation_steps=grad_accum,
                    max_steps=max_steps,
                )
                # LoRA returns a merged model — must update reference
                if "merged_model" in ft_stats:
                    model = ft_stats.pop("merged_model")
            else:
                print(f"    [ft] All params frozen — skipping FT")
                ft_stats = {"skipped": True, "reason": "all_frozen"}

        # ── Step 5: Move to eval device → evaluate ────────────────────
        model.to(device)
        model.eval()
        post = eval_all_behaviors(model, tokenizer, behaviors, probes_dict, device)

        # ── Compute deltas from baseline ──────────────────────────────
        behavior_results = {}
        for behavior in behaviors:
            delta = post[behavior]["rho"] - baseline[behavior]["rho"]
            arrow = "↑" if delta > 0.005 else "↓" if delta < -0.005 else "="
            print(f"    [{behavior}] delta={delta:+.4f} {arrow}")
            behavior_results[behavior] = {
                "baseline": baseline[behavior],
                "compressed": post[behavior],
                "delta": delta,
            }

        all_results[key] = {
            "model": model_name,
            "model_id": model_id,
            "compress_ratio": compress_ratio,
            "freeze_ratio": freeze_ratio,
            "fine_tuned": do_ft and not ft_stats.get("skipped", False),
            "ft_stats": ft_stats,
            "behaviors": behavior_results,
            "layer_info": layer_info,
            "compress_stats": {
                "n_compressed": n_compressed,
                "compress_time": compress_time,
                "n_layers": layer_info["n_layers"],
                **{k: v for k, v in freeze_stats.items()
                   if k not in ("n_layers",)},
            },
            "timestamp": datetime.now().isoformat(),
        }
        _save(all_results, output_path)

    del model, tokenizer, original_state, calib_dataset
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _save(results, path):
    """Save results JSON (atomic-ish)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=float)
    tmp.rename(path)


# ── Full sweep ────────────────────────────────────────────────────────────

def run_sweep(models, compress_ratio, freeze_ratios, behaviors, device,
              resume_path=None, do_ft=True, calib_size=DEFAULT_CALIB_SIZE,
              lr=DEFAULT_LR, epochs=DEFAULT_EPOCHS, max_steps=DEFAULT_MAX_STEPS):
    """Run the full freeze-ratio sweep."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LOADING PROBES")
    print("=" * 70)
    probes_dict = load_all_behavioral_probes(behaviors, seed=42)

    # Load or init results — use v2 filename
    suffix = "sweep_v2.json" if do_ft else "sweep_noft.json"
    output_path = RESULTS_DIR / suffix
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
        ft_label = "with FT" if do_ft else "NO FT"
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name} ({model_id})")
        print(f"COMPRESS: {compress_ratio:.0%} | FREEZE SWEEP: {freeze_ratios}")
        print(f"MODE: {ft_label} | CALIB: {calib_size} examples | LR: {lr}")
        print(f"{'=' * 70}")
        sweep_one_model(
            model_name, model_id, compress_ratio, freeze_ratios,
            behaviors, probes_dict, device, all_results, output_path,
            do_ft=do_ft, calib_size=calib_size, lr=lr, epochs=epochs,
            max_steps=max_steps,
        )

    print(f"\n{'=' * 70}")
    print(f"SWEEP COMPLETE — {len(all_results)} entries")
    print(f"Results: {output_path}")
    print(f"{'=' * 70}")
    return all_results


# ── Analysis ──────────────────────────────────────────────────────────────

def analyze(results):
    """Analyze freeze-ratio sweep and print behavioral localization table."""
    from collections import defaultdict

    table = defaultdict(lambda: defaultdict(dict))
    compress_ratios_seen = set()

    for key, data in results.items():
        if "_baseline" in key:
            continue
        model = data["model"]
        freeze_ratio = data.get("freeze_ratio", 0.0)
        compress_ratio = data.get("compress_ratio", 0.7)
        compress_ratios_seen.add(compress_ratio)
        ft = data.get("fine_tuned", False)

        for behavior, bdata in data.get("behaviors", {}).items():
            if isinstance(bdata, str) or "delta" not in bdata:
                continue
            table[model][behavior][freeze_ratio] = {
                "delta": bdata["delta"],
                "ft": ft,
            }

    for model in sorted(table):
        print(f"\n{'=' * 90}")
        print(f"MODEL: {model}  (compress={', '.join(f'{c:.0%}' for c in sorted(compress_ratios_seen))})")
        print(f"{'=' * 90}")

        behaviors = sorted(table[model])
        freeze_ratios = sorted(set(fr for b in behaviors for fr in table[model][b]))

        # Print header
        header = f"{'Behavior':<14}"
        for fr in freeze_ratios:
            header += f" | f={fr:>4.0%}"
        print(header)
        print("-" * len(header))

        for behavior in behaviors:
            row = f"{behavior:<14}"
            for fr in freeze_ratios:
                entry = table[model][behavior].get(fr)
                if entry is not None:
                    d = entry["delta"]
                    ft_marker = "†" if entry["ft"] else " "
                    if d > 0.01:
                        row += f" | {d:>+5.3f}↑{ft_marker}"
                    elif d < -0.01:
                        row += f" | {d:>+5.3f}↓{ft_marker}"
                    else:
                        row += f" | {d:>+5.3f} {ft_marker}"
                else:
                    row += f" |       — "
            print(row)

        print(f"\n  † = after gentle fine-tuning")

        # ── Behavioral localization analysis ──────────────────────────
        print(f"\n--- Behavioral Localization Analysis ---")
        for behavior in behaviors:
            entries_by_fr = table[model][behavior]
            if not entries_by_fr:
                continue

            deltas_by_fr = {fr: e["delta"] for fr, e in entries_by_fr.items()}
            best_fr = max(deltas_by_fr, key=lambda fr: deltas_by_fr[fr])
            best_delta = deltas_by_fr[best_fr]

            high_freeze = [d for fr, d in deltas_by_fr.items() if fr >= 0.75]
            low_freeze = [d for fr, d in deltas_by_fr.items() if fr <= 0.25]

            if high_freeze and low_freeze:
                high_mean = np.mean(high_freeze)
                low_mean = np.mean(low_freeze)
                if high_mean > low_mean + 0.01:
                    location = "EARLY-LAYER (survives high freeze)"
                elif low_mean > high_mean + 0.01:
                    location = "LATE-LAYER (needs unfrozen late layers)"
                else:
                    location = "DISTRIBUTED (similar across freeze levels)"
            else:
                location = "INSUFFICIENT DATA"

            marker = " **" if best_delta > 0 else ""
            print(f"  {behavior:<14}: best={best_fr:.0%} (delta={best_delta:+.4f}){marker}")
            print(f"    └─ {location}")

        # ── Denoising zone ────────────────────────────────────────────
        print(f"\n--- Denoising Zone ---")
        sweet_spot = []
        for behavior in behaviors:
            entries_by_fr = table[model][behavior]
            candidates = [entries_by_fr[fr]["delta"]
                          for fr in [0.50, 0.75, 0.90]
                          if fr in entries_by_fr]
            if candidates and max(candidates) > 0:
                sweet_spot.append(behavior)
                print(f"  {behavior}: denoising detected "
                      f"(best in zone: {max(candidates):+.4f})")

        n = len(sweet_spot)
        total = len(behaviors)
        print(f"\n  {n}/{total} behaviors show denoising in 50-90% freeze zone")


# ── Validation ────────────────────────────────────────────────────────────

def validate():
    """Quick pipeline validation on Qwen-0.5B with FT."""
    print("=" * 70)
    print("VALIDATION (Qwen-0.5B, reduced probes, 3 freeze ratios, WITH FT)")
    print("=" * 70)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model("Qwen/Qwen2.5-0.5B", DEVICE)

    # Small probe sets
    probes = {"factual": get_all_probes()[:10]}
    for b in ["toxicity", "bias", "sycophancy"]:
        try:
            probes[b] = load_behavioral_probes(b, n=20, seed=42)
            print(f"  {b}: {len(probes[b])} probes")
        except Exception as e:
            print(f"  {b}: FAILED — {e}")

    available = list(probes.keys())

    # Small calibration dataset
    print("\n  Loading calibration data (200 examples)...")
    calib = load_calibration_data(n=200, seed=42, tokenizer=tokenizer)

    print("\n--- Baseline (no compression) ---")
    baseline = {}
    for b in available:
        r = evaluate_behavior(b, model, tokenizer, probes[b], DEVICE)
        baseline[b] = r
        print(f"  {b}: rho={r['rho']:.4f}")

    original_state = copy.deepcopy(model.state_dict())

    print(f"\n--- Freeze-ratio sweep (compress=70%, with gentle FT) ---")
    for freeze_ratio in [0.0, 0.50, 0.90]:
        model.load_state_dict(original_state)
        model.eval()

        # Compress
        compress_qko(model, ratio=0.70)

        # Freeze
        unfreeze_all(model)
        if freeze_ratio > 0:
            freeze_layers(model, ratio=freeze_ratio)

        # Gentle FT
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  freeze={freeze_ratio:.0%} ({n_trainable/1e6:.1f}M trainable):")

        if n_trainable > 0:
            ft_stats = gentle_finetune(
                model, tokenizer, calib,
                epochs=1, lr=2e-4, batch_size=2,
                device=DEVICE, max_steps=50,  # just 50 steps for validation
            )
            # LoRA returns merged model
            if "merged_model" in ft_stats:
                model = ft_stats.pop("merged_model")
            print(f"    FT: loss={ft_stats['loss']:.4f}, time={ft_stats['time']:.1f}s")
        else:
            print(f"    FT: skipped (all frozen)")

        model.to(DEVICE)
        model.eval()

        for b in available:
            r = evaluate_behavior(b, model, tokenizer, probes[b], DEVICE)
            d = r["rho"] - baseline[b]["rho"]
            print(f"    {b}: rho={r['rho']:.4f} (delta={d:+.4f})")

    del model, tokenizer, original_state
    gc.collect()
    print("\n✓ Validation complete — freeze ratios should now show different deltas!")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Freeze-ratio sweep v2: layer localization with post-compression FT"
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model keys (e.g., qwen2.5-7b llama3.1-8b)")
    parser.add_argument("--compress-ratio", type=float, default=DEFAULT_COMPRESS_RATIO)
    parser.add_argument("--freeze-ratios", default=None,
                        help="Comma-separated freeze ratios")
    parser.add_argument("--behaviors", nargs="+", default=None)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--resume", default=None)

    # FT options
    parser.add_argument("--no-ft", action="store_true",
                        help="Skip fine-tuning (v1 behavior, for comparison)")
    parser.add_argument("--calib-size", type=int, default=DEFAULT_CALIB_SIZE,
                        help=f"Calibration dataset size (default: {DEFAULT_CALIB_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max training steps per condition (overrides epochs)")

    parser.add_argument("--validate", action="store_true",
                        help="Quick validation on Qwen-0.5B with FT")
    parser.add_argument("--analyze", default=None,
                        help="Analyze existing results JSON")
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

    freeze_ratios = (
        [float(r) for r in args.freeze_ratios.split(",")]
        if args.freeze_ratios
        else FREEZE_RATIOS
    )

    results = run_sweep(
        models=models,
        compress_ratio=args.compress_ratio,
        freeze_ratios=freeze_ratios,
        behaviors=args.behaviors or BEHAVIORS,
        device=args.device,
        resume_path=args.resume,
        do_ft=not args.no_ft,
        calib_size=args.calib_size,
        lr=args.lr,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
    analyze(results)


if __name__ == "__main__":
    main()
