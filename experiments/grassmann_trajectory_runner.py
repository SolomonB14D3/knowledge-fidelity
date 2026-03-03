#!/usr/bin/env python3
"""Grassmann angle trajectory experiment: constant γ vs γ-annealing.

Three conditions:
  1. Constant γ=0.03 (control — peak bias improvement from γ* sweep)
  2. Constant γ=0.10 (control — safe operating point)
  3. Annealing γ=0.10→0 at step 110 (hypothesis: early protection + late freedom)

Hypothesis (from 34M scale ladder, Mar 2):
  bias↔sycophancy Grassmann angles widen in the first third of training, then
  plateau. High γ is only needed during this early separation phase. Dropping
  γ→0 after step 110 (of 219) gives the optimizer maximum gradient freedom for
  the remainder, combining early protection with late spontaneous alignment.

Full pipeline:
  1. Phase A: baseline + compressed subspace extraction (~30 min)
  2. Surgery × 3 conditions with gradient telemetry + LoRA checkpoints (~7.5h)
  3. Phase B: post-hoc Grassmann extraction for all conditions (~8h)
  4. Analysis: compare trajectories, summarize findings

Usage:
    python experiments/grassmann_trajectory_runner.py Qwen/Qwen2.5-7B-Instruct
    python experiments/grassmann_trajectory_runner.py Qwen/Qwen2.5-7B-Instruct --phase-a-only
    python experiments/grassmann_trajectory_runner.py Qwen/Qwen2.5-7B-Instruct --phase-b-only
    python experiments/grassmann_trajectory_runner.py Qwen/Qwen2.5-7B-Instruct --surgery-only
    python experiments/grassmann_trajectory_runner.py Qwen/Qwen2.5-7B-Instruct --analysis-only
    python experiments/grassmann_trajectory_runner.py Qwen/Qwen2.5-7B-Instruct --conditions const_0.03 anneal
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR = "results/grassmann_trajectory"

# Dense checkpoints around step 110 (annealing transition point)
CHECKPOINT_STEPS = [0, 25, 50, 75, 100, 105, 110, 115, 120, 150, 219]

# ── Condition definitions ──────────────────────────────────────────────
# Each condition: (label, gamma_schedule callable, description)
# The gamma_schedule maps global_step → γ value.

ANNEAL_STEP = 110  # Midpoint of 219 steps; 34M data shows separation completes in first third

CONDITIONS = {
    "const_0.03": {
        "label": "gamma_const_0.03",
        "schedule": lambda s: 0.03,
        "gamma_weight": 0.03,  # fallback for display
        "description": "Constant γ=0.03 (peak bias improvement from γ* sweep)",
    },
    "const_0.10": {
        "label": "gamma_const_0.10",
        "schedule": lambda s: 0.10,
        "gamma_weight": 0.10,
        "description": "Constant γ=0.10 (safe operating point)",
    },
    "anneal": {
        "label": "gamma_anneal_0.10_to_0",
        "schedule": lambda s: 0.10 if s < ANNEAL_STEP else 0.0,
        "gamma_weight": 0.10,  # initial value (for display)
        "description": f"Annealing γ=0.10→0 at step {ANNEAL_STEP}",
    },
}

DEFAULT_CONDITIONS = ["const_0.03", "const_0.10", "anneal"]


def run_surgery_with_telemetry(
    model_name: str,
    condition_key: str,
    output_dir: str,
):
    """Run rho-surgery with gradient telemetry and LoRA checkpointing."""
    import mlx_lm
    from rho_eval.alignment.mlx_trainer import mlx_rho_guided_sft, mlx_svd_compress
    from rho_eval.alignment.dataset import BehavioralContrastDataset

    cond = CONDITIONS[condition_key]
    label = cond["label"]
    schedule = cond["schedule"]
    gamma_display = cond["gamma_weight"]

    gamma_dir = Path(output_dir) / label
    gamma_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if (gamma_dir / "surgery_result.json").exists():
        print(f"\n  Surgery [{label}] already complete, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"  SURGERY: {cond['description']}")
    print(f"  Label: {label}")
    print(f"  Model: {model_name}")
    print(f"  Checkpoints: {CHECKPOINT_STEPS}")
    print(f"  Output: {gamma_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # Load model
    print("  Loading model (MLX)...", flush=True)
    model, tokenizer = mlx_lm.load(model_name)

    # SVD compress
    print("  SVD compression (ratio=0.7)...", flush=True)
    n_compressed = mlx_svd_compress(model, keep_ratio=0.7)
    print(f"  Compressed {n_compressed} matrices")

    # Build SFT dataset (same as standard surgery)
    print("  Building SFT dataset...", flush=True)
    import random as _random
    from rho_eval.alignment.dataset import _build_trap_texts, _load_alpaca_texts
    rng = _random.Random(42)
    sft_texts = []
    trap_texts = _build_trap_texts(["sycophancy"], seed=42)
    rng.shuffle(trap_texts)
    sft_texts.extend(trap_texts[:400])
    try:
        alpaca_texts = _load_alpaca_texts(1600, seed=42)
        sft_texts.extend(alpaca_texts)
    except Exception as e:
        print(f"  Alpaca load failed ({e}), using traps only", flush=True)
    rng.shuffle(sft_texts)
    sft_texts = sft_texts[:2000]
    print(f"  {len(sft_texts)} SFT texts")

    # Build contrast dataset (sycophancy)
    print("  Building contrast dataset (sycophancy)...", flush=True)
    contrast_dataset = BehavioralContrastDataset(behaviors=["sycophancy"], seed=42)
    print(f"  {len(contrast_dataset)} sycophancy contrast pairs")

    # Build protection dataset (bias — same categories as previous runs)
    print("  Building protection dataset (bias)...", flush=True)
    protection_categories = [
        "Age", "Gender_biology", "Race_ethnicity",
        "Sexual_orientation_biology", "Religion",
    ]
    protection_dataset = BehavioralContrastDataset(
        behaviors=["bias"],
        categories=protection_categories,
        seed=42,
    )
    print(f"  {len(protection_dataset)} protection pairs")

    # Run SFT with telemetry + gamma_schedule
    print(f"\n  Starting LoRA SFT: rho=0.2, {cond['description']}", flush=True)
    print(f"  Gradient telemetry → {gamma_dir / 'gradient_telemetry.csv'}")
    print(f"  LoRA checkpoints at steps {CHECKPOINT_STEPS}")

    sft_result = mlx_rho_guided_sft(
        model, tokenizer, sft_texts, contrast_dataset,
        rho_weight=0.2,
        gamma_weight=gamma_display,  # initial value (overridden by schedule)
        gamma_schedule=schedule,
        protection_dataset=protection_dataset,
        epochs=1,
        lr=2e-4,
        batch_size=2,
        gradient_accumulation_steps=4,
        lora_rank=8,
        lora_alpha=16,
        margin=0.1,
        save_path=str(gamma_dir / "model"),
        checkpoint_steps=CHECKPOINT_STEPS,
        checkpoint_dir=str(gamma_dir),
        gradient_log_path=str(gamma_dir / "gradient_telemetry.csv"),
    )

    elapsed = time.time() - t0

    # Save result (without the model object)
    result = {
        "model": model_name,
        "condition": condition_key,
        "label": label,
        "description": cond["description"],
        "gamma_weight": gamma_display,
        "anneal_step": ANNEAL_STEP if "anneal" in condition_key else None,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "sft_result": {k: v for k, v in sft_result.items() if k != "merged_model"},
        "total_elapsed_sec": elapsed,
    }
    (gamma_dir / "surgery_result.json").write_text(json.dumps(result, indent=2))
    print(f"\n  Surgery complete: {elapsed:.0f}s → {gamma_dir / 'surgery_result.json'}")

    # Cleanup
    del model, sft_result
    import gc
    gc.collect()


def run_analysis(output_dir: str, condition_keys: list[str] = None):
    """Compare gradient telemetry and Grassmann trajectories across conditions."""
    import csv

    out = Path(output_dir)
    if condition_keys is None:
        condition_keys = DEFAULT_CONDITIONS

    print(f"\n{'='*60}")
    print(f"  ANALYSIS: Comparing {len(condition_keys)} conditions")
    print(f"{'='*60}\n")

    analysis = {
        "conditions": condition_keys,
        "anneal_step": ANNEAL_STEP,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "findings": [],
    }

    for cond_key in condition_keys:
        cond = CONDITIONS.get(cond_key)
        if cond is None:
            print(f"  [WARN] Unknown condition: {cond_key}")
            continue

        label = cond["label"]
        gamma_dir = out / label

        print(f"\n  ── {label} ({cond['description']}) ──")

        # Load gradient telemetry
        csv_path = gamma_dir / "gradient_telemetry.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if rows:
                cos_rg_values = [float(r["cos_rho_gamma"]) for r in rows]
                gamma_values = [float(r["gamma_value"]) for r in rows]

                mean_cos = sum(cos_rg_values) / len(cos_rg_values)
                early_cos = sum(cos_rg_values[:50]) / min(50, len(cos_rg_values))
                late_cos = sum(cos_rg_values[-50:]) / min(50, len(cos_rg_values))

                # For annealing: show pre/post transition stats
                pre_anneal = [c for i, c in enumerate(cos_rg_values) if i < ANNEAL_STEP]
                post_anneal = [c for i, c in enumerate(cos_rg_values) if i >= ANNEAL_STEP]

                print(f"    Mean cos(∇_rho, ∇_gamma) = {mean_cos:.4f}")
                print(f"    Early  (steps 0-49)       = {early_cos:.4f}")
                print(f"    Late   (steps 170-219)    = {late_cos:.4f}")
                if post_anneal:
                    pre_mean = sum(pre_anneal) / len(pre_anneal)
                    post_mean = sum(post_anneal) / max(1, len(post_anneal))
                    print(f"    Pre-{ANNEAL_STEP} cos               = {pre_mean:.4f}")
                    print(f"    Post-{ANNEAL_STEP} cos              = {post_mean:.4f}")
                print(f"    Trend: {'increasing' if late_cos > early_cos else 'decreasing'}")

                # Verify gamma schedule logged correctly
                if "anneal" in cond_key:
                    pre_gamma = [g for i, g in enumerate(gamma_values) if i < ANNEAL_STEP]
                    post_gamma = [g for i, g in enumerate(gamma_values) if i >= ANNEAL_STEP]
                    if pre_gamma and post_gamma:
                        print(f"    γ values: pre-{ANNEAL_STEP} mean={sum(pre_gamma)/len(pre_gamma):.4f}, "
                              f"post-{ANNEAL_STEP} mean={sum(post_gamma)/len(post_gamma):.4f}")

                analysis[f"{label}_gradient"] = {
                    "mean_cos_rho_gamma": mean_cos,
                    "early_cos_rho_gamma": early_cos,
                    "late_cos_rho_gamma": late_cos,
                    "pre_anneal_cos": sum(pre_anneal) / max(1, len(pre_anneal)) if pre_anneal else None,
                    "post_anneal_cos": sum(post_anneal) / max(1, len(post_anneal)) if post_anneal else None,
                    "n_steps": len(rows),
                }

        # Load Grassmann trajectory
        traj_path = out / f"grassmann_trajectory_{label}.json"
        if traj_path.exists():
            trajectory = json.loads(traj_path.read_text())
            print(f"\n    Grassmann trajectory ({len(trajectory)} checkpoints):")

            angles_by_step = {}
            for entry in trajectory:
                step = entry["step"]
                # Find sycophancy↔bias angle at deepest layer (34M data says deep layers drive separation)
                deepest_layer = None
                deepest_angle = None
                for layer_str, om in entry["overlap"].items():
                    behaviors = om["behaviors"]
                    if "sycophancy" in behaviors and "bias" in behaviors:
                        si = behaviors.index("sycophancy")
                        bi = behaviors.index("bias")
                        angle = om["subspace_angles"][si][bi]
                        layer_idx = int(layer_str)
                        if deepest_layer is None or layer_idx > deepest_layer:
                            deepest_layer = layer_idx
                            deepest_angle = angle
                if deepest_angle is not None:
                    marker = " ← anneal transition" if step == ANNEAL_STEP else ""
                    print(f"      step {step:3d}, layer {deepest_layer}: "
                          f"syc↔bias = {deepest_angle:.1f}°{marker}")
                    angles_by_step[step] = deepest_angle

            if angles_by_step:
                analysis[f"{label}_grassmann"] = {
                    "angles_by_step": angles_by_step,
                    "initial_angle": angles_by_step.get(0),
                    "final_angle": angles_by_step.get(219) or angles_by_step.get(max(angles_by_step)),
                    "transition_angle": angles_by_step.get(ANNEAL_STEP),
                }

    # ── Cross-condition comparison ──
    print(f"\n  ── Cross-Condition Summary ──")
    for cond_key in condition_keys:
        label = CONDITIONS[cond_key]["label"]
        g_key = f"{label}_grassmann"
        if g_key in analysis:
            g = analysis[g_key]
            init = g.get("initial_angle", "?")
            final = g.get("final_angle", "?")
            trans = g.get("transition_angle", "?")
            delta = f"{final - init:+.1f}°" if isinstance(init, (int, float)) and isinstance(final, (int, float)) else "?"
            print(f"    {label:35s}: {init}° → {final}° (Δ={delta}), at step {ANNEAL_STEP}: {trans}°")

    # Key verification checks
    findings = []
    labels = [CONDITIONS[k]["label"] for k in condition_keys]
    g_keys = [f"{l}_grassmann" for l in labels]

    # Check 1: step-0 angles should be identical across conditions
    step0_angles = [analysis[gk]["initial_angle"] for gk in g_keys if gk in analysis and analysis[gk].get("initial_angle")]
    if len(step0_angles) >= 2:
        spread = max(step0_angles) - min(step0_angles)
        check = "PASS" if spread < 1.0 else "FAIL"
        findings.append(f"Step-0 angle consistency: spread={spread:.2f}° [{check}]")
        print(f"\n    ✓ Step-0 consistency: {spread:.2f}° spread [{check}]")

    # Check 2: annealing vs const_0.03 final angle
    anneal_key = "gamma_anneal_0.10_to_0_grassmann"
    const03_key = "gamma_const_0.03_grassmann"
    if anneal_key in analysis and const03_key in analysis:
        a_final = analysis[anneal_key].get("final_angle")
        c_final = analysis[const03_key].get("final_angle")
        if a_final and c_final:
            verdict = "CONFIRMED" if a_final >= c_final else "REJECTED"
            findings.append(f"Annealing ≥ const_0.03: {a_final:.1f}° vs {c_final:.1f}° [{verdict}]")
            print(f"    {'✓' if verdict == 'CONFIRMED' else '✗'} Annealing hypothesis: "
                  f"{a_final:.1f}° vs {c_final:.1f}° [{verdict}]")

    analysis["findings"] = findings

    # Save analysis
    analysis_path = out / "trajectory_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2))
    print(f"\n  Analysis saved → {analysis_path}")


def main():
    parser = argparse.ArgumentParser(description="Grassmann trajectory experiment (3-condition)")
    parser.add_argument("model", nargs="?", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("-o", "--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--phase-a-only", action="store_true",
                        help="Run only Phase A (baseline extraction)")
    parser.add_argument("--phase-b-only", action="store_true",
                        help="Run only Phase B (checkpoint extraction)")
    parser.add_argument("--surgery-only", action="store_true",
                        help="Run only surgery")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Run only analysis on existing results")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS,
                        choices=list(CONDITIONS.keys()),
                        help=f"Conditions to run (default: {DEFAULT_CONDITIONS})")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    if args.analysis_only:
        run_analysis(args.output_dir, args.conditions)
        return

    if args.phase_a_only:
        from grassmann_trajectory import phase_a
        phase_a(args.model, args.output_dir)
        return

    if args.phase_b_only:
        from grassmann_trajectory import phase_b
        for cond_key in args.conditions:
            label = CONDITIONS[cond_key]["label"]
            gamma_dir = out / label
            if gamma_dir.exists():
                phase_b(args.model, str(gamma_dir), args.output_dir,
                        gamma_label=label)
        return

    if args.surgery_only:
        for cond_key in args.conditions:
            run_surgery_with_telemetry(args.model, cond_key, args.output_dir)
        return

    # ── Full pipeline ──

    # Step 1: Phase A — baseline + compressed extraction
    print("\n" + "█" * 60)
    print("  STEP 1: Phase A — Baseline Subspace Extraction")
    print("█" * 60)
    from grassmann_trajectory import phase_a
    phase_a(args.model, args.output_dir)

    # Steps 2-4: Surgery at each condition
    for i, cond_key in enumerate(args.conditions):
        cond = CONDITIONS[cond_key]
        print(f"\n" + "█" * 60)
        print(f"  STEP {i+2}: Surgery — {cond['description']}")
        print("█" * 60)
        run_surgery_with_telemetry(args.model, cond_key, args.output_dir)

    # Step 5: Phase B — Grassmann extraction from checkpoints
    print(f"\n" + "█" * 60)
    print(f"  STEP {len(args.conditions)+2}: Phase B — Grassmann Trajectory Extraction")
    print("█" * 60)
    from grassmann_trajectory import phase_b
    for cond_key in args.conditions:
        label = CONDITIONS[cond_key]["label"]
        gamma_dir = out / label
        phase_b(args.model, str(gamma_dir), args.output_dir,
                gamma_label=label)

    # Step 6: Analysis
    print(f"\n" + "█" * 60)
    print(f"  STEP {len(args.conditions)+3}: Analysis")
    print("█" * 60)
    run_analysis(args.output_dir, args.conditions)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  COMPLETE: {total/3600:.1f}h total")
    print(f"  Results: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
