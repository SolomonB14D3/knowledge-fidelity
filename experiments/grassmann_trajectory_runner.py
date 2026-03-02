#!/usr/bin/env python3
"""Grassmann angle trajectory experiment: γ=0.03 vs γ=0.10.

Full pipeline:
  1. Phase A: baseline + compressed subspace extraction (~30 min)
  2. Surgery at γ=0.03 with gradient telemetry + LoRA checkpoints (~2.5h)
  3. Surgery at γ=0.10 with gradient telemetry + LoRA checkpoints (~2.5h)
  4. Phase B: post-hoc Grassmann extraction for both γ runs (~3h)
  5. Analysis: compare trajectories, summarize findings

Usage:
    python experiments/grassmann_trajectory_runner.py Qwen/Qwen2.5-7B-Instruct
    python experiments/grassmann_trajectory_runner.py Qwen/Qwen2.5-7B-Instruct --phase-a-only
    python experiments/grassmann_trajectory_runner.py Qwen/Qwen2.5-7B-Instruct --phase-b-only
    python experiments/grassmann_trajectory_runner.py Qwen/Qwen2.5-7B-Instruct --analysis-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR = "results/grassmann_trajectory"
CHECKPOINT_STEPS = [0, 25, 50, 100, 150, 219]
GAMMA_VALUES = [0.03, 0.10]


def run_surgery_with_telemetry(model_name: str, gamma: float, output_dir: str):
    """Run rho-surgery with gradient telemetry and LoRA checkpointing."""
    import mlx_lm
    from rho_eval.alignment.mlx_trainer import mlx_rho_guided_sft, mlx_svd_compress
    from rho_eval.alignment.dataset import BehavioralContrastDataset
    from rho_eval.surgical_planner import SurgicalPlan

    gamma_dir = Path(output_dir) / f"gamma_{gamma:.2f}"
    gamma_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if (gamma_dir / "surgery_result.json").exists():
        print(f"\n  Surgery at γ={gamma} already complete, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"  SURGERY: γ={gamma}")
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
    from rho_eval.alignment.dataset import load_sft_texts
    sft_texts = load_sft_texts(max_samples=2000)
    print(f"  {len(sft_texts)} SFT texts")

    # Build contrast dataset (sycophancy)
    print("  Building contrast dataset (sycophancy)...", flush=True)
    contrast_dataset = BehavioralContrastDataset(behaviors=["sycophancy"])
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
    )
    print(f"  {len(protection_dataset)} protection pairs")

    # Run SFT with telemetry
    print(f"\n  Starting LoRA SFT: rho=0.2, gamma={gamma}", flush=True)
    print(f"  Gradient telemetry → {gamma_dir / 'gradient_telemetry.csv'}")
    print(f"  LoRA checkpoints at steps {CHECKPOINT_STEPS}")

    sft_result = mlx_rho_guided_sft(
        model, tokenizer, sft_texts, contrast_dataset,
        rho_weight=0.2,
        gamma_weight=gamma,
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
        "gamma": gamma,
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


def run_analysis(output_dir: str):
    """Compare gradient telemetry and Grassmann trajectories across γ values."""
    out = Path(output_dir)

    print(f"\n{'='*60}")
    print(f"  ANALYSIS: Comparing γ trajectories")
    print(f"{'='*60}\n")

    analysis = {"gamma_values": GAMMA_VALUES, "findings": []}

    for gamma in GAMMA_VALUES:
        gamma_dir = out / f"gamma_{gamma:.2f}"

        # Load gradient telemetry
        csv_path = gamma_dir / "gradient_telemetry.csv"
        if csv_path.exists():
            import csv
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if rows:
                cos_rg_values = [float(r["cos_rho_gamma"]) for r in rows]
                mean_cos = sum(cos_rg_values) / len(cos_rg_values)
                early_cos = sum(cos_rg_values[:50]) / min(50, len(cos_rg_values))
                late_cos = sum(cos_rg_values[-50:]) / min(50, len(cos_rg_values))

                print(f"  γ={gamma}:")
                print(f"    Mean cos(∇_rho, ∇_gamma) = {mean_cos:.4f}")
                print(f"    Early (steps 0-49)        = {early_cos:.4f}")
                print(f"    Late  (steps 170-219)     = {late_cos:.4f}")
                print(f"    Trend: {'increasing' if late_cos > early_cos else 'decreasing'}")

                analysis[f"gamma_{gamma:.2f}_gradient"] = {
                    "mean_cos_rho_gamma": mean_cos,
                    "early_cos_rho_gamma": early_cos,
                    "late_cos_rho_gamma": late_cos,
                    "n_steps": len(rows),
                }

        # Load Grassmann trajectory
        traj_path = out / f"grassmann_trajectory_{gamma:.2f}.json"
        if traj_path.exists():
            trajectory = json.loads(traj_path.read_text())
            print(f"\n    Grassmann trajectory ({len(trajectory)} checkpoints):")

            for entry in trajectory:
                step = entry["step"]
                # Find sycophancy↔bias angle at representative layers
                for layer_str, om in entry["overlap"].items():
                    behaviors = om["behaviors"]
                    if "sycophancy" in behaviors and "bias" in behaviors:
                        si = behaviors.index("sycophancy")
                        bi = behaviors.index("bias")
                        angle = om["subspace_angles"][si][bi]
                        print(f"      step {step:3d}, layer {layer_str}: syc↔bias = {angle:.1f}°")
                        break  # Just show one representative layer

    # Save analysis
    analysis_path = out / "trajectory_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2))
    print(f"\n  Analysis saved → {analysis_path}")


def main():
    parser = argparse.ArgumentParser(description="Grassmann trajectory experiment")
    parser.add_argument("model", nargs="?", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("-o", "--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--phase-a-only", action="store_true",
                        help="Run only Phase A (baseline extraction)")
    parser.add_argument("--phase-b-only", action="store_true",
                        help="Run only Phase B (checkpoint extraction)")
    parser.add_argument("--surgery-only", action="store_true",
                        help="Run only surgery (steps 2-3)")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Run only analysis on existing results")
    parser.add_argument("--gammas", type=float, nargs="+", default=GAMMA_VALUES,
                        help="Gamma values to test (default: 0.03 0.10)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    if args.analysis_only:
        run_analysis(args.output_dir)
        return

    if args.phase_a_only:
        from grassmann_trajectory import phase_a
        phase_a(args.model, args.output_dir)
        return

    if args.phase_b_only:
        from grassmann_trajectory import phase_b
        for gamma in args.gammas:
            gamma_dir = out / f"gamma_{gamma:.2f}"
            if gamma_dir.exists():
                phase_b(args.model, str(gamma_dir), args.output_dir,
                        gamma_label=f"{gamma:.2f}")
        return

    if args.surgery_only:
        for gamma in args.gammas:
            run_surgery_with_telemetry(args.model, gamma, args.output_dir)
        return

    # ── Full pipeline ──

    # Step 1: Phase A — baseline + compressed extraction
    print("\n" + "█" * 60)
    print("  STEP 1: Phase A — Baseline Subspace Extraction")
    print("█" * 60)
    from grassmann_trajectory import phase_a
    phase_a(args.model, args.output_dir)

    # Steps 2-3: Surgery at each γ
    for i, gamma in enumerate(args.gammas):
        print(f"\n" + "█" * 60)
        print(f"  STEP {i+2}: Surgery at γ={gamma}")
        print("█" * 60)
        run_surgery_with_telemetry(args.model, gamma, args.output_dir)

    # Step 4: Phase B — Grassmann extraction from checkpoints
    print(f"\n" + "█" * 60)
    print(f"  STEP {len(args.gammas)+2}: Phase B — Grassmann Trajectory Extraction")
    print("█" * 60)
    from grassmann_trajectory import phase_b
    for gamma in args.gammas:
        gamma_dir = out / f"gamma_{gamma:.2f}"
        phase_b(args.model, str(gamma_dir), args.output_dir,
                gamma_label=f"{gamma:.2f}")

    # Step 5: Analysis
    print(f"\n" + "█" * 60)
    print(f"  STEP {len(args.gammas)+3}: Analysis")
    print("█" * 60)
    run_analysis(args.output_dir)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  COMPLETE: {total/3600:.1f}h total")
    print(f"  Results: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
