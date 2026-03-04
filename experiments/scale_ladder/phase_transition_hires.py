#!/usr/bin/env python3
"""High-resolution phase transition experiment for behavioral subspace emergence.

Re-trains the 7M model with fine-grained checkpoints around the overrefusal
phase transition (steps 1220-2440) to capture the geometric reorganization
at high temporal resolution.

The original 7M training showed:
  - step 1220: overrefusal ρ=0.000 (absent)
  - step 2440: overrefusal ρ=0.920 (nearly saturated)
  - step 3660: overrefusal ρ=1.000 (fully saturated)

This experiment saves checkpoints every 100 steps from 800-3000 (23 snapshots),
then audits each for both behavioral ρ scores AND subspace geometry (effective
dimensionality + Grassmann angles). This reveals whether the transition is a
smooth rotation of the subspace into alignment or a sudden crystallization.

Also tracks factual ρ decline: the original trajectory showed factual peaking
at step 2440 (ρ=0.393) then declining to ρ=0.330, suggesting the LM objective
actively destroys factual structure — direct motivation for geometric priors.

Usage:
    # Full pipeline: train + audit (CPU-safe, runs alongside MPS scale ladder)
    python experiments/scale_ladder/phase_transition_hires.py

    # Audit only (if training already done)
    python experiments/scale_ladder/phase_transition_hires.py --audit-only

    # Quick audit (no subspace extraction, just behavioral ρ)
    python experiments/scale_ladder/phase_transition_hires.py --audit-only --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# High-res checkpoint schedule: every 100 steps from 800-3000
# Covers the transition zone (1220-2440) with ±400-step margins
HIRES_STEPS = list(range(800, 3001, 100))  # 23 checkpoints

OUTPUT_DIR = "results/scale_ladder/7M_seed42_hires"


def train_hires(device: str = "cpu"):
    """Re-train 7M model with fine-grained checkpoints."""
    from experiments.scale_ladder.train_model import train

    print(f"\n{'█' * 60}")
    print(f"  HIGH-RES PHASE TRANSITION EXPERIMENT")
    print(f"  Model: 7M, Seed: 42 (identical to original run)")
    print(f"  Checkpoints: every 100 steps, {HIRES_STEPS[0]}-{HIRES_STEPS[-1]}")
    print(f"  Total snapshots: {len(HIRES_STEPS)}")
    print(f"  Device: {device}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'█' * 60}\n")

    t0 = time.time()

    train(
        size="7M",
        seed=42,
        device_str=device,
        output_dir=OUTPUT_DIR,
        dataset="openwebtext",
        checkpoint_steps=HIRES_STEPS,
    )

    elapsed = time.time() - t0
    print(f"\n  Training complete: {elapsed / 60:.1f} min")

    return elapsed


def audit_checkpoints(device: str = "cpu", quick: bool = False):
    """Audit all high-res checkpoints for behavioral ρ + subspace geometry."""
    from experiments.scale_ladder.scale_audit import (
        load_checkpoint,
        run_rho_audit,
        run_subspace_extraction,
    )
    from experiments.scale_ladder.configs import SUBSPACE_BEHAVIORS

    out = Path(OUTPUT_DIR)
    ckpt_dirs = sorted(out.glob("checkpoint_*"))

    if not ckpt_dirs:
        print("  ERROR: No checkpoints found. Run training first.")
        return

    print(f"\n{'█' * 60}")
    print(f"  AUDITING {len(ckpt_dirs)} CHECKPOINTS")
    print(f"  Mode: {'quick (ρ only)' if quick else 'full (ρ + subspaces)'}")
    print(f"{'█' * 60}\n")

    trajectory = []
    t_total = time.time()

    for i, ckpt_dir in enumerate(ckpt_dirs):
        step = int(ckpt_dir.name.split("_")[-1])

        # Skip if already audited
        audit_path = ckpt_dir / "audit_report.json"
        subspace_path = ckpt_dir / "subspace_report.json"
        if audit_path.exists() and (quick or subspace_path.exists()):
            print(f"  [{i+1}/{len(ckpt_dirs)}] Step {step}: already audited, loading...")
            audit_data = json.loads(audit_path.read_text())
            entry = {"step": step, "audit": audit_data}
            if subspace_path.exists():
                entry["subspace"] = json.loads(subspace_path.read_text())
            trajectory.append(entry)
            continue

        print(f"\n  [{i+1}/{len(ckpt_dirs)}] Step {step} {'─' * 40}")
        t0 = time.time()

        # Load model
        model, tokenizer = load_checkpoint(str(ckpt_dir), device=device)

        # Behavioral ρ audit
        report = run_rho_audit(model, tokenizer, device=device)

        # Save audit
        import torch
        audit_data = {
            "checkpoint": str(ckpt_dir),
            "step": step,
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_layers": model.config.n_layer,
            "d_model": model.config.n_embd,
            "results": [
                {
                    "behavior": r.behavior,
                    "rho": r.rho,
                    "positive_count": r.positive_count,
                    "total": r.total,
                    "status": r.status,
                }
                for r in report.behaviors.values()
            ],
        }
        audit_path.write_text(json.dumps(audit_data, indent=2))

        entry = {"step": step, "audit": audit_data}

        # Subspace extraction (unless quick mode)
        if not quick:
            subspaces, overlap, eff_dim_data = run_subspace_extraction(
                model, tokenizer, device=device
            )

            # Serialize overlap
            overlap_serializable = {}
            for key, matrix in overlap.items():
                if hasattr(matrix, "tolist"):
                    overlap_serializable[key] = matrix.tolist()
                elif isinstance(matrix, dict):
                    overlap_serializable[key] = {
                        str(k): v.tolist() if hasattr(v, "tolist") else v
                        for k, v in matrix.items()
                    }
                else:
                    overlap_serializable[key] = str(matrix)

            subspace_data = {
                "checkpoint": str(ckpt_dir),
                "step": step,
                "behaviors": SUBSPACE_BEHAVIORS,
                "n_layers": model.config.n_layer,
                "effective_dimensionality": {
                    beh: {str(k): v for k, v in layers.items()}
                    for beh, layers in eff_dim_data.items()
                },
                "overlap": overlap_serializable,
            }
            subspace_path.write_text(json.dumps(subspace_data, indent=2, default=str))
            entry["subspace"] = subspace_data

        trajectory.append(entry)

        elapsed = time.time() - t0
        print(f"  Step {step} audited in {elapsed:.0f}s")

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save combined trajectory
    traj_path = out / "phase_transition_trajectory.json"
    traj_path.write_text(json.dumps(trajectory, indent=2, default=str))

    total_elapsed = time.time() - t_total
    print(f"\n{'═' * 60}")
    print(f"  AUDIT COMPLETE: {len(trajectory)} checkpoints, {total_elapsed / 60:.1f} min")
    print(f"  Trajectory: {traj_path}")
    print(f"{'═' * 60}")

    # Print summary table
    print(f"\n  {'Step':>6s} | {'factual':>8s} | {'overref':>8s} | {'toxicity':>8s} | {'refusal':>8s} | {'bias':>6s} | {'syc':>6s}")
    print(f"  {'─' * 65}")
    for entry in trajectory:
        step = entry["step"]
        results = {r["behavior"]: r["rho"] for r in entry["audit"]["results"]}
        print(
            f"  {step:6d} | "
            f"{results.get('factual', 0):8.3f} | "
            f"{results.get('overrefusal', 0):8.3f} | "
            f"{results.get('toxicity', 0):8.3f} | "
            f"{results.get('refusal', 0):8.3f} | "
            f"{results.get('bias', 0):6.3f} | "
            f"{results.get('sycophancy', 0):6.3f}"
        )

    return trajectory


def main():
    parser = argparse.ArgumentParser(
        description="High-resolution phase transition experiment (7M)"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--audit-only", action="store_true",
                        help="Skip training, just audit existing checkpoints")
    parser.add_argument("--quick", action="store_true",
                        help="Quick audit: behavioral ρ only, skip subspace extraction")
    args = parser.parse_args()

    t_start = time.time()

    if not args.audit_only:
        train_hires(device=args.device)

    audit_checkpoints(device=args.device, quick=args.quick)

    total = time.time() - t_start
    print(f"\n  TOTAL TIME: {total / 60:.1f} min ({total / 3600:.1f}h)")


if __name__ == "__main__":
    main()
