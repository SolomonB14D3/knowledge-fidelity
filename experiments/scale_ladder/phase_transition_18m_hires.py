#!/usr/bin/env python3
"""High-resolution phase transition experiment for BIAS emergence at 18M.

Bias is the first "complex" behavioral signal (rank-1 subspace) to emerge in the
scale ladder. At 7M, bias ρ = 0.000; at 18M, bias ρ = 0.133. The coarse bracket
from quick_audit_18m.py shows:

    step 2441 (20%): bias = 0.000
    step 4882 (40%): bias = 0.003
    step 7323 (60%): bias = 0.187  ← first signal
    step 9764 (80%): bias = 0.177
    step 12205 (100%): bias = 0.133 (final, with slight decline)

The emergence window is steps 4882–7323. This script re-trains the 18M model
with fine-grained checkpoints every 100 steps in that window, plus margin points
on both sides, to capture the bias ignition at high temporal resolution.

Key question: Does bias exhibit the same sharp sigmoid as overrefusal at 7M
(width = 22 steps), or is its emergence gradual?

Usage:
    # Full pipeline: train + audit
    python experiments/scale_ladder/phase_transition_18m_hires.py --device cpu

    # Audit only (if training already done)
    python experiments/scale_ladder/phase_transition_18m_hires.py --audit-only

    # Quick audit (no subspace extraction, just behavioral ρ)
    python experiments/scale_ladder/phase_transition_18m_hires.py --audit-only --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# High-res checkpoint schedule: every 100 steps in the bias emergence window
# Bracket: 4882–7323 with margins on both sides
# Also include a few early/late points for context
HIRES_STEPS = (
    [4000, 4200, 4400, 4600]  # Pre-emergence baseline (4 points)
    + list(range(4800, 7400, 100))  # Dense window: 4800-7300, 26 points
    + [7500, 7800, 8100, 8500, 9000]  # Post-emergence (5 points)
)

OUTPUT_DIR = "results/scale_ladder/18M_seed42_hires"


def train_hires(device: str = "cpu"):
    """Re-train 18M model with fine-grained checkpoints."""
    from experiments.scale_ladder.train_model import train

    print(f"\n{'█' * 60}")
    print(f"  HIGH-RES BIAS EMERGENCE EXPERIMENT")
    print(f"  Model: 18M (6 layers, d=256), Seed: 42")
    print(f"  Checkpoints: {len(HIRES_STEPS)} total")
    print(f"  Dense window: 4800-7300 (every 100 steps)")
    print(f"  Device: {device}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'█' * 60}\n")

    t0 = time.time()

    train(
        size="18M",
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
    """Audit all hires checkpoints for behavioral ρ + subspace geometry."""
    from experiments.scale_ladder.scale_audit import (
        SUBSPACE_BEHAVIORS,
        load_checkpoint,
        run_rho_audit,
        run_subspace_extraction,
    )

    results_dir = Path(OUTPUT_DIR)
    ckpt_dirs = sorted(results_dir.glob("checkpoint_*"), key=lambda d: int(d.name.split("_")[-1]))

    if not ckpt_dirs:
        print(f"\n  ERROR: No checkpoints found in {OUTPUT_DIR}")
        print(f"  Run training first: python {__file__}")
        return []

    print(f"\n{'=' * 60}")
    print(f"  Auditing {len(ckpt_dirs)} checkpoints {'(QUICK mode)' if quick else '(full + subspace)'}")
    print(f"{'=' * 60}")

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
        # Print key results immediately
        bias_rho = next((r.rho for r in report.behaviors.values() if r.behavior == "bias"), 0)
        print(f"  Step {step} audited in {elapsed:.0f}s — bias ρ = {bias_rho:.3f}"
              f"{'  ★ SIGNAL' if bias_rho > 0.05 else ''}")

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - t_total
    print(f"\n  All {len(trajectory)} checkpoints audited in {total_elapsed / 60:.1f} min")

    # Save combined trajectory
    traj_path = Path(OUTPUT_DIR) / "phase_transition_trajectory.json"
    traj_path.write_text(json.dumps(trajectory, indent=2, default=str))
    print(f"  Saved: {traj_path}")

    return trajectory


def main():
    parser = argparse.ArgumentParser(
        description="High-resolution bias emergence experiment at 18M scale"
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu or mps)")
    parser.add_argument("--audit-only", action="store_true", help="Skip training, audit existing checkpoints")
    parser.add_argument("--quick", action="store_true", help="Quick audit (behavioral ρ only, no subspace extraction)")
    args = parser.parse_args()

    if not args.audit_only:
        train_hires(device=args.device)

    trajectory = audit_checkpoints(device=args.device, quick=args.quick)

    if trajectory:
        # Print summary table
        print(f"\n{'═' * 70}")
        print(f"  18M BIAS EMERGENCE — HIGH RESOLUTION")
        print(f"{'═' * 70}")
        print(f"\n  {'Step':>6s} {'bias':>8s} {'factual':>8s} {'overref':>8s} {'toxicity':>8s} {'sycoph':>8s}")
        print(f"  {'─' * 55}")
        for entry in sorted(trajectory, key=lambda e: e["step"]):
            step = entry["step"]
            results = {r["behavior"]: r["rho"] for r in entry["audit"]["results"]}
            bias = results.get("bias", 0)
            fact = results.get("factual", 0)
            oref = results.get("overrefusal", 0)
            tox = results.get("toxicity", 0)
            syc = results.get("sycophancy", 0)
            marker = " ★" if bias > 0.05 else ""
            print(f"  {step:>6d} {bias:>8.3f} {fact:>8.3f} {oref:>8.3f} {tox:>8.3f} {syc:>8.3f}{marker}")

        # Try sigmoid fit
        steps = [e["step"] for e in trajectory]
        bias_vals = [next((r["rho"] for r in e["audit"]["results"] if r["behavior"] == "bias"), 0)
                     for e in trajectory]

        try:
            from experiments.scale_ladder.analyze_phase_transition import fit_sigmoid
            fit = fit_sigmoid(steps, bias_vals, "bias")
            if fit and "error" not in fit and fit.get("fit_quality_r2", 0) > 0.5:
                print(f"\n  Sigmoid fit: midpoint={fit['midpoint']:.0f}, "
                      f"width={fit['width']:.0f} steps, R²={fit['fit_quality_r2']:.3f}")
                if fit['width'] < 50:
                    print(f"  → SHARP transition (width < 50 steps)")
                elif fit['width'] < 200:
                    print(f"  → MODERATE transition")
                else:
                    print(f"  → GRADUAL transition")
        except ImportError:
            pass

        print(f"\n{'═' * 70}\n")


if __name__ == "__main__":
    main()
