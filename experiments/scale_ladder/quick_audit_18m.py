#!/usr/bin/env python3
"""Quick audit of existing 18M checkpoints to bracket bias emergence.

Runs behavioral ρ only (no subspace extraction) on the 5 existing checkpoints
to find where bias first shows signal. This tells us where to place the
high-res window for a detailed phase transition analysis.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale_ladder.scale_audit import load_checkpoint, run_rho_audit


def main():
    results_dir = Path("results/scale_ladder/18M_seed42")
    ckpt_dirs = sorted(results_dir.glob("checkpoint_*"), key=lambda d: int(d.name.split("_")[-1]))

    print(f"\n{'=' * 60}")
    print(f"  Quick Audit: 18M Checkpoints (bias emergence bracket)")
    print(f"  Found {len(ckpt_dirs)} checkpoints")
    print(f"{'=' * 60}\n")

    trajectory = []

    for i, ckpt_dir in enumerate(ckpt_dirs):
        step = int(ckpt_dir.name.split("_")[-1])
        audit_path = ckpt_dir / "audit_report.json"

        # Skip if already audited
        if audit_path.exists():
            print(f"  [{i+1}/{len(ckpt_dirs)}] Step {step}: already audited, loading...")
            audit_data = json.loads(audit_path.read_text())
            trajectory.append({"step": step, "audit": audit_data})
            continue

        print(f"\n  [{i+1}/{len(ckpt_dirs)}] Step {step} {'─' * 40}")
        t0 = time.time()

        model, tokenizer = load_checkpoint(str(ckpt_dir), device="cpu")
        report = run_rho_audit(model, tokenizer, device="cpu")

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

        trajectory.append({"step": step, "audit": audit_data})

        elapsed = time.time() - t0
        print(f"  Step {step} audited in {elapsed:.0f}s")

        # Print immediate results for this checkpoint
        for r in report.behaviors.values():
            if r.rho > 0.05:
                print(f"    {r.behavior}: ρ = {r.rho:.3f} {'← SIGNAL' if r.rho > 0.1 else ''}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Also include final model audit if exists
    final_audit_path = results_dir / "audit_report.json"
    if final_audit_path.exists():
        final_data = json.loads(final_audit_path.read_text())
        final_step = max(int(d.name.split("_")[-1]) for d in ckpt_dirs) + 2441  # approximate
        # Actually get exact step from training_metrics
        metrics_path = results_dir / "training_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            final_step = metrics.get("total_steps", final_step)
        trajectory.append({"step": final_step, "audit": final_data, "is_final": True})

    # Print summary
    print(f"\n{'═' * 60}")
    print(f"  BIAS EMERGENCE BRACKET — 18M MODEL")
    print(f"{'═' * 60}")
    print(f"\n  {'Step':>8s} {'bias':>8s} {'factual':>8s} {'overref':>8s} {'toxicity':>8s} {'sycoph':>8s}")
    print(f"  {'─' * 55}")

    for entry in sorted(trajectory, key=lambda e: e["step"]):
        step = entry["step"]
        results = {r["behavior"]: r["rho"] for r in entry["audit"]["results"]}
        bias = results.get("bias", 0)
        factual = results.get("factual", 0)
        overref = results.get("overrefusal", 0)
        tox = results.get("toxicity", 0)
        syc = results.get("sycophancy", 0)
        marker = " ← BIAS SIGNAL" if bias > 0.05 else ""
        print(f"  {step:>8d} {bias:>8.3f} {factual:>8.3f} {overref:>8.3f} {tox:>8.3f} {syc:>8.3f}{marker}")

    # Save trajectory
    output_path = results_dir / "bias_bracket_trajectory.json"
    output_path.write_text(json.dumps(trajectory, indent=2, default=str))
    print(f"\n  Saved: {output_path}")

    # Determine bracket
    prev_step = None
    for entry in sorted(trajectory, key=lambda e: e["step"]):
        results = {r["behavior"]: r["rho"] for r in entry["audit"]["results"]}
        if results.get("bias", 0) > 0.05:
            if prev_step:
                print(f"\n  ★ BIAS EMERGENCE BRACKET: steps {prev_step} → {entry['step']}")
                print(f"    High-res window should span this range")
            else:
                print(f"\n  ★ Bias already present at earliest checkpoint (step {entry['step']})")
            break
        prev_step = entry["step"]
    else:
        print(f"\n  ★ Bias not detected at any intermediate checkpoint")
        print(f"    Emerged between step {prev_step} and final model")

    print(f"\n{'═' * 60}\n")


if __name__ == "__main__":
    main()
