#!/usr/bin/env python3
"""Run full subspace extraction (with Grassmann angles) on hires checkpoints.

Fills in subspace_report.json for any checkpoint that has audit_report.json
but is missing subspace data. Works on any scale (7M, 18M, 34M, 64M).

Usage:
    python experiments/scale_ladder/audit_hires_subspaces.py results/scale_ladder/18M_seed42_hires
    python experiments/scale_ladder/audit_hires_subspaces.py results/scale_ladder/7M_seed42_hires
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scale_ladder.scale_audit import (
    load_checkpoint,
    run_rho_audit,
    run_subspace_extraction,
)
from experiments.scale_ladder.configs import SUBSPACE_BEHAVIORS


def main():
    if len(sys.argv) < 2:
        print("Usage: python audit_hires_subspaces.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"

    ckpt_dirs = sorted(results_dir.glob("checkpoint_*"))
    if not ckpt_dirs:
        print(f"No checkpoints found in {results_dir}")
        sys.exit(1)

    # Find checkpoints needing subspace extraction
    needs_work = []
    for ckpt_dir in ckpt_dirs:
        subspace_path = ckpt_dir / "subspace_report.json"
        if not subspace_path.exists():
            needs_work.append(ckpt_dir)

    # Also find ones that have subspace but no Grassmann angles
    for ckpt_dir in ckpt_dirs:
        subspace_path = ckpt_dir / "subspace_report.json"
        if subspace_path.exists() and ckpt_dir not in needs_work:
            data = json.loads(subspace_path.read_text())
            if "overlap" not in data or not data["overlap"]:
                needs_work.append(ckpt_dir)

    needs_work = sorted(set(needs_work))

    print(f"\n{'=' * 60}")
    print(f"  SUBSPACE EXTRACTION: {results_dir.name}")
    print(f"  Total checkpoints: {len(ckpt_dirs)}")
    print(f"  Need extraction: {len(needs_work)}")
    print(f"  Device: {device}")
    print(f"{'=' * 60}\n")

    if not needs_work:
        print("  All checkpoints already have subspace data.")
        return

    t_total = time.time()

    for i, ckpt_dir in enumerate(needs_work):
        step = int(ckpt_dir.name.split("_")[-1])
        print(f"\n  [{i+1}/{len(needs_work)}] Step {step} {'─' * 40}")
        t0 = time.time()

        model, tokenizer = load_checkpoint(str(ckpt_dir), device=device)

        # Run audit if missing
        audit_path = ckpt_dir / "audit_report.json"
        if not audit_path.exists():
            report = run_rho_audit(model, tokenizer, device=device)
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
            print(f"    Audit saved.")

        # Subspace extraction
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
        subspace_path = ckpt_dir / "subspace_report.json"
        subspace_path.write_text(json.dumps(subspace_data, indent=2, default=str))

        elapsed = time.time() - t0
        print(f"    Subspace saved. ({elapsed:.0f}s)")

        # Free memory
        del model, tokenizer, subspaces, overlap, eff_dim_data

    total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE: {len(needs_work)} checkpoints in {total / 60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
