"""Rho-Surgery — End-to-end surgical intervention pipeline.

Runs the full 4-stage Rho-Surgery process:
  1. DIAGNOSE: Generate a SurgicalPlan from audit data
  2. PRE-PATCH: (Phase B — SAE pre-patch, not yet implemented)
  3. SURGICAL TRAIN: Run hybrid pipeline with γ protection
  4. VERIFY: Check outcome against plan expectations

Usage:
    # Auto-generate plan and run surgery
    python experiments/rho_surgery.py Qwen/Qwen2.5-7B-Instruct

    # Use a saved plan
    python experiments/rho_surgery.py Qwen/Qwen2.5-7B-Instruct --plan surgical_plan.json

    # Generate plan only (no GPU needed)
    python experiments/rho_surgery.py Qwen/Qwen2.5-7B-Instruct --plan-only

    # Verify only (from existing results)
    python experiments/rho_surgery.py Qwen/Qwen2.5-7B-Instruct --verify-only results/surgery/

    # Conservative strategy (stronger protection, may limit target improvement)
    python experiments/rho_surgery.py Qwen/Qwen2.5-7B-Instruct --strategy conservative
"""

import argparse
import json
import time
from pathlib import Path


def stage_diagnose(
    model_name: str,
    *,
    audit_json_path: str | None = None,
    layer_heatmap_path: str | None = None,
    probe_landscape_path: str | None = None,
    strategy: str = "balanced",
    output_dir: Path,
) -> "SurgicalPlan":
    """Stage 1: Generate a surgical plan from diagnostics."""
    from rho_eval.surgical_planner import generate_surgical_plan

    print(f"\n{'='*70}")
    print(f"  STAGE 1: DIAGNOSE")
    print(f"  Model:    {model_name}")
    print(f"  Strategy: {strategy}")
    print(f"{'='*70}")

    plan = generate_surgical_plan(
        model_name,
        target_behavior="sycophancy",
        audit_json_path=audit_json_path,
        layer_heatmap_path=layer_heatmap_path,
        probe_landscape_path=probe_landscape_path,
        strategy=strategy,
    )

    # Save plan
    plan_path = output_dir / "surgical_plan.json"
    plan.to_json(plan_path)
    print(f"\n{plan.summary()}")
    print(f"\n  Plan saved to {plan_path}")

    return plan


def stage_surgical_train(
    model_name: str,
    plan: "SurgicalPlan",
    output_dir: Path,
) -> "HybridResult":
    """Stage 3: Run hybrid pipeline with γ protection from plan."""
    from rho_eval.hybrid import apply_hybrid_control

    print(f"\n{'='*70}")
    print(f"  STAGE 3: SURGICAL TRAIN")
    print(f"  Config: SVD={plan.compress_ratio}, SAE={plan.sae_layer}, "
          f"ρ={plan.rho_weight}, γ={plan.gamma_weight}")
    print(f"  Protection: {plan.protection_categories or 'none'}")
    print(f"{'='*70}")

    config = plan.to_hybrid_config()
    run_dir = output_dir / "hybrid_run"

    t0 = time.time()
    result = apply_hybrid_control(
        model_name, config,
        output_dir=str(run_dir),
    )
    elapsed = time.time() - t0

    print(f"\n  Completed in {elapsed/3600:.1f}h")
    print(f"  {result.summary()}")
    print(f"\n{result.to_table()}")

    return result


def stage_verify(
    plan: "SurgicalPlan",
    result: "HybridResult",
    output_dir: Path,
    *,
    max_collateral: float = 0.05,
) -> dict:
    """Stage 4: Verify surgical outcome."""
    from rho_eval.surgical_planner import verify_surgical_outcome

    print(f"\n{'='*70}")
    print(f"  STAGE 4: VERIFY")
    print(f"{'='*70}")

    verification = verify_surgical_outcome(
        plan, result,
        max_collateral_per_category=max_collateral,
    )

    print(f"\n  {verification['summary']}")

    # Target check
    tc = verification["target_check"]
    marker = "✓" if tc["passed"] else "✗"
    print(f"\n  Target: {marker} improvement={tc['improvement']:+.4f} "
          f"(threshold: {tc['threshold']:+.4f})")

    # Category checks
    for cat, cc in verification["category_checks"].items():
        marker = "✓" if cc["passed"] else "✗"
        print(f"  {cat}: {marker} {cc['before']:.1%} → {cc['after']:.1%} "
              f"(Δ={cc['delta']:+.1%}, max={cc['threshold']:+.1%})")

    # Overall check
    oc = verification["overall_check"]
    marker = "✓" if oc["passed"] else "✗"
    print(f"  Overall: {marker} regression={oc['regression']:+.4f} "
          f"(threshold: {oc['threshold']:+.4f})")

    # Suggestions
    if verification["suggestions"]:
        print(f"\n  Suggestions for next iteration:")
        for s in verification["suggestions"]:
            print(f"    → {s}")

    # Save verification
    verify_path = output_dir / "verification.json"
    verify_path.write_text(json.dumps(verification, indent=2, default=str))
    print(f"\n  Verification saved to {verify_path}")

    return verification


def find_cached_audit(model_name: str) -> str | None:
    """Find the most recent cached audit for this model.

    Looks in common locations:
      - results/hybrid_sweep/{model_short}/ (from previous sweeps)
      - results/ (from audit runs)
    """
    model_short = model_name.split("/")[-1]
    candidates = [
        Path("results/hybrid_sweep") / model_short,
        Path("results"),
    ]

    for base in candidates:
        if not base.exists():
            continue
        for json_path in sorted(base.rglob("hybrid_result.json"), reverse=True):
            try:
                data = json.loads(json_path.read_text())
                if data.get("model_name", "").endswith(model_short):
                    # Extract baseline phase report
                    phases = data.get("phases", [])
                    if phases and phases[0].get("phase") == "baseline":
                        return str(json_path)
            except (json.JSONDecodeError, KeyError):
                continue

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Rho-Surgery — Surgical behavioral intervention pipeline",
    )
    parser.add_argument("model", help="HuggingFace model ID")
    parser.add_argument(
        "--plan", type=str, default=None,
        help="Path to existing SurgicalPlan JSON (skip Stage 1)",
    )
    parser.add_argument(
        "--audit", type=str, default=None,
        help="Path to audit JSON for plan generation",
    )
    parser.add_argument(
        "--heatmap", type=str, default=None,
        help="Path to layer heatmap JSON",
    )
    parser.add_argument(
        "--landscape", type=str, default=None,
        help="Path to probe landscape JSON",
    )
    parser.add_argument(
        "--strategy", type=str, default="balanced",
        choices=["conservative", "balanced", "aggressive"],
        help="Protection strategy (default: balanced)",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Only generate plan (no GPU needed)",
    )
    parser.add_argument(
        "--verify-only", type=str, default=None,
        help="Path to results dir — skip training, just verify",
    )
    parser.add_argument(
        "--max-collateral", type=float, default=0.05,
        help="Max acceptable per-category collateral (default: 0.05)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/surgery",
        help="Output directory (default: results/surgery)",
    )
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    output_dir = Path(args.output) / model_short
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRho-Surgery Pipeline")
    print(f"  Model:    {args.model}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Output:   {output_dir}")

    # ── Stage 1: Diagnose ─────────────────────────────────────────────
    if args.plan:
        from rho_eval.surgical_planner import SurgicalPlan
        plan = SurgicalPlan.from_json(args.plan)
        print(f"\n  Loaded plan from {args.plan}")
        print(f"\n{plan.summary()}")
    else:
        # Try to find cached audit data
        audit_path = args.audit
        if audit_path is None:
            audit_path = find_cached_audit(args.model)
            if audit_path:
                print(f"\n  Found cached audit: {audit_path}")

        # Auto-detect known paths
        heatmap = args.heatmap
        landscape = args.landscape or "docs/probe_landscape.json"
        if not Path(landscape).exists():
            landscape = None

        plan = stage_diagnose(
            args.model,
            audit_json_path=audit_path,
            layer_heatmap_path=heatmap,
            probe_landscape_path=landscape,
            strategy=args.strategy,
            output_dir=output_dir,
        )

    if args.plan_only:
        print("\n  --plan-only: stopping after Stage 1 (Diagnose)")
        return

    # ── Stage 3: Surgical Train ───────────────────────────────────────
    if args.verify_only:
        # Load existing result for verification
        verify_dir = Path(args.verify_only)
        result_path = verify_dir / "hybrid_run" / "hybrid_result.json"
        if not result_path.exists():
            result_path = verify_dir / "hybrid_result.json"
        if not result_path.exists():
            print(f"  ERROR: No hybrid_result.json found in {verify_dir}")
            return

        from rho_eval.hybrid.schema import HybridResult, HybridConfig, PhaseResult
        data = json.loads(result_path.read_text())

        # Reconstruct HybridResult from JSON
        config_data = data.get("config", {})
        for key in ("compress_targets", "target_behaviors", "eval_behaviors",
                     "protection_behaviors", "protection_categories"):
            if key in config_data and isinstance(config_data[key], list):
                config_data[key] = tuple(config_data[key])
        config = HybridConfig(**config_data)

        result = HybridResult(config=config, model_name=data.get("model_name", args.model))
        result.audit_before = data.get("audit_before", {})
        result.audit_after = data.get("audit_after", {})
        result.collateral_damage = data.get("collateral_damage", {})
        result.total_elapsed_sec = data.get("total_elapsed_sec", 0)
        result.phases = [
            PhaseResult(
                phase=p["phase"],
                elapsed_sec=p.get("elapsed_sec", 0),
                details=p.get("details", {}),
                error=p.get("error"),
            )
            for p in data.get("phases", [])
        ]

        print(f"\n  Loaded result from {result_path}")
        print(f"  {result.summary()}")
    else:
        result = stage_surgical_train(args.model, plan, output_dir)

    # ── Stage 4: Verify ───────────────────────────────────────────────
    verification = stage_verify(
        plan, result, output_dir,
        max_collateral=args.max_collateral,
    )

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SURGERY COMPLETE: {'PASS ✓' if verification['passed'] else 'FAIL ✗'}")
    print(f"{'='*70}")

    if not verification["passed"] and verification["suggestions"]:
        print(f"\n  Next iteration: address {len(verification['suggestions'])} suggestions")
        print(f"  Re-run with --plan {output_dir / 'surgical_plan.json'} "
              f"after adjusting parameters")


if __name__ == "__main__":
    main()
