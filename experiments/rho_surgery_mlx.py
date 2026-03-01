"""Rho-Surgery — End-to-end surgical intervention pipeline (MLX backend).

MLX version of rho_surgery.py. Runs entirely on Apple Silicon GPU via MLX,
avoiding the MPS deadlock issues with PyTorch on 7B+ models.

Pipeline stages:
  1. DIAGNOSE: Generate a SurgicalPlan from audit/diagnostic data
  2. PRE-PATCH: (Phase B — not yet implemented)
  3. SURGICAL TRAIN: SVD compress + LoRA SFT with γ protection (MLX)
  4. VERIFY: Compare per-category bias before/after

Usage:
    # Auto-generate plan and run surgery
    python experiments/rho_surgery_mlx.py Qwen/Qwen2.5-7B-Instruct

    # Generate plan only (no GPU needed)
    python experiments/rho_surgery_mlx.py Qwen/Qwen2.5-7B-Instruct --plan-only

    # Use a saved plan
    python experiments/rho_surgery_mlx.py Qwen/Qwen2.5-7B-Instruct --plan surgical_plan.json

    # Conservative strategy (stronger γ protection)
    python experiments/rho_surgery_mlx.py Qwen/Qwen2.5-7B-Instruct --strategy conservative

    # Use diagnostic results from a previous MLX run
    python experiments/rho_surgery_mlx.py Qwen/Qwen2.5-7B-Instruct \
        --diag-result results/hybrid_sweep/diagnostics/cr0.7_saeNone_rho0.2_diag_mlx/diagnostic_result.json
"""

import argparse
import json
import random as _random
import time
import sys
from pathlib import Path

import numpy as np

# Ensure src/ on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ── MLX SVD Compression (from diagnose_bias_7b_mlx.py) ──────────────────


def mlx_svd_compress(model, ratio: float = 0.7) -> int:
    """SVD compress Q/K/O attention projections in an MLX model."""
    import mlx.core as mx

    compressed = 0
    safe_projections = ["q_proj", "k_proj", "o_proj"]

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for proj_name in safe_projections:
            if not hasattr(attn, proj_name):
                continue

            proj = getattr(attn, proj_name)
            W_mx = proj.weight
            # Convert to float32 numpy for SVD (MLX bfloat16 can't go direct to numpy)
            W = np.array(W_mx.astype(mx.float32))

            if len(W.shape) != 2 or min(W.shape) <= 10:
                continue

            rank = max(1, int(min(W.shape) * ratio))

            try:
                U, S, Vh = np.linalg.svd(W, full_matrices=False)
                W_approx = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
                proj.weight = mx.array(W_approx).astype(W_mx.dtype)
                compressed += 1
            except Exception as e:
                print(f"    SVD failed for layer {i} {proj_name}: {e}", flush=True)
                continue

    mx.eval(model.parameters())
    return compressed


def mlx_freeze_layers(model, ratio: float = 0.75) -> dict:
    """Return freeze info (for LoRA layer selection). In MLX, freezing is
    handled by only applying LoRA to unfrozen layers."""
    n_layers = len(model.model.layers)
    n_freeze = int(n_layers * ratio)

    return {
        "n_layers": n_layers,
        "n_frozen": n_freeze,
        "freeze_ratio": ratio,
        "frozen_layer_indices": list(range(n_freeze)),
        "trainable_layer_indices": list(range(n_freeze, n_layers)),
    }


# ── Bias Audit (MLX) ────────────────────────────────────────────────────


def run_bias_audit(model, tokenizer, label: str) -> dict:
    """Run bias audit on MLX model and return structured results."""
    from rho_eval.audit import audit

    t0 = time.time()
    report = audit(model=model, tokenizer=tokenizer, behaviors=["bias"], device="mlx")
    elapsed = time.time() - t0

    bias_result = report.behaviors["bias"]
    print(f"  [{label}] Bias audit: {elapsed:.1f}s", flush=True)
    print(f"    rho={bias_result.rho:.4f}, "
          f"{bias_result.positive_count}/{bias_result.total} "
          f"({bias_result.retention:.1%})", flush=True)

    # Category breakdown
    meta = bias_result.metadata or {}
    cat_metrics = meta.get("category_metrics", {})
    if cat_metrics:
        print(f"    {'Category':<30s} {'Accuracy':>8s} {'N':>4s}", flush=True)
        for cat, data in sorted(cat_metrics.items(), key=lambda x: -x[1]["accuracy"]):
            print(f"    {cat:<30s} {data['accuracy']:>7.1%} {data['n']:>4d}", flush=True)

    return {
        "rho": bias_result.rho,
        "positive_count": bias_result.positive_count,
        "total": bias_result.total,
        "retention": bias_result.retention,
        "category_metrics": cat_metrics,
        "elapsed": elapsed,
    }


# ── Stage 1: Diagnose ───────────────────────────────────────────────────


def stage_diagnose(
    model_name: str,
    *,
    audit_json_path: str | None = None,
    diag_result_path: str | None = None,
    layer_heatmap_path: str | None = None,
    probe_landscape_path: str | None = None,
    strategy: str = "balanced",
    output_dir: Path,
) -> "SurgicalPlan":
    """Stage 1: Generate a surgical plan from diagnostics.

    Can consume either:
      - A standard audit JSON (AuditReport or HybridResult format)
      - A diagnostic_result.json from diagnose_bias_7b_mlx.py
    """
    from rho_eval.surgical_planner import generate_surgical_plan, SurgicalPlan

    print(f"\n{'='*70}", flush=True)
    print(f"  STAGE 1: DIAGNOSE", flush=True)
    print(f"  Model:    {model_name}", flush=True)
    print(f"  Strategy: {strategy}", flush=True)
    print(f"{'='*70}", flush=True)

    # If we have a diagnostic result (from MLX diagnostic script), convert it
    # to the format generate_surgical_plan() expects
    effective_audit_path = audit_json_path

    if diag_result_path and not audit_json_path:
        diag_data = json.loads(Path(diag_result_path).read_text())
        # Extract baseline bias data and wrap in audit-compatible format
        baseline = diag_data.get("baseline_bias", {})
        cat_metrics = baseline.get("category_metrics", {})

        if cat_metrics:
            # Build a minimal audit-like structure from diagnostic results
            audit_wrapper = {
                "model": model_name,
                "behaviors": {
                    "bias": {
                        "rho": baseline.get("rho", 0.0),
                        "positive_count": baseline.get("positive_count", 0),
                        "total": baseline.get("total", 0),
                        "retention": baseline.get("retention", 0.0),
                        "metadata": {
                            "category_metrics": cat_metrics,
                        },
                    },
                },
            }
            # Save wrapper to temp file for plan generation
            wrapper_path = output_dir / "_diag_audit_wrapper.json"
            wrapper_path.write_text(json.dumps(audit_wrapper, indent=2))
            effective_audit_path = str(wrapper_path)
            print(f"  Using diagnostic result: {diag_result_path}", flush=True)

    plan = generate_surgical_plan(
        model_name,
        target_behavior="sycophancy",
        audit_json_path=effective_audit_path,
        layer_heatmap_path=layer_heatmap_path,
        probe_landscape_path=probe_landscape_path,
        strategy=strategy,
    )

    # Save plan
    plan_path = output_dir / "surgical_plan.json"
    plan.to_json(plan_path)
    print(f"\n{plan.summary()}", flush=True)
    print(f"\n  Plan saved to {plan_path}", flush=True)

    return plan


# ── Stage 3: Surgical Train (MLX) ───────────────────────────────────────


def stage_surgical_train_mlx(
    model_name: str,
    plan: "SurgicalPlan",
    output_dir: Path,
) -> dict:
    """Stage 3: Run SVD + γ-protected LoRA SFT entirely on MLX.

    Returns a result dict with baseline and final bias audits.
    """
    import mlx_lm
    import mlx.core as mx

    run_dir = output_dir / "surgery_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  STAGE 3: SURGICAL TRAIN (MLX)", flush=True)
    print(f"  Config: SVD={plan.compress_ratio}, "
          f"ρ={plan.rho_weight}, γ={plan.gamma_weight}", flush=True)
    print(f"  Protection: {plan.protection_categories or 'none'}", flush=True)
    print(f"  Output: {run_dir}", flush=True)
    print(f"{'='*70}", flush=True)

    t_start = time.time()

    # ── Load model ────────────────────────────────────────────────
    print(f"\n  Loading {model_name} via MLX...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(model_name)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # ── Phase 1: Baseline bias audit ──────────────────────────────
    print(f"\n  Phase 1: Baseline audit", flush=True)
    baseline = run_bias_audit(model, tokenizer, "baseline")

    # ── Phase 2: SVD compression ──────────────────────────────────
    if plan.compress_ratio < 1.0:
        print(f"\n  Phase 2: SVD compression (ratio={plan.compress_ratio})", flush=True)
        t0 = time.time()
        n_compressed = mlx_svd_compress(model, ratio=plan.compress_ratio)
        freeze_info = mlx_freeze_layers(model, ratio=plan.freeze_fraction)
        mx.eval(model.parameters())
        print(f"  Compressed {n_compressed} matrices, "
              f"frozen {freeze_info['n_frozen']}/{freeze_info['n_layers']} layers "
              f"in {time.time()-t0:.1f}s", flush=True)
    else:
        print(f"\n  Phase 2: SVD compression skipped (ratio=1.0)", flush=True)

    # ── Phase 4: LoRA SFT with γ protection ───────────────────────
    if plan.rho_weight > 0:
        print(f"\n  Phase 4: LoRA SFT (ρ={plan.rho_weight}, γ={plan.gamma_weight})",
              flush=True)
        t0 = time.time()

        from rho_eval.alignment.dataset import (
            _build_trap_texts, _load_alpaca_texts, BehavioralContrastDataset,
        )
        from rho_eval.alignment.mlx_trainer import mlx_rho_guided_sft

        # ── Build SFT text data ────────────────────────────────
        rng = _random.Random(42)
        sft_texts = []
        trap_texts = _build_trap_texts(["sycophancy"], seed=42)
        rng.shuffle(trap_texts)
        sft_texts.extend(trap_texts[:400])  # 20% of 2000
        try:
            alpaca_texts = _load_alpaca_texts(1600, seed=42)
            sft_texts.extend(alpaca_texts)
        except Exception as e:
            print(f"    Alpaca load failed ({e}), using traps only", flush=True)
        rng.shuffle(sft_texts)
        sft_texts = sft_texts[:2000]
        print(f"  {len(sft_texts)} SFT texts loaded", flush=True)

        # ── Build contrast dataset (sycophancy — target behavior) ──
        contrast_dataset = BehavioralContrastDataset(
            behaviors=["sycophancy"], seed=42,
        )

        # ── Build protection dataset (bias — protected behavior) ──
        protection_dataset = None
        if plan.gamma_weight > 0 and plan.protection_categories:
            protection_dataset = BehavioralContrastDataset(
                behaviors=["bias"],
                categories=list(plan.protection_categories),
                seed=42,
            )
            print(f"  Protection dataset: {len(protection_dataset)} pairs "
                  f"from categories: {plan.protection_categories}", flush=True)

        # ── Run SFT ──────────────────────────────────────────────
        sft_result = mlx_rho_guided_sft(
            model, tokenizer,
            sft_texts,
            contrast_dataset=contrast_dataset,
            rho_weight=plan.rho_weight,
            gamma_weight=plan.gamma_weight,
            protection_dataset=protection_dataset,
            epochs=1,
            lr=2e-4,
            margin=0.1,
        )
        sft_elapsed = time.time() - t0
        print(f"  SFT complete in {sft_elapsed:.1f}s "
              f"({sft_result['steps']} steps)", flush=True)
    else:
        print(f"\n  Phase 4: SFT skipped (rho_weight=0)", flush=True)
        sft_result = None

    # ── Phase 5: Final bias audit ─────────────────────────────────
    print(f"\n  Phase 5: Final audit", flush=True)
    final = run_bias_audit(model, tokenizer, "final")

    # ── Results ───────────────────────────────────────────────────
    total_elapsed = time.time() - t_start

    print(f"\n  {'='*60}", flush=True)
    print(f"  RESULTS", flush=True)
    print(f"  Total time: {total_elapsed/60:.1f} min", flush=True)
    print(f"  Bias: {baseline['rho']:.4f} → {final['rho']:.4f} "
          f"(Δ={final['rho']-baseline['rho']:+.4f})", flush=True)

    # Per-category comparison
    if baseline["category_metrics"] and final["category_metrics"]:
        print(f"\n  {'Category':<30s} {'Before':>8s} {'After':>8s} {'Delta':>8s}",
              flush=True)
        print(f"  {'-'*58}", flush=True)
        for cat in sorted(final["category_metrics"].keys()):
            before = baseline["category_metrics"].get(cat, {}).get("accuracy", 0)
            after = final["category_metrics"][cat]["accuracy"]
            delta = after - before
            marker = "▼" if delta < -0.05 else ("▲" if delta > 0.05 else " ")
            print(f"  {cat:<30s} {before:>7.1%} {after:>7.1%} {delta:>+7.1%} {marker}",
                  flush=True)

    # Save results
    result_data = {
        "config": {
            "compress_ratio": plan.compress_ratio,
            "freeze_fraction": plan.freeze_fraction,
            "rho_weight": plan.rho_weight,
            "gamma_weight": plan.gamma_weight,
            "protection_categories": list(plan.protection_categories),
            "strategy": plan.strategy,
        },
        "model": model_name,
        "backend": "mlx",
        "total_elapsed_sec": total_elapsed,
        "baseline_bias": baseline,
        "final_bias": final,
        "sft_result": {
            k: v for k, v in (sft_result or {}).items()
            if k != "merged_model"
        },
    }
    result_path = run_dir / "surgery_result.json"
    result_path.write_text(json.dumps(result_data, indent=2, default=str))
    print(f"\n  Saved to {result_path}", flush=True)

    return result_data


# ── Stage 4: Verify ─────────────────────────────────────────────────────


def stage_verify_mlx(
    plan: "SurgicalPlan",
    result_data: dict,
    output_dir: Path,
    *,
    max_collateral: float = 0.05,
    min_target_improvement: float = 0.0,
) -> dict:
    """Stage 4: Verify surgical outcome against the plan.

    MLX-native verification that works with the dict-based result format
    (since we don't use HybridResult in the MLX pipeline).
    """
    print(f"\n{'='*70}", flush=True)
    print(f"  STAGE 4: VERIFY", flush=True)
    print(f"{'='*70}", flush=True)

    suggestions: list[str] = []

    # ── 1. Overall bias change ─────────────────────────────────────
    baseline_rho = result_data["baseline_bias"]["rho"]
    final_rho = result_data["final_bias"]["rho"]
    bias_delta = final_rho - baseline_rho

    overall_passed = bias_delta >= -max_collateral * 2  # Allow 2× per-cat for overall
    overall_check = {
        "passed": overall_passed,
        "baseline_rho": round(baseline_rho, 4),
        "final_rho": round(final_rho, 4),
        "delta": round(bias_delta, 4),
        "threshold": round(-max_collateral * 2, 4),
    }

    if not overall_passed:
        suggestions.append(
            f"Overall bias regression ({bias_delta:+.4f}) exceeds threshold. "
            f"Try increasing gamma_weight or using 'conservative' strategy."
        )

    # ── 2. Per-category checks ─────────────────────────────────────
    category_checks: dict[str, dict] = {}
    n_category_failures = 0

    baseline_cats = result_data["baseline_bias"].get("category_metrics", {})
    final_cats = result_data["final_bias"].get("category_metrics", {})

    for cat in plan.protection_categories:
        before = baseline_cats.get(cat, {}).get("accuracy", 0.0)
        after = final_cats.get(cat, {}).get("accuracy", 0.0)
        delta = after - before
        passed = delta >= -max_collateral

        category_checks[cat] = {
            "passed": passed,
            "before": round(before, 4),
            "after": round(after, 4),
            "delta": round(delta, 4),
            "threshold": round(-max_collateral, 4),
        }

        if not passed:
            n_category_failures += 1
            suggestions.append(
                f"Category '{cat}' dropped {delta:+.1%} "
                f"(threshold: {-max_collateral:+.1%}). "
                f"Try: increase gamma_weight or add more '{cat}' probes."
            )

    # ── 3. Unprotected category checks (informational) ─────────────
    unprotected_checks: dict[str, dict] = {}
    for cat in sorted(final_cats.keys()):
        if cat in plan.protection_categories:
            continue
        before = baseline_cats.get(cat, {}).get("accuracy", 0.0)
        after = final_cats[cat]["accuracy"]
        delta = after - before
        unprotected_checks[cat] = {
            "before": round(before, 4),
            "after": round(after, 4),
            "delta": round(delta, 4),
        }

    # ── 4. Suggestions ─────────────────────────────────────────────
    if n_category_failures > 0 and plan.gamma_weight < 0.3:
        suggestions.append(
            f"Suggestion: increase gamma_weight from {plan.gamma_weight} to "
            f"{min(plan.gamma_weight * 1.5, 0.3):.2f}"
        )

    if n_category_failures > 2:
        suggestions.append(
            "Multiple category failures — consider 'conservative' strategy "
            "or reducing rho_weight."
        )

    # ── 5. Verdict ─────────────────────────────────────────────────
    all_passed = overall_passed and n_category_failures == 0
    n_cats_checked = len(category_checks)
    n_cats_passed = sum(1 for c in category_checks.values() if c["passed"])

    summary = (
        f"{'PASS' if all_passed else 'FAIL'}: "
        f"bias Δ={bias_delta:+.4f}, "
        f"categories {n_cats_passed}/{n_cats_checked} passed, "
        f"overall {'✓' if overall_passed else '✗'}"
    )

    print(f"\n  {summary}", flush=True)

    # Protected category details
    for cat, cc in category_checks.items():
        marker = "✓" if cc["passed"] else "✗"
        print(f"  {cat}: {marker} {cc['before']:.1%} → {cc['after']:.1%} "
              f"(Δ={cc['delta']:+.1%}, max={cc['threshold']:+.1%})", flush=True)

    # Unprotected category details
    if unprotected_checks:
        print(f"\n  Unprotected categories (informational):", flush=True)
        for cat, uc in sorted(unprotected_checks.items(),
                              key=lambda x: x[1]["delta"]):
            marker = "▼" if uc["delta"] < -0.05 else ("▲" if uc["delta"] > 0.05 else " ")
            print(f"  {cat:<30s} {uc['before']:.1%} → {uc['after']:.1%} "
                  f"Δ={uc['delta']:+.1%} {marker}", flush=True)

    # Suggestions
    if suggestions:
        print(f"\n  Suggestions for next iteration:", flush=True)
        for s in suggestions:
            print(f"    → {s}", flush=True)

    # Save verification
    verification = {
        "passed": all_passed,
        "overall_check": overall_check,
        "category_checks": category_checks,
        "unprotected_checks": unprotected_checks,
        "suggestions": suggestions,
        "summary": summary,
    }
    verify_path = output_dir / "verification.json"
    verify_path.write_text(json.dumps(verification, indent=2, default=str))
    print(f"\n  Verification saved to {verify_path}", flush=True)

    return verification


# ── Diagnostic result finder ─────────────────────────────────────────────


def find_diagnostic_result(model_name: str) -> str | None:
    """Find the most recent MLX diagnostic result for this model."""
    model_short = model_name.split("/")[-1]
    base = Path("results/hybrid_sweep/diagnostics")
    if not base.exists():
        return None

    # Look for star config diagnostic (most useful for surgery planning)
    candidates = sorted(base.glob("*_diag_mlx/diagnostic_result.json"), reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text())
            if data.get("model", "").endswith(model_short):
                return str(path)
        except (json.JSONDecodeError, KeyError):
            continue

    return None


def find_cached_audit(model_name: str) -> str | None:
    """Find the most recent cached audit or hybrid result for this model."""
    model_short = model_name.split("/")[-1]
    candidates = [
        Path("results/hybrid_sweep") / model_short,
        Path("results"),
    ]

    for base_dir in candidates:
        if not base_dir.exists():
            continue
        for json_path in sorted(base_dir.rglob("hybrid_result.json"), reverse=True):
            try:
                data = json.loads(json_path.read_text())
                if data.get("model_name", "").endswith(model_short):
                    phases = data.get("phases", [])
                    if phases and phases[0].get("phase") == "baseline":
                        return str(json_path)
            except (json.JSONDecodeError, KeyError):
                continue

    return None


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Rho-Surgery — Surgical behavioral intervention (MLX backend)",
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
        "--diag-result", type=str, default=None,
        help="Path to diagnostic_result.json from diagnose_bias_7b_mlx.py",
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
        help="Path to surgery_result.json — skip training, just verify",
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

    print(f"\nRho-Surgery Pipeline (MLX)", flush=True)
    print(f"  Model:    {args.model}", flush=True)
    print(f"  Strategy: {args.strategy}", flush=True)
    print(f"  Output:   {output_dir}", flush=True)

    # ── Stage 1: Diagnose ──────────────────────────────────────────
    if args.plan:
        from rho_eval.surgical_planner import SurgicalPlan
        plan = SurgicalPlan.from_json(args.plan)
        print(f"\n  Loaded plan from {args.plan}", flush=True)
        print(f"\n{plan.summary()}", flush=True)
    else:
        # Try to find cached data sources
        audit_path = args.audit
        diag_result_path = args.diag_result

        # Auto-discover diagnostic results if not provided
        if audit_path is None and diag_result_path is None:
            diag_result_path = find_diagnostic_result(args.model)
            if diag_result_path:
                print(f"\n  Found diagnostic result: {diag_result_path}", flush=True)
            else:
                audit_path = find_cached_audit(args.model)
                if audit_path:
                    print(f"\n  Found cached audit: {audit_path}", flush=True)

        # Auto-detect known paths
        landscape = args.landscape or "docs/probe_landscape.json"
        if not Path(landscape).exists():
            landscape = None

        plan = stage_diagnose(
            args.model,
            audit_json_path=audit_path,
            diag_result_path=diag_result_path,
            layer_heatmap_path=args.heatmap,
            probe_landscape_path=landscape,
            strategy=args.strategy,
            output_dir=output_dir,
        )

    if args.plan_only:
        print("\n  --plan-only: stopping after Stage 1 (Diagnose)", flush=True)
        return

    # ── Stage 3: Surgical Train ────────────────────────────────────
    if args.verify_only:
        verify_path = Path(args.verify_only)
        if verify_path.is_dir():
            verify_path = verify_path / "surgery_run" / "surgery_result.json"
        if not verify_path.exists():
            print(f"  ERROR: No surgery_result.json found at {verify_path}", flush=True)
            return
        result_data = json.loads(verify_path.read_text())
        print(f"\n  Loaded result from {verify_path}", flush=True)
    else:
        result_data = stage_surgical_train_mlx(args.model, plan, output_dir)

    # ── Stage 4: Verify ────────────────────────────────────────────
    verification = stage_verify_mlx(
        plan, result_data, output_dir,
        max_collateral=args.max_collateral,
    )

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"  SURGERY COMPLETE: {'PASS ✓' if verification['passed'] else 'FAIL ✗'}",
          flush=True)
    print(f"{'='*70}", flush=True)

    if not verification["passed"] and verification["suggestions"]:
        print(f"\n  Next iteration: address {len(verification['suggestions'])} "
              f"suggestions", flush=True)
        plan_path = output_dir / "surgical_plan.json"
        print(f"  Re-run with --plan {plan_path} after adjusting parameters",
              flush=True)


if __name__ == "__main__":
    main()
