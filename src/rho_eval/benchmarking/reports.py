"""Certificate generation — the Fidelity-Bench orchestrator.

Runs the complete Fidelity-Bench 2.0 pipeline:
1. Standard audit() for baseline ρ
2. Build pressure suites for all domains
3. Score all pressure probes
4. Build pressure curves
5. Compute truth gaps per domain + overall
6. Compute fidelity score with bootstrap CIs
7. Assign grade and generate FidelityCertificate
"""

from __future__ import annotations

import time
from typing import Optional

from .schema import (
    BenchmarkConfig,
    FidelityCertificate,
    FidelityScore,
    TruthGap,
    _grade_from_composite,
    BENCHMARK_VERSION,
)
from .adversarial import build_pressure_suite
from .scorers import (
    score_pressure_suite,
    build_pressure_curves,
    compute_truth_gap,
    compute_fidelity_score,
    bootstrap_fidelity_score,
)
from .loader import (
    load_all_bench_probes,
    compute_probe_hash,
)


def generate_certificate(
    model_name_or_path: Optional[str] = None,
    *,
    model=None,
    tokenizer=None,
    config: Optional[BenchmarkConfig] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> FidelityCertificate:
    """Run the complete Fidelity-Bench 2.0 and generate a certificate.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        model: Pre-loaded model (optional).
        tokenizer: Pre-loaded tokenizer (optional).
        config: Benchmark configuration.
        device: Device string. None → auto-detect.
        verbose: Print progress.

    Returns:
        FidelityCertificate with complete benchmark results.
    """
    from ..audit import audit
    from ..utils import get_device

    config = config or BenchmarkConfig()
    t0 = time.time()

    # Resolve device
    if device is None:
        device = config.device or str(get_device())

    model_id = model_name_or_path or "unknown"

    # ── Step 1: Baseline audit ─────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}", flush=True)
        print(f"Fidelity-Bench 2.0 — {model_id}", flush=True)
        print(f"{'='*60}", flush=True)
        print("\n[1/6] Running baseline audit...", flush=True)

    baseline_report = audit(
        model_name_or_path,
        model=model,
        tokenizer=tokenizer,
        behaviors="all",
        seed=config.seed,
        device=device,
        trust_remote_code=config.trust_remote_code,
    )

    # Capture the loaded model/tokenizer for reuse in pressure testing
    # (audit() loads the model if not provided)
    if model is None or tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        if verbose:
            print("  Loading model for pressure testing...", flush=True)
        dev = torch.device(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=config.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.float16 if dev.type != "cpu" else torch.float32,
        ).to(dev).eval()

    behavior_baselines = {
        name: result.rho
        for name, result in baseline_report.behaviors.items()
    }

    if verbose:
        for name, rho in sorted(behavior_baselines.items()):
            print(f"  {name:<12s}  ρ={rho:+.4f}", flush=True)

    # ── Step 2: Load bench probes ──────────────────────────────────────
    if verbose:
        print(f"\n[2/6] Loading bench probes...", flush=True)

    probes = load_all_bench_probes(
        domains=list(config.domains),
        n_per_domain=config.n_probes_per_domain,
        seed=config.seed,
    )
    probe_hash = compute_probe_hash(probes)

    if verbose:
        domain_counts = {}
        for p in probes:
            d = p.get("domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1
        for d, c in sorted(domain_counts.items()):
            print(f"  {d}: {c} probes", flush=True)

    # ── Step 3: Build and score pressure suites ────────────────────────
    if verbose:
        print(f"\n[3/6] Running pressure tests ({config.pressure_levels} levels)...", flush=True)

    suite = build_pressure_suite(
        probes,
        n_levels=config.pressure_levels,
        seed=config.seed,
    )

    if verbose:
        print(f"  Total probes to score: {len(suite)}", flush=True)

    results = score_pressure_suite(model, tokenizer, suite, device, verbose=verbose)

    # ── Step 4: Build pressure curves ──────────────────────────────────
    if verbose:
        print(f"\n[4/6] Computing pressure curves...", flush=True)

    curves = build_pressure_curves(results)

    # ── Step 5: Compute truth gaps ─────────────────────────────────────
    if verbose:
        print(f"\n[5/6] Computing truth gaps...", flush=True)

    truth_gaps = {}
    for domain in config.domains:
        tg = compute_truth_gap(curves, domain=domain)
        truth_gaps[domain] = tg
        if verbose:
            print(
                f"  {domain:<10s}  ΔF={tg.delta_f:+.4f}  "
                f"(baseline={tg.rho_baseline:.3f} → pressured={tg.rho_pressured:.3f})  "
                f"unbreakable={tg.pct_unbreakable:.0%}",
                flush=True,
            )

    # Overall truth gap
    truth_gaps["overall"] = compute_truth_gap(curves)

    # ── Step 6: Compute fidelity score ─────────────────────────────────
    if verbose:
        print(f"\n[6/6] Computing fidelity score...", flush=True)

    bias_rho = behavior_baselines.get("bias", 0.0)
    syc_rho = behavior_baselines.get("sycophancy", 0.0)

    fidelity_score = compute_fidelity_score(truth_gaps, bias_rho, syc_rho)

    # Bootstrap CIs
    ci_lower, ci_upper = bootstrap_fidelity_score(
        curves, bias_rho, syc_rho,
        n_bootstrap=config.n_bootstrap,
        ci_level=config.ci_level,
        seed=config.seed,
    )
    fidelity_score.ci_lower = ci_lower
    fidelity_score.ci_upper = ci_upper

    grade = _grade_from_composite(fidelity_score.composite)

    elapsed = time.time() - t0

    if verbose:
        print(f"\n  Grade: {grade}  Composite: {fidelity_score.composite:.3f} "
              f"[{ci_lower:.3f}, {ci_upper:.3f}]", flush=True)
        print(f"  Total time: {elapsed:.1f}s\n", flush=True)

    # ── Assemble certificate ───────────────────────────────────────────

    cert = FidelityCertificate(
        model=model_id,
        benchmark_version=BENCHMARK_VERSION,
        fidelity_score=fidelity_score,
        truth_gaps=truth_gaps,
        pressure_curves=curves,
        behavior_baselines=behavior_baselines,
        elapsed_seconds=elapsed,
        grade=grade,
        probe_hash=probe_hash,
        device=device,
    )

    return cert
