"""
rho-eval — Behavioral auditing toolkit for LLMs.

Audit any model across 5 behavioral dimensions (factual, toxicity, bias,
sycophancy, reasoning) using teacher-forced confidence probes. No internet
required — all 806 probes ship with the package.

Quick start:
    from rho_eval import audit

    report = audit("Qwen/Qwen2.5-7B-Instruct")
    print(report)   # mean ρ, PASS/WARN/FAIL per behavior

    # Or with a pre-loaded model:
    report = audit(model=model, tokenizer=tokenizer, behaviors=["factual", "bias"])

Compare two models:
    from rho_eval import compare

    baseline = audit("Qwen/Qwen2.5-7B-Instruct")
    compressed = audit("my-compressed-model")
    delta = compare(compressed, baseline)
    print(delta.to_table())

List available behaviors and probes:
    from rho_eval import list_behaviors
    from rho_eval.probes import list_probe_sets, get_probe_counts

Also includes SVD compression and calibration fine-tuning:
    from rho_eval.svd import compress_qko, freeze_layers
    from rho_eval.cartography import analyze_confidence
    from rho_eval.calibration import gentle_finetune
"""

# ── New v2 API ─────────────────────────────────────────────────────────────
from .audit import audit
from .behaviors import list_behaviors, get_behavior, get_all_behaviors
from .behaviors.base import ABCBehavior, BehaviorResult
from .output import AuditReport, compare

# ── Interpretability (v2.1) ───────────────────────────────────────────────
from .interpretability import (
    extract_subspaces,
    compute_overlap,
    head_attribution,
    InterpretabilityReport,
)

# ── Alignment (v2.2) ─────────────────────────────────────────────────────
from .alignment import (
    rho_guided_sft,
    contrastive_confidence_loss,
    rho_auxiliary_loss,
)

# ── Steering (v2.3) — opt-in, not auto-imported ────────────────────────────
# SAE-based behavioral steering: from rho_eval.steering import GatedSAE, ...

# ── Benchmarking (v2.5) — Fidelity-Bench 2.0 ─────────────────────────────
from .benchmarking import (
    generate_certificate,
    FidelityCertificate,
    TruthGap,
    FidelityScore,
    BenchmarkConfig,
    BENCHMARK_VERSION,
)

# ── Legacy v1 API (backward compatible) ────────────────────────────────────
from .core import compress_and_audit, audit_model
from .denoise import find_optimal_denoise_ratio
from .probes import (
    get_default_probes,
    get_mandela_probes,
    get_medical_probes,
    get_commonsense_probes,
    get_truthfulqa_probes,
    get_all_probes,
    load_probes,
)
from .behavioral import load_behavioral_probes, evaluate_behavior
from .calibration import load_calibration_data, gentle_finetune

__version__ = "2.2.0"

__all__ = [
    # ── New v2 API ──
    "audit",
    "list_behaviors",
    "get_behavior",
    "get_all_behaviors",
    "ABCBehavior",
    "BehaviorResult",
    "AuditReport",
    "compare",
    # ── Interpretability ──
    "extract_subspaces",
    "compute_overlap",
    "head_attribution",
    "InterpretabilityReport",
    # ── Alignment ──
    "rho_guided_sft",
    "contrastive_confidence_loss",
    "rho_auxiliary_loss",
    # ── Benchmarking ──
    "generate_certificate",
    "FidelityCertificate",
    "TruthGap",
    "FidelityScore",
    "BenchmarkConfig",
    "BENCHMARK_VERSION",
    # ── Legacy v1 API ──
    "compress_and_audit",
    "audit_model",
    "find_optimal_denoise_ratio",
    "get_default_probes",
    "get_mandela_probes",
    "get_medical_probes",
    "get_commonsense_probes",
    "get_truthfulqa_probes",
    "get_all_probes",
    "load_probes",
    "load_behavioral_probes",
    "evaluate_behavior",
    "load_calibration_data",
    "gentle_finetune",
]
