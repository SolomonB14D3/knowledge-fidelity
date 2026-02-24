"""Fidelity-Bench 2.0 — The Open Behavioral Benchmark Suite.

A standardized adversarial benchmark that measures how much truth a model
sacrifices under social pressure. The signature metric is the Truth-Gap:

    ΔF = ρ_baseline − ρ_pressured

Usage:
    from rho_eval.benchmarking import generate_certificate

    cert = generate_certificate("Qwen/Qwen2.5-7B-Instruct")
    print(cert.to_markdown())   # Model Fidelity Certificate
    print(cert.grade)           # "B"
    print(cert.fidelity_score)  # Composite + components

    # Just the adversarial wrapping:
    from rho_eval.benchmarking.adversarial import wrap_probe, build_pressure_suite
"""

from .schema import (
    BenchmarkConfig,
    BENCHMARK_VERSION,
    PressureResult,
    PressureCurve,
    TruthGap,
    FidelityScore,
    FidelityCertificate,
)
from .reports import generate_certificate
from .adversarial import (
    wrap_probe,
    build_pressure_suite,
    PRESSURE_TEMPLATES,
    MAX_PRESSURE_LEVEL,
)
from .scorers import (
    compute_truth_gap,
    compute_fidelity_score,
    build_pressure_curves,
    bootstrap_fidelity_score,
)
from .loader import (
    load_bench_probes,
    load_all_bench_probes,
    compute_probe_hash,
    get_bench_metadata,
    validate_version,
)

__all__ = [
    # Schema
    "BenchmarkConfig",
    "BENCHMARK_VERSION",
    "PressureResult",
    "PressureCurve",
    "TruthGap",
    "FidelityScore",
    "FidelityCertificate",
    # Reports
    "generate_certificate",
    # Adversarial
    "wrap_probe",
    "build_pressure_suite",
    "PRESSURE_TEMPLATES",
    "MAX_PRESSURE_LEVEL",
    # Scorers
    "compute_truth_gap",
    "compute_fidelity_score",
    "build_pressure_curves",
    "bootstrap_fidelity_score",
    # Loader
    "load_bench_probes",
    "load_all_bench_probes",
    "compute_probe_hash",
    "get_bench_metadata",
    "validate_version",
]
