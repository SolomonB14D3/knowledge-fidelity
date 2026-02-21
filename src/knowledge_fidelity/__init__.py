"""
Knowledge Fidelity — Compress LLMs while auditing what they still know.

The first toolkit that lets you compress an LLM while auditing whether it
still knows truth vs popular myths, using the same factual probes for both
structural importance scoring and behavioral false-belief detection.

Quick start:
    from knowledge_fidelity import compress_and_audit

    report = compress_and_audit("Qwen/Qwen2.5-7B-Instruct", ratio=0.7)
    print(f"Retention: {report['retention']:.0%} | "
          f"rho: {report['rho_before']:.3f} -> {report['rho_after']:.3f}")

Or use the components independently:

    # SVD compression only
    from knowledge_fidelity.svd import compress_qko, freeze_layers

    # Confidence analysis only
    from knowledge_fidelity.cartography import analyze_confidence

    # Shared probes
    from knowledge_fidelity.probes import get_default_probes, get_mandela_probes

Built on:
    - Intelligent SVD (github.com/SolomonB14D3/intelligent-svd)
    - Confidence Cartography (github.com/SolomonB14D3/confidence-cartography)
"""

from .core import compress_and_audit, audit_model
from .probes import (
    get_default_probes,
    get_mandela_probes,
    get_medical_probes,
    get_all_probes,
    load_probes,
)

__version__ = "0.1.1"

__all__ = [
    # Main API
    "compress_and_audit",
    "audit_model",
    # Probes
    "get_default_probes",
    "get_mandela_probes",
    "get_medical_probes",
    "get_all_probes",
    "load_probes",
]
