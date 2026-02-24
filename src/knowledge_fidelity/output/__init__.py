"""Standardized output formats for behavioral audits.

Provides AuditReport (multi-behavior container) and export utilities
for JSON, Markdown, CSV, and table rendering.

Usage:
    from knowledge_fidelity.output import AuditReport
    from knowledge_fidelity.output.exporters import to_json, to_markdown, to_csv
    from knowledge_fidelity.output.comparator import compare
"""

from .schema import AuditReport
from .comparator import compare

__all__ = ["AuditReport", "compare"]
