"""Standardized output formats for behavioral audits.

Provides AuditReport (multi-behavior container) and export utilities
for JSON, Markdown, CSV, and table rendering.

Usage:
    from rho_eval.output import AuditReport
    from rho_eval.output.exporters import to_json, to_markdown, to_csv
    from rho_eval.output.comparator import compare
"""

from .schema import AuditReport
from .comparator import compare

__all__ = ["AuditReport", "compare"]
