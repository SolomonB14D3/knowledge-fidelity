"""Export utilities for AuditReport.

Formats: JSON, Markdown, CSV, plain-text table.
"""

from __future__ import annotations

import csv
import io
from typing import Optional

from .schema import AuditReport


def to_json(report: AuditReport, indent: int = 2, include_details: bool = True) -> str:
    """Export report as JSON string."""
    return report.to_json(indent=indent, include_details=include_details)


def to_markdown(report: AuditReport) -> str:
    """Export report as a Markdown table.

    Returns:
        Markdown string with summary header and behavior table.
    """
    lines = [
        f"# Behavioral Audit: {report.model}",
        "",
        f"**Status:** {report.overall_status}  ",
        f"**Mean ρ:** {report.mean_rho:.4f}  ",
        f"**Probes:** {report.total_probes}  ",
        f"**Device:** {report.device}  ",
        f"**Time:** {report.elapsed_seconds:.1f}s  ",
        f"**Timestamp:** {report.timestamp}  ",
        "",
        "| Behavior | ρ | Retention | Score | Status | Time |",
        "|----------|---:|----------:|------:|:------:|-----:|",
    ]

    for name, result in sorted(report.behaviors.items()):
        lines.append(
            f"| {name} | {result.rho:.4f} | {result.retention:.1%} | "
            f"{result.positive_count}/{result.total} | "
            f"{result.status} | {result.elapsed:.1f}s |"
        )

    # ── Category breakdowns (for behaviors that provide them) ────────
    for name, result in sorted(report.behaviors.items()):
        cat_metrics = result.metadata.get("category_metrics")
        if cat_metrics:
            lines.append("")
            lines.append(f"### {name.title()} by Category")
            lines.append("")
            lines.append("| Category | Accuracy | N | Biased % |")
            lines.append("|----------|----------|---|----------|")
            for cat, data in sorted(cat_metrics.items(), key=lambda x: -x[1]["accuracy"]):
                lines.append(
                    f"| {cat} | {data['accuracy']:.1%} | {data['n']} | "
                    f"{data.get('biased_rate', 0):.1%} |"
                )

        src_metrics = result.metadata.get("source_metrics")
        if src_metrics:
            lines.append("")
            lines.append(f"### {name.title()} by Source")
            lines.append("")
            lines.append("| Source | Accuracy | N |")
            lines.append("|--------|----------|---|")
            for src, data in sorted(src_metrics.items(), key=lambda x: -x[1]["accuracy"]):
                lines.append(
                    f"| {src} | {data['accuracy']:.1%} | {data['n']} |"
                )

    return "\n".join(lines) + "\n"


def to_csv(report: AuditReport) -> str:
    """Export report as CSV string.

    Returns:
        CSV with header row and one row per behavior.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "model", "behavior", "rho", "retention",
        "positive_count", "total", "status", "elapsed_s",
    ])

    for name, result in sorted(report.behaviors.items()):
        writer.writerow([
            report.model,
            name,
            f"{result.rho:.4f}",
            f"{result.retention:.4f}",
            result.positive_count,
            result.total,
            result.status,
            f"{result.elapsed:.1f}",
        ])

    return output.getvalue()


def to_table(report: AuditReport, color: bool = True) -> str:
    """Export report as a plain-text table with optional ANSI color.

    Args:
        report: AuditReport to format.
        color: If True, use ANSI escape codes for PASS/WARN/FAIL.

    Returns:
        Formatted table string.
    """
    # ANSI color codes
    if color:
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
    else:
        GREEN = YELLOW = RED = BOLD = RESET = ""

    STATUS_COLORS = {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED}

    lines = []

    # Header
    overall_color = STATUS_COLORS.get(report.overall_status, "")
    lines.append(f"{BOLD}Behavioral Audit: {report.model}{RESET}")
    lines.append(
        f"  Status: {overall_color}{report.overall_status}{RESET}  "
        f"Mean ρ: {report.mean_rho:.4f}  "
        f"Probes: {report.total_probes}  "
        f"Time: {report.elapsed_seconds:.1f}s"
    )
    lines.append("")

    # Table header
    header = f"  {'Behavior':<12s}  {'ρ':>8s}  {'Retention':>10s}  {'Score':>8s}  {'Status':>6s}  {'Time':>6s}"
    lines.append(header)
    lines.append("  " + "─" * (len(header) - 2))

    # Rows
    for name, result in sorted(report.behaviors.items()):
        sc = STATUS_COLORS.get(result.status, "")
        lines.append(
            f"  {name:<12s}  {result.rho:>+8.4f}  "
            f"{result.retention:>9.1%}  "
            f"{result.positive_count:>3d}/{result.total:<4d}  "
            f"{sc}{result.status:>6s}{RESET}  "
            f"{result.elapsed:>5.1f}s"
        )

    return "\n".join(lines) + "\n"
