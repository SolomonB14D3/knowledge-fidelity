"""Compare two AuditReports and produce a delta table.

Usage:
    from knowledge_fidelity.output import compare

    delta = compare(report_after, report_before)
    print(delta.to_table())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BehaviorDelta:
    """Change in a single behavior between two reports."""
    behavior: str
    rho_before: float
    rho_after: float
    delta_rho: float
    status_before: str
    status_after: str
    label: str  # "IMPROVED", "DEGRADED", "UNCHANGED"


@dataclass
class ComparisonResult:
    """Comparison of two AuditReports."""
    model_before: str
    model_after: str
    deltas: list[BehaviorDelta] = field(default_factory=list)
    mean_delta: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model_before": self.model_before,
            "model_after": self.model_after,
            "mean_delta": self.mean_delta,
            "deltas": [
                {
                    "behavior": d.behavior,
                    "rho_before": d.rho_before,
                    "rho_after": d.rho_after,
                    "delta_rho": d.delta_rho,
                    "status_before": d.status_before,
                    "status_after": d.status_after,
                    "label": d.label,
                }
                for d in self.deltas
            ],
        }

    def to_table(self, color: bool = True) -> str:
        """Format comparison as a plain-text table."""
        if color:
            GREEN = "\033[92m"
            RED = "\033[91m"
            BOLD = "\033[1m"
            RESET = "\033[0m"
        else:
            GREEN = RED = BOLD = RESET = ""

        lines = [
            f"{BOLD}Comparison: {self.model_before} → {self.model_after}{RESET}",
            f"  Mean Δρ: {self.mean_delta:+.4f}",
            "",
            f"  {'Behavior':<12s}  {'Before':>8s}  {'After':>8s}  {'Δρ':>8s}  {'Label':>10s}",
            "  " + "─" * 54,
        ]

        for d in self.deltas:
            if d.delta_rho > 0.001:
                c = GREEN
                arrow = "↑"
            elif d.delta_rho < -0.001:
                c = RED
                arrow = "↓"
            else:
                c = ""
                arrow = "="

            lines.append(
                f"  {d.behavior:<12s}  {d.rho_before:>+8.4f}  "
                f"{d.rho_after:>+8.4f}  "
                f"{c}{d.delta_rho:>+8.4f}{RESET}  "
                f"{arrow} {d.label}"
            )

        return "\n".join(lines) + "\n"

    def to_markdown(self) -> str:
        """Format comparison as Markdown table."""
        lines = [
            f"## Comparison: {self.model_before} → {self.model_after}",
            "",
            f"**Mean Δρ:** {self.mean_delta:+.4f}",
            "",
            "| Behavior | Before | After | Δρ | Label |",
            "|----------|-------:|------:|---:|:------|",
        ]

        for d in self.deltas:
            lines.append(
                f"| {d.behavior} | {d.rho_before:.4f} | "
                f"{d.rho_after:.4f} | {d.delta_rho:+.4f} | "
                f"{d.label} |"
            )

        return "\n".join(lines) + "\n"


def compare(report_after, report_before, threshold: float = 0.01) -> ComparisonResult:
    """Compare two AuditReports and produce a delta table.

    Args:
        report_after: The newer report (e.g., after compression/fine-tuning).
        report_before: The baseline report.
        threshold: Minimum |Δρ| to count as IMPROVED/DEGRADED.

    Returns:
        ComparisonResult with per-behavior deltas.
    """
    all_behaviors = sorted(
        set(report_before.behaviors.keys()) | set(report_after.behaviors.keys())
    )

    deltas = []
    for name in all_behaviors:
        rho_before = report_before.behaviors[name].rho if name in report_before.behaviors else 0.0
        rho_after = report_after.behaviors[name].rho if name in report_after.behaviors else 0.0
        status_before = report_before.behaviors[name].status if name in report_before.behaviors else "N/A"
        status_after = report_after.behaviors[name].status if name in report_after.behaviors else "N/A"

        delta_rho = rho_after - rho_before

        if delta_rho > threshold:
            label = "IMPROVED"
        elif delta_rho < -threshold:
            label = "DEGRADED"
        else:
            label = "UNCHANGED"

        deltas.append(BehaviorDelta(
            behavior=name,
            rho_before=rho_before,
            rho_after=rho_after,
            delta_rho=delta_rho,
            status_before=status_before,
            status_after=status_after,
            label=label,
        ))

    mean_delta = sum(d.delta_rho for d in deltas) / len(deltas) if deltas else 0.0

    return ComparisonResult(
        model_before=report_before.model,
        model_after=report_after.model,
        deltas=deltas,
        mean_delta=mean_delta,
    )
