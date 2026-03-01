"""AuditReport — top-level container for a full behavioral audit.

Wraps multiple BehaviorResult objects with model metadata, timestamps,
and serialization helpers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..behaviors.base import BehaviorResult


@dataclass
class AuditReport:
    """Complete audit report for one model.

    Attributes:
        model: Model identifier (HuggingFace repo or path).
        behaviors: Mapping of behavior name → BehaviorResult.
        timestamp: ISO-8601 UTC timestamp of the audit.
        device: Device used for evaluation (e.g., "mps", "cuda:0", "cpu").
        elapsed_seconds: Total wall-clock time for the full audit.
        metadata: Extra info (e.g., model revision, quantization, etc.).
    """
    model: str
    behaviors: dict[str, BehaviorResult] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    device: str = "cpu"
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Summary properties ────────────────────────────────────────────

    @property
    def mean_rho(self) -> float:
        """Mean rho across all behaviors."""
        if not self.behaviors:
            return 0.0
        return sum(r.rho for r in self.behaviors.values()) / len(self.behaviors)

    @property
    def overall_status(self) -> str:
        """PASS if all behaviors PASS, FAIL if any FAIL, else WARN."""
        if not self.behaviors:
            return "FAIL"
        statuses = [r.status for r in self.behaviors.values()]
        if all(s == "PASS" for s in statuses):
            return "PASS"
        if any(s == "FAIL" for s in statuses):
            return "FAIL"
        return "WARN"

    @property
    def total_probes(self) -> int:
        """Total number of probes across all behaviors."""
        return sum(r.total for r in self.behaviors.values())

    # ── Add results ───────────────────────────────────────────────────

    def add_result(self, result: BehaviorResult) -> None:
        """Add a BehaviorResult to the report."""
        self.behaviors[result.behavior] = result

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self, include_details: bool = True) -> dict:
        """Convert to a JSON-serializable dict.

        Args:
            include_details: If False, omit per-probe details to reduce size.
        """
        d = {
            "model": self.model,
            "timestamp": self.timestamp,
            "device": self.device,
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": self.metadata,
            "summary": {
                "mean_rho": self.mean_rho,
                "overall_status": self.overall_status,
                "total_probes": self.total_probes,
                "n_behaviors": len(self.behaviors),
            },
            "behaviors": {},
        }
        for name, result in sorted(self.behaviors.items()):
            rd = result.to_dict()
            if not include_details:
                rd.pop("details", None)
            d["behaviors"][name] = rd
        return d

    def to_json(self, indent: int = 2, include_details: bool = True) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(include_details=include_details), indent=indent)

    def save(self, path: str | Path, include_details: bool = True) -> Path:
        """Save report to a JSON file.

        Args:
            path: Output file path.
            include_details: If False, omit per-probe details.

        Returns:
            The resolved output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(include_details=include_details), f, indent=2)
        return path.resolve()

    @classmethod
    def load(cls, path: str | Path) -> "AuditReport":
        """Load a report from a JSON file.

        Args:
            path: Path to JSON file saved by .save().

        Returns:
            AuditReport instance.
        """
        with open(path) as f:
            d = json.load(f)

        report = cls(
            model=d["model"],
            timestamp=d.get("timestamp", ""),
            device=d.get("device", "unknown"),
            elapsed_seconds=d.get("elapsed_seconds", 0.0),
            metadata=d.get("metadata", {}),
        )

        for name, bd in d.get("behaviors", {}).items():
            report.behaviors[name] = BehaviorResult(
                behavior=bd.get("behavior", name),
                rho=bd["rho"],
                retention=bd["retention"],
                positive_count=bd["positive_count"],
                total=bd["total"],
                elapsed=bd.get("elapsed", 0.0),
                metadata=bd.get("metadata", {}),
                details=bd.get("details", []),
            )

        return report

    # ── Category disaggregation ──────────────────────────────────────

    def category_report(self, behavior_name: str = "bias") -> dict[str, dict]:
        """Extract per-category metrics for a behavior.

        Args:
            behavior_name: Name of the behavior to disaggregate.

        Returns:
            Dict mapping category name → {accuracy, n, biased_rate}.
            Empty dict if behavior not found or no category_metrics.
        """
        if behavior_name not in self.behaviors:
            return {}
        result = self.behaviors[behavior_name]
        return result.metadata.get("category_metrics", {})

    def source_report(self, behavior_name: str = "bias") -> dict[str, dict]:
        """Extract per-source metrics for a behavior.

        Args:
            behavior_name: Name of the behavior to disaggregate.

        Returns:
            Dict mapping source name → {accuracy, n}.
        """
        if behavior_name not in self.behaviors:
            return {}
        result = self.behaviors[behavior_name]
        return result.metadata.get("source_metrics", {})

    def categories_summary_table(self, behavior_name: str = "bias") -> str:
        """ASCII table of per-category metrics.

        Args:
            behavior_name: Behavior to disaggregate.

        Returns:
            Formatted table string, or "No category metrics available."
        """
        metrics = self.category_report(behavior_name)
        if not metrics:
            return "No category metrics available."

        lines = [
            f"  {'Category':<25s}  {'Acc':>6s}  {'N':>4s}  {'Biased%':>7s}",
            "  " + "─" * 46,
        ]
        for cat, data in sorted(metrics.items(), key=lambda x: -x[1]["accuracy"]):
            lines.append(
                f"  {cat:<25s}  {data['accuracy']:>5.1%}  "
                f"{data['n']:>4d}  {data['biased_rate']:>6.1%}"
            )
        return "\n".join(lines)

    # ── DataFrame export (optional pandas) ────────────────────────────

    def to_dataframe(self):
        """Convert summary to a pandas DataFrame. Requires pandas.

        Returns:
            DataFrame with one row per behavior.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        rows = []
        for name, result in sorted(self.behaviors.items()):
            rows.append({
                "behavior": name,
                "rho": result.rho,
                "retention": result.retention,
                "positive": result.positive_count,
                "total": result.total,
                "status": result.status,
                "elapsed_s": result.elapsed,
            })
        return pd.DataFrame(rows)

    # ── Repr ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n = len(self.behaviors)
        return (
            f"<AuditReport model={self.model!r} "
            f"behaviors={n} mean_ρ={self.mean_rho:.4f} "
            f"status={self.overall_status}>"
        )
