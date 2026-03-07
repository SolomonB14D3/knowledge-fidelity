"""Four-quadrant behavioral diagnosis for rho-unlock.

Combines two axes:
  Axis 1 — Knowledge metric: does the model discriminate correctly?
           - Behavioral ρ (Spearman correlation from rho-eval): threshold 0.3
           - Logit accuracy (argmax over answer tokens for benchmarks):
             threshold = chance + margin (e.g., 40% for 4-choice at 15% margin)
  Axis 2 — Expression gap: can the model express what it knows?

Four quadrants:
  HEALTHY:      knows AND gap < threshold → no action needed
  UNLOCK:       knows AND gap ≥ threshold → contrastive decoding rescues
  RETRAIN:      !knows AND gap < threshold → expressive but wrong, needs training
  BOTH_NEEDED:  !knows AND gap ≥ threshold → mute and wrong, needs training + unlock
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


class Quadrant(str, Enum):
    """Four-quadrant classification for behavioral diagnosis."""
    HEALTHY = "HEALTHY"
    UNLOCK = "UNLOCK"
    RETRAIN = "RETRAIN"
    BOTH_NEEDED = "BOTH_NEEDED"
    UNKNOWN = "UNKNOWN"  # For behaviors where gap is not measurable

    @property
    def action(self) -> str:
        """Human-readable remediation action."""
        return {
            Quadrant.HEALTHY: "No action needed",
            Quadrant.UNLOCK: "Apply contrastive decoding (rho-unlock unlock)",
            Quadrant.RETRAIN: "Fine-tune with behavioral data (rho-surgery)",
            Quadrant.BOTH_NEEDED: "Fine-tune first, then apply contrastive decoding",
            Quadrant.UNKNOWN: "Expression gap N/A — use ρ score for guidance",
        }[self]

    @property
    def symbol(self) -> str:
        """Unicode symbol for CLI display."""
        return {
            Quadrant.HEALTHY: "✓",
            Quadrant.UNLOCK: "🔓",
            Quadrant.RETRAIN: "🔧",
            Quadrant.BOTH_NEEDED: "🔧🔓",
            Quadrant.UNKNOWN: "?",
        }[self]


class MetricType(str, Enum):
    """Type of Axis-1 'knows' metric."""
    BEHAVIORAL_RHO = "behavioral_rho"      # Spearman ρ from rho-eval
    LOGIT_ACCURACY = "logit_accuracy"      # Logit argmax accuracy (benchmarks)


def compute_knows_threshold(
    metric_type: MetricType,
    n_choices: int = 4,
    rho_threshold: float = 0.3,
    above_chance_margin: float = 0.15,
) -> float:
    """Compute the 'knows' threshold based on metric type.

    For behavioral ρ: returns rho_threshold directly (default 0.3).
    For logit accuracy: returns chance (1/n_choices) + margin.

    Examples:
        behavioral ρ:  threshold = 0.3
        4-choice MCQ:  threshold = 0.25 + 0.15 = 0.40
        3-choice MCQ:  threshold = 0.333 + 0.15 = 0.483
        2-choice MCQ:  threshold = 0.50 + 0.15 = 0.65
    """
    if metric_type == MetricType.BEHAVIORAL_RHO:
        return rho_threshold
    chance = 1.0 / n_choices
    return chance + above_chance_margin


@dataclass
class BehaviorDiagnosis:
    """Diagnosis result for a single behavior or benchmark."""
    behavior: str
    axis1_score: Optional[float]          # ρ or logit accuracy
    gap: Optional[float]
    parse_rate: Optional[float]
    quadrant: Quadrant
    action: str
    metric_type: MetricType = MetricType.BEHAVIORAL_RHO
    knows_threshold: Optional[float] = None
    logit_accuracy: Optional[float] = None
    gen_accuracy: Optional[float] = None

    # Backward-compatible alias
    @property
    def rho(self) -> Optional[float]:
        return self.axis1_score

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quadrant"] = self.quadrant.value
        d["metric_type"] = self.metric_type.value
        # Include rho alias for backward compat
        d["rho"] = self.axis1_score
        return d


def classify_quadrant(
    axis1_score: float,
    gap: Optional[float],
    metric_type: MetricType = MetricType.BEHAVIORAL_RHO,
    n_choices: int = 4,
    rho_threshold: float = 0.3,
    above_chance_margin: float = 0.15,
    gap_threshold: float = 0.05,
) -> tuple[Quadrant, float]:
    """Classify a single behavior into one of four quadrants.

    Args:
        axis1_score: Axis-1 metric (ρ for behaviors, logit_accuracy for benchmarks).
        gap: Expression gap (logit_acc - gen_acc). None if not measurable.
        metric_type: Whether axis1_score is behavioral ρ or logit accuracy.
        n_choices: Number of MC choices (only used for LOGIT_ACCURACY).
        rho_threshold: ρ above this = "knows" (behavioral only, default: 0.3).
        above_chance_margin: Margin above chance for benchmarks (default: 0.15).
        gap_threshold: Gap above this = "can't express" (default: 5%).

    Returns:
        (quadrant, threshold_used) tuple.
    """
    threshold = compute_knows_threshold(
        metric_type, n_choices, rho_threshold, above_chance_margin,
    )

    if gap is None:
        return Quadrant.UNKNOWN, threshold

    knows = axis1_score >= threshold
    can_express = gap < gap_threshold

    if knows and can_express:
        return Quadrant.HEALTHY, threshold
    elif knows and not can_express:
        return Quadrant.UNLOCK, threshold
    elif not knows and can_express:
        return Quadrant.RETRAIN, threshold
    else:
        return Quadrant.BOTH_NEEDED, threshold


def diagnose(
    rho_scores: dict[str, float],
    expression_gaps: dict[str, Optional[float]],
    parse_rates: Optional[dict[str, Optional[float]]] = None,
    logit_accuracies: Optional[dict[str, Optional[float]]] = None,
    gen_accuracies: Optional[dict[str, Optional[float]]] = None,
    benchmark_scores: Optional[dict[str, float]] = None,
    benchmark_n_choices: Optional[dict[str, int]] = None,
    rho_threshold: float = 0.3,
    above_chance_margin: float = 0.15,
    gap_threshold: float = 0.05,
) -> dict[str, BehaviorDiagnosis]:
    """Classify each behavior into one of four quadrants.

    Handles both behavioral ρ scores (from rho-eval) and benchmark
    logit accuracies with type-appropriate "knows" thresholds.

    Args:
        rho_scores: Dict mapping behavior name → ρ score.
        expression_gaps: Dict mapping behavior name → gap (or None).
        parse_rates: Optional dict mapping behavior → parse rate.
        logit_accuracies: Optional dict mapping behavior → logit accuracy.
        gen_accuracies: Optional dict mapping behavior → generation accuracy.
        benchmark_scores: Dict mapping benchmark name → logit accuracy.
            These use chance + margin as the "knows" threshold.
        benchmark_n_choices: Dict mapping benchmark name → number of choices.
        rho_threshold: ρ above this = "knows" (behavioral only).
        above_chance_margin: Margin above chance for benchmarks.
        gap_threshold: Gap above this = "can't express".

    Returns:
        Dict mapping behavior name → BehaviorDiagnosis.
    """
    parse_rates = parse_rates or {}
    logit_accuracies = logit_accuracies or {}
    gen_accuracies = gen_accuracies or {}
    benchmark_scores = benchmark_scores or {}
    benchmark_n_choices = benchmark_n_choices or {}

    all_names = sorted(
        set(rho_scores.keys()) | set(expression_gaps.keys()) | set(benchmark_scores.keys())
    )

    results = {}
    for behavior in all_names:
        gap = expression_gaps.get(behavior)

        # Determine metric type and axis-1 score
        if behavior in benchmark_scores:
            metric_type = MetricType.LOGIT_ACCURACY
            axis1 = benchmark_scores[behavior]
            n_choices = benchmark_n_choices.get(behavior, 4)
        else:
            metric_type = MetricType.BEHAVIORAL_RHO
            axis1 = rho_scores.get(behavior, 0.0)
            n_choices = 4  # unused for behavioral ρ

        quadrant, threshold = classify_quadrant(
            axis1, gap, metric_type, n_choices,
            rho_threshold, above_chance_margin, gap_threshold,
        )

        results[behavior] = BehaviorDiagnosis(
            behavior=behavior,
            axis1_score=axis1,
            gap=gap,
            parse_rate=parse_rates.get(behavior),
            quadrant=quadrant,
            action=quadrant.action,
            metric_type=metric_type,
            knows_threshold=threshold,
            logit_accuracy=logit_accuracies.get(behavior),
            gen_accuracy=gen_accuracies.get(behavior),
        )

    return results


def format_diagnosis_table(diagnoses: dict[str, BehaviorDiagnosis]) -> str:
    """Format diagnoses as a human-readable ASCII table.

    Shows metric-type-aware formatting: ρ values for behaviors,
    percentages for benchmark logit accuracies.

    Returns:
        Multi-line string table suitable for terminal output.
    """
    lines = []
    header = f"  {'Behavior':<14s} {'Knows?':>8s} {'Gap':>8s} {'Parse%':>8s} {'Quadrant':<14s}"
    sep = "  " + "─" * (len(header) - 2)

    lines.append(header)
    lines.append(sep)

    has_benchmarks = False

    for name, diag in sorted(diagnoses.items()):
        # Format axis-1 score based on metric type
        if diag.axis1_score is not None:
            if diag.metric_type == MetricType.LOGIT_ACCURACY:
                score_str = f"{diag.axis1_score:.1%}"
                has_benchmarks = True
            else:
                score_str = f"ρ={diag.axis1_score:.3f}"
        else:
            score_str = "N/A"

        gap_str = f"{diag.gap:+.1%}" if diag.gap is not None else "N/A"
        parse_str = f"{diag.parse_rate:.0%}" if diag.parse_rate is not None else "—"

        quadrant_str = diag.quadrant.value
        if diag.quadrant == Quadrant.UNLOCK:
            quadrant_str += " ←"

        lines.append(
            f"  {name:<14s} {score_str:>8s} {gap_str:>8s} {parse_str:>8s} {quadrant_str:<14s}"
        )

    # Footnote explaining thresholds
    if has_benchmarks:
        lines.append("")
        example_thresh = [
            d for d in diagnoses.values()
            if d.metric_type == MetricType.LOGIT_ACCURACY
        ]
        if example_thresh:
            t = example_thresh[0]
            lines.append(
                f"  Benchmarks: knows = accuracy > {t.knows_threshold:.0%} "
                f"(chance + {t.knows_threshold - 1.0 / 4:.0%} margin)"
            )

    return "\n".join(lines)
