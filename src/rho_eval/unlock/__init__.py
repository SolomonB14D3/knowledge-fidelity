"""rho-unlock: Two-axis behavioral diagnostic + contrastive decoding unlock.

Axis 1 (ρ scores from rho-eval): Does the model discriminate correctly?
Axis 2 (expression gap): Can the model express what it knows?

Four quadrants:
  HEALTHY:      knows + can express → no action
  UNLOCK:       knows + can't express → contrastive decoding rescues
  RETRAIN:      doesn't know + can express → needs fine-tuning
  BOTH_NEEDED:  doesn't know + can't express → needs both
"""

from .contrastive import (
    contrastive_generate,
    contrastive_logit_classify,
    detect_amateur,
    get_answer_token_ids,
    parse_generated_answer,
    AMATEUR_MAP,
)

from .expression_gap import (
    ExpressionGapResult,
    measure_expression_gap,
    measure_all_gaps,
    MC_BEHAVIORS,
    BENCHMARK_ONLY,
    BEHAVIOR_N_CHOICES,
)

from .diagnosis import (
    BehaviorDiagnosis,
    MetricType,
    Quadrant,
    classify_quadrant,
    compute_knows_threshold,
    diagnose,
    format_diagnosis_table,
)

__all__ = [
    # Contrastive decoding
    "contrastive_generate",
    "contrastive_logit_classify",
    "detect_amateur",
    "get_answer_token_ids",
    "parse_generated_answer",
    "AMATEUR_MAP",
    # Expression gap
    "ExpressionGapResult",
    "measure_expression_gap",
    "measure_all_gaps",
    "MC_BEHAVIORS",
    "BENCHMARK_ONLY",
    "BEHAVIOR_N_CHOICES",
    # Diagnosis
    "BehaviorDiagnosis",
    "MetricType",
    "Quadrant",
    "classify_quadrant",
    "compute_knows_threshold",
    "diagnose",
    "format_diagnosis_table",
]
