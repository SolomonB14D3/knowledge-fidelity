"""Mechanistic interpretability of behavioral subspaces.

Extract, analyze, and visualize the internal directions within transformer
layers that encode specific behavioral traits (factual, sycophancy, bias,
toxicity, reasoning).

Quick start:
    from rho_eval.interpretability import extract_subspaces, compute_overlap

    subspaces = extract_subspaces(model, tokenizer, ["factual", "sycophancy", "bias"])
    overlaps = compute_overlap(subspaces)

Surgical intervention:
    from rho_eval.interpretability import orthogonal_project, evaluate_surgical

    result = evaluate_surgical(
        model, tokenizer, subspaces,
        target_behavior="sycophancy", layer_idx=17,
        eval_behaviors=["factual", "sycophancy", "bias"],
        orthogonal_to=["bias"],
    )
"""

from .subspaces import extract_subspaces
from .overlap import compute_overlap
from .heads import head_attribution
from .surgical import (
    orthogonal_project,
    rank_k_steer,
    evaluate_surgical,
    evaluate_baseline,
)
from .schema import (
    SubspaceResult,
    OverlapMatrix,
    HeadImportance,
    SurgicalResult,
    InterpretabilityReport,
)

__all__ = [
    # Core analysis
    "extract_subspaces",
    "compute_overlap",
    "head_attribution",
    # Surgical interventions
    "orthogonal_project",
    "rank_k_steer",
    "evaluate_surgical",
    "evaluate_baseline",
    # Data schemas
    "SubspaceResult",
    "OverlapMatrix",
    "HeadImportance",
    "SurgicalResult",
    "InterpretabilityReport",
]
