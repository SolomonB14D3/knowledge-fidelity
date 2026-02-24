"""Schema dataclasses for mechanistic interpretability results.

Defines structured containers for subspace extraction, overlap analysis,
head attribution, and surgical intervention results. Follows the pattern
from rho_eval.output.schema (AuditReport) and rho_eval.behaviors.base
(BehaviorResult).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch


# ── Core Result Types ─────────────────────────────────────────────────────


@dataclass
class SubspaceResult:
    """Principal directions of a behavioral subspace at a specific layer.

    Stores the full SVD decomposition of the centered difference matrix
    between positive and negative activation vectors.

    Attributes:
        behavior: Name of the behavior (e.g., "factual", "sycophancy").
        layer_idx: Transformer layer index.
        n_pairs: Number of contrast pairs used for extraction.
        directions: Top-k principal directions, shape (k, hidden_dim).
        singular_values: Corresponding singular values.
        explained_variance: Cumulative explained variance ratios.
        effective_dim: Number of directions for 90% variance.
        mean_pos: Mean positive activation, shape (hidden_dim,).
        mean_neg: Mean negative activation, shape (hidden_dim,).
        steering_vector: Mean-diff steering vector, shape (hidden_dim,).
    """
    behavior: str
    layer_idx: int
    n_pairs: int
    directions: torch.Tensor          # (k, hidden_dim)
    singular_values: list[float]
    explained_variance: list[float]    # cumulative
    effective_dim: int
    mean_pos: torch.Tensor             # (hidden_dim,)
    mean_neg: torch.Tensor             # (hidden_dim,)
    steering_vector: torch.Tensor      # (hidden_dim,)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (tensors become summary stats)."""
        return {
            "behavior": self.behavior,
            "layer_idx": self.layer_idx,
            "n_pairs": self.n_pairs,
            "effective_dim": self.effective_dim,
            "singular_values": self.singular_values[:20],  # top-20 only
            "explained_variance": self.explained_variance[:20],
            "steering_vector_norm": float(self.steering_vector.norm()),
            "n_directions": self.directions.shape[0],
            "hidden_dim": self.directions.shape[1],
        }


@dataclass
class OverlapMatrix:
    """Pairwise overlap between behavioral subspaces at a layer.

    Three metrics capture different aspects of subspace similarity:
    - cosine: alignment of top-1 principal directions
    - shared_variance: proportion of variance in shared subspace (Grassmann)
    - subspace_angles: mean principal angle between subspaces

    Attributes:
        layer_idx: Transformer layer index.
        behaviors: Ordered list of behavior names.
        cosine_matrix: (n, n) cosine similarity of top-1 directions.
        shared_variance: (n, n) shared variance metric at given rank.
        subspace_angles: (n, n) mean principal angles in degrees.
        rank_used: Number of directions used for shared_variance/angles.
    """
    layer_idx: int
    behaviors: list[str]
    cosine_matrix: list[list[float]]
    shared_variance: list[list[float]]
    subspace_angles: list[list[float]]
    rank_used: int = 10

    def to_dict(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "behaviors": self.behaviors,
            "cosine_matrix": self.cosine_matrix,
            "shared_variance": self.shared_variance,
            "subspace_angles": self.subspace_angles,
            "rank_used": self.rank_used,
        }


@dataclass
class HeadImportance:
    """Per-head importance for a specific behavior at a specific layer.

    Importance is the ratio of per-head activation norm difference to
    the full-layer activation norm difference.

    Attributes:
        behavior: Behavior name.
        layer_idx: Layer index.
        head_idx: Attention head index.
        importance_score: Normalized importance (> 1/n_heads means over-represented).
        n_heads: Total number of heads in this layer.
    """
    behavior: str
    layer_idx: int
    head_idx: int
    importance_score: float
    n_heads: int

    def to_dict(self) -> dict:
        return {
            "behavior": self.behavior,
            "layer_idx": self.layer_idx,
            "head_idx": self.head_idx,
            "importance_score": round(self.importance_score, 6),
            "n_heads": self.n_heads,
        }


@dataclass
class SurgicalResult:
    """Result of a surgical intervention experiment.

    Records the intervention type, configuration, and resulting rho scores
    across all evaluated behaviors.

    Attributes:
        intervention: Type of intervention (e.g., "rank_1", "orthogonal_project").
        target_behavior: The behavior being steered.
        rho_scores: {behavior_name: rho} for all evaluated behaviors.
        config: Configuration dict (layer, alpha, rank, removed behaviors, etc.).
        baseline_rho_scores: Optional baseline rho scores for comparison.
    """
    intervention: str
    target_behavior: str
    rho_scores: dict[str, float]
    config: dict[str, Any]
    baseline_rho_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "intervention": self.intervention,
            "target_behavior": self.target_behavior,
            "rho_scores": {k: round(v, 6) if v is not None else None
                           for k, v in self.rho_scores.items()},
            "config": self.config,
            "baseline_rho_scores": {k: round(v, 6) if v is not None else None
                                    for k, v in self.baseline_rho_scores.items()},
        }


# ── Top-Level Report ──────────────────────────────────────────────────────


@dataclass
class InterpretabilityReport:
    """Complete interpretability analysis for one model.

    Top-level container analogous to AuditReport. Holds subspace extraction
    results, overlap matrices, head importance scores, and surgical
    intervention outcomes.

    Attributes:
        model: Model identifier.
        subspaces: {behavior: {layer_idx: SubspaceResult}}.
        overlaps: {layer_idx: OverlapMatrix}.
        head_importance: {behavior: [HeadImportance]}.
        surgical_results: List of SurgicalResult.
        timestamp: ISO-8601 UTC timestamp.
        elapsed_seconds: Total wall-clock time.
        metadata: Extra info (n_layers, hidden_dim, n_heads, etc.).
    """
    model: str
    subspaces: dict[str, dict[int, SubspaceResult]] = field(default_factory=dict)
    overlaps: dict[int, OverlapMatrix] = field(default_factory=dict)
    head_importance: dict[str, list[HeadImportance]] = field(default_factory=dict)
    surgical_results: list[SurgicalResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Serialization (JSON summary — no tensors) ─────────────────────

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict.

        Tensor data (directions, mean activations) is NOT included in the
        JSON output. Use save_tensors() separately for tensor data.
        """
        d = {
            "model": self.model,
            "timestamp": self.timestamp,
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": self.metadata,
            "subspaces": {},
            "overlaps": {},
            "head_importance": {},
            "surgical_results": [],
        }

        for behavior, layers in sorted(self.subspaces.items()):
            d["subspaces"][behavior] = {
                str(layer_idx): sr.to_dict()
                for layer_idx, sr in sorted(layers.items())
            }

        for layer_idx, om in sorted(self.overlaps.items()):
            d["overlaps"][str(layer_idx)] = om.to_dict()

        for behavior, heads in sorted(self.head_importance.items()):
            d["head_importance"][behavior] = [h.to_dict() for h in heads]

        d["surgical_results"] = [sr.to_dict() for sr in self.surgical_results]

        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize summary to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> Path:
        """Save report summary to JSON file.

        For tensor data (directions, mean activations), call save_tensors()
        separately with the same base path.

        Args:
            path: Output JSON file path.

        Returns:
            Resolved output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path.resolve()

    def save_tensors(self, path: str | Path) -> Path:
        """Save tensor data (directions, mean activations) to a .pt file.

        Args:
            path: Output .pt file path.

        Returns:
            Resolved output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tensor_data = {}
        for behavior, layers in self.subspaces.items():
            tensor_data[behavior] = {}
            for layer_idx, sr in layers.items():
                tensor_data[behavior][layer_idx] = {
                    "directions": sr.directions,
                    "mean_pos": sr.mean_pos,
                    "mean_neg": sr.mean_neg,
                    "steering_vector": sr.steering_vector,
                }

        torch.save(tensor_data, path)
        return path.resolve()

    @classmethod
    def load(cls, path: str | Path, tensor_path: str | Path | None = None) -> "InterpretabilityReport":
        """Load report from JSON, optionally with tensor data.

        Args:
            path: JSON file saved by .save().
            tensor_path: Optional .pt file saved by .save_tensors().

        Returns:
            InterpretabilityReport instance.
        """
        with open(path) as f:
            d = json.load(f)

        # Load tensor data if available
        tensors = {}
        if tensor_path is not None:
            tensors = torch.load(tensor_path, weights_only=True)

        report = cls(
            model=d["model"],
            timestamp=d.get("timestamp", ""),
            elapsed_seconds=d.get("elapsed_seconds", 0.0),
            metadata=d.get("metadata", {}),
        )

        # Reconstruct subspaces (summary only unless tensors provided)
        for behavior, layers in d.get("subspaces", {}).items():
            report.subspaces[behavior] = {}
            for layer_str, sd in layers.items():
                layer_idx = int(layer_str)
                hidden_dim = sd.get("hidden_dim", 0)
                n_dirs = sd.get("n_directions", 0)

                # Use tensor data if available
                t = tensors.get(behavior, {}).get(layer_idx, {})
                directions = t.get("directions", torch.zeros(n_dirs, hidden_dim))
                mean_pos = t.get("mean_pos", torch.zeros(hidden_dim))
                mean_neg = t.get("mean_neg", torch.zeros(hidden_dim))
                steering_vector = t.get("steering_vector", torch.zeros(hidden_dim))

                report.subspaces[behavior][layer_idx] = SubspaceResult(
                    behavior=behavior,
                    layer_idx=layer_idx,
                    n_pairs=sd.get("n_pairs", 0),
                    directions=directions,
                    singular_values=sd.get("singular_values", []),
                    explained_variance=sd.get("explained_variance", []),
                    effective_dim=sd.get("effective_dim", 0),
                    mean_pos=mean_pos,
                    mean_neg=mean_neg,
                    steering_vector=steering_vector,
                )

        # Reconstruct overlaps
        for layer_str, od in d.get("overlaps", {}).items():
            layer_idx = int(layer_str)
            report.overlaps[layer_idx] = OverlapMatrix(
                layer_idx=layer_idx,
                behaviors=od["behaviors"],
                cosine_matrix=od["cosine_matrix"],
                shared_variance=od["shared_variance"],
                subspace_angles=od["subspace_angles"],
                rank_used=od.get("rank_used", 10),
            )

        # Reconstruct head importance
        for behavior, heads in d.get("head_importance", {}).items():
            report.head_importance[behavior] = [
                HeadImportance(
                    behavior=h["behavior"],
                    layer_idx=h["layer_idx"],
                    head_idx=h["head_idx"],
                    importance_score=h["importance_score"],
                    n_heads=h["n_heads"],
                )
                for h in heads
            ]

        # Reconstruct surgical results
        for sr in d.get("surgical_results", []):
            report.surgical_results.append(SurgicalResult(
                intervention=sr["intervention"],
                target_behavior=sr["target_behavior"],
                rho_scores=sr["rho_scores"],
                config=sr["config"],
                baseline_rho_scores=sr.get("baseline_rho_scores", {}),
            ))

        return report

    def __repr__(self) -> str:
        n_behaviors = len(self.subspaces)
        n_layers = len(self.overlaps)
        n_surgical = len(self.surgical_results)
        return (
            f"<InterpretabilityReport model={self.model!r} "
            f"behaviors={n_behaviors} layers={n_layers} "
            f"surgical={n_surgical}>"
        )
