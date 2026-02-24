"""Schema dataclasses for SAE-based behavioral steering.

Defines structured containers for SAE configuration, activation data,
per-feature behavioral profiles, and steering experiment results.
Follows the pattern from rho_eval.interpretability.schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch


# ── Configuration ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SAEConfig:
    """Configuration for Gated SAE training.

    Attributes:
        hidden_dim: Model hidden dimension (input/output size of SAE).
        expansion_factor: SAE width multiplier (n_features = hidden_dim * expansion_factor).
        sparsity_lambda: L1 penalty weight on gate activations.
        lr: Learning rate for Adam optimizer.
        batch_size: Training batch size.
        n_epochs: Number of training epochs over the activation dataset.
        device: Torch device string.
    """
    hidden_dim: int
    expansion_factor: int = 8
    sparsity_lambda: float = 1e-3
    lr: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 5
    device: str = "cpu"

    @property
    def n_features(self) -> int:
        return self.hidden_dim * self.expansion_factor

    def to_dict(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "expansion_factor": self.expansion_factor,
            "n_features": self.n_features,
            "sparsity_lambda": self.sparsity_lambda,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "device": self.device,
        }


# ── Activation Data ──────────────────────────────────────────────────────


@dataclass
class ActivationData:
    """Collected activations with behavior labels and polarities.

    Stores (n_samples, hidden_dim) activations from model residual stream,
    along with metadata identifying which behavior and polarity (positive
    vs negative) each sample came from.

    Attributes:
        activations: Stacked activation vectors, shape (n_samples, hidden_dim).
        labels: Behavior name for each sample (e.g., "factual", "toxicity").
        polarities: "positive" or "negative" for each sample.
        layer_idx: Transformer layer the activations were captured from.
        model_name: Model identifier.
    """
    activations: torch.Tensor        # (n_samples, hidden_dim)
    labels: list[str]
    polarities: list[str]
    layer_idx: int
    model_name: str = ""

    def __post_init__(self):
        n = self.activations.shape[0]
        assert len(self.labels) == n, f"labels length {len(self.labels)} != n_samples {n}"
        assert len(self.polarities) == n, f"polarities length {len(self.polarities)} != n_samples {n}"

    @property
    def n_samples(self) -> int:
        return self.activations.shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.activations.shape[1]

    @property
    def behaviors(self) -> list[str]:
        """Unique behaviors in order of first appearance."""
        seen = set()
        result = []
        for label in self.labels:
            if label not in seen:
                seen.add(label)
                result.append(label)
        return result

    def save(self, path: str | Path) -> Path:
        """Save activation data to a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "activations": self.activations,
            "labels": self.labels,
            "polarities": self.polarities,
            "layer_idx": self.layer_idx,
            "model_name": self.model_name,
        }, path)
        return path.resolve()

    @classmethod
    def load(cls, path: str | Path) -> "ActivationData":
        """Load activation data from a .pt file."""
        data = torch.load(path, weights_only=False)
        return cls(
            activations=data["activations"],
            labels=data["labels"],
            polarities=data["polarities"],
            layer_idx=data["layer_idx"],
            model_name=data.get("model_name", ""),
        )


# ── Feature Report ───────────────────────────────────────────────────────


@dataclass
class FeatureReport:
    """Behavioral profile for a single SAE feature.

    Describes how strongly a feature responds to each behavior and whether
    it discriminates between positive and negative examples.

    Attributes:
        feature_idx: Index in the SAE latent space.
        behavior_scores: {behavior: selectivity_score} — how much this
            feature's activation differs between positive and negative
            examples for each behavior.
        dominant_behavior: Behavior with highest selectivity, or None.
        mean_activation_pos: Mean activation on positive examples (overall).
        mean_activation_neg: Mean activation on negative examples (overall).
        selectivity: How specific this feature is to one behavior
            (max score / mean of all scores). Higher = more monosemantic.
    """
    feature_idx: int
    behavior_scores: dict[str, float]
    dominant_behavior: Optional[str] = None
    mean_activation_pos: float = 0.0
    mean_activation_neg: float = 0.0
    selectivity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "feature_idx": self.feature_idx,
            "behavior_scores": {k: round(v, 6) for k, v in self.behavior_scores.items()},
            "dominant_behavior": self.dominant_behavior,
            "mean_activation_pos": round(self.mean_activation_pos, 6),
            "mean_activation_neg": round(self.mean_activation_neg, 6),
            "selectivity": round(self.selectivity, 4),
        }


# ── Top-Level Report ─────────────────────────────────────────────────────


@dataclass
class SAESteeringReport:
    """Complete SAE steering analysis for one model at one layer.

    Top-level container analogous to InterpretabilityReport. Holds the SAE
    configuration, per-feature behavioral profiles, behavioral feature
    assignments, and steering experiment results.

    Attributes:
        model: Model identifier.
        layer_idx: Transformer layer used for SAE.
        sae_config: SAE training configuration.
        feature_reports: Per-feature behavioral profiles.
        behavioral_features: {behavior: [feature_indices]} for steering.
        overlap_before: SVD-based overlap matrix (for comparison).
        steering_results: Per-intervention audit results.
        train_stats: SAE training statistics.
        timestamp: ISO-8601 UTC timestamp.
        elapsed_seconds: Total wall-clock time.
    """
    model: str
    layer_idx: int
    sae_config: Optional[SAEConfig] = None
    feature_reports: list[FeatureReport] = field(default_factory=list)
    behavioral_features: dict[str, list[int]] = field(default_factory=dict)
    overlap_before: dict[str, dict[str, float]] = field(default_factory=dict)
    steering_results: list[dict] = field(default_factory=list)
    train_stats: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    elapsed_seconds: float = 0.0

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "model": self.model,
            "layer_idx": self.layer_idx,
            "sae_config": self.sae_config.to_dict() if self.sae_config else None,
            "n_behavioral_features": sum(len(v) for v in self.behavioral_features.values()),
            "behavioral_features": self.behavioral_features,
            "feature_reports": [fr.to_dict() for fr in self.feature_reports],
            "overlap_before": self.overlap_before,
            "steering_results": self.steering_results,
            "train_stats": {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in self.train_stats.items()
            },
            "timestamp": self.timestamp,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> Path:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path.resolve()

    @classmethod
    def load(cls, path: str | Path) -> "SAESteeringReport":
        """Load report from JSON file."""
        with open(path) as f:
            d = json.load(f)

        config_d = d.get("sae_config")
        config = None
        if config_d:
            # Remove computed fields
            config_d.pop("n_features", None)
            config = SAEConfig(**config_d)

        report = cls(
            model=d["model"],
            layer_idx=d["layer_idx"],
            sae_config=config,
            behavioral_features=d.get("behavioral_features", {}),
            overlap_before=d.get("overlap_before", {}),
            steering_results=d.get("steering_results", []),
            train_stats=d.get("train_stats", {}),
            timestamp=d.get("timestamp", ""),
            elapsed_seconds=d.get("elapsed_seconds", 0.0),
        )

        for fr_d in d.get("feature_reports", []):
            report.feature_reports.append(FeatureReport(
                feature_idx=fr_d["feature_idx"],
                behavior_scores=fr_d["behavior_scores"],
                dominant_behavior=fr_d.get("dominant_behavior"),
                mean_activation_pos=fr_d.get("mean_activation_pos", 0.0),
                mean_activation_neg=fr_d.get("mean_activation_neg", 0.0),
                selectivity=fr_d.get("selectivity", 0.0),
            ))

        return report

    def __repr__(self) -> str:
        n_features = sum(len(v) for v in self.behavioral_features.values())
        n_results = len(self.steering_results)
        return (
            f"<SAESteeringReport model={self.model!r} layer={self.layer_idx} "
            f"behavioral_features={n_features} results={n_results}>"
        )
