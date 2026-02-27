"""Hybrid control framework — configuration and result schemas.

All dataclasses for the hybrid pipeline: config, per-phase results,
and the aggregate hybrid result with collateral-damage tracking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ── Config ────────────────────────────────────────────────────────────────

@dataclass
class HybridConfig:
    """Configuration for the hybrid weight + activation control pipeline.

    Three control surfaces, each optional:

    1. Weight-space control (SVD + freeze):
       - compress_ratio: fraction of singular values to retain (1.0 = skip)
       - freeze_fraction: fraction of layers to freeze from bottom (0.0 = skip)
       - compress_targets: which projection matrices to compress

    2. Activation-space control (SAE steering):
       - sae_layer: layer index for SAE training (None = auto-detect)
       - sae_expansion: hidden dimension multiplier for Gated SAE
       - target_behaviors: which behaviors to steer via SAE features
       - scale_factor: steering vector magnitude (higher = stronger)

    3. Training-time control (Rho-guided SFT):
       - rho_weight: weight of contrastive confidence loss (0.0 = skip)
       - sft_epochs: number of fine-tuning epochs
       - sft_lr: learning rate for LoRA fine-tuning
       - margin: contrastive loss margin (positive - negative gap)

    Set any section's main parameter to its skip value (1.0/None/0.0) to
    disable that control surface entirely.
    """

    # ── Weight-space control ──────────────────────────────────────────────
    compress_ratio: float = 0.7
    """Fraction of singular values to retain in SVD compression.
    1.0 = no compression. Default 0.7 matches CF90 validated range."""

    freeze_fraction: float = 0.75
    """Fraction of layers to freeze (bottom-up). 0.0 = no freezing.
    Default 0.75 protects factual knowledge in lower layers."""

    compress_targets: tuple[str, ...] = ("q", "k", "o")
    """Which attention projection matrices to compress.
    Safety rule: NEVER include 'v' below 90% or MLP layers."""

    # ── Activation-space control ──────────────────────────────────────────
    sae_layer: Optional[int] = None
    """Layer index for SAE training. None = auto-detect based on
    behavioral activation variance analysis."""

    sae_expansion: int = 8
    """Hidden dimension multiplier for Gated SAE.
    d_sae = d_model * sae_expansion."""

    target_behaviors: tuple[str, ...] = ("sycophancy",)
    """Behaviors to target with SAE feature steering."""

    scale_factor: float = 4.0
    """Steering vector magnitude. Higher = stronger behavioral shift,
    but increases collateral damage risk."""

    # ── Training-time control ─────────────────────────────────────────────
    rho_weight: float = 0.2
    """Weight of contrastive confidence loss in SFT objective.
    Total loss = (1 - rho_weight) * CE + rho_weight * contrastive.
    0.0 = no rho-guided training."""

    sft_epochs: int = 1
    """Number of SFT epochs. CF90 validated at 1 epoch."""

    sft_lr: float = 2e-4
    """Learning rate for LoRA/full fine-tuning."""

    margin: float = 0.1
    """Contrastive loss margin: desired gap between positive
    and negative pair confidence."""

    # ── Evaluation ────────────────────────────────────────────────────────
    eval_behaviors: tuple[str, ...] = ("all",)
    """Behaviors to evaluate in before/after audit.
    ("all",) evaluates all 8 registered behaviors."""

    device: Optional[str] = None
    """Device for computation. None = auto-detect (CUDA > MPS > CPU)."""

    trust_remote_code: bool = False
    """Whether to trust remote code when loading models."""

    seed: int = 42
    """Random seed for reproducibility."""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: Path | str) -> "HybridConfig":
        path = Path(path)
        data = json.loads(path.read_text())
        # Convert lists back to tuples for frozen fields
        for key in ("compress_targets", "target_behaviors", "eval_behaviors"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)

    @property
    def weight_space_enabled(self) -> bool:
        return self.compress_ratio < 1.0 or self.freeze_fraction > 0.0

    @property
    def activation_space_enabled(self) -> bool:
        return self.sae_layer is not None

    @property
    def training_time_enabled(self) -> bool:
        return self.rho_weight > 0.0

    @property
    def enabled_phases(self) -> list[str]:
        phases = []
        if self.weight_space_enabled:
            phases.append("weight_space")
        if self.activation_space_enabled:
            phases.append("activation_space")
        if self.training_time_enabled:
            phases.append("training_time")
        return phases


# ── Results ───────────────────────────────────────────────────────────────

@dataclass
class PhaseResult:
    """Result from one phase of the hybrid pipeline."""

    phase: str                          # "weight_space" / "activation_space" / "training_time"
    elapsed_sec: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None         # None = success

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HybridResult:
    """Aggregate result from the full hybrid control pipeline.

    Tracks before/after audit scores across all 8 behaviors,
    per-phase timing and details, and a collateral damage matrix.
    """

    config: HybridConfig
    model_name: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Audit snapshots
    audit_before: dict[str, float] = field(default_factory=dict)
    """Per-behavior rho scores BEFORE any intervention."""

    audit_after: dict[str, float] = field(default_factory=dict)
    """Per-behavior rho scores AFTER all interventions."""

    # Per-phase results
    phases: list[PhaseResult] = field(default_factory=list)

    # Collateral damage tracking
    collateral_damage: dict[str, float] = field(default_factory=dict)
    """Per-behavior delta: audit_after - audit_before.
    Negative = regression (collateral damage).
    Positive = improvement."""

    total_elapsed_sec: float = 0.0

    def compute_collateral(self) -> None:
        """Compute collateral damage from before/after audits."""
        self.collateral_damage = {}
        for behavior in self.audit_before:
            if behavior in self.audit_after:
                delta = self.audit_after[behavior] - self.audit_before[behavior]
                self.collateral_damage[behavior] = round(delta, 4)

    @property
    def target_improvement(self) -> float:
        """Mean improvement on targeted behaviors."""
        targets = self.config.target_behaviors
        deltas = [
            self.collateral_damage.get(b, 0.0)
            for b in targets
            if b in self.collateral_damage
        ]
        return sum(deltas) / len(deltas) if deltas else 0.0

    @property
    def non_target_regression(self) -> float:
        """Mean regression on non-targeted behaviors (negative = bad)."""
        targets = set(self.config.target_behaviors)
        deltas = [
            v for k, v in self.collateral_damage.items()
            if k not in targets and v < 0
        ]
        return sum(deltas) / len(deltas) if deltas else 0.0

    def summary(self) -> str:
        """One-line summary of the hybrid control result."""
        n_improved = sum(1 for v in self.collateral_damage.values() if v > 0)
        n_regressed = sum(1 for v in self.collateral_damage.values() if v < 0)
        return (
            f"Hybrid control on {self.model_name}: "
            f"{n_improved} improved, {n_regressed} regressed, "
            f"target Δ={self.target_improvement:+.4f}, "
            f"collateral={self.non_target_regression:+.4f}"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def to_table(self) -> str:
        """Formatted table of before/after/delta per behavior."""
        lines = [
            f"{'Behavior':<15} {'Before':>8} {'After':>8} {'Delta':>8}  Status",
            "─" * 55,
        ]
        for b in sorted(self.audit_before.keys()):
            before = self.audit_before.get(b, 0.0)
            after = self.audit_after.get(b, 0.0)
            delta = self.collateral_damage.get(b, 0.0)
            status = "✓" if delta >= 0 else "⚠ REGRESSED"
            lines.append(
                f"{b:<15} {before:>8.4f} {after:>8.4f} {delta:>+8.4f}  {status}"
            )
        return "\n".join(lines)
