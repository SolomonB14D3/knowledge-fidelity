"""Rho-guided alignment via auxiliary behavioral losses during SFT.

Provides differentiable proxy losses derived from behavioral contrast pairs
that can be added alongside standard cross-entropy during fine-tuning.

The core idea: instead of making rho itself differentiable, we use the
underlying teacher-forced CE loss on positive vs negative text pairs as
a contrastive margin loss. This drives the same behavioral signal that
rho measures, but through a fully differentiable path.

Usage:
    from rho_eval.alignment import rho_guided_sft, contrastive_confidence_loss

    # Quick: run rho-guided SFT on a model
    result = rho_guided_sft(model, tokenizer, sft_dataset, contrast_dataset)

    # Low-level: compute the auxiliary loss for a single pair
    loss = contrastive_confidence_loss(model, tokenizer, "Paris is in France", "Paris is in Germany")
"""

from .losses import (
    differentiable_ce_loss,
    contrastive_confidence_loss,
    rho_auxiliary_loss,
)
from .dataset import (
    load_sft_dataset,
    BehavioralContrastDataset,
)
from .trainer import rho_guided_sft

__all__ = [
    "differentiable_ce_loss",
    "contrastive_confidence_loss",
    "rho_auxiliary_loss",
    "load_sft_dataset",
    "BehavioralContrastDataset",
    "rho_guided_sft",
]
