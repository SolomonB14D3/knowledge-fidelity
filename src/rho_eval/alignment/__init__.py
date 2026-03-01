"""Rho-guided alignment via auxiliary behavioral losses during SFT.

Provides differentiable proxy losses derived from behavioral contrast pairs
that can be added alongside standard cross-entropy during fine-tuning.

The core idea: instead of making rho itself differentiable, we use the
underlying teacher-forced CE loss on positive vs negative text pairs as
a contrastive margin loss. This drives the same behavioral signal that
rho measures, but through a fully differentiable path.

Two backends are available:

  PyTorch (default) — works on any hardware:
    from rho_eval.alignment import rho_guided_sft
    result = rho_guided_sft(model, tokenizer, sft_dataset, contrast_dataset)

  MLX (Apple Silicon only) — ~10x faster on M-series Macs:
    from rho_eval.alignment import mlx_rho_guided_sft  # requires: pip install mlx mlx-lm
    result = mlx_rho_guided_sft(model, tokenizer, sft_texts, contrast_dataset)

Both produce the same training signal (CE + contrastive margin loss on
Q/K/O LoRA). The MLX backend avoids PyTorch MPS NaN gradient bugs by
using Apple's native ML framework directly.

Check ``_HAS_MLX`` to see if the MLX backend is available:
    from rho_eval.alignment import _HAS_MLX  # True if mlx is installed
"""

from .losses import (
    differentiable_ce_loss,
    contrastive_confidence_loss,
    rho_auxiliary_loss,
    gamma_protection_loss,
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
    "gamma_protection_loss",
    "load_sft_dataset",
    "BehavioralContrastDataset",
    "rho_guided_sft",
]

# ── MLX backend (Apple Silicon only) ─────────────────────────────────
try:
    from .mlx_losses import (
        mlx_ce_loss,
        mlx_contrastive_confidence_loss,
        mlx_rho_auxiliary_loss,
    )
    from .mlx_trainer import mlx_rho_guided_sft

    _HAS_MLX = True
    __all__ += [
        "mlx_ce_loss",
        "mlx_contrastive_confidence_loss",
        "mlx_rho_auxiliary_loss",
        "mlx_rho_guided_sft",
        "_HAS_MLX",
    ]
except ImportError:
    _HAS_MLX = False
