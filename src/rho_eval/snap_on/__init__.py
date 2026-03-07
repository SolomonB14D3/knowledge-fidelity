"""Snap-On Communication Module.

A tiny adapter that learns to adjust frozen base model outputs to produce
instruction-following behavior without modifying base model weights.

Two modes:
  hidden:  logits = lm_head(h + adapter(h))
           Adapter perturbs hidden states before unembedding.

  logit:   logits = lm_head(h) + adapter(lm_head(h))
           Adapter reshapes the output distribution without perturbing
           the knowledge pathway (lm_head(h)).

Usage:
    from rho_eval.snap_on import (
        SnapOnConfig, create_adapter,
        train, load_alpaca_data, save_adapter, load_adapter,
        generate_with_adapter, generate_base_only, evaluate_mmlu,
    )
"""

from .module import (
    SnapOnConfig,
    SnapOnMLP,
    SnapOnLogitMLP,
    SnapOnTransformer,
    create_adapter,
)

from .training import (
    ALPACA_TEMPLATE,
    train,
    load_alpaca_data,
    evaluate_loss,
    save_adapter,
    load_adapter,
)

from .inference import (
    generate_with_adapter,
    generate_base_only,
    evaluate_mmlu,
)

__all__ = [
    # Config & architectures
    "SnapOnConfig",
    "SnapOnMLP",
    "SnapOnLogitMLP",
    "SnapOnTransformer",
    "create_adapter",
    # Training
    "ALPACA_TEMPLATE",
    "train",
    "load_alpaca_data",
    "evaluate_loss",
    "save_adapter",
    "load_adapter",
    # Inference
    "generate_with_adapter",
    "generate_base_only",
    "evaluate_mmlu",
]
