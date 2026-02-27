"""Hybrid Weight + Activation Control Framework.

Combines three complementary control surfaces for behavioral repair:

  1. **Weight-space** (SVD compression + freeze) — removes capacity for
     storing false beliefs by truncating low-importance directions and
     freezing factual layers.

  2. **Activation-space** (SAE steering) — trains a Gated Sparse Autoencoder
     on model residual-stream activations, identifies behavioral features,
     and steers them at inference or during training.

  3. **Training-time** (Rho-guided SFT) — adds a contrastive confidence
     loss (positive/negative text pairs) alongside standard CE to reinforce
     truthful confident behavior.

Quick start:
    from rho_eval.hybrid import HybridConfig, apply_hybrid_control

    config = HybridConfig(
        compress_ratio=0.7,
        freeze_fraction=0.75,
        sae_layer=17,
        target_behaviors=["sycophancy"],
        rho_weight=0.2,
    )

    result = apply_hybrid_control("Qwen/Qwen2.5-7B-Instruct", config)
    print(result.summary())   # Before/after audit across all 8 behaviors

CLI:
    rho-hybrid Qwen/Qwen2.5-7B --compress 0.7 --freeze 0.75 \\
               --sae-layer 17 --target sycophancy --rho-weight 0.2
"""

from .schema import HybridConfig, HybridResult, PhaseResult
from .pipeline import apply_hybrid_control

__all__ = [
    "HybridConfig",
    "HybridResult",
    "PhaseResult",
    "apply_hybrid_control",
]
