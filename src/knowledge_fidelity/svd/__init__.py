"""SVD compression module for knowledge-fidelity.

Provides knowledge-preserving SVD compression of LLM attention projections.
Adapted from the Intelligent SVD project (github.com/SolomonB14D3/intelligent-svd).

Core method (CF90):
  1. Compress Q, K, O attention projections at 70% rank via truncated SVD
  2. Freeze 75% of layers from the bottom up
  3. Fine-tune gently (1 epoch, LoRA rank 8, lr=2e-4)

Builds on truncation-aware SVD (SVD-LLM, Wang et al. 2024, arXiv:2403.07378)
and activation-aware rank allocation (ASVD, Yuan et al. 2023, arXiv:2312.05821).
We extend these with importance-guided truncation scored on factual probes
and behavioral auditing via teacher-forced confidence (rho metric).

Safety rules (validated on Qwen 0.5B-32B, Llama 2 7B, Mistral 7B):
  - Q, K, O: safe at 70% rank
  - V: safe at 90-95% only
  - MLP: NEVER compress
"""

from .compress import compress_qko, compress_qko_importance, SAFE_PROJECTIONS
from .freeze import freeze_layers, freeze_hierarchical, unfreeze_all
from .importance import compute_importance

__all__ = [
    "compress_qko",
    "compress_qko_importance",
    "freeze_layers",
    "freeze_hierarchical",
    "unfreeze_all",
    "compute_importance",
    "SAFE_PROJECTIONS",
]
