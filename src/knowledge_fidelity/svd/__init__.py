"""SVD compression module for knowledge-fidelity.

Provides knowledge-preserving SVD compression of LLM attention projections.
Adapted from the Intelligent SVD project (github.com/SolomonB14D3/intelligent-svd).

Core method (CF90):
  1. Compress Q, K, O attention projections at 70% rank via truncated SVD
  2. Freeze 75% of layers from the bottom up
  3. Fine-tune gently (1 epoch, lr=1e-5)

Safety rules (validated on Qwen 0.5B-32B, Llama 2 7B):
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
