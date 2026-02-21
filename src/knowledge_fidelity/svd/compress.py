"""SVD compression for transformer attention projections.

Two modes:
  1. Standard SVD (compress_qko) -- fast, good for the CF90 protection pipeline
  2. Importance-guided SVD (compress_qko_importance) -- better at aggressive
     compression (50%+), uses gradient info to pick which singular values to keep

Safety rules (validated experimentally):
  - Q, K, O projections: safe to compress at 70% rank
  - V projections: safe only at 90-95% (marginal gains, not worth the risk)
  - MLP layers: NEVER compress (destroys model at any compression level)
"""

import torch
from typing import Optional

from ..utils import get_layers, get_attention


# Projection names safe to compress (Q, K, O only -- never V, never MLP)
SAFE_PROJECTIONS = ('q_proj', 'k_proj', 'o_proj')


def compress_qko(
    model,
    ratio: float = 0.7,
    n_layers: Optional[int] = None,
) -> int:
    """Compress Q, K, O attention projections using standard SVD.

    For each weight matrix W, computes W ≈ U[:,:k] @ diag(S[:k]) @ Vh[:k,:]
    where k = int(min_dim * ratio).

    Args:
        model: HuggingFace causal LM model
        ratio: Fraction of singular values to keep (0.7 = keep 70% of rank)
        n_layers: Number of layers to compress (default: all)

    Returns:
        Number of matrices compressed
    """
    layers = get_layers(model)
    if n_layers is None:
        n_layers = len(layers)

    compressed = 0
    for i in range(min(n_layers, len(layers))):
        attn = get_attention(layers[i])
        if attn is None:
            continue

        for proj_name in SAFE_PROJECTIONS:
            if not hasattr(attn, proj_name):
                continue

            proj = getattr(attn, proj_name)
            W = proj.weight.data
            if len(W.shape) != 2 or min(W.shape) <= 10:
                continue

            rank = max(1, int(min(W.shape) * ratio))

            try:
                U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)
                W_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
                proj.weight.data = W_approx.to(W.dtype).to(W.device)
                compressed += 1
            except Exception:
                continue

    return compressed


def compress_qko_importance(
    model,
    importance: dict,
    ratio: float = 0.7,
    n_layers: Optional[int] = None,
) -> int:
    """Compress Q, K, O projections using importance-guided SVD.

    Instead of keeping the top-k singular values by magnitude, scores each
    singular value by how much it contributes to gradient-important directions.
    At 50% compression, this preserves 3x more factual knowledge than standard SVD.

    Args:
        model: HuggingFace causal LM model
        importance: Dict of {param_name: importance_tensor} from compute_importance()
        ratio: Fraction of singular values to keep
        n_layers: Number of layers to compress (default: all)

    Returns:
        Number of matrices compressed
    """
    layers = get_layers(model)
    if n_layers is None:
        n_layers = len(layers)

    compressed = 0
    for i in range(min(n_layers, len(layers))):
        attn = get_attention(layers[i])
        if attn is None:
            continue

        for proj_name in SAFE_PROJECTIONS:
            if not hasattr(attn, proj_name):
                continue

            proj = getattr(attn, proj_name)
            W = proj.weight.data
            if len(W.shape) != 2 or min(W.shape) <= 10:
                continue

            rank = max(1, int(min(W.shape) * ratio))

            # Find importance weights for this parameter
            imp = None
            for key in importance:
                if proj_name in key and f".{i}." in key:
                    imp = importance[key]
                    break

            try:
                U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)

                if imp is not None:
                    # Score each singular value by its contribution to
                    # gradient-important directions
                    sv_importance = torch.zeros(len(S))
                    for j in range(min(len(S), rank * 2)):
                        contrib = S[j] * torch.outer(U[:, j], Vh[j, :])
                        sv_importance[j] = (contrib.abs() * imp.float().cpu()).sum()
                    top_indices = sv_importance.argsort(descending=True)[:rank].sort().values
                else:
                    # Fall back to standard top-k
                    top_indices = torch.arange(rank)

                W_approx = U[:, top_indices] @ torch.diag(S[top_indices]) @ Vh[top_indices, :]
                proj.weight.data = W_approx.to(W.dtype).to(W.device)
                compressed += 1
            except Exception:
                continue

    return compressed
