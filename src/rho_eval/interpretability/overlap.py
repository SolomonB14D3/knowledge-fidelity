"""Pairwise overlap analysis between behavioral subspaces.

Computes three complementary overlap metrics between behavioral subspaces
at each layer, answering the question: are behavioral representations
orthogonal or entangled?
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .schema import SubspaceResult, OverlapMatrix


def _cosine_sim(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Cosine similarity between two vectors."""
    dot = torch.dot(v1.float(), v2.float())
    norm1 = v1.float().norm()
    norm2 = v2.float().norm()
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(dot / (norm1 * norm2))


def _shared_variance(V1: torch.Tensor, V2: torch.Tensor) -> float:
    """Shared variance metric (Grassmann-inspired).

    Computes ||V1^T @ V2||_F / k where V1, V2 are (k, d) orthonormal bases.
    Returns 1.0 for identical subspaces, 0.0 for orthogonal.
    """
    V1f = V1.float()
    V2f = V2.float()
    k = min(V1f.shape[0], V2f.shape[0])
    if k == 0:
        return 0.0
    # Cross-correlation matrix
    C = V1f[:k] @ V2f[:k].T  # (k, k)
    return float(C.norm() / math.sqrt(k))


def _principal_angles(V1: torch.Tensor, V2: torch.Tensor) -> float:
    """Mean principal angle between two subspaces in degrees.

    Principal angles are computed via SVD of V1^T @ V2. The singular
    values are cos(theta_i) where theta_i are the principal angles.
    """
    V1f = V1.float()
    V2f = V2.float()
    k = min(V1f.shape[0], V2f.shape[0])
    if k == 0:
        return 90.0  # Orthogonal by convention

    C = V1f[:k] @ V2f[:k].T  # (k, k)
    # Clamp for numerical stability
    S = torch.linalg.svdvals(C)
    S = S.clamp(-1.0, 1.0)

    # Convert to angles
    angles = torch.acos(S)  # radians
    mean_angle = float(angles.mean()) * 180.0 / math.pi
    return mean_angle


def compute_overlap(
    subspaces: dict[str, dict[int, SubspaceResult]],
    layers: list[int] | None = None,
    top_k: int = 10,
    verbose: bool = True,
) -> dict[int, OverlapMatrix]:
    """Compute pairwise overlap between behavioral subspaces.

    Three overlap metrics per (behavior_i, behavior_j, layer) triple:

    1. **Cosine similarity** of top-1 directions: How aligned are the
       primary behavioral directions? High cosine means they share the
       same dominant activation pattern.

    2. **Shared variance** (Grassmann distance): How much of the top-k
       subspace is shared? Uses Frobenius norm of cross-correlation.
       1.0 = identical subspaces, 0.0 = orthogonal.

    3. **Principal angles**: Mean angle between subspaces. 0° = parallel,
       90° = orthogonal. More robust than cosine for multi-dimensional
       comparison.

    Args:
        subspaces: Output of extract_subspaces(). {behavior: {layer: SubspaceResult}}.
        layers: Which layers to analyze (None = all available).
        top_k: Number of principal directions for shared variance and angles.
        verbose: Print progress.

    Returns:
        Dict mapping layer_idx → OverlapMatrix.
    """
    # Determine behaviors and layers
    behaviors = sorted(subspaces.keys())
    if len(behaviors) < 2:
        raise ValueError(f"Need at least 2 behaviors for overlap, got {behaviors}")

    if layers is None:
        # Use intersection of layers available across all behaviors
        layer_sets = [set(subspaces[b].keys()) for b in behaviors]
        common_layers = sorted(set.intersection(*layer_sets))
    else:
        common_layers = layers

    if verbose:
        print(f"\n  Computing overlap for {behaviors} at {len(common_layers)} layers (top-{top_k})")

    results: dict[int, OverlapMatrix] = {}
    n = len(behaviors)

    for layer_idx in common_layers:
        cosine_mat = [[0.0] * n for _ in range(n)]
        shared_var_mat = [[0.0] * n for _ in range(n)]
        angle_mat = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                sr_i = subspaces[behaviors[i]][layer_idx]
                sr_j = subspaces[behaviors[j]][layer_idx]

                if i == j:
                    cosine_mat[i][j] = 1.0
                    shared_var_mat[i][j] = 1.0
                    angle_mat[i][j] = 0.0
                else:
                    # Top-1 cosine
                    cosine_mat[i][j] = _cosine_sim(
                        sr_i.directions[0], sr_j.directions[0]
                    )

                    # Shared variance at top-k
                    k = min(top_k, sr_i.directions.shape[0], sr_j.directions.shape[0])
                    shared_var_mat[i][j] = _shared_variance(
                        sr_i.directions[:k], sr_j.directions[:k]
                    )

                    # Principal angles at top-k
                    angle_mat[i][j] = _principal_angles(
                        sr_i.directions[:k], sr_j.directions[:k]
                    )

        # Round for cleaner output
        cosine_mat = [[round(v, 4) for v in row] for row in cosine_mat]
        shared_var_mat = [[round(v, 4) for v in row] for row in shared_var_mat]
        angle_mat = [[round(v, 2) for v in row] for row in angle_mat]

        results[layer_idx] = OverlapMatrix(
            layer_idx=layer_idx,
            behaviors=behaviors,
            cosine_matrix=cosine_mat,
            shared_variance=shared_var_mat,
            subspace_angles=angle_mat,
            rank_used=top_k,
        )

        if verbose:
            print(f"    Layer {layer_idx:3d}: ", end="")
            # Print the most interesting off-diagonal pairs
            for i in range(n):
                for j in range(i + 1, n):
                    print(
                        f"{behaviors[i][:4]}-{behaviors[j][:4]} "
                        f"cos={cosine_mat[i][j]:+.3f} ",
                        end=""
                    )
            print()

    return results
