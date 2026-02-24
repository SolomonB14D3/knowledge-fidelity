"""Surgical interventions for behavioral steering.

Implements low-rank and orthogonal projection interventions that operate
on the behavioral subspaces extracted by subspaces.py. The key experiment
is orthogonal projection: removing one behavior's subspace from another's
steering vector to test whether coupled behaviors (e.g., sycophancy-bias
at Layer 17) can be disentangled.
"""

from __future__ import annotations

from typing import Optional, Any

import torch

from .activation import SteeringHook
from .schema import SubspaceResult, SurgicalResult


# ── Low-Rank Steering ────────────────────────────────────────────────────


def rank_k_steer(
    subspace: SubspaceResult,
    rank: int = 1,
) -> torch.Tensor:
    """Project steering vector onto top-k principal directions.

    Instead of the full mean-diff vector, returns the component that lies
    within the top-k subspace:
        v_k = sum_{i=0}^{k-1} (v · d_i) * d_i

    This tests whether rank-1 (or low-rank) interventions are sufficient,
    i.e., whether the behavioral signal is concentrated in a few directions.

    Args:
        subspace: SubspaceResult with directions and steering_vector.
        rank: Number of principal directions to use.

    Returns:
        Low-rank steering vector, shape (hidden_dim,).
    """
    v = subspace.steering_vector.float()
    D = subspace.directions[:rank].float()  # (rank, hidden_dim)

    # Project: v_k = D^T @ (D @ v)
    coeffs = D @ v  # (rank,)
    v_k = coeffs @ D  # (hidden_dim,)

    return v_k


# ── Orthogonal Projection ───────────────────────────────────────────────


def orthogonal_project(
    vector: torch.Tensor,
    subspace_to_remove: SubspaceResult,
    n_directions: int = 5,
) -> torch.Tensor:
    """Remove one behavior's subspace from a steering vector.

    Projects the vector into the null space of the subspace_to_remove:
        v_clean = v - sum_{i=0}^{n-1} (v · d_i) * d_i

    This is the key operation for testing whether orthogonalized
    sycophancy vectors avoid bias collapse at Layer 17.

    Args:
        vector: Steering vector to clean, shape (hidden_dim,).
        subspace_to_remove: SubspaceResult whose directions will be removed.
        n_directions: Number of top directions to project out.

    Returns:
        Cleaned steering vector with the removed subspace's component
        subtracted, shape (hidden_dim,).
    """
    v = vector.float()
    k = min(n_directions, subspace_to_remove.directions.shape[0])
    D = subspace_to_remove.directions[:k].float()  # (k, hidden_dim)

    # Remove component: v_clean = v - D^T @ (D @ v)
    coeffs = D @ v  # (k,)
    projection = coeffs @ D  # (hidden_dim,)
    v_clean = v - projection

    return v_clean


# ── Combined Evaluation ──────────────────────────────────────────────────


def evaluate_surgical(
    model,
    tokenizer,
    subspaces: dict[str, dict[int, SubspaceResult]],
    target_behavior: str,
    layer_idx: int,
    eval_behaviors: list[str],
    alpha: float = 4.0,
    device: str = "cpu",
    orthogonal_to: list[str] | None = None,
    rank: int | None = None,
    verbose: bool = True,
) -> SurgicalResult:
    """Run a surgical intervention experiment.

    Combines rank_k_steer and orthogonal_project:
    1. Start with the target behavior's steering vector at the given layer.
    2. If orthogonal_to is specified, remove those behaviors' subspaces.
    3. If rank is specified, project onto top-k directions.
    4. Apply the resulting vector via SteeringHook and evaluate all behaviors.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        subspaces: Output of extract_subspaces().
        target_behavior: Behavior to steer (e.g., "sycophancy").
        layer_idx: Layer to apply intervention.
        eval_behaviors: Behaviors to evaluate after intervention.
        alpha: Steering strength multiplier.
        device: Torch device.
        orthogonal_to: Behaviors whose subspaces to remove from the vector.
        rank: If set, use only top-k directions of the target subspace.
        verbose: Print progress.

    Returns:
        SurgicalResult with intervention type, rho scores, and config.
    """
    from ..behavioral import evaluate_behavior, load_behavioral_probes
    from ..probes import get_all_probes

    if target_behavior not in subspaces:
        raise ValueError(
            f"No subspace for {target_behavior}. "
            f"Available: {list(subspaces.keys())}"
        )
    if layer_idx not in subspaces[target_behavior]:
        raise ValueError(
            f"No subspace for {target_behavior} at layer {layer_idx}. "
            f"Available: {list(subspaces[target_behavior].keys())}"
        )

    target_subspace = subspaces[target_behavior][layer_idx]

    # Determine intervention type and build steering vector
    config: dict[str, Any] = {
        "layer": layer_idx,
        "alpha": alpha,
        "target": target_behavior,
    }

    if rank is not None:
        # Low-rank steering
        vector = rank_k_steer(target_subspace, rank=rank)
        intervention_name = f"rank_{rank}"
        config["rank"] = rank
    else:
        # Full-rank steering vector
        vector = target_subspace.steering_vector.float()
        intervention_name = "full_rank"

    if orthogonal_to is not None and len(orthogonal_to) > 0:
        # Remove other behaviors' subspaces
        for remove_behavior in orthogonal_to:
            if remove_behavior not in subspaces:
                if verbose:
                    print(f"    WARNING: No subspace for {remove_behavior} to remove — skipping")
                continue
            if layer_idx not in subspaces[remove_behavior]:
                if verbose:
                    print(f"    WARNING: No subspace for {remove_behavior} at layer {layer_idx}")
                continue
            vector = orthogonal_project(
                vector,
                subspaces[remove_behavior][layer_idx],
                n_directions=5,
            )
        intervention_name = f"ortho_{'_'.join(orthogonal_to)}"
        if rank is not None:
            intervention_name = f"rank_{rank}_ortho_{'_'.join(orthogonal_to)}"
        config["orthogonal_to"] = orthogonal_to
        config["n_directions_removed"] = 5

    if verbose:
        orig_norm = target_subspace.steering_vector.norm().item()
        new_norm = vector.norm().item()
        print(f"    {intervention_name}: ||v||={orig_norm:.2f} → {new_norm:.2f} "
              f"(retained {new_norm/orig_norm:.1%})")

    # Apply steering and evaluate
    hook = SteeringHook(model, layer_idx, vector, alpha)

    rho_scores: dict[str, float] = {}
    try:
        for eval_beh in eval_behaviors:
            if eval_beh == "factual":
                probes = get_all_probes()
            else:
                probes = load_behavioral_probes(eval_beh, seed=42)

            result = evaluate_behavior(eval_beh, model, tokenizer, probes, device)
            rho_scores[eval_beh] = result["rho"]

            if verbose:
                print(f"      {eval_beh}: ρ={result['rho']:.4f}")
    finally:
        hook.remove()

    return SurgicalResult(
        intervention=intervention_name,
        target_behavior=target_behavior,
        rho_scores=rho_scores,
        config=config,
    )


def evaluate_baseline(
    model,
    tokenizer,
    eval_behaviors: list[str],
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate baseline rho scores (no steering).

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        eval_behaviors: Behaviors to evaluate.
        device: Torch device.
        verbose: Print progress.

    Returns:
        Dict mapping behavior → baseline rho.
    """
    from ..behavioral import evaluate_behavior, load_behavioral_probes
    from ..probes import get_all_probes

    if verbose:
        print("\n  Computing baselines...")

    baselines: dict[str, float] = {}
    for behavior in eval_behaviors:
        if behavior == "factual":
            probes = get_all_probes()
        else:
            probes = load_behavioral_probes(behavior, seed=42)

        result = evaluate_behavior(behavior, model, tokenizer, probes, device)
        baselines[behavior] = result["rho"]

        if verbose:
            print(f"    {behavior}: ρ={result['rho']:.4f}")

    return baselines
