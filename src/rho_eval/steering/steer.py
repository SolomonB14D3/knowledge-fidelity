"""SAE-based behavioral steering.

Instead of adding a raw steering vector to the residual stream (SVD approach),
SAE steering works in the feature space:
  1. Encode the residual stream through the SAE to get sparse features
  2. Scale specific behavioral features up or down
  3. Decode back to the residual stream and replace the activation

This allows targeting individual behavioral features while leaving other
features (including those from entangled behaviors) untouched.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..utils import get_layers
from .schema import ActivationData, FeatureReport, SAESteeringReport, SAEConfig
from .sae import GatedSAE


# ── SAE Steering Hook ────────────────────────────────────────────────────


class SAESteeringHook:
    """Apply behavioral steering through the SAE feature space.

    Unlike SteeringHook (which adds a vector directly), this hook:
    1. Encodes the activation through the SAE
    2. Scales specific features by a multiplicative factor
    3. Decodes back and replaces the activation

    This is the key mechanism for disentangled steering — modifying
    sycophancy features without touching the bias features that share
    the same subspace in linear SVD analysis.

    Usage:
        hook = SAESteeringHook(model, sae, layer_idx=17,
                               feature_indices=[42, 87, 103], scale=2.0)
        model.generate(...)  # steered generation
        hook.remove()
    """

    def __init__(
        self,
        model,
        sae: GatedSAE,
        layer_idx: int,
        feature_indices: list[int],
        scale: float = 2.0,
    ):
        self.sae = sae
        self.feature_indices = feature_indices
        self.scale = scale
        self._sae_device = sae.device

        layers = get_layers(model)
        self._hook = layers[layer_idx].register_forward_hook(self._steer)

    def _steer(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        # Move SAE to activation device if needed
        sae_device = self._sae_device
        h_device = h.device
        h_dtype = h.dtype

        # Encode through SAE (in float32 for stability)
        h_float = h.float()
        if sae_device != h_device:
            self.sae = self.sae.to(h_device)
            self._sae_device = h_device

        with torch.no_grad():
            z, _ = self.sae.encode(h_float)  # (..., n_features)

            # Scale target features
            for idx in self.feature_indices:
                z[..., idx] = z[..., idx] * self.scale

            # Decode back
            h_new = self.sae.decode(z)  # (..., hidden_dim)

        h_new = h_new.to(h_dtype)

        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    def remove(self):
        """Remove the hook."""
        self._hook.remove()


# ── Convenience Functions ────────────────────────────────────────────────


def steer_features(
    model,
    sae: GatedSAE,
    layer_idx: int,
    feature_indices: list[int],
    scale: float = 2.0,
) -> SAESteeringHook:
    """Attach an SAE steering hook to the model.

    Convenience wrapper around SAESteeringHook.

    Args:
        model: HuggingFace causal LM.
        sae: Trained GatedSAE.
        layer_idx: Layer to intercept.
        feature_indices: SAE feature indices to modify.
        scale: Multiplicative factor for target features.
            >1.0 amplifies, <1.0 suppresses, 0.0 ablates.

    Returns:
        SAESteeringHook (call .remove() when done).
    """
    return SAESteeringHook(model, sae, layer_idx, feature_indices, scale)


def evaluate_sae_steering(
    model,
    tokenizer,
    sae: GatedSAE,
    behavioral_features: dict[str, list[int]],
    target_behavior: str,
    layer_idx: int,
    eval_behaviors: list[str],
    scales: list[float] | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> list[dict]:
    """Evaluate SAE steering across multiple scale factors.

    For each scale factor, attaches an SAESteeringHook that modifies
    the target behavior's features, then evaluates all behaviors via
    the rho-eval audit system. Records rho scores and collateral
    damage (changes in non-target behaviors).

    This is the key comparison function: SAE steering should show
    less collateral damage than SVD steering.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        sae: Trained GatedSAE.
        behavioral_features: {behavior: [feature_indices]} from
            identify_behavioral_features().
        target_behavior: Behavior to steer.
        layer_idx: Layer to apply steering.
        eval_behaviors: Behaviors to evaluate.
        scales: Scale factors to test (default: [0.0, 1.5, 2.0, 3.0, 4.0]).
        device: Torch device.
        verbose: Print progress.

    Returns:
        List of result dicts, one per scale:
            scale: Scale factor applied.
            rho_scores: {behavior: rho} from evaluation.
            collateral: {behavior: delta_from_baseline} for non-target behaviors.
    """
    from ..behavioral import evaluate_behavior, load_behavioral_probes
    from ..probes import get_all_probes

    if scales is None:
        scales = [0.0, 1.5, 2.0, 3.0, 4.0]

    if target_behavior not in behavioral_features:
        raise ValueError(
            f"No features for {target_behavior}. "
            f"Available: {list(behavioral_features.keys())}"
        )

    feature_indices = behavioral_features[target_behavior]
    if not feature_indices:
        raise ValueError(f"No features assigned to {target_behavior}")

    if verbose:
        print(f"\n  SAE steering: {target_behavior} at layer {layer_idx}")
        print(f"  Features: {len(feature_indices)} features")
        print(f"  Scales: {scales}")

    # ── Evaluate baseline (scale=1.0, i.e., no modification) ─────
    if verbose:
        print(f"\n  [baseline] Evaluating...")

    baseline_scores: dict[str, float] = {}
    for eval_beh in eval_behaviors:
        if eval_beh == "factual":
            probes = get_all_probes()
        else:
            probes = load_behavioral_probes(eval_beh, seed=42)
        result = evaluate_behavior(eval_beh, model, tokenizer, probes, device)
        baseline_scores[eval_beh] = result["rho"]
        if verbose:
            print(f"    {eval_beh}: rho={result['rho']:.4f}")

    # ── Evaluate at each scale ───────────────────────────────────
    results = []

    for scale in scales:
        if verbose:
            print(f"\n  [scale={scale:.1f}] Steering {len(feature_indices)} features...")

        hook = SAESteeringHook(model, sae, layer_idx, feature_indices, scale)

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
                    delta = result["rho"] - baseline_scores.get(eval_beh, 0)
                    marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else "=")
                    print(f"    {eval_beh}: rho={result['rho']:.4f} ({delta:+.4f}) {marker}")
        finally:
            hook.remove()

        # Compute collateral damage
        collateral = {}
        for beh in eval_behaviors:
            if beh != target_behavior:
                collateral[beh] = rho_scores.get(beh, 0) - baseline_scores.get(beh, 0)

        results.append({
            "scale": scale,
            "rho_scores": rho_scores,
            "baseline_scores": baseline_scores,
            "collateral": collateral,
            "n_features": len(feature_indices),
            "feature_indices": feature_indices[:20],  # truncate for JSON
        })

    return results
