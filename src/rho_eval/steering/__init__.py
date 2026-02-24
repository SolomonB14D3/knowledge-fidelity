"""SAE-based behavioral steering for LLMs.

Train a Gated Sparse Autoencoder on model activations to discover
disentangled behavioral features, then steer individual features
without the collateral damage caused by SVD-based linear steering.

Quick start:
    from rho_eval.steering import train_behavioral_sae, identify_behavioral_features
    from rho_eval.steering import steer_features, evaluate_sae_steering

    # Train SAE on behavioral activations
    sae, act_data, stats = train_behavioral_sae(
        model, tokenizer,
        behaviors=["factual", "toxicity", "sycophancy", "bias"],
        layer_idx=17, device="cpu",
    )

    # Identify which features encode which behaviors
    reports, features = identify_behavioral_features(sae, act_data)

    # Steer specific behavioral features
    hook = steer_features(model, sae, layer_idx=17,
                          feature_indices=features["sycophancy"], scale=2.0)
    model.generate(...)
    hook.remove()
"""

# ── Schema ───────────────────────────────────────────────────────────────
from .schema import (
    SAEConfig,
    ActivationData,
    FeatureReport,
    SAESteeringReport,
)

# ── SAE Architecture ─────────────────────────────────────────────────────
from .sae import GatedSAE

# ── Activation Collection ────────────────────────────────────────────────
from .collect import collect_activations

# ── Training ─────────────────────────────────────────────────────────────
from .train import train_sae, train_behavioral_sae

# ── Analysis ─────────────────────────────────────────────────────────────
from .analyze import identify_behavioral_features, feature_overlap_matrix

# ── Steering ─────────────────────────────────────────────────────────────
from .steer import SAESteeringHook, steer_features, evaluate_sae_steering


__all__ = [
    # Schema
    "SAEConfig",
    "ActivationData",
    "FeatureReport",
    "SAESteeringReport",
    # SAE
    "GatedSAE",
    # Collection
    "collect_activations",
    # Training
    "train_sae",
    "train_behavioral_sae",
    # Analysis
    "identify_behavioral_features",
    "feature_overlap_matrix",
    # Steering
    "SAESteeringHook",
    "steer_features",
    "evaluate_sae_steering",
]
