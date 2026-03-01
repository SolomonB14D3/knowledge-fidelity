"""SAE-based behavioral steering for LLMs.

Train a Gated Sparse Autoencoder on model activations to discover
disentangled behavioral features, then steer individual features
without the collateral damage caused by SVD-based linear steering.

Quick start (full rho-eval integration):
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

Standalone usage (no probe registry needed, just torch + transformers):
    from rho_eval.steering import GatedSAE, collect_activations_from_texts, steer

    # 1. Prepare contrast pairs (raw text)
    pairs = [
        {"positive": "The Earth orbits the Sun.",
         "negative": "The Sun orbits the Earth.",
         "behavior": "factual"},
    ]

    # 2. Collect activations
    act_data = collect_activations_from_texts(model, tokenizer, pairs, layer_idx=17)

    # 3. Train SAE
    from rho_eval.steering import train_sae
    sae, stats = train_sae(act_data.activations, hidden_dim=model.config.hidden_size)

    # 4. Steer
    hook = steer(model, sae, layer_idx=17, feature_indices=[42], scale=2.0)
    model.generate(...)
    hook.remove()

    # 5. Save/load
    sae.save("my_sae.pt")
    sae = GatedSAE.load("my_sae.pt")
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
from .collect import collect_activations, collect_activations_from_texts

# ── Training ─────────────────────────────────────────────────────────────
from .train import train_sae, train_behavioral_sae

# ── Analysis ─────────────────────────────────────────────────────────────
from .analyze import identify_behavioral_features, feature_overlap_matrix

# ── Steering ─────────────────────────────────────────────────────────────
from .steer import SAESteeringHook, steer_features, evaluate_sae_steering

# ── Convenience alias ────────────────────────────────────────────────────
steer = steer_features  # one-word entry point


__all__ = [
    # Schema
    "SAEConfig",
    "ActivationData",
    "FeatureReport",
    "SAESteeringReport",
    # SAE
    "GatedSAE",
    # Collection (full + standalone)
    "collect_activations",
    "collect_activations_from_texts",
    # Training
    "train_sae",
    "train_behavioral_sae",
    # Analysis
    "identify_behavioral_features",
    "feature_overlap_matrix",
    # Steering
    "SAESteeringHook",
    "steer_features",
    "steer",
    "evaluate_sae_steering",
]
