"""SAE training loop for behavioral feature disentanglement.

Trains a Gated SAE on collected model activations. Includes dead feature
detection and optional warmup scheduling.

Two entry points:
  - train_sae(): low-level, takes a pre-built SAE and ActivationData
  - train_behavioral_sae(): high-level convenience wrapper that handles
    activation collection and SAE construction
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from .schema import SAEConfig, ActivationData
from .sae import GatedSAE


def train_sae(
    sae: GatedSAE,
    activation_data: ActivationData,
    config: SAEConfig,
    verbose: bool = True,
) -> dict:
    """Train a Gated SAE on collected activations.

    Standard SAE training: MSE reconstruction + L1 sparsity on gate.
    Adam optimizer with linear warmup over first 10% of steps.
    Decoder columns are normalized to unit norm after each step.

    Args:
        sae: GatedSAE instance (will be trained in-place).
        activation_data: Collected activations to train on.
        config: Training configuration.
        verbose: Print training progress.

    Returns:
        Dict with training statistics:
            sae: Trained GatedSAE (same object, for convenience).
            final_mse: Final epoch MSE loss.
            final_l1: Final epoch L1 loss.
            final_total: Final epoch total loss.
            n_active_features: Number of non-dead features.
            dead_features: List of dead feature indices.
            steps: Total training steps.
            time: Training wall-clock time in seconds.
    """
    device = torch.device(config.device)
    sae = sae.to(device)
    sae.train()

    activations = activation_data.activations.to(device)
    dataset = TensorDataset(activations)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=False,
    )

    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)

    # Linear warmup schedule
    total_steps = config.n_epochs * len(loader)
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    t0 = time.time()
    step = 0
    epoch_mse = 0.0
    epoch_l1 = 0.0
    epoch_total = 0.0
    epoch_batches = 0

    # Track feature activations for dead feature detection
    feature_sum = torch.zeros(sae.n_features, device=device)
    feature_count = 0

    for epoch in range(config.n_epochs):
        epoch_mse = 0.0
        epoch_l1 = 0.0
        epoch_total = 0.0
        epoch_batches = 0

        # Reset dead feature tracking for last epoch
        if epoch == config.n_epochs - 1:
            feature_sum.zero_()
            feature_count = 0

        for (batch_x,) in loader:
            x_hat, z, gate_pre = sae(batch_x)
            total, mse, l1 = GatedSAE.compute_loss(
                batch_x, x_hat, gate_pre, config.sparsity_lambda,
            )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            scheduler.step()

            # Normalize decoder columns after each step
            sae.normalize_decoder()

            # Track statistics
            epoch_mse += mse.item()
            epoch_l1 += l1.item()
            epoch_total += total.item()
            epoch_batches += 1
            step += 1

            # Dead feature tracking (last epoch only)
            if epoch == config.n_epochs - 1:
                with torch.no_grad():
                    feature_sum += z.abs().mean(dim=0)
                    feature_count += 1

        # Average over epoch
        if epoch_batches > 0:
            epoch_mse /= epoch_batches
            epoch_l1 /= epoch_batches
            epoch_total /= epoch_batches

        if verbose:
            print(f"  Epoch {epoch + 1}/{config.n_epochs}: "
                  f"loss={epoch_total:.6f} (mse={epoch_mse:.6f}, l1={epoch_l1:.6f})")

    elapsed = time.time() - t0

    # Dead feature detection
    if feature_count > 0:
        feature_mean = feature_sum / feature_count
    else:
        feature_mean = feature_sum
    dead_mask = feature_mean < 1e-6
    dead_features = dead_mask.nonzero(as_tuple=True)[0].cpu().tolist()
    n_active = sae.n_features - len(dead_features)

    sae.eval()

    if verbose:
        print(f"\n  Training complete: {step} steps in {elapsed:.1f}s")
        print(f"  Active features: {n_active}/{sae.n_features} "
              f"({len(dead_features)} dead)")

    return {
        "sae": sae,
        "final_mse": epoch_mse,
        "final_l1": epoch_l1,
        "final_total": epoch_total,
        "n_active_features": n_active,
        "dead_features": dead_features,
        "steps": step,
        "time": elapsed,
    }


def train_behavioral_sae(
    model,
    tokenizer,
    behaviors: list[str],
    layer_idx: int,
    config: Optional[SAEConfig] = None,
    device: str = "cpu",
    max_probes: int | None = None,
    verbose: bool = True,
) -> tuple[GatedSAE, ActivationData, dict]:
    """Convenience wrapper: collect activations → build SAE → train.

    High-level entry point that handles the full pipeline:
    1. Collect last-token activations from behavioral contrast pairs
    2. Construct a GatedSAE with appropriate dimensions
    3. Train the SAE on the collected activations

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        behaviors: Behavior names to collect activations for.
        layer_idx: Transformer layer to capture.
        config: SAE training config (auto-configured if None).
        device: Torch device string.
        max_probes: Cap on contrast pairs per behavior.
        verbose: Print progress.

    Returns:
        Tuple of (trained_sae, activation_data, train_stats).
    """
    from .collect import collect_activations

    # Step 1: Collect activations
    if verbose:
        print("\n[1/2] Collecting activations...")
    act_data = collect_activations(
        model, tokenizer, behaviors, layer_idx,
        device=device, max_probes=max_probes, verbose=verbose,
    )

    # Step 2: Build config if not provided
    if config is None:
        config = SAEConfig(
            hidden_dim=act_data.hidden_dim,
            device=device,
        )
    elif config.hidden_dim != act_data.hidden_dim:
        raise ValueError(
            f"Config hidden_dim={config.hidden_dim} does not match "
            f"activation dim={act_data.hidden_dim}"
        )

    # Step 3: Build and train SAE
    if verbose:
        print(f"\n[2/2] Training GatedSAE "
              f"(dim={config.hidden_dim}, features={config.n_features})...")
    sae = GatedSAE(config.hidden_dim, config.expansion_factor)
    train_stats = train_sae(sae, act_data, config, verbose=verbose)

    return train_stats["sae"], act_data, train_stats
