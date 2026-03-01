"""Gated Sparse Autoencoder for behavioral feature disentanglement.

Anthropic-style Gated SAE with separate gating and magnitude paths.
The gate determines which features are active (via sigmoid), while the
magnitude path determines their activation strength (via ReLU). This
separation allows the L1 sparsity penalty to be applied to the gate
logits rather than the feature activations, avoiding the shrinkage
bias inherent in vanilla L1 on ReLU outputs.

Architecture:
    gate   = sigmoid(W_gate @ x + b_gate)
    mag    = ReLU(W_mag @ x + b_mag)
    z      = gate * mag                     # sparse latent
    x_hat  = W_dec @ z + b_dec              # reconstruction
    loss   = MSE(x, x_hat) + lambda * L1(sigmoid(gate_pre))

Decoder columns are projected to unit norm after each optimizer step
(hard constraint, not regularization).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedSAE(nn.Module):
    """Gated Sparse Autoencoder.

    Args:
        hidden_dim: Input/output dimension (model residual stream width).
        expansion_factor: SAE width multiplier.
            n_features = hidden_dim * expansion_factor.
    """

    def __init__(self, hidden_dim: int, expansion_factor: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_factor = expansion_factor
        self.n_features = hidden_dim * expansion_factor

        # ── Gate path (determines which features fire) ───────────────
        self.W_gate = nn.Parameter(torch.empty(self.n_features, hidden_dim))
        self.b_gate = nn.Parameter(torch.zeros(self.n_features))

        # ── Magnitude path (determines activation strength) ──────────
        self.W_mag = nn.Parameter(torch.empty(self.n_features, hidden_dim))
        self.b_mag = nn.Parameter(torch.zeros(self.n_features))

        # ── Decoder ──────────────────────────────────────────────────
        self.W_dec = nn.Parameter(torch.empty(hidden_dim, self.n_features))
        self.b_dec = nn.Parameter(torch.zeros(hidden_dim))

        # ── Initialize ───────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform for weight matrices, zero for biases."""
        # Standard deviation for Kaiming uniform
        std = 1.0 / math.sqrt(self.hidden_dim)
        nn.init.uniform_(self.W_gate, -std, std)
        nn.init.uniform_(self.W_mag, -std, std)
        nn.init.uniform_(self.W_dec, -std, std)

        # Normalize decoder columns to unit norm
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        """Project decoder columns to unit norm (hard constraint).

        Called after each optimizer step to prevent decoder norm from
        absorbing sparsity — forces the latent activations to carry
        the actual magnitude information.
        """
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to sparse latent representation.

        Args:
            x: Input activations, shape (..., hidden_dim).

        Returns:
            z: Sparse latent activations, shape (..., n_features).
            gate_pre: Pre-sigmoid gate logits, shape (..., n_features).
                Used for L1 penalty during training.
        """
        # Gate path: determines which features are active
        gate_pre = F.linear(x, self.W_gate, self.b_gate)  # (..., n_features)
        gate = torch.sigmoid(gate_pre)

        # Magnitude path: determines activation strength
        mag = F.relu(F.linear(x, self.W_mag, self.b_mag))  # (..., n_features)

        # Sparse latent: element-wise product
        z = gate * mag  # (..., n_features)

        return z, gate_pre

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse latent back to input space.

        Args:
            z: Sparse latent activations, shape (..., n_features).

        Returns:
            x_hat: Reconstructed activations, shape (..., hidden_dim).
        """
        return F.linear(z, self.W_dec, self.b_dec)  # (..., hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → decode.

        Args:
            x: Input activations, shape (..., hidden_dim).

        Returns:
            x_hat: Reconstructed activations, shape (..., hidden_dim).
            z: Sparse latent activations, shape (..., n_features).
            gate_pre: Pre-sigmoid gate logits for L1 penalty.
        """
        z, gate_pre = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, gate_pre

    @staticmethod
    def compute_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        gate_pre: torch.Tensor,
        sparsity_lambda: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute SAE training loss.

        Loss = MSE(x, x_hat) + lambda * mean(sigmoid(gate_pre))

        The L1 penalty is applied to sigmoid(gate_pre) rather than the
        latent activations z. This avoids the shrinkage bias problem:
        with L1 on z = gate * mag, the penalty pushes both gate and mag
        toward zero, biasing magnitude estimates downward. With L1 on
        the gate alone, the magnitude path can learn accurate values.

        Args:
            x: Original activations, shape (..., hidden_dim).
            x_hat: Reconstructed activations, shape (..., hidden_dim).
            gate_pre: Pre-sigmoid gate logits, shape (..., n_features).
            sparsity_lambda: L1 penalty weight.

        Returns:
            total: Total loss (MSE + lambda * L1).
            mse: Reconstruction MSE.
            l1: Sparsity L1 penalty (before lambda weighting).
        """
        mse = F.mse_loss(x_hat, x)
        # L1 on gate probabilities (sigmoid output)
        l1 = torch.sigmoid(gate_pre).mean()
        total = mse + sparsity_lambda * l1
        return total, mse, l1

    @property
    def device(self) -> torch.device:
        return self.W_gate.device

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save SAE weights and config to a single file.

        The saved file contains both the state dict and the constructor
        arguments, so it can be loaded without knowing the original config.

        Args:
            path: File path (typically .pt or .safetensors).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "hidden_dim": self.hidden_dim,
                    "expansion_factor": self.expansion_factor,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "GatedSAE":
        """Load a saved SAE.

        Args:
            path: File saved by :meth:`save`.
            device: Device to load weights onto.

        Returns:
            Reconstructed GatedSAE with loaded weights.
        """
        data = torch.load(path, map_location=device, weights_only=False)
        sae = cls(**data["config"])
        sae.load_state_dict(data["state_dict"])
        sae = sae.to(device)
        return sae

    def __repr__(self) -> str:
        return (
            f"GatedSAE(hidden_dim={self.hidden_dim}, "
            f"expansion_factor={self.expansion_factor}, "
            f"n_features={self.n_features})"
        )
