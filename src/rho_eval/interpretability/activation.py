"""Activation capture utilities for mechanistic interpretability.

Provides hook-based extraction of residual stream activations and per-head
attention outputs. Also includes contrast pair construction for behavioral
subspace analysis.

This is a library-level promotion of the ActivationCapture pattern from
experiments/steering_vectors.py, extended with multi-layer capture and
per-head output extraction.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..utils import get_layers, get_attention


# ── Multi-Layer Residual Stream Capture ──────────────────────────────────


class LayerActivationCapture:
    """Capture residual stream activations from multiple layers simultaneously.

    Registers forward hooks on multiple transformer layers and captures
    their output hidden states in a single forward pass.

    Usage:
        cap = LayerActivationCapture(model, [7, 17, 24])
        model(**inputs)
        h17 = cap.get(17)         # (batch, seq_len, hidden_dim)
        all_h = cap.get_all()     # {7: tensor, 17: tensor, 24: tensor}
        cap.remove()
    """

    def __init__(self, model, layer_indices: list[int]):
        self._activations: dict[int, torch.Tensor] = {}
        self._hooks = []
        layers = get_layers(model)

        for idx in layer_indices:
            # Use a factory to avoid late-binding closure bug
            hook = layers[idx].register_forward_hook(self._make_capture(idx))
            self._hooks.append(hook)

    def _make_capture(self, layer_idx: int):
        """Create a capture closure for a specific layer index."""
        def _capture(module, input, output):
            if isinstance(output, tuple):
                self._activations[layer_idx] = output[0].detach()
            else:
                self._activations[layer_idx] = output.detach()
        return _capture

    def get(self, layer_idx: int) -> torch.Tensor:
        """Return captured activations for one layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Tensor of shape (batch, seq_len, hidden_dim).

        Raises:
            KeyError: If layer was not captured or no forward pass ran.
        """
        if layer_idx not in self._activations:
            raise KeyError(
                f"No activations for layer {layer_idx}. "
                f"Available: {sorted(self._activations.keys())}"
            )
        return self._activations[layer_idx]

    def get_all(self) -> dict[int, torch.Tensor]:
        """Return all captured activations.

        Returns:
            Dict mapping layer_idx → tensor (batch, seq_len, hidden_dim).
        """
        return dict(self._activations)

    def clear(self):
        """Clear cached activations (hooks remain active)."""
        self._activations.clear()

    def remove(self):
        """Remove all hooks and free cached activations."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activations.clear()


# ── Per-Head Output Capture ──────────────────────────────────────────────


class HeadOutputCapture:
    """Capture individual attention head outputs from a layer.

    Hooks the output projection's input to capture per-head activations
    before they are projected back to the residual dimension.

    For Qwen/Llama/Mistral: hooks self_attn.o_proj's input, which has
    shape (batch, seq_len, n_heads * head_dim). This is reshaped to
    (batch, n_heads, seq_len, head_dim).

    Usage:
        cap = HeadOutputCapture(model, layer_idx=17, n_heads=28, head_dim=128)
        model(**inputs)
        heads = cap.get()  # (batch, n_heads, seq_len, head_dim)
        cap.remove()
    """

    def __init__(self, model, layer_idx: int, n_heads: int, head_dim: int):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self._head_outputs: Optional[torch.Tensor] = None

        layers = get_layers(model)
        attn = get_attention(layers[layer_idx])
        if attn is None:
            raise ValueError(f"No attention module found at layer {layer_idx}")

        # Hook the o_proj input (which is the concatenated head outputs)
        if hasattr(attn, 'o_proj'):
            self._hook = attn.o_proj.register_forward_hook(self._capture)
        else:
            raise ValueError(
                f"Attention module at layer {layer_idx} has no 'o_proj'. "
                f"Available: {[n for n, _ in attn.named_children()]}"
            )

    def _capture(self, module, input, output):
        # input[0] is the concatenated head outputs: (batch, seq_len, n_heads * head_dim)
        x = input[0].detach()
        batch, seq_len, _ = x.shape
        # Reshape to (batch, n_heads, seq_len, head_dim)
        self._head_outputs = x.view(batch, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def get(self) -> torch.Tensor:
        """Return per-head outputs.

        Returns:
            Tensor of shape (batch, n_heads, seq_len, head_dim).

        Raises:
            RuntimeError: If no forward pass has been run.
        """
        if self._head_outputs is None:
            raise RuntimeError("No head outputs captured. Run a forward pass first.")
        return self._head_outputs

    def remove(self):
        """Remove the hook and free cached outputs."""
        self._hook.remove()
        self._head_outputs = None


# ── Steering Hook (library-level) ────────────────────────────────────────


class SteeringHook:
    """Apply a steering vector during forward pass.

    Adds a scaled vector to the residual stream at a specific layer,
    applied at all token positions (CAA-style: Rimsky et al., 2023).

    Usage:
        hook = SteeringHook(model, layer_idx=17, vector=v, alpha=4.0)
        model.generate(...)  # steered generation
        hook.remove()
    """

    def __init__(self, model, layer_idx: int, vector: torch.Tensor, alpha: float = 1.0):
        self.vector = vector
        self.alpha = alpha
        layers = get_layers(model)
        self._hook = layers[layer_idx].register_forward_hook(self._steer)

    def _steer(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + self.alpha * self.vector.to(h.device, h.dtype)
            return (h,) + output[1:]
        else:
            return output + self.alpha * self.vector.to(output.device, output.dtype)

    def remove(self):
        """Remove the hook."""
        self._hook.remove()


# ── Contrast Pair Construction ───────────────────────────────────────────


def build_contrast_pairs(behavior: str, probes: list[dict]) -> list[dict]:
    """Build contrast pairs from behavioral probes.

    Each pair has a 'positive' text (desired behavior) and 'negative' text
    (undesired behavior). The specific construction depends on the behavior.

    Supported behaviors: factual, sycophancy, bias, toxicity.

    Args:
        behavior: Name of the behavior.
        probes: List of probe dicts from the probe loader.

    Returns:
        List of dicts with keys: positive, negative, id.

    Raises:
        ValueError: If behavior is not supported for contrast pair construction.
    """
    pairs = []

    if behavior == "factual":
        for p in probes:
            if "text" in p and "false" in p:
                pairs.append({
                    "positive": p["text"],
                    "negative": p["false"],
                    "id": p.get("id", ""),
                })

    elif behavior == "sycophancy":
        for p in probes:
            text = p["text"]
            pairs.append({
                "positive": f"{text}\n{p['truthful_answer']}",
                "negative": f"{text}\n{p['sycophantic_answer']}",
                "id": p.get("id", ""),
            })

    elif behavior == "bias":
        for p in probes:
            text = p["text"]
            pairs.append({
                "positive": f"{text} {p['correct_answer']}",
                "negative": f"{text} {p['biased_answer']}",
                "id": p.get("id", ""),
            })

    elif behavior == "toxicity":
        benign = [p for p in probes if not p.get("is_toxic", False)]
        toxic = [p for p in probes if p.get("is_toxic", False)]
        n = min(len(benign), len(toxic))
        for i in range(n):
            pairs.append({
                "positive": benign[i]["text"],
                "negative": toxic[i]["text"],
                "id": f"tox_pair_{i}",
            })

    elif behavior == "refusal":
        # positive = benign request + helpful response (model should be confident)
        # negative = harmful request + compliant response (model should not)
        for p in probes:
            if "text" in p and "harmful_version" in p:
                pairs.append({
                    "positive": p["text"],
                    "negative": p["harmful_version"],
                    "id": p.get("id", ""),
                })

    elif behavior == "deception":
        # positive = honest/helpful response (model should be confident)
        # negative = deceptive/evasive response (model should not)
        for p in probes:
            if "honest" in p and "deceptive" in p:
                pairs.append({
                    "positive": p["honest"],
                    "negative": p["deceptive"],
                    "id": p.get("id", ""),
                })

    else:
        raise ValueError(
            f"No contrast pair construction for behavior: {behavior}. "
            f"Supported: factual, sycophancy, bias, toxicity, refusal, deception"
        )

    return pairs
