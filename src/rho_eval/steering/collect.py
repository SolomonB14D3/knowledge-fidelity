"""Activation collection for SAE training.

Collects last-token residual stream activations from behavioral contrast
pairs, labeled by behavior and polarity. Follows the activation capture
pattern from interpretability/subspaces.py but stores activations flat
(not grouped by behavior) for efficient SAE training.

Two entry points:
  - collect_activations(): Full integration with rho-eval probe registry.
  - collect_activations_from_texts(): Standalone — takes raw text pairs,
    requires only torch + transformers (no probe registry needed).
"""

from __future__ import annotations

from typing import Optional

import torch

from .schema import ActivationData


# ── Inlined helpers (standalone, no rho_eval imports) ────────────────────


def _get_layers(model):
    """Get transformer layers from a HuggingFace causal LM.

    Supports:
      - Qwen, Llama, Mistral: model.model.layers
      - GPT-2 style: model.transformer.h
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    else:
        raise ValueError(
            f"Unknown model architecture: {type(model).__name__}. "
            "Expected model.model.layers or model.transformer.h"
        )


class _LayerCapture:
    """Lightweight activation capture for one or more layers.

    Inlined from interpretability.activation.LayerActivationCapture for
    standalone use without the full rho_eval dependency tree.
    """

    def __init__(self, model, layer_indices: list[int]):
        self._activations: dict[int, torch.Tensor] = {}
        self._hooks = []
        layers = _get_layers(model)

        for idx in layer_indices:
            hook = layers[idx].register_forward_hook(self._make_capture(idx))
            self._hooks.append(hook)

    def _make_capture(self, layer_idx: int):
        def _capture(module, input, output):
            if isinstance(output, tuple):
                self._activations[layer_idx] = output[0].detach()
            else:
                self._activations[layer_idx] = output.detach()
        return _capture

    def get(self, layer_idx: int) -> torch.Tensor:
        if layer_idx not in self._activations:
            raise KeyError(
                f"No activations for layer {layer_idx}. "
                f"Available: {sorted(self._activations.keys())}"
            )
        return self._activations[layer_idx]

    def clear(self):
        self._activations.clear()

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activations.clear()


def _build_contrast_pairs(behavior: str, probes: list[dict]) -> list[dict]:
    """Build contrast pairs from behavioral probes.

    Inlined from interpretability.activation.build_contrast_pairs for
    standalone use. Supports all behavior types.
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
        for p in probes:
            if "text" in p and "harmful_version" in p:
                pairs.append({
                    "positive": p["text"],
                    "negative": p["harmful_version"],
                    "id": p.get("id", ""),
                })

    elif behavior == "deception":
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


# ── Internal helpers ─────────────────────────────────────────────────────


def _load_probes_for_behavior(behavior: str) -> list[dict]:
    """Load probes using rho-eval's behavior plugin system.

    Supports all registered behaviors (factual, toxicity, bias,
    sycophancy, reasoning, refusal, deception, overrefusal).

    NOTE: This function uses the full rho-eval probe registry.
    For standalone use, see collect_activations_from_texts().
    """
    from ..behaviors import get_behavior
    behavior_obj = get_behavior(behavior)
    return behavior_obj.load_probes(n=999, seed=42)


def _run_pair_collection(
    model,
    tokenizer,
    pairs: list[dict],
    cap: _LayerCapture,
    layer_idx: int,
    behavior: str,
    device: str,
    verbose: bool,
) -> tuple[list[torch.Tensor], list[str], list[str]]:
    """Run forward passes on contrast pairs and collect activations.

    Shared between collect_activations() and collect_activations_from_texts().
    """
    activations = []
    labels = []
    polarities = []

    for i, pair in enumerate(pairs):
        # ── Positive example ─────────────────────────────────
        inputs = tokenizer(
            pair["positive"], return_tensors="pt",
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        model(**inputs)
        h = cap.get(layer_idx)  # (1, seq_len, hidden_dim)
        activations.append(h[0, -1, :].cpu())
        labels.append(behavior)
        polarities.append("positive")

        # ── Negative example ─────────────────────────────────
        inputs = tokenizer(
            pair["negative"], return_tensors="pt",
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        model(**inputs)
        h = cap.get(layer_idx)
        activations.append(h[0, -1, :].cpu())
        labels.append(behavior)
        polarities.append("negative")

        cap.clear()

        if verbose and (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(pairs)} pairs")

    return activations, labels, polarities


# ── Public API: Full rho-eval integration ────────────────────────────────


@torch.no_grad()
def collect_activations(
    model,
    tokenizer,
    behaviors: list[str],
    layer_idx: int,
    device: str = "cpu",
    max_probes: int | None = None,
    verbose: bool = True,
) -> ActivationData:
    """Collect last-token activations from behavioral contrast pairs.

    Uses the full rho-eval probe registry to load probes and build
    contrast pairs. For standalone use without the probe registry,
    see :func:`collect_activations_from_texts`.

    Args:
        model: HuggingFace causal LM (on device, in eval mode).
        tokenizer: Corresponding tokenizer.
        behaviors: Behavior names (e.g., ["factual", "toxicity"]).
        layer_idx: Transformer layer to capture activations from.
        device: Torch device string.
        max_probes: Cap on contrast pairs per behavior (None = use all).
        verbose: Print progress.

    Returns:
        ActivationData with all activations stacked, labeled by behavior
        and polarity.
    """
    model.eval()
    model_name = getattr(model, "name_or_path", getattr(model.config, "_name_or_path", ""))

    # Validate layer index
    n_layers = len(_get_layers(model))
    if layer_idx < 0 or layer_idx >= n_layers:
        raise ValueError(
            f"layer_idx={layer_idx} out of range for model with {n_layers} layers"
        )

    if verbose:
        print(f"  Collecting activations at layer {layer_idx} for {behaviors}")

    all_activations = []
    all_labels = []
    all_polarities = []

    cap = _LayerCapture(model, [layer_idx])

    for behavior in behaviors:
        if verbose:
            print(f"\n  [{behavior}] Loading probes...")

        probes = _load_probes_for_behavior(behavior)
        pairs = _build_contrast_pairs(behavior, probes)

        if max_probes is not None and len(pairs) > max_probes:
            import random
            rng = random.Random(42)
            pairs = rng.sample(pairs, max_probes)

        if len(pairs) < 3:
            if verbose:
                print(f"    WARNING: Only {len(pairs)} pairs — skipping")
            continue

        if verbose:
            print(f"    {len(pairs)} contrast pairs")

        acts, labs, pols = _run_pair_collection(
            model, tokenizer, pairs, cap, layer_idx, behavior, device, verbose
        )
        all_activations.extend(acts)
        all_labels.extend(labs)
        all_polarities.extend(pols)

    cap.remove()

    if not all_activations:
        raise RuntimeError("No activations collected — check behaviors and probes")

    activations = torch.stack(all_activations)  # (n_total, hidden_dim)

    if verbose:
        print(f"\n  Collected {activations.shape[0]} activations "
              f"(dim={activations.shape[1]}) from {len(behaviors)} behaviors")

    return ActivationData(
        activations=activations,
        labels=all_labels,
        polarities=all_polarities,
        layer_idx=layer_idx,
        model_name=model_name,
    )


# ── Public API: Standalone (no probe registry) ──────────────────────────


@torch.no_grad()
def collect_activations_from_texts(
    model,
    tokenizer,
    text_pairs: list[dict],
    layer_idx: int,
    device: str = "cpu",
    verbose: bool = True,
) -> ActivationData:
    """Collect activations from raw text pairs — no probe registry needed.

    This is the standalone entry point for users who want to use the
    steering module without the full rho-eval behavior system. Requires
    only torch + transformers.

    Args:
        model: HuggingFace causal LM (on device, in eval mode).
        tokenizer: Corresponding tokenizer.
        text_pairs: List of dicts, each with keys:
            - ``positive`` (str): Text representing desired behavior.
            - ``negative`` (str): Text representing undesired behavior.
            - ``behavior`` (str): Label for this pair's behavior category.
        layer_idx: Transformer layer to capture activations from.
        device: Torch device string.
        verbose: Print progress.

    Returns:
        ActivationData with all activations stacked.

    Example::

        pairs = [
            {"positive": "The Earth orbits the Sun.",
             "negative": "The Sun orbits the Earth.",
             "behavior": "factual"},
            {"positive": "I'm not sure about that claim.",
             "negative": "You're absolutely right!",
             "behavior": "sycophancy"},
        ]
        act_data = collect_activations_from_texts(
            model, tokenizer, pairs, layer_idx=17, device="cpu"
        )
    """
    model.eval()
    model_name = getattr(model, "name_or_path", getattr(model.config, "_name_or_path", ""))

    # Validate layer index
    n_layers = len(_get_layers(model))
    if layer_idx < 0 or layer_idx >= n_layers:
        raise ValueError(
            f"layer_idx={layer_idx} out of range for model with {n_layers} layers"
        )

    # Group pairs by behavior for progress reporting
    from collections import defaultdict
    by_behavior: dict[str, list[dict]] = defaultdict(list)
    for pair in text_pairs:
        beh = pair.get("behavior", "unknown")
        by_behavior[beh].append(pair)

    if verbose:
        print(f"  Collecting activations at layer {layer_idx} "
              f"for {len(text_pairs)} pairs across {list(by_behavior.keys())}")

    all_activations = []
    all_labels = []
    all_polarities = []

    cap = _LayerCapture(model, [layer_idx])

    for behavior, pairs in by_behavior.items():
        if verbose:
            print(f"\n  [{behavior}] {len(pairs)} pairs")

        acts, labs, pols = _run_pair_collection(
            model, tokenizer, pairs, cap, layer_idx, behavior, device, verbose
        )
        all_activations.extend(acts)
        all_labels.extend(labs)
        all_polarities.extend(pols)

    cap.remove()

    if not all_activations:
        raise RuntimeError("No activations collected — check text_pairs")

    activations = torch.stack(all_activations)

    if verbose:
        print(f"\n  Collected {activations.shape[0]} activations "
              f"(dim={activations.shape[1]}) from {len(by_behavior)} behaviors")

    return ActivationData(
        activations=activations,
        labels=all_labels,
        polarities=all_polarities,
        layer_idx=layer_idx,
        model_name=model_name,
    )
