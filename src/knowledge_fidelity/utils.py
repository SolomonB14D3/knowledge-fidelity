"""Shared utilities for knowledge-fidelity toolkit."""

import gc
import torch
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths (relative to project root, resolved at import time)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
PROBES_DIR = DATA_DIR / "probes"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = _PROJECT_ROOT / "figures"


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [RESULTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Pick the best available device: MPS > CUDA > CPU.

    Note: MPS has matmul issues with some models (Qwen) and NaN gradients
    when training with frozen layers. Use CPU for training, MPS for inference.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def free_memory():
    """Free GPU/MPS memory. Call between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Model architecture helpers
# ---------------------------------------------------------------------------

def get_layers(model):
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


def get_attention(layer):
    """Get the attention module from a transformer layer."""
    if hasattr(layer, 'self_attn'):
        return layer.self_attn
    if hasattr(layer, 'attn'):
        return layer.attn
    return None
