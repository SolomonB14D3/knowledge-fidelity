"""Shared utilities for knowledge-fidelity toolkit."""

import gc
import sys
from typing import Optional, Tuple, Union

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


def _has_mlx() -> bool:
    """Check if MLX is available (Apple Silicon only)."""
    if sys.platform != "darwin":
        return False
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def load_model(
    model_name_or_path: str,
    device: str = "auto",
    trust_remote_code: bool = False,
    dtype=None,
) -> Tuple:
    """Load model + tokenizer, auto-selecting MLX or PyTorch backend.

    Platform priority for device="auto":
      1. MLX (Apple Silicon, if mlx + mlx-lm installed)
      2. CUDA (if torch.cuda.is_available())
      3. CPU (fallback)

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        device: "auto", "mlx", "cuda", "cpu". Default auto-detects.
        trust_remote_code: Passed to HuggingFace model loading.
        dtype: Torch dtype for PyTorch loading (e.g., torch.float16).

    Returns:
        Tuple of (model, tokenizer, backend) where backend is "mlx" or "torch".
    """
    # Resolve device
    use_mlx = False
    if device == "auto":
        use_mlx = _has_mlx()
    elif device == "mlx":
        if not _has_mlx():
            raise RuntimeError(
                "MLX requested but not available. "
                "Install with: pip install mlx mlx-lm (Apple Silicon only)"
            )
        use_mlx = True
    # "cuda" and "cpu" fall through to PyTorch

    if use_mlx:
        import mlx_lm
        model, tokenizer = mlx_lm.load(model_name_or_path)
        return model, tokenizer, "mlx"

    # PyTorch path
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    elif torch.cuda.is_available() and device in ("auto", "cuda"):
        load_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, **load_kwargs,
    )

    # Move to device
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        model = model.to("cuda")
    # Note: we don't move to MPS by default (deadlocks during training)

    return model, tokenizer, "torch"


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
