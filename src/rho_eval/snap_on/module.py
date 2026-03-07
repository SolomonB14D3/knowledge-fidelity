"""Snap-On Communication Module architectures.

A tiny adapter that learns to adjust frozen base model outputs to produce
instruction-following behavior.

Modes:
  hidden:  Adapter operates on hidden states (h).
           logits = lm_head(h + adapter(h))
  logit:   Adapter operates on logits (post-lm_head).
           logits = lm_head(h) + adapter(lm_head(h))
           Knowledge pathway (lm_head(h)) is never perturbed.

Architecture Options:
  MLP  (n_layers=0):  SwiGLU adapter
  Transformer (n_layers>0):  down-proj -> N causal transformer blocks -> up-proj
"""

import json
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, asdict, field
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SnapOnConfig:
    """Configuration for a snap-on adapter."""
    d_model: int = 3584        # Base model hidden_size
    d_inner: int = 1024        # Adapter internal dimension
    n_layers: int = 0          # 0 = MLP only, >0 = transformer layers
    n_heads: int = 8           # Attention heads (transformer variant)
    mode: str = "hidden"       # "hidden" or "logit"
    vocab_size: int = 0        # Required for logit mode

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SnapOnConfig":
        with open(path) as f:
            data = json.load(f)
            # Backward compat: old configs don't have mode/vocab_size
            return cls(**{k: v for k, v in data.items()
                         if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Hidden-space MLP Adapter  (~11M params at d_inner=1024, d_model=3584)
# ---------------------------------------------------------------------------

class SnapOnMLP(nn.Module):
    """SwiGLU MLP adapter with zero-initialized output.

    h_base  ->  [gate * silu(up) -> down]  ->  adjustment
    logits  =  lm_head(h_base + adjustment)

    At init, down_proj.weight is zero -> adjustment is zero -> pure base model.
    """

    def __init__(self, config: SnapOnConfig):
        super().__init__()
        self.config = config
        d, di = config.d_model, config.d_inner
        self.gate_proj = nn.Linear(d, di, bias=False)
        self.up_proj   = nn.Linear(d, di, bias=False)
        self.down_proj = nn.Linear(di, d, bias=False)
        # Zero-init output -> identity at start
        self.down_proj.weight = mx.zeros((d, di))

    def __call__(self, h: mx.array) -> mx.array:
        """Return adjustment to add to h.  Shape: same as h."""
        return self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))


# ---------------------------------------------------------------------------
# Logit-space MLP Adapter  (~29M params at d_inner=64, vocab=152064)
# ---------------------------------------------------------------------------

class SnapOnLogitMLP(nn.Module):
    """SwiGLU MLP adapter operating in logit space.

    base_logits = lm_head(h)                      # knowledge preserved
    logit_adj   = adapter(base_logits)             # learned distribution reshape
    final       = base_logits + logit_adj

    Like a learned version of contrastive decoding: operates purely on the
    output distribution, never perturbing the hidden->logit knowledge pathway.

    At init, down_proj.weight is zero -> logit_adj is zero -> pure base model.
    """

    def __init__(self, config: SnapOnConfig):
        super().__init__()
        self.config = config
        v, di = config.vocab_size, config.d_inner
        assert v > 0, "vocab_size must be set for logit mode"
        self.gate_proj = nn.Linear(v, di, bias=False)
        self.up_proj   = nn.Linear(v, di, bias=False)
        self.down_proj = nn.Linear(di, v, bias=False)
        # Zero-init output -> no adjustment at start
        self.down_proj.weight = mx.zeros((v, di))

    def __call__(self, logits: mx.array) -> mx.array:
        """Return logit adjustment. Shape: same as logits."""
        return self.down_proj(nn.silu(self.gate_proj(logits)) * self.up_proj(logits))


# ---------------------------------------------------------------------------
# Transformer Adapter  (~12M params at d_inner=512, n_layers=2)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm causal transformer block with SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn  = nn.MultiHeadAttention(d_model, n_heads, bias=False)
        self.norm2 = nn.RMSNorm(d_model)
        d_ff = d_model * 4
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up   = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Self-attention with pre-norm
        r = x
        x = self.norm1(x)
        x = self.attn(x, x, x, mask=mask)
        x = r + x
        # FFN (SwiGLU) with pre-norm
        r = x
        x = self.norm2(x)
        x = self.ffn_down(nn.silu(self.ffn_gate(x)) * self.ffn_up(x))
        return r + x


class SnapOnTransformer(nn.Module):
    """Transformer adapter: project down -> N blocks -> project up.

    More expressive than MLP -- can attend across sequence positions.
    Zero-initialized output for stable training start.
    """

    def __init__(self, config: SnapOnConfig):
        super().__init__()
        self.config = config
        d, di = config.d_model, config.d_inner
        self.proj_in = nn.Linear(d, di, bias=False)
        self.layers  = [TransformerBlock(di, config.n_heads)
                        for _ in range(config.n_layers)]
        self.norm    = nn.RMSNorm(di)
        self.proj_out = nn.Linear(di, d, bias=False)
        # Zero-init output
        self.proj_out.weight = mx.zeros((d, di))

    def __call__(self, h: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = self.proj_in(h)
        if mask is None and h.ndim >= 2 and h.shape[-2] > 1:
            L = h.shape[-2]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            mask = mask.astype(x.dtype)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.proj_out(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_adapter(config: SnapOnConfig) -> nn.Module:
    """Create an adapter from config."""
    if config.mode == "logit":
        return SnapOnLogitMLP(config)
    if config.n_layers == 0:
        return SnapOnMLP(config)
    return SnapOnTransformer(config)
