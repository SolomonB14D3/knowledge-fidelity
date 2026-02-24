"""Core confidence extraction engine for causal language models.

Teacher-forced confidence analysis: feed text through a model in one forward
pass and measure how much probability it assigns to each actual next token.
Low confidence at specific tokens indicates the model is uncertain or holds
a belief that conflicts with the text.
"""

from __future__ import annotations

import gc
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from .schema import TokenAnalysis, ConfidenceRecord
from ..utils import get_device


# ---------------------------------------------------------------------------
# Model singleton (caches loaded model to avoid reloading between calls)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_device = None
_current_model_name = None
_current_revision = None


def load_model(
    model_name: str = "EleutherAI/pythia-160m",
    revision: str = "main",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple:
    """Load model and tokenizer (cached singleton).

    Reloads if model_name or revision changes.
    """
    global _model, _tokenizer, _device, _current_model_name, _current_revision

    if (_model is not None
            and _current_model_name == model_name
            and _current_revision == revision):
        return _model, _tokenizer, _device

    if _model is not None:
        unload_model()

    _device = device or get_device()
    use_dtype = dtype or torch.float32
    print(f"Loading {model_name} (revision={revision}, dtype={use_dtype}) on {_device}...")

    _tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    try:
        _model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=use_dtype,
        ).to(_device)
    except OSError:
        _model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=use_dtype,
            use_safetensors=False,
        ).to(_device)
    _model.eval()
    _current_revision = revision
    _current_model_name = model_name

    print(f"  Loaded. Vocab={_tokenizer.vocab_size}, Device={_device}")
    return _model, _tokenizer, _device


def unload_model():
    """Free the cached model."""
    global _model, _tokenizer, _device, _current_model_name, _current_revision
    _model = None
    _tokenizer = None
    _device = None
    _current_model_name = None
    _current_revision = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Teacher-forced confidence analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_confidence(
    text: str,
    category: str = "",
    label: str = "",
    model_name: str = "EleutherAI/pythia-160m",
    revision: str = "main",
    top_k: int = 5,
    dtype: Optional[torch.dtype] = None,
    model=None,
    tokenizer=None,
    device=None,
) -> ConfidenceRecord:
    """Teacher-forced confidence analysis of fixed text.

    Feed the entire text through the model in one forward pass. At each
    position t, the model outputs a distribution over the next token. We
    measure how much probability it assigns to the ACTUAL token at t+1.

    Args:
        text: The text to analyze
        category: Category label (e.g., "true_fact", "false_belief")
        label: Short identifier (e.g., "capital_france")
        model_name: HuggingFace model name (ignored if model is provided)
        revision: Model revision
        top_k: Number of top predictions to record per position
        dtype: Model dtype (default: float32)
        model: Pre-loaded model (optional, skips loading)
        tokenizer: Pre-loaded tokenizer (optional)
        device: Device (optional)

    Returns:
        ConfidenceRecord with per-token analysis
    """
    if model is None or tokenizer is None:
        model, tokenizer, device = load_model(model_name, revision, dtype=dtype)
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    if seq_len < 2:
        raise ValueError(f"Text too short ({seq_len} tokens): '{text}'")

    outputs = model(**inputs)
    logits = outputs.logits

    # logits[0, t, :] predicts token at position t+1
    logits_cpu = logits[0, :-1, :].cpu().float()
    probs = torch.softmax(logits_cpu, dim=-1)
    target_ids = input_ids[0, 1:].cpu()

    token_analyses = []
    top1_probs_list = []
    entropies_list = []

    for t in range(seq_len - 1):
        target_id = target_ids[t].item()
        target_str = tokenizer.decode([target_id])
        prob_dist = probs[t]

        actual_prob = prob_dist[target_id].item()
        sorted_indices = torch.argsort(prob_dist, descending=True)
        rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item()
        entropy = -(prob_dist * torch.log2(prob_dist + 1e-12)).sum().item()

        topk_probs_t, topk_ids_t = torch.topk(prob_dist, k=top_k)
        topk_strs = [tokenizer.decode([tid]) for tid in topk_ids_t.tolist()]

        ta = TokenAnalysis(
            position=t,
            token_id=target_id,
            token_str=target_str,
            top1_prob=actual_prob,
            top1_rank=rank,
            entropy=entropy,
            top5_tokens=topk_strs,
            top5_probs=topk_probs_t.tolist(),
            top5_ids=topk_ids_t.tolist(),
        )
        token_analyses.append(ta)
        top1_probs_list.append(actual_prob)
        entropies_list.append(entropy)

    probs_arr = np.array(top1_probs_list)
    ent_arr = np.array(entropies_list)
    min_idx = int(np.argmin(probs_arr))

    used_model_name = model_name
    if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
        used_model_name = model.config._name_or_path

    return ConfidenceRecord(
        text=text,
        category=category,
        label=label,
        mode="fixed",
        num_tokens=seq_len,
        tokens=token_analyses,
        mean_top1_prob=float(probs_arr.mean()),
        mean_entropy=float(ent_arr.mean()),
        std_top1_prob=float(probs_arr.std()),
        std_entropy=float(ent_arr.std()),
        min_confidence_pos=min_idx,
        min_confidence_token=token_analyses[min_idx].token_str,
        min_confidence_value=float(probs_arr[min_idx]),
        model_name=used_model_name,
        model_revision=revision,
    )
