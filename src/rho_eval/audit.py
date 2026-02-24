"""High-level audit orchestrator.

The main entry point for running behavioral audits on a model.

Usage:
    from rho_eval import audit

    # From model name (auto-loads from HuggingFace)
    report = audit("Qwen/Qwen2.5-7B-Instruct")

    # From pre-loaded model
    report = audit(model=model, tokenizer=tokenizer, behaviors=["factual", "bias"])

    # Specific behaviors and probe counts
    report = audit("my-model", behaviors=["factual", "toxicity"], n=50)
"""

from __future__ import annotations

import time
from typing import Optional, Union

from .behaviors import get_behavior, get_all_behaviors, list_behaviors
from .behaviors.base import BehaviorResult
from .output.schema import AuditReport
from .utils import get_device, free_memory


def audit(
    model_name_or_path: Optional[str] = None,
    *,
    model=None,
    tokenizer=None,
    behaviors: Union[str, list[str]] = "all",
    n: Optional[int] = None,
    seed: int = 42,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    dtype=None,
) -> AuditReport:
    """Run a full behavioral audit on a model.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
            Ignored if model and tokenizer are provided directly.
        model: Pre-loaded HuggingFace CausalLM (optional).
        tokenizer: Pre-loaded tokenizer (optional).
        behaviors: "all" for all registered behaviors, or a list of
            behavior names (e.g., ["factual", "toxicity"]).
        n: Number of probes per behavior. None → behavior defaults.
        seed: Random seed for probe sampling.
        device: Device string ("cpu", "cuda", "mps"). None → auto-detect.
        trust_remote_code: Passed to AutoModel.from_pretrained().
        dtype: Torch dtype for model loading (e.g., torch.float16).

    Returns:
        AuditReport with results for each requested behavior.

    Raises:
        ValueError: If no model source is provided, or unknown behavior name.

    Examples:
        >>> report = audit("Qwen/Qwen2.5-0.5B")
        >>> print(report.overall_status)
        'PASS'

        >>> report = audit(model=my_model, tokenizer=my_tok, behaviors=["factual"])
        >>> print(report.behaviors["factual"].rho)
        0.72
    """
    import torch

    t0 = time.time()

    # Resolve device
    if device is None:
        dev = get_device()
        device_str = str(dev)
    else:
        device_str = device
        dev = torch.device(device)

    # Load model if not provided
    model_id = model_name_or_path or "unknown"
    if model is None or tokenizer is None:
        if model_name_or_path is None:
            raise ValueError(
                "Either model_name_or_path or (model, tokenizer) must be provided."
            )
        model, tokenizer = _load_model(
            model_name_or_path,
            device=dev,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
        )
        model_id = model_name_or_path

    model.eval()

    # Resolve behavior list
    if behaviors == "all":
        behavior_names = list_behaviors()
    elif isinstance(behaviors, str):
        behavior_names = [behaviors]
    else:
        behavior_names = list(behaviors)

    # Create report
    report = AuditReport(
        model=model_id,
        device=device_str,
    )

    # Run each behavior
    for bname in behavior_names:
        behavior = get_behavior(bname)
        probes = behavior.load_probes(n=n, seed=seed)
        result = behavior.evaluate(model, tokenizer, probes, device=device_str)
        report.add_result(result)

    report.elapsed_seconds = time.time() - t0
    return report


def _load_model(model_name, device, trust_remote_code=False, dtype=None):
    """Load a HuggingFace model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": trust_remote_code,
    }
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    elif device.type != "cpu":
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model = model.to(device)
    model.eval()

    return model, tokenizer
