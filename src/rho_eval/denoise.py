"""Denoise mode: find the compression ratio that maximizes factual signal.

SVD compression can sometimes *improve* a model's ability to discriminate
truth from myth by stripping noise from attention projections. This module
finds the optimal ratio for that effect.

    from rho_eval import find_optimal_denoise_ratio

    result = find_optimal_denoise_ratio("Qwen/Qwen2.5-7B-Instruct")
    print(f"Best ratio: {result['optimal_ratio']} -> rho {result['optimal_rho']:.3f}")
"""

import copy
import time
import torch
from typing import Optional

from .svd.compress import compress_qko
from .core import _run_audit
from .probes import get_mandela_probes, get_medical_probes, get_default_probes


PROBE_SET_MAP = {
    "mandela": get_mandela_probes,
    "medical": get_medical_probes,
    "default": get_default_probes,
}


def find_optimal_denoise_ratio(
    model_name_or_path: str,
    probe_set: str = "mandela",
    ratios: Optional[list[float]] = None,
    freeze_ratio: float = 0.75,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> dict:
    """Find the compression ratio that maximizes factual signal on a probe set.

    Loads the model once, then for each candidate ratio:
      1. Restores original weights
      2. Compresses Q/K/O at that ratio
      3. Audits confidence on the chosen probe set
      4. Records rho

    Returns the ratio with the highest rho (best truth/myth discrimination).

    Args:
        model_name_or_path: HuggingFace model name or local path
        probe_set: Which probes to optimize for ("mandela", "medical", "default")
        ratios: List of ratios to test (default: [0.5, 0.6, 0.7, 0.8, 0.9])
        freeze_ratio: Layer freeze ratio for each attempt
        device: Device to use
        dtype: Model dtype (default: float32)

    Returns:
        Dict with:
          - optimal_ratio: float, best ratio found
          - optimal_rho: float, rho at optimal ratio
          - baseline_rho: float, rho with no compression
          - improvement: float, optimal_rho - baseline_rho (positive = denoising helped)
          - denoising_detected: bool, True if any ratio improved over baseline
          - all_results: list of {ratio, rho, rho_p, n_positive, retention} dicts
          - model: the model compressed at optimal ratio
          - tokenizer: the tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .svd.freeze import freeze_layers

    if ratios is None:
        ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    use_dtype = dtype or torch.float32

    # Get probes
    if probe_set in PROBE_SET_MAP:
        probes = PROBE_SET_MAP[probe_set]()
    else:
        raise ValueError(f"Unknown probe_set '{probe_set}'. Use: {list(PROBE_SET_MAP.keys())}")

    start_time = time.time()

    # Load model once
    print(f"[denoise] Loading {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=use_dtype, trust_remote_code=True,
    ).to(device)
    model.eval()

    # Save original state dict
    original_state = copy.deepcopy(model.state_dict())

    # Baseline (no compression)
    print(f"[denoise] Measuring baseline ({probe_set} probes)...")
    baseline_audit = _run_audit(model, tokenizer, probes, device)
    baseline_rho = baseline_audit["rho"]
    print(f"  Baseline rho: {baseline_rho:.3f}")

    # Test each ratio
    all_results = []
    best_rho = baseline_rho
    best_ratio = 1.0  # no compression

    for ratio in sorted(ratios):
        print(f"[denoise] Testing ratio={ratio:.0%}...")

        # Restore original weights
        model.load_state_dict(original_state)
        model.eval()

        # Compress
        n_compressed = compress_qko(model, ratio=ratio)
        freeze_layers(model, ratio=freeze_ratio)
        model.eval()

        # Audit
        audit = _run_audit(model, tokenizer, probes, device)
        rho = audit["rho"]

        # Compute retention vs baseline
        n_retained = 0
        for i in range(len(probes)):
            delta_base = baseline_audit["true_confs"][i] - baseline_audit["false_confs"][i]
            delta_now = audit["true_confs"][i] - audit["false_confs"][i]
            if delta_base > 0 and delta_now > 0:
                n_retained += 1
            elif delta_base <= 0:
                n_retained += 1
        retention = n_retained / len(probes) if probes else 0.0

        improved = rho > baseline_rho
        marker = " ** IMPROVED **" if improved else ""
        print(f"  ratio={ratio:.0%}: rho={rho:.3f} (retention={retention:.0%}){marker}")

        result = {
            "ratio": ratio,
            "rho": rho,
            "rho_p": audit["rho_p"],
            "n_positive": audit["n_positive_delta"],
            "n_probes": audit["n_probes"],
            "retention": retention,
            "n_compressed": n_compressed,
            "improved_over_baseline": improved,
        }
        all_results.append(result)

        if rho > best_rho:
            best_rho = rho
            best_ratio = ratio

    # Compress model at optimal ratio for return
    model.load_state_dict(original_state)
    model.eval()
    if best_ratio < 1.0:
        compress_qko(model, ratio=best_ratio)
        freeze_layers(model, ratio=freeze_ratio)
        model.eval()

    denoising_detected = best_rho > baseline_rho
    improvement = best_rho - baseline_rho
    elapsed = time.time() - start_time

    print(f"\n[denoise] {'='*50}")
    if denoising_detected:
        print(f"  DENOISING DETECTED on {probe_set} probes!")
        print(f"  Optimal ratio: {best_ratio:.0%} -> rho {best_rho:.3f} "
              f"(+{improvement:.3f} vs baseline {baseline_rho:.3f})")
    else:
        print(f"  No denoising effect found on {probe_set} probes.")
        print(f"  Best ratio: {best_ratio:.0%} (rho={best_rho:.3f}, "
              f"baseline={baseline_rho:.3f})")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"[denoise] {'='*50}")

    return {
        "optimal_ratio": best_ratio,
        "optimal_rho": best_rho,
        "baseline_rho": baseline_rho,
        "improvement": improvement,
        "denoising_detected": denoising_detected,
        "probe_set": probe_set,
        "all_results": all_results,
        "model": model,
        "tokenizer": tokenizer,
        "elapsed_seconds": elapsed,
    }
