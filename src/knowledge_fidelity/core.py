"""Core unified API for knowledge-fidelity.

The main entry point: compress a model with CF90, then audit whether it
still knows truth vs myths using the same factual probes for both steps.

    from knowledge_fidelity import compress_and_audit
    report = compress_and_audit("meta-llama/Llama-3.1-8B-Instruct", ratio=0.7)
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .svd import compress_qko, compress_qko_importance, freeze_layers
from .svd.importance import compute_importance
from .cartography.engine import analyze_confidence
from .probes import get_default_probes, get_importance_prompts


def compress_and_audit(
    model_name_or_path: str,
    ratio: float = 0.7,
    freeze_ratio: float = 0.75,
    use_importance: bool = False,
    probes: Optional[list[dict]] = None,
    output_dir: Optional[str] = None,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> dict:
    """Compress an LLM and audit its knowledge fidelity in one call.

    Pipeline:
      1. Load model
      2. Run confidence audit BEFORE compression (baseline)
      3. Compute importance scores (optional, for aggressive compression)
      4. Apply CF90: SVD compress Q/K/O + freeze layers
      5. Run confidence audit AFTER compression
      6. Compare: how much false-belief signal was preserved?

    Args:
        model_name_or_path: HuggingFace model name or local path
        ratio: SVD compression ratio (0.7 = keep 70% of singular values)
        freeze_ratio: Fraction of layers to freeze (0.75 = 75%)
        use_importance: If True, use gradient-based importance for SVD
                       (slower but 3x better at aggressive compression <70%)
        probes: List of probe dicts (default: built-in 20 probes).
                Each probe has "text" (true) and "false" keys.
        output_dir: Directory to save compressed model (optional)
        device: Device to use ("cpu", "cuda", "mps")
        dtype: Model dtype (default: float32, recommended for compression)

    Returns:
        Dict with:
          - model: the compressed model object
          - tokenizer: the tokenizer
          - compression: SVD compression stats
          - freeze: layer freeze stats
          - audit_before: confidence audit results before compression
          - audit_after: confidence audit results after compression
          - retention: float, fraction of probes where confidence was preserved
          - rho_before: Spearman rho (true vs false confidence) before
          - rho_after: Spearman rho after
          - summary: human-readable summary string
    """
    from scipy import stats as sp_stats

    start_time = time.time()

    if probes is None:
        probes = get_default_probes()
    use_dtype = dtype or torch.float32

    # --- Step 1: Load model ---
    print(f"Loading {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=use_dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"  Loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    # --- Step 2: Audit BEFORE compression ---
    print("Auditing confidence BEFORE compression...")
    audit_before = _run_audit(model, tokenizer, probes, device)
    rho_before = audit_before["rho"]
    print(f"  Spearman rho (true vs false): {rho_before:.3f} "
          f"(p={audit_before['rho_p']:.4f})")

    # --- Step 3: Compress ---
    print(f"Compressing Q/K/O at {ratio:.0%} rank...")
    if use_importance:
        importance_prompts = get_importance_prompts(probes)
        importance = compute_importance(model, tokenizer, prompts=importance_prompts)
        n_compressed = compress_qko_importance(model, importance, ratio=ratio)
    else:
        n_compressed = compress_qko(model, ratio=ratio)

    freeze_stats = freeze_layers(model, ratio=freeze_ratio)
    print(f"  Compressed {n_compressed} matrices, "
          f"frozen {freeze_stats['n_frozen']}/{freeze_stats['n_layers']} layers")

    # --- Step 4: Audit AFTER compression ---
    print("Auditing confidence AFTER compression...")
    model.eval()
    audit_after = _run_audit(model, tokenizer, probes, device)
    rho_after = audit_after["rho"]
    print(f"  Spearman rho (true vs false): {rho_after:.3f} "
          f"(p={audit_after['rho_p']:.4f})")

    # --- Step 5: Compute retention ---
    # A probe's confidence is "retained" if the true/false delta didn't flip
    n_retained = 0
    for i in range(len(probes)):
        delta_before = audit_before["true_confs"][i] - audit_before["false_confs"][i]
        delta_after = audit_after["true_confs"][i] - audit_after["false_confs"][i]
        # Retained if delta stayed positive (model still more confident on truth)
        if delta_before > 0 and delta_after > 0:
            n_retained += 1
        elif delta_before <= 0:
            # If model was already wrong before, count as retained if still same
            n_retained += 1

    retention = n_retained / len(probes) if probes else 0.0

    # --- Step 6: Save if requested ---
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_path)
        tokenizer.save_pretrained(out_path)
        print(f"  Saved compressed model to {out_path}")

    elapsed = time.time() - start_time

    summary = (
        f"Compressed {model_name_or_path} at {ratio:.0%} rank | "
        f"{n_compressed} matrices compressed | "
        f"{freeze_stats['n_frozen']}/{freeze_stats['n_layers']} layers frozen | "
        f"Retention: {retention:.0%} | "
        f"rho: {rho_before:.3f} -> {rho_after:.3f} | "
        f"{elapsed:.1f}s"
    )
    print(f"\n  {summary}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "compression": {
            "n_compressed": n_compressed,
            "ratio": ratio,
            "use_importance": use_importance,
        },
        "freeze": freeze_stats,
        "audit_before": audit_before,
        "audit_after": audit_after,
        "retention": retention,
        "rho_before": rho_before,
        "rho_after": rho_after,
        "elapsed_seconds": elapsed,
        "summary": summary,
    }


def audit_model(
    model,
    tokenizer,
    probes: Optional[list[dict]] = None,
    device: Optional[str] = None,
) -> dict:
    """Run a confidence audit on a pre-loaded model.

    Useful for auditing a model you've already compressed yourself.

    Args:
        model: HuggingFace causal LM model
        tokenizer: Corresponding tokenizer
        probes: Probe list (default: built-in)
        device: Device (auto-detected if not provided)

    Returns:
        Audit results dict with rho, p-value, per-probe confidence scores
    """
    if probes is None:
        probes = get_default_probes()
    if device is None:
        device = str(next(model.parameters()).device)

    return _run_audit(model, tokenizer, probes, device)


def _run_audit(model, tokenizer, probes: list[dict], device: str) -> dict:
    """Internal: run confidence analysis on true/false probe pairs.

    Returns dict with:
      - true_confs: list of mean confidence on true statements
      - false_confs: list of mean confidence on false statements
      - deltas: true_conf - false_conf per probe
      - rho: Spearman correlation between true and false confidence
      - rho_p: p-value for rho
      - records: list of ConfidenceRecord objects (for detailed analysis)
    """
    from scipy import stats as sp_stats

    true_confs = []
    false_confs = []
    records = []

    for probe in probes:
        try:
            rec_true = analyze_confidence(
                text=probe["text"],
                category="true",
                label=probe.get("id", ""),
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            rec_false = analyze_confidence(
                text=probe["false"],
                category="false",
                label=probe.get("id", ""),
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            true_confs.append(rec_true.mean_top1_prob)
            false_confs.append(rec_false.mean_top1_prob)
            records.extend([rec_true, rec_false])
        except Exception as e:
            print(f"  Warning: probe '{probe.get('id', '?')}' failed: {e}")
            true_confs.append(0.0)
            false_confs.append(0.0)

    true_arr = np.array(true_confs)
    false_arr = np.array(false_confs)
    deltas = true_arr - false_arr

    # Spearman correlation between true and false confidence
    if len(true_confs) >= 3:
        rho, rho_p = sp_stats.spearmanr(true_arr, false_arr)
    else:
        rho, rho_p = 0.0, 1.0

    return {
        "true_confs": true_confs,
        "false_confs": false_confs,
        "deltas": deltas.tolist(),
        "mean_delta": float(deltas.mean()),
        "rho": float(rho),
        "rho_p": float(rho_p),
        "n_probes": len(probes),
        "n_positive_delta": int((deltas > 0).sum()),
        "records": records,
    }
