"""Hybrid control pipeline — orchestrates weight, activation, and training control.

The pipeline runs up to five phases:

  Phase 1: Baseline audit — snapshot rho across all 8 behaviors
  Phase 2: Weight-space control — SVD compress + freeze layers
  Phase 3: Activation-space control — train SAE, identify features, install hook
  Phase 4: Training-time control — Rho-guided SFT with SAE hook active
  Phase 5: Final audit — measure improvement and collateral damage

Each phase reuses battle-tested components:
  - SVD: rho_eval.svd.compress_qko(), freeze_layers()
  - SAE: rho_eval.steering.train_behavioral_sae(), identify_behavioral_features()
  - SFT: rho_eval.alignment.rho_guided_sft() / mlx_rho_guided_sft()
  - Audit: rho_eval.audit.audit()
"""

from __future__ import annotations

import time
import logging
from typing import Any, Optional

from .schema import HybridConfig, HybridResult, PhaseResult

logger = logging.getLogger(__name__)


def _resolve_behaviors(config: HybridConfig) -> str | list[str]:
    """Normalize config.eval_behaviors for audit().

    audit() expects behaviors="all" (string) or a list of names.
    HybridConfig stores ("all",) as a tuple.
    """
    if config.eval_behaviors == ("all",) or config.eval_behaviors == ["all"]:
        return "all"
    return list(config.eval_behaviors)


def _load_model_and_tokenizer(
    model_name: str,
    config: HybridConfig,
) -> tuple[Any, Any]:
    """Load a HuggingFace model and tokenizer.

    Uses the same loading logic as audit._load_model() but returns
    the objects so they can be threaded through all pipeline phases.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = config.device
    if device is None:
        from ..utils import get_device
        device = str(get_device())

    load_kwargs = {
        "trust_remote_code": config.trust_remote_code,
    }

    if device == "cuda" or (device and device.startswith("cuda:")):
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"
    elif device == "mps":
        load_kwargs["torch_dtype"] = torch.float32
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if device == "mps":
        model = model.to("mps")

    model.eval()
    return model, tokenizer


def apply_hybrid_control(
    model_name: str,
    config: Optional[HybridConfig] = None,
    *,
    model: Any = None,
    tokenizer: Any = None,
    sft_dataset: Any = None,
    contrast_dataset: Any = None,
    output_dir: Optional[str] = None,
) -> HybridResult:
    """Apply the full hybrid control pipeline to a model.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    config : HybridConfig, optional
        Pipeline configuration. Defaults to HybridConfig() if not provided.
    model : Any, optional
        Pre-loaded model (PyTorch). If None, loaded from model_name.
    tokenizer : Any, optional
        Pre-loaded tokenizer. If None, loaded from model_name.
    sft_dataset : Any, optional
        Dataset for supervised fine-tuning phase. If None, auto-constructed
        from Alpaca + behavioral trap examples.
    contrast_dataset : Any, optional
        Dataset of (positive, negative) text pairs for contrastive loss.
        If None, auto-constructed from behavioral probes.
    output_dir : str, optional
        Directory to save intermediate artifacts.

    Returns
    -------
    HybridResult
        Full result with before/after audits, per-phase details,
        and collateral damage matrix.

    Example
    -------
    >>> from rho_eval.hybrid import HybridConfig, apply_hybrid_control
    >>> config = HybridConfig(
    ...     compress_ratio=0.7, freeze_fraction=0.75,
    ...     sae_layer=17, target_behaviors=("sycophancy",),
    ...     rho_weight=0.2,
    ... )
    >>> result = apply_hybrid_control("Qwen/Qwen2.5-7B-Instruct", config)
    >>> print(result.to_table())
    """
    if config is None:
        config = HybridConfig()

    result = HybridResult(config=config, model_name=model_name)
    t_start = time.time()

    # ── Load model/tokenizer up front ─────────────────────────────────
    if model is None or tokenizer is None:
        logger.info(f"Loading model: {model_name}")
        model, tokenizer = _load_model_and_tokenizer(model_name, config)
    else:
        logger.info("Using pre-loaded model and tokenizer")

    device = config.device
    if device is None:
        from ..utils import get_device
        device = str(get_device())

    # ── Phase 1: Baseline audit ───────────────────────────────────────
    logger.info("Phase 1: Baseline audit")
    phase1 = _phase_baseline(model, tokenizer, config, device=device)
    result.phases.append(phase1)
    result.audit_before = phase1.details.get("scores", {})

    # ── Phase 2: Weight-space control (SVD + freeze) ──────────────────
    if config.weight_space_enabled:
        logger.info("Phase 2: Weight-space control (SVD compress + freeze)")
        phase2 = _phase_weight_space(model, config)
        result.phases.append(phase2)
    else:
        logger.info("Phase 2: Skipped (weight-space control disabled)")

    # ── Phase 3: Activation-space control (SAE) ───────────────────────
    sae = None
    behavioral_features = None
    if config.activation_space_enabled:
        logger.info("Phase 3: Activation-space control (SAE train + identify)")
        phase3, sae, behavioral_features = _phase_activation_space(
            model, tokenizer, config, device=device,
        )
        result.phases.append(phase3)
    else:
        logger.info("Phase 3: Skipped (activation-space control disabled)")

    # ── Phase 4: Training-time control (Rho-guided SFT) ───────────────
    if config.training_time_enabled:
        logger.info("Phase 4: Rho-guided SFT with contrastive loss")
        phase4 = _phase_training_time(
            model, tokenizer, config,
            device=device,
            sae=sae,
            behavioral_features=behavioral_features,
            sft_dataset=sft_dataset,
            contrast_dataset=contrast_dataset,
        )
        result.phases.append(phase4)
    else:
        logger.info("Phase 4: Skipped (training-time control disabled)")

    # ── Phase 5: Final audit ──────────────────────────────────────────
    logger.info("Phase 5: Final audit")
    phase5 = _phase_final_audit(model, tokenizer, config, device=device)
    result.phases.append(phase5)
    result.audit_after = phase5.details.get("scores", {})

    # ── Compute collateral damage ─────────────────────────────────────
    result.compute_collateral()
    result.total_elapsed_sec = time.time() - t_start

    if output_dir:
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.to_json(out / "hybrid_result.json")
        logger.info(f"Results saved to {out / 'hybrid_result.json'}")

    logger.info(result.summary())
    return result


# ── Phase implementations ─────────────────────────────────────────────


def _phase_baseline(
    model: Any,
    tokenizer: Any,
    config: HybridConfig,
    *,
    device: str = "cpu",
) -> PhaseResult:
    """Phase 1: Run baseline audit across all behaviors."""
    t0 = time.time()
    from ..audit import audit

    report = audit(
        model=model,
        tokenizer=tokenizer,
        behaviors=_resolve_behaviors(config),
        device=device,
    )
    scores = {name: r.rho for name, r in report.behaviors.items()}

    return PhaseResult(
        phase="baseline",
        elapsed_sec=time.time() - t0,
        details={"scores": scores, "report": report.to_dict()},
    )


def _phase_weight_space(
    model: Any,
    config: HybridConfig,
) -> PhaseResult:
    """Phase 2: SVD compression + layer freezing.

    Mutates model in-place (replaces weight matrices, freezes parameters).
    """
    t0 = time.time()
    from ..svd import compress_qko, freeze_layers

    n_compressed = compress_qko(model, ratio=config.compress_ratio)
    freeze_stats = freeze_layers(model, ratio=config.freeze_fraction)

    return PhaseResult(
        phase="weight_space",
        elapsed_sec=time.time() - t0,
        details={
            "n_compressed": n_compressed,
            "freeze_stats": freeze_stats,
        },
    )


def _phase_activation_space(
    model: Any,
    tokenizer: Any,
    config: HybridConfig,
    *,
    device: str = "cpu",
) -> tuple[PhaseResult, Any, dict]:
    """Phase 3: Train SAE on target layer, identify behavioral features.

    Returns:
        Tuple of (PhaseResult, trained_sae, behavioral_features_dict).
        The SAE and features are passed to Phase 4 for steering during SFT.
    """
    t0 = time.time()
    from ..steering import train_behavioral_sae, identify_behavioral_features

    sae, act_data, train_stats = train_behavioral_sae(
        model, tokenizer,
        behaviors=list(config.target_behaviors),
        layer_idx=config.sae_layer,
        device=device,
    )
    reports, features = identify_behavioral_features(sae, act_data)

    # Serialize-safe stats (exclude the SAE object itself)
    safe_stats = {k: v for k, v in train_stats.items() if k != "sae"}

    phase_result = PhaseResult(
        phase="activation_space",
        elapsed_sec=time.time() - t0,
        details={
            "sae_layer": config.sae_layer,
            "n_features": {b: len(f) for b, f in features.items()},
            "train_stats": safe_stats,
        },
    )
    return phase_result, sae, features


def _phase_training_time(
    model: Any,
    tokenizer: Any,
    config: HybridConfig,
    *,
    device: str = "cpu",
    sae: Any = None,
    behavioral_features: Optional[dict] = None,
    sft_dataset: Any = None,
    contrast_dataset: Any = None,
) -> PhaseResult:
    """Phase 4: Rho-guided SFT with contrastive confidence loss.

    If SAE and features from Phase 3 are provided, installs a steering
    hook that amplifies target behavioral features during training.
    The model learns to self-steer through the SAE features.
    """
    t0 = time.time()

    # Install SAE steering hook if Phase 3 ran
    hook = None
    if sae is not None and behavioral_features is not None:
        from ..steering import steer_features
        all_feature_indices = []
        for behavior in config.target_behaviors:
            all_feature_indices.extend(behavioral_features.get(behavior, []))
        if all_feature_indices:
            hook = steer_features(
                model, sae,
                layer_idx=config.sae_layer,
                feature_indices=all_feature_indices,
                scale=config.scale_factor,
            )
            logger.info(
                f"  SAE hook installed: {len(all_feature_indices)} features "
                f"at scale={config.scale_factor}"
            )

    # Auto-construct datasets if not provided
    if sft_dataset is None:
        from ..alignment.dataset import load_sft_dataset
        sft_dataset = load_sft_dataset(
            tokenizer,
            n=2000,
            include_traps=True,
            behaviors=list(config.target_behaviors),
            seed=config.seed,
        )

    if contrast_dataset is None:
        from ..alignment.dataset import BehavioralContrastDataset
        contrast_dataset = BehavioralContrastDataset(
            behaviors=list(config.target_behaviors),
            seed=config.seed,
        )

    # Run SFT
    try:
        from ..alignment import _HAS_MLX
        if _HAS_MLX and 'mlx' in str(type(model).__module__):
            from ..alignment import mlx_rho_guided_sft
            # MLX trainer expects raw text list
            if hasattr(sft_dataset, 'texts'):
                sft_texts = sft_dataset.texts
            else:
                sft_texts = [sft_dataset[i] for i in range(len(sft_dataset))]
            sft_result = mlx_rho_guided_sft(
                model, tokenizer,
                sft_texts,
                contrast_dataset=contrast_dataset,
                rho_weight=config.rho_weight,
                epochs=config.sft_epochs,
                lr=config.sft_lr,
                margin=config.margin,
            )
        else:
            from ..alignment import rho_guided_sft
            sft_result = rho_guided_sft(
                model, tokenizer,
                sft_dataset,
                contrast_dataset=contrast_dataset,
                rho_weight=config.rho_weight,
                epochs=config.sft_epochs,
                lr=config.sft_lr,
                margin=config.margin,
                device=device,
            )
    finally:
        # Always remove hook, even if SFT fails
        if hook is not None:
            hook.remove()
            logger.info("  SAE hook removed")

    # Extract serializable stats from result
    safe_result = {
        k: v for k, v in sft_result.items()
        if k not in ("merged_model",)
    }

    return PhaseResult(
        phase="training_time",
        elapsed_sec=time.time() - t0,
        details={"train_result": safe_result},
    )


def _phase_final_audit(
    model: Any,
    tokenizer: Any,
    config: HybridConfig,
    *,
    device: str = "cpu",
) -> PhaseResult:
    """Phase 5: Run final audit to measure improvement and collateral damage."""
    t0 = time.time()
    from ..audit import audit

    report = audit(
        model=model,
        tokenizer=tokenizer,
        behaviors=_resolve_behaviors(config),
        device=device,
    )
    scores = {name: r.rho for name, r in report.behaviors.items()}

    return PhaseResult(
        phase="final_audit",
        elapsed_sec=time.time() - t0,
        details={"scores": scores, "report": report.to_dict()},
    )
