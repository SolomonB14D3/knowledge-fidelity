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
        Pre-loaded model (PyTorch or MLX). If None, loaded from model_name.
    tokenizer : Any, optional
        Pre-loaded tokenizer. If None, loaded from model_name.
    sft_dataset : Any, optional
        Dataset for supervised fine-tuning phase. If None, uses default
        behavioral contrast dataset from rho_eval.alignment.
    contrast_dataset : Any, optional
        Dataset of (positive, negative) text pairs for contrastive loss.
        If None, generated from probe text for target_behaviors.
    output_dir : str, optional
        Directory to save intermediate artifacts (SAE checkpoints,
        audit reports, collateral damage matrix).

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

    # ── Phase 1: Load model + baseline audit ─────────────────────────────
    logger.info("Phase 1: Baseline audit")
    phase1 = _phase_baseline(model_name, config, model=model, tokenizer=tokenizer)
    result.phases.append(phase1)
    result.audit_before = phase1.details.get("scores", {})

    # TODO: store model/tokenizer references from phase1 for reuse

    # ── Phase 2: Weight-space control (SVD + freeze) ─────────────────────
    if config.weight_space_enabled:
        logger.info("Phase 2: Weight-space control (SVD compress + freeze)")
        phase2 = _phase_weight_space(model_name, config)
        result.phases.append(phase2)
    else:
        logger.info("Phase 2: Skipped (weight-space control disabled)")

    # ── Phase 3: Activation-space control (SAE) ──────────────────────────
    if config.activation_space_enabled:
        logger.info("Phase 3: Activation-space control (SAE train + identify)")
        phase3 = _phase_activation_space(model_name, config)
        result.phases.append(phase3)
    else:
        logger.info("Phase 3: Skipped (activation-space control disabled)")

    # ── Phase 4: Training-time control (Rho-guided SFT) ──────────────────
    if config.training_time_enabled:
        logger.info("Phase 4: Rho-guided SFT with contrastive loss")
        phase4 = _phase_training_time(
            model_name, config,
            sft_dataset=sft_dataset,
            contrast_dataset=contrast_dataset,
        )
        result.phases.append(phase4)
    else:
        logger.info("Phase 4: Skipped (training-time control disabled)")

    # ── Phase 5: Final audit ─────────────────────────────────────────────
    logger.info("Phase 5: Final audit")
    phase5 = _phase_final_audit(model_name, config)
    result.phases.append(phase5)
    result.audit_after = phase5.details.get("scores", {})

    # ── Compute collateral damage ────────────────────────────────────────
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


# ── Phase implementations (stubs) ────────────────────────────────────────


def _phase_baseline(
    model_name: str,
    config: HybridConfig,
    *,
    model: Any = None,
    tokenizer: Any = None,
) -> PhaseResult:
    """Phase 1: Load model and run baseline audit across all behaviors.

    Reuses: rho_eval.audit.audit()
    """
    # TODO: Implement
    # t0 = time.time()
    # from rho_eval.audit import audit
    # report = audit(model_name, model=model, tokenizer=tokenizer,
    #                behaviors=config.eval_behaviors,
    #                device=config.device,
    #                trust_remote_code=config.trust_remote_code)
    # scores = {b.behavior: b.rho for b in report.results}
    # return PhaseResult(
    #     phase="baseline",
    #     elapsed_sec=time.time() - t0,
    #     details={"scores": scores, "report": report.to_dict()},
    # )
    raise NotImplementedError(
        "Phase 1 (baseline audit) not yet implemented. "
        "See rho_eval.audit.audit() for the underlying API."
    )


def _phase_weight_space(model_name: str, config: HybridConfig) -> PhaseResult:
    """Phase 2: SVD compression + layer freezing.

    Reuses:
      - rho_eval.svd.compress_qko(model, ratio=config.compress_ratio,
                                    targets=config.compress_targets)
      - rho_eval.svd.freeze_layers(model, fraction=config.freeze_fraction)
    """
    # TODO: Implement
    # t0 = time.time()
    # from rho_eval.svd import compress_qko, freeze_layers
    # stats = compress_qko(model, ratio=config.compress_ratio,
    #                       targets=list(config.compress_targets))
    # n_frozen = freeze_layers(model, fraction=config.freeze_fraction)
    # return PhaseResult(
    #     phase="weight_space",
    #     elapsed_sec=time.time() - t0,
    #     details={"compress_stats": stats, "frozen_layers": n_frozen},
    # )
    raise NotImplementedError(
        "Phase 2 (weight-space control) not yet implemented. "
        "See rho_eval.svd.compress_qko() and freeze_layers()."
    )


def _phase_activation_space(model_name: str, config: HybridConfig) -> PhaseResult:
    """Phase 3: Train SAE on target layer, identify behavioral features.

    Reuses:
      - rho_eval.steering.train_behavioral_sae(
            model, tokenizer,
            behaviors=list(config.target_behaviors),
            layer_idx=config.sae_layer,
        )
      - rho_eval.steering.identify_behavioral_features(sae, act_data)
    """
    # TODO: Implement
    # t0 = time.time()
    # from rho_eval.steering import train_behavioral_sae, identify_behavioral_features
    # sae, act_data, train_stats = train_behavioral_sae(
    #     model, tokenizer,
    #     behaviors=list(config.target_behaviors),
    #     layer_idx=config.sae_layer,
    #     expansion=config.sae_expansion,
    # )
    # reports, features = identify_behavioral_features(sae, act_data)
    # return PhaseResult(
    #     phase="activation_space",
    #     elapsed_sec=time.time() - t0,
    #     details={
    #         "sae_layer": config.sae_layer,
    #         "n_features": {b: len(f) for b, f in features.items()},
    #         "train_stats": train_stats,
    #     },
    # )
    raise NotImplementedError(
        "Phase 3 (activation-space control) not yet implemented. "
        "See rho_eval.steering.train_behavioral_sae()."
    )


def _phase_training_time(
    model_name: str,
    config: HybridConfig,
    *,
    sft_dataset: Any = None,
    contrast_dataset: Any = None,
) -> PhaseResult:
    """Phase 4: Rho-guided SFT with contrastive confidence loss.

    If SAE hook is active from Phase 3, it remains installed during
    training — the model learns to self-steer through the SAE features.

    Reuses:
      - rho_eval.alignment.rho_guided_sft() (PyTorch)
      - rho_eval.alignment.mlx_rho_guided_sft() (Apple Silicon)
    """
    # TODO: Implement
    # t0 = time.time()
    # from rho_eval.alignment import rho_guided_sft, _HAS_MLX
    # if _HAS_MLX and _is_mlx_model(model):
    #     from rho_eval.alignment import mlx_rho_guided_sft
    #     result = mlx_rho_guided_sft(
    #         model, tokenizer, sft_texts,
    #         contrast_dataset=contrast_dataset,
    #         rho_weight=config.rho_weight,
    #         epochs=config.sft_epochs,
    #         lr=config.sft_lr,
    #         margin=config.margin,
    #     )
    # else:
    #     result = rho_guided_sft(
    #         model, tokenizer, sft_dataset,
    #         contrast_dataset=contrast_dataset,
    #         rho_weight=config.rho_weight,
    #         epochs=config.sft_epochs,
    #         lr=config.sft_lr,
    #         margin=config.margin,
    #     )
    # return PhaseResult(
    #     phase="training_time",
    #     elapsed_sec=time.time() - t0,
    #     details={"train_loss": result.get("loss", None)},
    # )
    raise NotImplementedError(
        "Phase 4 (training-time control) not yet implemented. "
        "See rho_eval.alignment.rho_guided_sft()."
    )


def _phase_final_audit(model_name: str, config: HybridConfig) -> PhaseResult:
    """Phase 5: Run final audit to measure improvement and collateral damage.

    Reuses: rho_eval.audit.audit()
    """
    # TODO: Implement
    # t0 = time.time()
    # from rho_eval.audit import audit
    # report = audit(model=model, tokenizer=tokenizer,
    #                behaviors=config.eval_behaviors,
    #                device=config.device)
    # scores = {b.behavior: b.rho for b in report.results}
    # return PhaseResult(
    #     phase="final_audit",
    #     elapsed_sec=time.time() - t0,
    #     details={"scores": scores, "report": report.to_dict()},
    # )
    raise NotImplementedError(
        "Phase 5 (final audit) not yet implemented. "
        "See rho_eval.audit.audit() for the underlying API."
    )
