# Changelog

All notable changes to this project will be documented in this file.

## [2.6.0] - 2026-03-08

### Added

- **Snap-On v2 adapter** — cross-scale logit-space adapters with vocabulary-aware transfer
  - Phase 4: v1 adapter (29M params, trained on Qwen2.5-1.5B) transfers to 3B-Instruct with zero MMLU degradation
  - Cross-architecture transfer: Qwen→Llama-3.1-8B with -0.2% MMLU delta via vocab truncation
  - Phase 3: style adapters (code/eli5/formal) — null style result, perfect MMLU preservation
- **Paper 9 — STEM Truth Oracle** (DOI: 10.5281/zenodo.19005729) — log-probability MC ranking reveals and corrects scale-invariant factual biases
- **Operation Destroyer** — logit-space adapter experiments for factual amplification
  - Margin oracle: positive margin → 100% correct (0 FP), negative → 0% (0 FN)
  - 4 bias patterns identified: positivity, linearity, missing-constant, truncation
  - Mixed adapter required for cross-pattern robustness
- **Operation Frontier** — autonomous oracle wrapper + conservation law discovery pipeline
- NoetherSolve autonomy loop with problem generation

### Removed

- **Leaderboard stubs** — removed unimplemented `rho-leaderboard` CLI, `leaderboard.py`, and `probe_registry.py` (all were `NotImplementedError` scaffolding)

### Fixed

- Tied word embeddings handling in snap-on adapters (v2.5.1)
- Synced `__version__` with pyproject.toml

## [2.5.0] - 2026-03-07

### Added

- **`snap-on` CLI tool** — train and apply frozen logit-space adapters on any base model
  - `snap-on train --model MODEL --mode logit` — train a modular adapter (zero knowledge damage)
  - `snap-on apply --model MODEL --adapter PATH` — apply adapter at inference
  - v1 architecture: `d_inner=64`, trained on 5K examples, 29M params for 1.5B model
- Papers 7-9 added to README and CITATION.cff

## [2.4.0] - 2026-03-06

### Added

- **`rho-unlock` CLI tool** — two-axis behavioral diagnostic + contrastive decoding unlock
  - `rho-unlock diagnose <model>` — measures behavioral discrimination (ρ) and expression gap per behavior/benchmark, classifies into four quadrants (HEALTHY/UNLOCK/RETRAIN/BOTH_NEEDED)
  - `rho-unlock unlock <model>` — applies contrastive decoding to rescue hidden capability in UNLOCK behaviors
  - Supports 4 benchmarks (MMLU, TruthfulQA, ARC-Challenge, HellaSwag) alongside rho-eval behavioral dimensions
  - Auto-detects amateur model for contrastive decoding
- **Benchmark-aware diagnosis** — `MetricType` enum distinguishes behavioral ρ (threshold 0.3) from logit accuracy (threshold = chance + margin)
  - `compute_knows_threshold()` for type-appropriate thresholds
  - `--above-chance-margin` CLI flag (default 0.15)
- Expression gap measurement (`expression_gap.py`) — logit accuracy vs generation accuracy with parse rate tracking
- Contrastive decoding module (`contrastive.py`) — logit-level and generation-level CD with amateur model auto-detection

## [2.3.1] - 2026-03-06

### Added

- **Paper 7: "Small Language Models Already Know More Than They Can Say"** — expression bottleneck paper
- Format-forced decoding scripts (`format_forced_decoding.py`, `format_forced_full_scale.py`)
- Contrastive decoding inference rescue (`contrastive_decoding.py`)
- Activation steering experiment (`activation_steering.py`)
- Generation trajectory analysis (`generation_trajectory.py`)
- Logit lens readout gate experiments (`logit_lens_readout_gate.py` v1-v3)
- OV circuit logit attribution (`ov_circuit_logit_attribution.py`)
- Residual budget decomposition (`residual_budget.py`)
- Confidence shift analysis (`confidence_shift.py`)
- Attention routing analysis (`attention_routing_analysis.py`)
- 3M and 5M model configurations for developmental scale sweep
- Developmental sweep runner (`experiments/developmental_sweep/`)
- Cross-transfer threshold discovery: absent at 3M, emerges at 5M
- Width sweep phase transition at d_model=96

### Changed

- Paper 6 revised: reframed as format generation enablement (not knowledge creation)
- Paper 6 citations updated, prose cleaned
- README updated with cross-transfer developmental onset finding

## [2.3.0] - 2026-03-05

### Added

- 64M contrastive injection results — scaling stabilization finding
- Cross-dimensional transfer experiments (single-behavior injection)
- Primitive evaluative hierarchy experiment infrastructure
  - 280 contrastive pairs grounded in developmental/comparative cognition paradigms
  - Curriculum schedule support (`--inject-schedule`) for ordering experiments
  - Injection cutoff (`--inject-until`) for cascade tests
- Custom pairs support (`--pairs-json`) in contrastive training pipeline
- Phase 1/2/3 runner scripts for primitive hierarchy experiments

### Changed

- README updated with 64M scaling stabilization and cross-transfer findings
- Paper 4 updated to v6 with 34M, 64M results, cross-transfer analysis, and scaling stabilization

## [2.2.2] - 2026-02-27

### Added

- External validation suite
- Hybrid control pipeline (`src/rho_eval/hybrid/pipeline.py`) — 5-phase automated sweep
- Attack/defense asymmetry experiment for SAE steering

### Fixed

- Behavior plugin API in `steering/collect.py` — use v2 probe loading
- Device propagation for MPS/CUDA in experiments
- `AuditReport` schema alignment in experiment scripts

## [2.1.1] - 2026-02-26

### Added

- Multi-seed ablation study
- Refusal behavior dimension with contrastive-only vs rho-guided comparison
- Margin ablation
- Safety stress test (`experiments/safety_stress_test.py`)
- Combined kill zone heatmap visualization

### Fixed

- Corrected 6 paper citations with wrong co-author lists

### Changed

- Paper updated with multi-seed results

## [2.1.0] - 2026-02-25

### Added

- Rho-Guided SFT research paper
- TruthfulQA MC2 evaluation with corrected scoring methodology
- Multi-condition ablation study
- Out-of-domain validation
- Calibration evaluation
- Statistical analysis scripts
- Completion-only logprob scoring

### Fixed

- TruthfulQA scoring: chat template + sum logprob matching lm-eval-harness standard

## [2.0.0] - 2026-02-23

### Major: rho-eval rebrand

Package renamed from `knowledge-fidelity` to **`rho-eval`**.

### Added

- Behavior plugin architecture (`ABCBehavior` + `@register`)
- Pre-sampled probe data (806 probes, no internet needed)
- Standardized output: `BehaviorResult`/`AuditReport` with PASS/WARN/FAIL
- 4 export formats (JSON, Markdown, CSV, colored table)
- One-line audit API: `rho_eval.audit("model", behaviors="all")`
- Comparison system with delta tables
- CLI overhaul with `--list-behaviors`, `--list-probes`, `--compare`
- 61 tests

### Backward Compatibility

- All v1.x imports still work
- `rho-audit` preserved as alias

---

## [1.2.0] - 2026-02-21

### Added

- Cross-architecture steering validation (Mistral-7B)
- Multi-vector steering cocktails
- Comparative Anatomy section in paper

## [1.1.0] - 2026-02-19

### Added

- Merge method audit (12 models, 2 architectures, 6 methods)
- Activation steering via CAA
- Behavioral localization via freeze-ratio sweeps
- `rho-audit` CLI

## [1.0.0] - 2026-02-17

### Added

- Initial release
- SVD compression with CF90 pipeline
- Teacher-forced confidence probes (Spearman rho)
- Importance-guided SVD
- 56 built-in probes across 5 categories
- Gradio demo app
