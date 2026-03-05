# Changelog

All notable changes to this project will be documented in this file.

## [2.3.1] - 2026-03-05

### Added

- 3M and 5M model configurations for developmental scale sweep
- Developmental sweep runner (`experiments/developmental_sweep/`)
- Cross-transfer threshold discovery: absent at 3M, emerges at 5M
- SVD spectrum extraction tool (`extract_svd_spectrum.py`)
- Calculator and subitizing contrastive pair generators for reasoning injection tests

### Changed

- Paper 4 updated to v7 with developmental sweep data, expanded cross-transfer table across 3 scales
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
