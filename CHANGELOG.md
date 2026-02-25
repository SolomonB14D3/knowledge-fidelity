# Changelog

All notable changes to this project will be documented in this file.

## [2.1.0] - 2026-02-25

### Added

- **Rho-Guided SFT research paper** — "Rho-Guided Supervised Fine-Tuning: Post-Training Repair of Calibration Damage in Large Language Models" (`paper/rho_guided_sft.md`). Documents the SFT inversion problem and the contrastive repair mechanism.
- **TruthfulQA MC2 evaluation** (`experiments/truthfulqa_mc2_mlx.py`) with corrected scoring methodology (chat template + sum logprob).
- **Ablation study** with 4 conditions (SFT-only, rho-guided, contrastive-only, shuffled-pairs) confirming the contrastive loss as the active ingredient.
- **OOD validation** on clinical, social, and logic domains showing transfer of calibration improvements.
- **Calibration evaluation** (ECE + Brier scores) across lambda values.
- **Statistical analysis scripts** (`experiments/analyze_ablation_stats.py`) with p-values and Cohen's d.
- **Overnight runner** (`experiments/run_overnight.sh`) for chaining experimental phases.
- **Completion-only logprob scoring** (`get_completion_logprob()` in `src/rho_eval/behaviors/metrics.py`) with sum/mean reduction for proper MC benchmark scoring.
- **Alignment result JSONs** committed to `results/alignment/`.

### Fixed

- **TruthfulQA scoring methodology**: Fixed chat template formatting and switched from mean to sum logprob for MC scoring, matching the lm-eval-harness standard. This fixed a 0.16-point measurement artifact (0.459 to 0.648 baseline MC2).

### Changed

- `.gitignore` updated to whitelist alignment result JSONs while ignoring logs and model weights.

## [2.0.0] - 2026-02-23

### Major: rho-eval rebrand

The package is renamed from `knowledge-fidelity` to **`rho-eval`** to better reflect its primary purpose: general-purpose behavioral auditing of LLMs.

### Added

- **Behavior plugin architecture** — `ABCBehavior` abstract base class with `@register` decorator. Add custom behaviors by subclassing and registering.
- **Pre-sampled probe data (806 probes, no internet needed)** — 9 JSON files ship with the package: 56 factual, 300 bias, 150 sycophancy, 200 toxicity, 100 reasoning.
- **Standardized output format** — `BehaviorResult` and `AuditReport` dataclasses with PASS/WARN/FAIL thresholds (ρ ≥ 0.5 / ≥ 0.2 / < 0.2).
- **4 export formats** — JSON, Markdown, CSV, and colored terminal table via `--format`.
- **One-line audit API** — `rho_eval.audit("model", behaviors="all")` returns an `AuditReport`.
- **Comparison system** — `rho_eval.compare(report_a, report_b)` produces delta tables with IMPROVED/DEGRADED labels.
- **Probe discovery API** — `list_probe_sets()`, `get_probes("factual/default")`, `get_probe_counts()`.
- **CLI overhaul** — `rho-eval` primary command, `--list-behaviors`, `--list-probes`, `--format`, `--compare`, CI-friendly exit codes.
- **61 tests** covering probes, registry, output, exporters, comparator, CLI, and metrics.

### Changed

- Default behavior set changed from `factual` to `all` (all 5 behaviors).
- `rho-eval` is now the primary CLI entry point (`rho-audit` still works as an alias).
- Version bumped to 2.0.0.

### Backward Compatibility

- All v1.x imports (`from knowledge_fidelity import ...`) continue to work.
- The `rho-audit` CLI command is preserved as an alias.
- `knowledge-fidelity` package included in the distribution for existing users.

---

## [1.2.0] - 2026-02-21

### Added

- Cross-architecture steering validation (Mistral-7B layer heatmap).
- Multi-vector steering cocktails and Layer 17 interference study.
- Mistral sensitivity map and layer heatmap figures.
- Comparative Anatomy section in paper.

## [1.1.0] - 2026-02-19

### Added

- Merge method audit across 12 models (2 architectures, 6 merge methods).
- Activation steering via Contrastive Activation Addition (CAA).
- Behavioral localization via freeze-ratio sweeps.
- Demo steering modes (Truth-Serum vs Social-Wrapper).
- `rho-audit` CLI for standalone behavioral auditing.
- Cross-behavioral denoising study (5 behavioral dimensions).

## [1.0.0] - 2026-02-17

### Added

- Initial release.
- SVD compression with CF90 pipeline (Q/K/O at 70% rank, freeze 75%, gentle fine-tune).
- Teacher-forced confidence probes (Spearman ρ) for factual auditing.
- Importance-guided SVD for aggressive compression ratios.
- Auto-denoising mode (`--denoise`).
- 56 built-in probes across 5 categories (default, mandela, medical, commonsense, truthfulqa).
- Gradio demo app.
