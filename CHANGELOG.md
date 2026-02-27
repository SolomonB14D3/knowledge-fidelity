# Changelog

All notable changes to this project will be documented in this file.

## [2.2.2] - 2026-02-27

### Added

- **3-seed × 3-λ external eval** on Qwen2.5-7B-Instruct with full 8-dimensional behavioral audit + TruthfulQA MC2. λ=0.5 recovers 29% of SFT truthfulness damage (up from 17% in prior 2-seed eval). Results in `results/alignment/external_eval_Qwen_Qwen2.5-7B-Instruct.json`.
- **Hybrid control pipeline** (`src/rho_eval/hybrid/pipeline.py`) — all 5 phases implemented: baseline audit, SVD compress + freeze, SAE train + identify, rho-guided SFT with steering hook, final audit. Ready for parameter sweeps.
- **Attack/defense asymmetry experiment** (`experiments/attack_defense_asymmetry.py`) — measures whether degrading safety behaviors (attack) is easier than improving them (defense) via SAE steering.

### Fixed

- **Behavior plugin API in `steering/collect.py`** — use `get_behavior(name).load_probes()` instead of legacy `load_behavioral_probes()` which only supported 4 of 8 behaviors.
- **Device propagation in `attack_defense_asymmetry.py`** — audit calls now pass `device=` parameter correctly for MPS/CUDA.
- **`AuditReport.results` → `.behaviors`** in attack_defense_asymmetry.py (3 locations) — matching the v2.0.0 schema.

### Changed

- Paper Section 4.4 updated to 3-seed TruthfulQA MC2 results with dose-response across λ∈{0.0, 0.2, 0.5}. Recovery figure strengthened from 17% to 29%.
- CITATION.cff version synced with pyproject.toml.

## [2.1.1] - 2026-02-26

### Added

- **5-seed ablation** expanding the original 2-seed study (seeds 42, 123, 456, 789, 1337) with updated effect sizes: d=10.8 toxicity, d=13.7 bias (p<0.0001).
- **Refusal behavior dimension** -- contrastive-only training erodes refusal (d=-8.4), rho-guided preserves it (d=+8.6). The "refusal buffer" hypothesis.
- **Margin ablation** (gamma=0 vs gamma=0.1) -- gamma=0 causes bias to go negative; gamma=0.1 is structurally necessary.
- **Safety stress test** (`experiments/safety_stress_test.py`) -- 25 jailbreak prompts across 10 categories, comparing all 4 training conditions.
- **Combined kill zone heatmap** (`experiments/plot_combined_killzone.py`) -- cross-architecture visualization.
- **Research notes** (`Research_Notes.md`) with hypotheses, open questions, and technical framing.
- **Dose-response with 5 seeds** -- variance collapse finding (factual sigma drops 63% from SFT-only to rho-guided).

### Fixed

- **6 fabricated paper citations** -- corrected co-author lists for ToxiGen (Hartvigsen et al.), LLM-KICK (Jaiswal et al.), CAA (Rimsky et al.), and corrected title/authors for TPLO (Fu et al.), Tian et al., and Gudibande et al.

### Changed

- Paper updated to 5-seed results throughout (abstract, contributions, ablation tables, discussion, conclusion).
- README Key Findings updated with refusal erosion, variance collapse, and margin findings.

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
