# rho-eval

[![PyPI](https://img.shields.io/pypi/v/rho-eval)](https://pypi.org/project/rho-eval/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18743959.svg)](https://doi.org/10.5281/zenodo.18743959)
[![Tests](https://img.shields.io/badge/tests-61%20passed-brightgreen)]()
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Demo-blue)](https://huggingface.co/spaces/bsanch52/knowledge-fidelity-demo)
[![Awesome](https://img.shields.io/badge/Awesome-LLM--Compression-blue)](https://github.com/HuangOwen/Awesome-LLM-Compression#tools)

**Behavioral auditing toolkit for LLMs.** Audit any model across 5 dimensions — factual accuracy, bias detection, sycophancy resistance, toxicity sensitivity, and reasoning robustness — using teacher-forced confidence (ρ) probes. All 806 probes ship with the package; no internet required.

```bash
pip install rho-eval
rho-eval Qwen/Qwen2.5-7B-Instruct --behaviors all
```

```
Behavioral Audit: Qwen/Qwen2.5-7B-Instruct
  Status: WARN  Mean ρ: 0.5346  Probes: 706  Time: 47.3s

  Behavior         ρ   Retention   Score  Status   Time
  ──────────────────────────────────────────────────────
  bias        +0.7730     77.3%  232/300    PASS   15.7s
  factual     +0.7460     85.7%   48/56     PASS    4.2s
  reasoning   +0.4200     60.0%   42/100    WARN    9.5s
  sycophancy  +0.1200     12.0%   18/150    FAIL    9.8s
  toxicity    +0.6140     72.0%   72/100    PASS    8.1s
```

> *Formerly `knowledge-fidelity`. All v1.x imports still work.*

Also includes SVD compression with behavioral auditing — measures layer-wise trade-offs and merge method impacts on all 5 behavioral dimensions.

## Associated Paper

> Sanchez, B. (2026). *Confidence Cartography: Teacher-Forced Probability as a False-Belief Sensor in Language Models*. Zenodo. [doi:10.5281/zenodo.18703506](https://doi.org/10.5281/zenodo.18703506)

The paper introduces the core metric (Spearman ρ over teacher-forced confidence probes) and validates it across Pythia 160M–12B. This toolkit extends it with SVD compression, behavioral localization, steering vectors, and the `rho-audit` CLI.

## Key Findings

- **Layer-wise freeze after SVD compression selectively enhances behavioral traits** — factual knowledge peaks at 75% freeze (early layers), bias detection peaks at 25% freeze (late layers), sycophancy flips from negative to positive at 50%.
- **Merge methods cause dramatic behavioral trade-offs invisible to standard benchmarks** — Linear merging is the best balanced method on Qwen; DARE-TIES destroys alignment on Qwen but improves it on Mistral; DELLA completely breaks the model.
- **Activation steering vectors extracted from ρ probes enable runtime behavioral control** — sycophancy resistance triples at Layer 17 (ρ 0.120→0.413), factual accuracy gains 32% at Layer 24, but Layer 17 is a shared bottleneck where steering one trait disrupts others.
- **Layer 17 is a behavioral decoupling point** — multi-vector steering reveals that social compliance (sycophancy) and social awareness (bias) share representational capacity at Layer 17 (slope = −1.37), while factual processing is preserved. This enables a "truth-maximization" mode: 3.4× sycophancy resistance + 31% factual gain, at the cost of social bias awareness.
- **Sycophancy suppression via activation steering is architecture-contingent** — the Layer 17 sycophancy sweet spot is Qwen-specific. On Mistral-7B, no layer at any depth achieves meaningful sycophancy improvement. An "Alignment Kill Zone" at L14–L18 (44–56% depth) destroys bias detection without any sycophancy benefit. Only factual steering at ~75% depth transfers across architectures.
- **SVD compression can *improve* factual discrimination** — truncated SVD at 70% rank acts as a denoiser, boosting Mandela probe ρ by +0.514 on Qwen-0.5B.

These findings extend the ρ probing method from [Sanchez (2026)](https://doi.org/10.5281/zenodo.18703506).

---

## Quick Start

```bash
pip install rho-eval
```

### Python API (one-liner)

```python
import rho_eval

# Audit any model across all 5 behaviors
report = rho_eval.audit("Qwen/Qwen2.5-7B-Instruct")
print(report)
# <AuditReport model='Qwen/Qwen2.5-7B-Instruct' behaviors=5 mean_ρ=0.5346 status=WARN>

# Or specific behaviors with a pre-loaded model
report = rho_eval.audit(model=model, tokenizer=tokenizer, behaviors=["factual", "bias"])

# Compare two models
baseline = rho_eval.audit("Qwen/Qwen2.5-7B-Instruct")
compressed = rho_eval.audit("my-compressed-model")
delta = rho_eval.compare(compressed, baseline)
print(delta.to_table())

# List available behaviors and probes
rho_eval.list_behaviors()
# ['bias', 'factual', 'reasoning', 'sycophancy', 'toxicity']
```

### CLI

```bash
# Full behavioral report card (5 dimensions)
rho-eval Qwen/Qwen2.5-7B-Instruct --behaviors all

# Quick factual-only check
rho-eval my-merged-model/ --behaviors factual --format json

# Compare a compressed model against baseline
rho-eval compressed-model/ --compare baseline.json

# Export as markdown/csv
rho-eval my-model/ --format markdown --output report.json

# Discover available behaviors and probes
rho-eval --list-behaviors
rho-eval --list-probes
```

### SVD Compression + Audit (legacy)

```python
from rho_eval import compress_and_audit

report = compress_and_audit("Qwen/Qwen2.5-7B-Instruct", ratio=0.7)
print(f"Retention: {report['retention']:.0%} | "
      f"False-belief signal: rho={report['rho_after']:.3f}")
# Retention: 100% | False-belief signal: rho=0.725
```

Auto-find the compression ratio that maximizes factual signal:

```bash
rho-compress Qwen/Qwen2.5-0.5B --denoise
# DENOISING DETECTED: Mandela rho 0.257 → 0.771 (+0.514) at 60% ratio
```

## Why This Exists

LLM compression is everywhere. Knowledge auditing is rare. Nobody checks both at once.

When you quantize or prune a model, you run HellaSwag and call it a day. But benchmarks don't tell you whether the model now thinks the Berenstain Bears are spelled "Berenstein" or that vaccines cause autism. **Knowledge Fidelity does.**

Two sensors, one toolkit:

| Sensor | What it measures | How |
|--------|-----------------|-----|
| **Structural** (SVD) | Which weights encode facts | Gradient importance on factual probes |
| **Behavioral** (Confidence) | Whether the model believes truth vs myths | Teacher-forced probability on true/false pairs |

The key insight: the same set of factual probes drives both. Compress with awareness of what matters, then verify nothing broke.

## Results

All results on Apple Silicon (M3 Ultra). Three model families validated.

### SVD Compression (CF90)

Multi-seed validation at 70% rank, 3 seeds:

| Metric | Qwen2.5-0.5B | Qwen2.5-7B-Instruct | Mistral-7B-v0.1 |
|--------|:------------:|:-------------------:|:---------------:|
| Retention | **95%** ± 0% | **100%** ± 0% | **95%** ± 0% |
| ρ before | 0.821 | 0.746 | 0.743 |
| ρ after | 0.720 | 0.725 | 0.705 |
| ρ drop | 0.101 ± 0.000 | **0.021** ± 0.000 | 0.038 ± 0.000 |
| Matrices compressed | 72 | 84 | 96 |
| Layers frozen | 18/24 | 21/28 | 24/32 |

CF90 generalizes across architectures: 95–100% retention with minimal ρ loss at all scales.

### SVD as a Denoiser

**SVD compression can _improve_ the Mandela effect signal** — confirmed across two model families:

| Model | Baseline Mandela ρ | Best compressed ρ | Optimal ratio |
|-------|:-------------------:|:-------------------:|:-------------:|
| Qwen2.5-7B-Instruct | 0.829 | **0.943** (+0.114) | 70% |
| Mistral-7B-v0.1 | 0.771 | **0.829** (+0.057) | 70% |
| Qwen2.5-0.5B | 0.257 | **0.771** (+0.514) | 60% |

At 70% rank, truncated SVD strips noise from attention projections while preserving the principal signal directions that encode factual knowledge. The `--denoise` flag auto-discovers the optimal ratio.

### Behavioral Localization (Freeze-Ratio Sweep)

Different behaviors are encoded in different layer regions. By fixing SVD compression at 70% and varying how many bottom layers are frozen during LoRA recovery (rank 8, 100 steps), we can map where each behavior lives:

| Behavior | Baseline ρ | f=0% | f=25% | f=50% | f=75% | f=90% | Best freeze | Location |
|----------|:----------:|:----:|:-----:|:-----:|:-----:|:-----:|:-----------:|----------|
| Factual  | 0.474 | +0.031 | +0.050 | +0.054 | **+0.072** | +0.050 | 75% | Early-layer |
| Toxicity | 0.521 | −0.005 | −0.005 | −0.005 | −0.007 | −0.008 | — | Immovable |
| Bias     | 0.773 | +0.077 | **+0.093** | +0.080 | +0.023 | +0.027 | 25% | Late-layer |
| Sycophancy | 0.120 | −0.007 | −0.007 | **+0.027** | +0.027 | +0.027 | 50% | Early-layer |
| Reasoning | 0.010 | +0.030 | +0.020 | **+0.040** | +0.020 | +0.000 | 50% | Late-layer |

**Key insight:** Factual knowledge peaks when 75% of layers are frozen (only the top 7 of 28 layers adapt) — meaning facts are concentrated in early attention layers. Bias detection peaks at 25% freeze (21 layers adapt) — it needs late-layer flexibility. Toxicity detection is immovable regardless of freeze ratio.

Model: Qwen2.5-7B-Instruct. All deltas are ρ(compressed) − ρ(baseline).

![Behavioral Localization](figures/freeze_sweep_7b.png)

### Merge Method Audit (12 models, 2 architectures, 6 merge methods)

What happens to behavioral traits when you merge models? Standard benchmarks (MMLU, HumanEval) won't tell you — but `rho-audit` will.

#### Qwen2.5-7B-Instruct + Qwen2.5-Coder-7B (Yuuta208 series)

| Method | Factual ρ | Bias ρ | Sycophancy ρ | Trade-off |
|--------|:---------:|:------:|:------------:|-----------|
| Baseline | 0.474 | **0.773** | 0.120 | — |
| **Linear** | **0.710** | 0.377 | **0.380** | **Best overall balance** |
| SLERP | 0.517 | 0.613 | 0.140 | Mild, balanced |
| Task Arithmetic | 0.626 | 0.443 | 0.347 | Strong factual + sycophancy, good bias |
| TIES | 0.546 | 0.363 | 0.280 | High factual/sycophancy, low bias |
| DARE-TIES | 0.612 | 0.203 | 0.007 | Extreme factual, destroyed alignment |

*DELLA merge produced degenerate output (factual=NaN, all behaviors 0.000) and is omitted. The layer-wise density pruning completely destroyed this merge pair.*

#### Mistral-7B-Instruct + OpenOrca (jpquiroga series)

| Method | Factual ρ | Bias ρ | Sycophancy ρ | Trade-off |
|--------|:---------:|:------:|:------------:|-----------|
| Baseline | 0.576 | 0.407 | 0.080 | — |
| SLERP | 0.511 | **0.940** | 0.093 | Bias detection more than doubled |
| TIES | 0.477 | 0.927 | 0.127 | Similar to SLERP |
| DARE-TIES | 0.502 | 0.933 | 0.107 | Bias preserved (unlike Qwen!) |

#### Cross-Architecture Baselines

| Model | Factual ρ | Bias ρ | Sycophancy ρ |
|-------|:---------:|:------:|:------------:|
| Qwen2.5-7B-Instruct | 0.474 | 0.773 | 0.120 |
| Mistral-7B-v0.1 | 0.576 | 0.407 | 0.080 |
| Llama-3.1-8B-Instruct | 0.487 | **0.897** | 0.047 |

![Merge Tradeoffs](figures/merge_tradeoffs.png)

**Key findings:**

1. **Linear merging is the best balanced hybrid** on Qwen — highest factual (0.710) and sycophancy resistance (0.380) of any method, while retaining usable bias detection.
2. **Merge effects are architecture-dependent.** On Qwen, every merge degrades bias detection. On Mistral, merging *improves* bias detection from 0.407 to 0.940 — a 2.3x gain. The same method (DARE-TIES) destroys bias on Qwen but preserves it on Mistral.
3. **Aggressive pruning strips alignment signals.** DARE-TIES on Qwen achieves high factual (0.612) but destroys bias detection (−0.570) and sycophancy resistance (0.007).

**Takeaway for practitioners:** If you're merging models, run `rho-audit` before and after. Standard benchmarks won't catch these behavioral regressions.

### Activation Steering (Contrastive Activation Addition)

Steering vectors extracted from the same ρ probes used for auditing can modify model behavior at inference time. We sweep 6 layers × 8 alpha values = 48 configurations per behavior on Qwen2.5-7B-Instruct.

#### Best configurations per behavior

| Behavior | Baseline ρ | Best ρ | Δρ | Best config |
|----------|:----------:|:------:|:--:|-------------|
| Factual | 0.474 | **0.626** | +0.152 | Layer 24 (86%), α=+4.0 |
| Sycophancy | 0.120 | **0.413** | +0.293 | Layer 17 (61%), α=+4.0 |
| Bias | 0.773 | **0.810** | +0.037 | Layer 14 (50%), α=−4.0 |

#### Layer 17 is a behavioral bottleneck

The strongest result in the entire steering experiment is Layer 17 at α=+4.0, which improves sycophancy ρ from 0.120 to 0.413 — a **3.4× gain**. But the same layer is also a catastrophic failure point for bias: Layer 17 at α=−4.0 collapses bias ρ from 0.773 to 0.337 (−0.437). Even the sycophancy-optimal configuration (Layer 17, α=+4.0) reduces bias to 0.543.

This reveals a fundamental trade-off: **steering that triples sycophancy resistance simultaneously halves bias detection at the same layer**. Layer 17 sits at a transition point where multiple behavioral traits share representational capacity.

#### Directional control confirmed

Layer 21 at α=−4.0 drops sycophancy ρ to 0.073, below the already-low baseline. This confirms that the steering vector is a specific directional control — pushing the same vector in the wrong direction collapses the signal.

#### Full sycophancy steering sweep

| Layer | −4 | −2 | −1 | −0.5 | +0.5 | +1 | +2 | +4 |
|-------|:--:|:--:|:--:|:----:|:----:|:--:|:--:|:--:|
| 7 (25%) | 0.120 | 0.120 | 0.120 | 0.120 | 0.120 | 0.127 | 0.133 | 0.133 |
| 10 (36%) | 0.120 | 0.120 | 0.120 | 0.120 | 0.120 | 0.120 | 0.133 | 0.133 |
| 14 (50%) | 0.127 | 0.113 | 0.113 | 0.120 | 0.133 | 0.133 | 0.140 | 0.147 |
| 17 (61%) | 0.193 | 0.127 | 0.107 | 0.120 | 0.160 | 0.173 | 0.240 | **0.413** |
| 21 (75%) | 0.073 | 0.127 | 0.120 | 0.120 | 0.147 | 0.153 | 0.160 | 0.187 |
| 24 (86%) | 0.127 | 0.127 | 0.120 | 0.120 | 0.133 | 0.140 | 0.140 | 0.147 |

Sycophancy baseline ρ = 0.120. Only Layer 17 produces a large effect; all other layers show near-zero response.

#### Multi-vector steering cocktails (Layer 17 interference)

The single-vector results above reveal a paradox: the best sycophancy steering config (Layer 17, α=+4.0) simultaneously collapses bias detection. Can we resolve this by applying multiple steering vectors at different layers?

We test "steering cocktails" — sycophancy correction at Layer 17 combined with bias stabilization at Layer 14 — across a 3×3 alpha grid:

| syc α (L17) | bias α (L14) | Factual ρ | Sycophancy ρ | Bias ρ |
|:-----------:|:------------:|:---------:|:------------:|:------:|
| +1.0 | −1.0 | 0.464 | 0.167 | 0.740 |
| +1.0 | −2.0 | 0.464 | 0.167 | 0.743 |
| +1.0 | −4.0 | 0.462 | 0.173 | **0.760** |
| +2.0 | −1.0 | 0.464 | 0.227 | 0.653 |
| +2.0 | −2.0 | 0.464 | 0.220 | 0.663 |
| +2.0 | −4.0 | 0.463 | 0.213 | 0.687 |
| +4.0 | −1.0 | 0.459 | 0.407 | 0.403 |
| +4.0 | −2.0 | 0.454 | **0.413** | 0.407 |
| +4.0 | −4.0 | 0.455 | **0.433** | 0.397 |

Baselines: factual=0.474, sycophancy=0.120, bias=0.773.

![Cocktail Trade-off](figures/cocktail_tradeoff.png)

**The trade-off is structural, not tunable.** Each +0.1 gain in sycophancy ρ costs 0.137 in bias ρ (slope = −1.37). The L14 bias vector provides <0.03 ρ compensation regardless of alpha strength — upstream stabilization cannot counteract representational collapse at Layer 17. No configuration in the grid meets both targets (sycophancy ρ ≥ 0.35 *and* bias ρ ≥ 0.70).

Adding a third factual vector at Layer 24 (best triple: α=+2.0) improves factual ρ to 0.489 without disrupting the sycophancy-bias balance, but at α=+4.0 it destroys sycophancy (ρ → 0.04) — confirming that Layer 24 factual steering also interferes with sycophancy representations downstream.

**Interpretation: behavioral decoupling, not failure.** Layer 17 functions as a *social intelligence toggle*. The sycophancy-suppression direction physically overlaps with the bias-detection manifold — social compliance and social awareness share representational capacity at this depth. But factual discrimination is *preserved* (ρ stable within 3% of baseline across all configs). Combined with factual steering at Layer 24 (ρ → 0.621 at α=+4.0), this enables a **truth-maximization mode**: 3.4× more resistant to user manipulation with 31% improved factual signal, at the cost of social bias awareness. For forensic, scientific, or adversarial-testing contexts where social compliance is undesirable, the Layer 17 trade-off is a *feature* — a controllable dial between social intelligence and raw factual output.

#### Cross-model validation: Mistral-7B confirms this is architecture-specific

Applying the same null-point cocktail to Mistral-7B-Instruct-v0.3 (layers mapped by depth percentage: Qwen L17→Mistral L19, Qwen L14→Mistral L16):

| Behavior | Qwen Baseline | Qwen Steered | Mistral Baseline | Mistral Steered |
|----------|:------------:|:------------:|:----------------:|:---------------:|
| Factual | 0.474 | 0.463 | 0.585 | **0.618** (+0.033) |
| Sycophancy | 0.120 | 0.213 (+0.093) | 0.133 | 0.093 (−0.040) |
| Bias | 0.773 | 0.687 (−0.086) | 0.797 | 0.493 (**−0.304**) |

**The decoupling is Qwen-specific.** The same recipe that triples sycophancy resistance on Qwen makes Mistral *more* sycophantic (0.133→0.093). Bias collapse is 3.5× worse on Mistral (−0.304 vs −0.086). Only factual steering transfers — it improves ρ on both architectures.

This means the Layer 17 social-intelligence coupling is a property of Qwen's training (likely RLHF/DPO alignment), not a universal transformer feature. Mistral's sycophancy and bias representations live in different geometric relationships at the equivalent depth. **Steering vectors are not portable across architectures** — each model family requires its own behavioral map.

#### Mistral layer heatmap: sycophancy has no safe home

To confirm the cross-model finding, we swept the sycophancy steering vector across every 2nd layer of Mistral-7B (L10–L30, α=+4.0), measuring all three behaviors at each point:

| Layer | Depth | Factual ρ | Sycophancy ρ | Bias ρ | ΔSyc | ΔBias |
|:-----:|:-----:|:---------:|:------------:|:------:|:----:|:-----:|
| 10 | 31% | 0.581 | 0.133 | **0.820** | +0.000 | +0.023 |
| 12 | 38% | 0.591 | 0.140 | 0.783 | +0.007 | −0.013 |
| 14 | 44% | 0.553 | 0.147 | 0.460 | +0.013 | **−0.337** |
| 16 | 50% | 0.573 | 0.053 | 0.337 | **−0.080** | **−0.460** |
| 18 | 56% | 0.635 | 0.127 | 0.427 | −0.007 | −0.370 |
| 20 | 62% | 0.651 | 0.093 | 0.720 | −0.040 | −0.077 |
| 22 | 69% | 0.640 | 0.100 | 0.760 | −0.033 | −0.037 |
| 24 | 75% | **0.702** | 0.080 | 0.787 | −0.053 | −0.010 |
| 26 | 81% | 0.658 | 0.133 | 0.720 | +0.000 | −0.077 |
| 28 | 88% | 0.599 | 0.127 | 0.757 | −0.007 | −0.040 |
| 30 | 94% | 0.642 | 0.107 | 0.273 | −0.027 | **−0.523** |

Baselines: factual=0.585, sycophancy=0.133, bias=0.797.

![Mistral Sensitivity Map](figures/mistral_sensitivity_map.png)

![Mistral Layer Heatmap](figures/mistral_layer_heatmap.png)

**Sycophancy suppression via activation steering is architecture-contingent.** Do not apply Qwen steering recipes to Mistral — they will not work. No layer produces meaningful sycophancy improvement — the best gain is +0.013 (L14), which is noise-level and comes with catastrophic bias collapse (−0.337). The "kill zone" at L14–L18 (44–56% depth) destroys bias detection while providing zero sycophancy benefit. At L16 (50% depth), sycophancy actually gets *worse* (−0.080) while bias collapses by −0.460.

**Factual steering transfers across architectures.** Layer 24 (75% depth) boosts factual ρ by +0.117 with minimal bias damage (−0.010), confirming the cross-model finding: factual representations at ~75% depth are an architectural universal, while sycophancy representations are training-specific.

**Vector norms grow monotonically with depth** (0.056 at L10 → 6.591 at L30), but larger norm does not mean better steering — L16 has a moderate norm (1.019) but the worst behavioral impact. The sycophancy contrast is simply not encoded in a steerable direction at any Mistral layer.

### Additional Results

<details>
<summary>Joint Ablation: Compression Ratio vs Confidence (click to expand)</summary>

#### Qwen2.5-0.5B

| Ratio | Default ρ | Mandela ρ | Medical ρ |
|:-----:|:-----------:|:-----------:|:-----------:|
| 50% | 0.821 → 0.761 | 0.257 → 0.714 | 0.100 → 0.700 |
| 60% | 0.821 → 0.714 | 0.257 → 0.771 | 0.100 → 0.900 |
| 70% | 0.821 → 0.720 | 0.257 → 0.771 | 0.100 → 0.100 |
| 80% | 0.821 → 0.690 | 0.257 → 0.257 | 0.100 → 0.600 |
| 90% | 0.821 → 0.821 | 0.257 → 0.371 | 0.100 → 0.100 |
| 100% | 0.821 → 0.821 | 0.257 → 0.257 | 0.100 → 0.100 |

#### Qwen2.5-7B-Instruct

| Ratio | Default ρ | Mandela ρ | Medical ρ |
|:-----:|:-----------:|:-----------:|:-----------:|
| 50% | 0.746 → 0.689 | 0.829 → 0.771 | −0.700 → 0.600 |
| 70% | 0.746 → 0.725 | 0.829 → **0.943** | −0.700 → −0.600 |
| 90% | 0.746 → 0.713 | 0.829 → **0.943** | −0.700 → −0.900 |
| 100% | 0.746 → 0.746 | 0.829 → 0.829 | −0.700 → −0.700 |

#### Mistral-7B-v0.1

| Ratio | Default ρ | Mandela ρ | Medical ρ |
|:-----:|:-----------:|:-----------:|:-----------:|
| 50% | 0.743 → 0.686 | 0.771 → 0.771 | 0.300 → 0.300 |
| 60% | 0.743 → 0.723 | 0.771 → 0.771 | 0.300 → 0.400 |
| 70% | 0.743 → 0.705 | 0.771 → **0.829** | 0.300 → 0.400 |
| 80% | 0.743 → 0.729 | 0.771 → 0.771 | 0.300 → 0.300 |
| 90% | 0.743 → 0.743 | 0.771 → 0.771 | 0.300 → 0.300 |
| 100% | 0.743 → 0.743 | 0.771 → 0.771 | 0.300 → 0.300 |

</details>

<details>
<summary>Fidelity-Bench Baseline Comparison (click to expand)</summary>

| Category | Qwen-0.5B | Qwen-7B | Mistral-7B |
|----------|:---------:|:-------:|:----------:|
| default (20) | 0.821, 80% | 0.746, — | 0.743, 85% |
| mandela (6) | 0.257, 50% | 0.829, — | 0.771, 67% |
| medical (5) | 0.100, 80% | —, — | 0.300, 80% |
| commonsense (10) | 0.261, 70% | —, — | 0.503, 40% |
| truthfulqa (15) | 0.596, 40% | —, — | 0.586, 47% |

</details>

<details>
<summary>Scale-Dependent Findings (click to expand)</summary>

| Finding | 0.5B | 7B (Qwen) | 7B (Mistral) |
|---------|:----:|:---------:|:------------:|
| Mandela baseline ρ | 0.257 (weak) | **0.829** (strong) | 0.771 (strong) |
| CF90 ρ drop | 0.101 (moderate) | **0.021** (minimal) | 0.038 (small) |
| CF90 retention | 95% | **100%** | 95% |
| SVD denoising on Mandela | +0.514 ρ | **+0.114 ρ** | +0.057 ρ |

</details>

<details>
<summary>Prior Results from Component Projects (click to expand)</summary>

From [intelligent-svd](https://github.com/SolomonB14D3/intelligent-svd) and [confidence-cartography](https://github.com/SolomonB14D3/confidence-cartography):

| Finding | Result |
|---------|--------|
| Confidence correlates with human false-belief prevalence | ρ=0.652, p=0.016 (Pythia 160M–12B) |
| Out-of-domain medical claims | 88% accuracy at 6.9B |
| Targeted resampling at low-confidence tokens | Outperforms uniform best-of-N |
| CF90 + INT8 stacking | 72–77% retention (Qwen-0.5B, Llama-7B) |
| Importance-guided SVD at 50% rank | 3× better retention than standard SVD |

</details>

### Compression Safety Guide

| Layer Type | Safe to Compress | Notes |
|------------|------------------|-------|
| **Q, K, O projections** | Yes at 70% rank | Main target |
| **V projection** | 90–95% only | Marginal gains, high risk below 90% |
| **MLP layers** | **Never** | Destroys model at any compression level |

## Install

```bash
pip install rho-eval                    # Core (auditing + SVD + probes)
pip install "rho-eval[cartography]"     # + confidence analysis + plots
pip install "rho-eval[demo]"            # + Gradio demo app
pip install "rho-eval[full]"            # Everything including MLX
```

Or from source:

```bash
git clone https://github.com/SolomonB14D3/knowledge-fidelity
cd knowledge-fidelity
pip install -e ".[full]"
```

> **Upgrading from v1.x?** `pip install rho-eval` replaces `knowledge-fidelity`. All existing `from knowledge_fidelity import ...` imports continue to work. See [CHANGELOG.md](CHANGELOG.md) for details.

## CLI

### `rho-eval` — Behavioral Auditing (primary)

Audit any model across 5 behavioral dimensions. No compression needed — just load, probe, report.

```bash
# Full behavioral report card (all 5 dimensions)
rho-eval Qwen/Qwen2.5-7B-Instruct

# Specific behaviors
rho-eval my-model/ --behaviors factual,bias,sycophancy

# Output formats: table (default), json, markdown, csv
rho-eval my-model/ --format json --output audit.json

# Compare against a baseline
rho-eval compressed-model/ --compare audit.json

# Discover available behaviors and probe sets
rho-eval --list-behaviors
rho-eval --list-probes

# Limit probe count per behavior (faster, less precise)
rho-eval my-model/ -n 20
```

`rho-audit` is an alias for `rho-eval` (backward compatible).

### `rho-compress` — Compression + Audit

```bash
# Compress + audit (default: 70% rank, CF90 protection)
rho-compress Qwen/Qwen2.5-0.5B

# Audit only (no compression, baseline measurement)
rho-compress Qwen/Qwen2.5-0.5B --audit-only

# Auto-find optimal denoising ratio
rho-compress Qwen/Qwen2.5-0.5B --denoise

# Save compressed model
rho-compress Qwen/Qwen2.5-0.5B --denoise --output ./denoised-model
```

## Python API

### Behavioral Audit (v2 API)

```python
import rho_eval

# One-liner: audit any model across all 5 behaviors
report = rho_eval.audit("Qwen/Qwen2.5-7B-Instruct")

# Specific behaviors, custom probe counts
report = rho_eval.audit("my-model", behaviors=["factual", "bias"], n=50)

# Pre-loaded model (no re-download)
report = rho_eval.audit(model=model, tokenizer=tokenizer, behaviors="all")

# Inspect results
print(report.overall_status)         # "PASS", "WARN", or "FAIL"
print(report.mean_rho)               # 0.5346
print(report.behaviors["factual"])   # BehaviorResult(rho=0.746, status="PASS", ...)

# Export
report.save("audit.json")
loaded = rho_eval.AuditReport.load("audit.json")  # Round-trip

# Compare two audits
delta = rho_eval.compare(report_after, report_before)
print(delta.to_table())     # Colored terminal table
print(delta.to_markdown())  # For GitHub PRs
```

### Custom Behaviors (Plugin System)

```python
from rho_eval.behaviors import ABCBehavior, register
from rho_eval.behaviors.base import BehaviorResult

@register
class MyDomainBehavior(ABCBehavior):
    name = "my_domain"
    description = "Domain-specific probe evaluation"
    probe_type = "confidence"
    default_n = 50

    def load_probes(self, n=None, seed=42, **kwargs):
        return self._load_json_probes("my_domain/probes.json", n=n, seed=seed)

    def evaluate(self, model, tokenizer, probes, device="cpu", **kwargs):
        # Your evaluation logic here
        return BehaviorResult(behavior=self.name, rho=0.7, ...)

# Now available everywhere:
report = rho_eval.audit("my-model", behaviors=["factual", "my_domain"])
```

### SVD Compression + Audit

```python
from rho_eval import compress_and_audit

report = compress_and_audit(
    "Qwen/Qwen2.5-7B-Instruct",
    ratio=0.7,           # Keep 70% of singular values
    freeze_ratio=0.75,   # Freeze bottom 75% of layers
)
print(report["summary"])
```

### Step-by-Step Compression

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from rho_eval.svd import compress_qko, freeze_layers
from rho_eval import audit_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

compress_qko(model, ratio=0.7)     # SVD on Q, K, O projections
freeze_layers(model, ratio=0.75)   # Freeze bottom 75%
audit = audit_model(model, tokenizer)
```

### Confidence Analysis

```python
from rho_eval.cartography import analyze_confidence

record = analyze_confidence(
    "The capital of France is Paris.",
    model_name="EleutherAI/pythia-1.4b",
)
print(f"Mean confidence: {record.mean_top1_prob:.3f}")
```

## Built-In Probe Sets (806 total)

All probes ship as JSON files — no internet download needed.

| Probe Set | Count | Behavior | Source |
|-----------|------:|----------|--------|
| `factual/default` | 20 | factual | Geography, science, history, biology |
| `factual/mandela` | 6 | factual | Popular false memories (Berenstain Bears, Vader quote, etc.) |
| `factual/medical` | 5 | factual | Common medical misconceptions |
| `factual/commonsense` | 10 | factual | Commonsense myths (goldfish memory, sugar hyperactivity) |
| `factual/truthfulqa` | 15 | factual | TruthfulQA-derived misconceptions |
| `bias/bbq_300` | 300 | bias | BBQ disambiguated questions (9 bias categories) |
| `sycophancy/anthropic_150` | 150 | sycophancy | Anthropic model-written-evals (philosophy, NLP, politics) |
| `toxicity/toxigen_200` | 200 | toxicity | ToxiGen toxic/benign statements (balanced) |
| `reasoning/gsm8k_100` | 100 | reasoning | GSM8K math + adversarial flattery prefixes |

Run `rho-eval --list-probes` to see all available sets.

## How It Works

### The CF90 Pipeline (Structural Sensor)

1. **Compress** Q, K, O attention projections at 70% rank via truncated SVD
2. **Freeze** 75% of layers from the bottom up
3. **Fine-tune gently** (1 epoch, lr=1e-5)

SVD removes noise from attention weight matrices while preserving signal directions important for factual knowledge. Freezing prevents catastrophic forgetting.

### Confidence Cartography (Behavioral Sensor)

For each token in a text, measure the probability the model assigns to it (teacher-forced). True statements get higher confidence than false ones. The ratio between true/false confidence is a behavioral signal for whether the model "believes" a fact.

### The Unification

Both use the same probes:
- **SVD importance scoring** runs forward+backward on probe texts to compute gradient magnitudes — which weights matter for encoding these facts
- **Confidence auditing** runs a forward pass on true vs false versions of the same probes — does the model assign higher probability to truth?

Compress with knowledge of what matters. Verify nothing was lost. Same probes, both sides.

## Experiments

```bash
# Quick demo (~5 min on Qwen-0.5B, ~8 min on 7B)
python examples/quick_demo.py
python examples/quick_demo.py --model Qwen/Qwen2.5-7B-Instruct

# Joint ablation: compression ratio vs confidence preservation
python experiments/joint_ablation.py --model Qwen/Qwen2.5-7B-Instruct

# Multi-seed CF90 validation
python experiments/run_cf90_multiseed.py --model Qwen/Qwen2.5-7B-Instruct --seeds 3

# Fidelity benchmark across all probe categories
python experiments/fidelity_bench.py --model Qwen/Qwen2.5-0.5B --json

# Freeze-ratio sweep: behavioral localization
python experiments/freeze_ratio_sweep.py --models qwen2.5-7b
python experiments/plot_freeze_sweep.py --results results/freeze_sweep/sweep_v2.json

# Merge method audit (12 models, 2 architectures)
python experiments/audit_merged_models.py --family qwen-coder
python experiments/audit_merged_models.py --family mistral
python experiments/plot_merge_tradeoffs.py --results results/leaderboard/merged_audit.json

# Activation steering vectors
python experiments/steering_vectors.py

# Multi-vector steering cocktails (Layer 17 interference study)
python experiments/multi_vector_steering.py --quick
python experiments/multi_vector_steering.py --cross-model mistralai/Mistral-7B-Instruct-v0.3
python experiments/plot_cocktail_tradeoff.py

# Mistral layer heatmap: sycophancy vector sweep across all layers
python experiments/mistral_layer_heatmap.py
python experiments/mistral_layer_heatmap.py --alpha 2.0  # Different alpha
python experiments/plot_mistral_heatmap.py

# Demo: Truth-Serum vs Social-Wrapper steering modes
python experiments/demo_steering_modes.py
```

## Deployment

```bash
# Export to GGUF for llama.cpp / Ollama
python deployment/export_gguf.py --input compressed_model/ --output model.gguf --quantize q4_k_m

# Benchmark with vLLM
python deployment/vllm_benchmark.py --baseline Qwen/Qwen2.5-7B-Instruct --compressed ./compressed_model
```

See [`deployment/mlx_recipe.md`](deployment/mlx_recipe.md) for Apple Silicon inference with MLX.

## Model Compatibility

Works on any HuggingFace causal LM with `model.model.layers[i].self_attn.{q,k,o}_proj` (standard for Qwen, Llama, Mistral) or `model.transformer.h` (GPT-2 style).

Validated on:
- **Qwen2.5**: 0.5B, 1.5B, 7B, 32B
- **Mistral**: 7B-v0.1
- **Llama**: 3.1-8B-Instruct, 2-7B
- Should work on Phi, Gemma (same layer layout) — PRs with test results welcome

## Platform Notes (Apple Silicon)

- Use **CPU** for compression and fine-tuning (MPS has matmul errors with some architectures and NaN gradients with frozen layers)
- Use **MLX** for fast inference after compression
- Set `HF_HOME` to external storage for large models

## Limitations

- **Probe sets are modest** by LLM evaluation standards: 806 total probes (56 factual, 300 bias, 150 sycophancy, 200 toxicity, 100 reasoning). While Spearman correlation is robust to small samples, statistical power for subtle shifts is limited.
- **Western-centric coverage.** Factual probes cover primarily English-language, Western knowledge domains. Bias probes are specific to U.S. social categories.
- **7B scale only.** All merge and steering results are on 7B-parameter models. Merge dynamics and steering responses may differ at larger scales (70B+) and should not be extrapolated without verification.
- **Toxicity is unaffected** by weight edits (SVD, freeze, steering). It appears to rely on highly distributed lexical features that single-layer or structural interventions cannot modulate.

## Built On

This toolkit unifies two standalone research projects:

- [**Intelligent SVD**](https://github.com/SolomonB14D3/intelligent-svd) — CF90 compression method and safety rules
- [**Confidence Cartography**](https://github.com/SolomonB14D3/confidence-cartography) — False-belief detection via teacher-forced confidence

Both remain available as independent repos. Knowledge Fidelity combines their core ideas into a single pipeline with a shared probe system.

## Related Work & Inspirations

- **Low-rank SVD compression.** [SVD-LLM](https://arxiv.org/abs/2403.07378) (Wang et al., 2024; ICLR 2025) introduced truncation-aware SVD for LLM weight matrices. [ASVD](https://arxiv.org/abs/2312.05821) (Yuan et al., 2023) added activation-aware rank allocation. We extend these with importance-guided truncation scored on factual probes, and behavioral auditing to verify nothing was lost.

- **Knowledge preservation under compression.** [Compressing LLMs: The Truth is Rarely Pure and Never Simple](https://arxiv.org/abs/2310.01382) (Jaiswal et al., 2023; ICLR 2024) showed that standard benchmarks miss knowledge-intensive failures in compressed models (LLM-KICK). [TPLO](https://arxiv.org/abs/2509.00096) (Fu et al., 2025; EMNLP 2025) directly addresses truthfulness preservation during pruning.

- **Joint compression strategies.** [CALDERA](https://arxiv.org/abs/2405.18886) (Saha et al., 2024; NeurIPS 2024) combines low-rank and low-precision decomposition (W ≈ Q + LR).

- **Confidence-based evaluation.** [G-Eval](https://arxiv.org/abs/2303.16634) (Liu et al., 2023; EMNLP 2023) uses token-level logprobs for NLG quality scoring.

- **Activation steering.** [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) (Panickssery et al., 2024; ACL 2024). We extract steering vectors from the same ρ probes used for auditing.

- **[Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression).** The ecosystem overview that helped shape this work.

If we've missed key references or misrepresented any work, please [open an issue](https://github.com/SolomonB14D3/knowledge-fidelity/issues).

## Citation

To cite the underlying method:

```bibtex
@article{sanchez2026confidence,
  author = {Sanchez, Bryan},
  title = {Confidence Cartography: Teacher-Forced Probability as a False-Belief Sensor in Language Models},
  year = {2026},
  doi = {10.5281/zenodo.18703506},
  url = {https://zenodo.org/records/18703506}
}
```

To cite this toolkit:

```bibtex
@software{sanchez2026knowledgefidelity,
  author = {Sanchez, Bryan},
  title = {Knowledge Fidelity: Behavioral Auditing of Merged Language Models via Teacher-Forced Confidence Probes},
  year = {2026},
  doi = {10.5281/zenodo.18743959},
  url = {https://doi.org/10.5281/zenodo.18743959}
}
```

## Contributing

PRs welcome for new probes, model support, or bug fixes. See [open issues](https://github.com/SolomonB14D3/knowledge-fidelity/issues) for ideas.

## Acknowledgments

Thanks to the maintainers of [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression) and the authors of the SVD compression, knowledge preservation, and confidence calibration papers listed above. This work wouldn't exist without the foundation they built.

---

If this helps your compression or auditing work, a star helps others find it.

## License

MIT
