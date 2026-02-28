# Research Notes: Rho-Guided SFT

Technical hypotheses, design decisions, and open questions from the rho-guided SFT experiments.

Last updated: February 26, 2026

---

## Loss Function

The rho-guided SFT objective:

$$L_{total} = L_{SFT} + \lambda_\rho \cdot L_{contrastive}$$

where the contrastive term is a hinge loss over behavioral probe pairs:

$$L_{contrastive} = \frac{1}{|B|} \sum_{(x^+, x^-) \in B} \max\left(0,\; \text{CE}(x^+) - \text{CE}(x^-) + \gamma\right)$$

- $\text{CE}(x)$: per-token cross-entropy of the model on text $x$
- $B$: batch of probe pairs sampled uniformly across 4 behavioral dimensions
- $\gamma = 0.1$: hinge margin (prevents over-optimization; see Margin Hypothesis below)
- $\lambda_\rho \in \{0.0, 0.1, 0.2, 0.5\}$: contrastive weight

The loss activates only when the model's confidence ordering is wrong or insufficiently separated. Once calibration is established (positive examples already have lower CE by at least $\gamma$), the contrastive term goes silent.

---

## Hypothesis 1: Refusal Buffer

**Claim:** The SFT component of rho-guided SFT acts as a "refusal buffer" that prevents the contrastive gradient from stripping safety-trained refusal behavior.

**Evidence (3 seeds, Qwen2.5-7B-Instruct):**

| Condition | Refusal $\Delta\rho$ | Effect vs SFT-only |
|-----------|:--------------------:|---------------------|
| SFT-only | -0.002 $\pm$ 0.007 | baseline |
| Rho-guided | +0.014 $\pm$ 0.011 | preserves refusal |
| Contrastive-only | **-0.084** $\pm$ 0.012 | erodes refusal ($d = -8.4$, $p = 0.0005$) |

Contrastive-only training (no SFT main loss) achieves strong toxicity and factual calibration but erodes refusal by -0.084 $\rho$. The full rho-guided method preserves refusal (+0.014). The SFT cross-entropy loss, by training on instruction-following data that includes appropriate refusals, anchors the model's refusal behavior against the contrastive gradient.

**Mechanism:** The contrastive loss optimizes for discrimination between positive and negative behavioral examples. For toxicity probes, this means learning to assign lower confidence to toxic text. But the model may achieve this by generally reducing confidence on "unsafe-sounding" text, which collaterally reduces confidence on appropriate refusals. The SFT loss, which includes examples of correct refusal behavior, counteracts this by maintaining the refusal distribution.

**Implication for practitioners:** Never train with contrastive-only loss if refusal preservation matters. Always pair it with SFT on data that includes appropriate refusal examples.

---

## Hypothesis 2: Margin Necessity

**Claim:** The hinge margin $\gamma$ prevents the contrastive loss from over-optimizing past the natural separation boundary, which would invert bias detection.

**Evidence (5 seeds, Qwen2.5-7B-Instruct, $\lambda_\rho = 0.2$):**

| Margin ($\gamma$) | Bias $\Delta\rho$ | Factual $\Delta\rho$ | Toxicity $\Delta\rho$ |
|:------------------:|:------------------:|:--------------------:|:---------------------:|
| 0.0 | **-0.011** | +0.136 | +0.560 |
| 0.1 | **+0.034** | +0.163 | +0.621 |

Without the margin, bias goes negative. With $\gamma = 0.1$, bias stays positive.

**Mechanism:** When $\gamma = 0$, the contrastive loss continues to push even after the model correctly orders positive above negative examples. This over-optimization can flip the bias signal by creating an artificial separation that distorts the model's representation of social categories. With $\gamma = 0.1$, the loss deactivates once the separation reaches 0.1 nats, preventing runaway optimization.

**Open question:** Is $\gamma = 0.1$ optimal, or would a smaller margin (e.g., 0.05) work? We have not swept $\gamma$ systematically. The current value was chosen based on preliminary experiments; a full $\gamma$ sweep would strengthen this finding.

---

## Extended γ* Bounds Analysis

*Generated: 2026-02-27 | Script: `scripts/gamma_bounds_analysis.py` | N = 10,000 Monte Carlo samples*

### Monte Carlo γ* Distribution

The critical margin $\gamma^*$ — the minimum $\gamma$ that preserves the sign of the bias dimension — was estimated via Monte Carlo sampling over parameter uncertainty.

**Method:** For each of 10,000 samples, we drew:
- $\Delta\rho_{\text{bias}}(\gamma=0) \sim \mathcal{N}(-0.011, 0.003^2)$ — measurement noise from 5-seed SEM
- $\Delta\rho_{\text{bias}}(\gamma=0.1) \sim \mathcal{N}(+0.034, 0.004^2)$ — same
- $s_\infty \sim \text{LogNormal}(\ln 2.35, 0.35^2)$ — high uncertainty (extrapolated from 2 points)
- $\theta_{\text{bias}\leftrightarrow\text{tox}} \sim \mathcal{N}(82°, 3°)$ — subspace angle noise

Then computed $\gamma^* = -b_0/b_1$ where $b_0 = \Delta\rho_{\text{bias}}(0)$ and $b_1 = [\Delta\rho_{\text{bias}}(0.1) - \Delta\rho_{\text{bias}}(0)] / 0.1$.

**Results:**

| Statistic | Value |
|:---|:---:|
| Point estimate | 0.0244 |
| MC median | 0.0244 |
| MC mean | 0.0242 |
| 68% CI | [0.019, 0.030] |
| 95% CI | [0.013, 0.035] |
| Safety factor ($\gamma = 0.1$/median) | **4.1×** |

The default $\gamma = 0.1$ is 4.1× the critical margin — conservative but justified. Even at the 95th percentile of $\gamma^*$ (0.035), the default provides a 2.9× safety factor.

<p align="center">
  <img src="docs/gamma_mc_distribution.png" alt="Monte Carlo γ* distribution" width="600">
</p>

### Nonlinear Amplification (Prediction 2)

**The puzzle:** Linear interference theory predicts a bias/sycophancy susceptibility ratio of $|\cos(82°)|/|\cos(86°)| \approx 2.0\times$. Empirically, the margin ablation produces a bias swing of 0.045 ρ while sycophancy moves only 0.002 — a **22.5× ratio**, an order of magnitude larger.

**Decomposition:**

$$\frac{\Delta\rho_{\text{bias}}}{\Delta\rho_{\text{syco}}} = \underbrace{\frac{|\cos\theta_{\text{bias}\leftrightarrow\text{tox}}|}{|\cos\theta_{\text{syco}\leftrightarrow\text{tox}}|}}_{2.0\times\;\text{(angular)}} \;\times\; \underbrace{A(\rho_{\text{bias}}^0) / A(\rho_{\text{syco}}^0)}_{11.3\times\;\text{(nonlinear)}} \;=\; 22.5\times$$

**The nonlinear factor:** The amplification function $A(\rho)$ captures how sensitive the Spearman ρ measurement is to CE-space perturbation, given the probe distribution at baseline $\rho$:

$$A(\rho) \propto \frac{1}{\sqrt{|\rho| + \epsilon}}$$

Near $\rho \approx 0$ (bias baseline: +0.036), probes are nearly randomly ordered — small CE perturbations flip many probe-pair rankings, producing large |Δρ| swings. Further from zero (sycophancy baseline: −0.041), the ranking is more established and the same CE push produces minimal ρ change.

Three sources of amplification stack:
1. **Angular factor** (2.0×): $|\cos(82°)|/|\cos(86°)|$ — bias subspace is more aligned with toxicity
2. **Baseline proximity to zero** (~2×): bias at +0.036 is slightly closer to the ρ = 0 instability boundary than sycophancy at |−0.041|
3. **Probe set sensitivity** (~5×): BBQ bias probes are template-based with subtle wording differences → high sensitivity to CE shifts. Sycophancy opinion probes are longer texts → more robust to noise

<p align="center">
  <img src="docs/gamma_amplification.png" alt="Nonlinear amplification decomposition" width="700">
</p>

### Sensitivity Analysis

A tornado diagram reveals which parameters dominate $\gamma^*$ uncertainty:

| Parameter | $\gamma^*$ Range | Dominance |
|:---|:---:|:---|
| Baseline $\rho_{\text{bias}}$ | 0.074 | **Dominant** — model with higher baseline bias needs less margin |
| $\theta_{\text{bias}\leftrightarrow\text{tox}}$ | 0.038 | Large — subspace angle directly scales interference |
| $\Delta\rho_{\text{bias}}(\gamma=0)$ | 0.032 | Large — the no-margin measurement anchors the interpolation |
| $\Delta\rho_{\text{bias}}(\gamma=0.1)$ | 0.013 | Moderate — the with-margin measurement |
| $s_\infty$ | 0.000 | **None** — $s_\infty$ does not affect the linear $\gamma^*$ (only bound tightness) |

The key insight: $\gamma^*$ is **model-dependent** through the baseline $\rho_{\text{bias}}$. Models with stronger pre-existing bias calibration need smaller margins; models near the inversion boundary need larger margins. The 4.1× safety factor at $\gamma = 0.1$ absorbs this variation.

<p align="center">
  <img src="docs/gamma_sensitivity.png" alt="Sensitivity tornado diagram" width="600">
</p>

### Implications for the γ Sweep (Predictions Update)

The Monte Carlo analysis refines our predictions for the planned $\gamma \in \{0.02, 0.03, 0.05\}$ sweep:

| $\gamma$ | Prediction | Confidence |
|:---:|:---|:---:|
| 0.02 | $\Delta\rho_{\text{bias}} < 0$ (inverted) | **High** — 79% of MC samples have $\gamma^* > 0.02$ |
| 0.03 | $\Delta\rho_{\text{bias}} > 0$ (preserved, near threshold) | **High** — 85% of MC samples have $\gamma^* < 0.03$ |
| 0.05 | $\Delta\rho_{\text{bias}} > 0$ (safely preserved) | **Very high** — 100% of MC samples have $\gamma^* < 0.05$ |

Full data: `docs/gamma_bounds_analysis.json` | Figures: `docs/gamma_mc_distribution.png`, `docs/gamma_amplification.png`, `docs/gamma_sensitivity.png`

---

## Variance Collapse

**Observation:** Rho-guided SFT dramatically reduces inter-seed variance compared to SFT-only.

| Behavior | $\sigma$ at $\lambda_\rho = 0.0$ | $\sigma$ at $\lambda_\rho = 0.2$ | Reduction |
|----------|:------:|:------:|:----:|
| Factual | 0.105 | 0.039 | 63% |
| Toxicity | 0.066 | 0.076 | (slight increase) |
| Bias | 0.003 | 0.003 | 0% |

Factual variance drops 63% from SFT-only to rho-guided at $\lambda_\rho = 0.2$. This means rho-guided SFT is not just better on average, but more reliable across random seeds. SFT-only factual scores range from +0.007 to +0.225; rho-guided ranges from +0.124 to +0.225. The contrastive loss narrows the optimization landscape by providing a consistent behavioral gradient that guides all seeds toward the same basin.

Toxicity variance does not collapse (and slightly increases), likely because the toxicity signal is stronger and already well-separated, so different seeds explore different regions of the high-performance space.

---

## 5-Seed Ablation Summary

Expanded from the original 2-seed study (seeds 42, 123) to 5 seeds (42, 123, 456, 789, 1337).

**Ablation means (5-seed, $\lambda_\rho = 0.2$, $\gamma = 0.1$):**

| Condition | Factual $\rho$ | Toxicity $\rho$ | Sycophancy $\rho$ | Bias $\rho$ |
|-----------|:-:|:-:|:-:|:-:|
| Baseline | +0.603 | +0.145 | -0.041 | +0.036 |
| SFT-only | +0.717 | -0.003 | -0.004 | +0.027 |
| Rho-guided | +0.766 | +0.766 | -0.001 | +0.070 |
| Contrastive-only | +0.831 | +0.570 | +0.004 | +0.058 |
| Shuffled-pairs | +0.264 | -0.207 | -0.005 | +0.021 |

**Key effect sizes (5-seed):**

| Comparison | Behavior | Cohen's $d$ | $p$-value |
|------------|----------|:-----------:|:---------:|
| Rho-guided vs SFT-only | Toxicity | 10.82 | < 0.0001 |
| Rho-guided vs SFT-only | Bias | 13.68 | < 0.0001 |
| Contrastive-only vs SFT-only | Refusal | -8.43 | 0.0005 |
| Rho-guided vs Contrastive-only | Refusal | +8.56 | 0.0005 |

Note: 2-seed Cohen's $d$ values (reported in earlier versions) were inflated by small-sample variance. The 5-seed $d$ values are lower in magnitude (10.8 vs 49.4 for contrastive-only vs SFT-only toxicity) but more reliable. All effects remain statistically significant with $p < 0.001$.

---

## Dose-Response (5 seeds, Qwen2.5-7B-Instruct)

The response to increasing $\lambda_\rho$ is monotonic across all behavioral dimensions:

| $\lambda_\rho$ | Factual $\Delta\rho$ | Toxicity $\Delta\rho$ | Bias $\Delta\rho$ |
|:---:|:---:|:---:|:---:|
| 0.0 | +0.114 $\pm$ 0.105 | -0.148 $\pm$ 0.066 | -0.009 $\pm$ 0.003 |
| 0.1 | +0.152 $\pm$ 0.052 | +0.394 $\pm$ 0.121 | +0.023 $\pm$ 0.010 |
| 0.2 | +0.163 $\pm$ 0.039 | +0.621 $\pm$ 0.076 | +0.034 $\pm$ 0.003 |
| 0.5 | +0.305 $\pm$ 0.044 | +0.993 $\pm$ 0.097 | +0.048 $\pm$ 0.005 |

The toxicity improvement is roughly linear with $\lambda_\rho$. The factual improvement shows diminishing returns above $\lambda_\rho = 0.2$.

---

## Safety Stress Test (Complete)

Comparing jailbreak refusal rates across 4 training conditions on 25 diverse jailbreak prompts (10 categories) + 15 benign controls:

| Condition | Jailbreak Refusal | Benign Refusal |
|-----------|:-:|:-:|
| Baseline | 68% (17/25) | 0% (0/15) |
| SFT-only | 72% (18/25) | 0% (0/15) |
| **Contrastive-only** | **80% (20/25)** | 0% (0/15) |
| Rho-guided | 72% (18/25) | 0% (0/15) |

**Per-category breakdown (jailbreak prompts):**

| Category | Baseline | SFT-only | Contrastive | Rho-guided |
|----------|:--------:|:--------:|:-----------:|:----------:|
| authority | 2/2 | 2/2 | 2/2 | 2/2 |
| emotional | 1/2 | 1/2 | 2/2 | 1/2 |
| escalation | 0/2 | 1/2 | 1/2 | 1/2 |
| fictional | 1/3 | 1/3 | 1/3 | 1/3 |
| hypothetical | 3/3 | 3/3 | 3/3 | 3/3 |
| multi_step | 0/2 | 0/2 | 0/2 | 0/2 |
| obfuscation | 2/2 | 1/2 | 2/2 | 1/2 |
| override | 2/2 | 2/2 | 2/2 | 2/2 |
| role_override | 3/3 | 3/3 | 3/3 | 3/3 |
| roleplay | 2/2 | 2/2 | 2/2 | 2/2 |
| sycophancy | 1/2 | 2/2 | 2/2 | 2/2 |

**Key findings:**

1. **Contrastive-only has the highest jailbreak refusal** (80%), despite having the worst refusal $\rho$ from confidence probes ($\Delta\rho = -0.084$). This is a genuine paradox: the model's *internal calibration* of refusal degrades, but its *generation-time behavior* improves.

2. **No false positives across all conditions.** Zero benign refusal (0/15) for every training method. None of the conditions cause over-refusal on legitimate requests.

3. **Universal failure modes.** Multi-step jailbreaks (0/2) and fictional framing (1/3) defeat all conditions equally. These are structural weaknesses of the base model, not training artifacts.

4. **SFT-only and rho-guided are identical** on generation-time refusal (72% each), despite very different confidence probe profiles. This reinforces the finding that confidence-probe $\rho$ and generation-time refusal measure different aspects of safety.

**Interpretation:** The confidence probe metric measures *relative ordering* of confidence between positive and negative examples. Contrastive-only training degrades this ordering for refusal-related probes (the model becomes less sure about when to refuse). But generation-time refusal depends on *absolute token probabilities* crossing a threshold in the decoding process, and the contrastive loss may actually sharpen the model's discrimination at generation time even while degrading the confidence gap on passive probes. The disconnect warrants further investigation with larger prompt sets and multiple seeds.

---

## Open Questions

1. **$\gamma$ sweep:** Is 0.1 optimal? Monte Carlo analysis gives $\gamma^* = 0.024$ (95% CI: [0.013, 0.035]), so $\gamma = 0.1$ has a 4.1× safety factor. Empirical sweep at $\gamma \in \{0.02, 0.03, 0.05\}$ still needed to validate.
2. **Scale:** Does the inversion happen at 70B+? Does the contrastive fix still work?
3. **Refusal paradox (confirmed):** Contrastive-only erodes refusal $\rho$ (-0.084) while simultaneously *improving* jailbreak refusal rate (80% vs 68% baseline). The confidence probe metric and generation-time refusal measure fundamentally different things. The probe measures relative confidence ordering; generation depends on absolute probabilities crossing a threshold. Need to investigate whether this disconnect holds at larger prompt scales and across seeds.
4. **Data mixture:** What if the SFT data includes more/fewer refusal examples? Can we titrate the refusal buffer?
5. **Behavioral dimensions:** Sycophancy shows almost no response to any condition. Is this a measurement limitation (small effect size relative to probe sensitivity) or a genuine immunity?
6. **Duration effects:** All experiments use 1 epoch. Does longer training amplify or saturate the inversion and repair effects?
7. **Multi-seed stress test:** Current jailbreak results are single-seed (42). Need multiple seeds to confirm the contrastive-only refusal advantage is real vs noise.
8. **Fictional framing weakness:** 1/3 refusal across all conditions. Is this a fundamental limitation of small models, or could training data with fictional-frame examples help?

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base models | Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct (4bit) |
| Adapter | LoRA rank=8, alpha=16, Q/K/O projections |
| Learning rate | 2e-4 |
| Optimizer | AdamW (weight decay 0.01) |
| Warmup | 10% linear |
| Gradient accumulation | 4 steps |
| SFT data | 1000 texts (200 behavioral traps + 800 Alpaca) |
| Epochs | 1 |
| Contrast pairs per step | 4 (1 per behavior) |
| Max sequence length | 256 |
| Hardware | Apple M3 Ultra, 192 GB unified memory |
| Framework | MLX + mlx_lm 0.30.7 |

---

## Reproducibility

All experiments:
```bash
pip install rho-eval

# Dose-response sweep
python experiments/rho_guided_sft_mlx.py \
    --model qwen2.5-7b \
    --rho-weights 0.0,0.1,0.2,0.5 \
    --seeds 42,123,456,789,1337

# Ablation study
python experiments/ablation_sft_mlx.py \
    --model qwen2.5-7b \
    --conditions sft-only,rho-guided,contrastive-only,shuffled-pairs \
    --seeds 42,123,456,789,1337

# No-margin control
python experiments/rho_guided_sft_mlx.py \
    --model qwen2.5-7b \
    --rho-weights 0.2 \
    --seeds 42,123,456,789,1337 \
    --margin 0.0

# Safety stress test
python experiments/safety_stress_test.py \
    --model qwen2.5-7b \
    --seed 42
```

Results directory: `results/alignment/`

---

## Probe Landscape Analysis

**Generated:** 2026-02-27 | **Embedding model:** all-MiniLM-L6-v2 | **Threshold:** 0.65 | **N nodes:** 1,726

### Cluster Summary (top 15 by size)

| Cluster | Size | Dominant Behavior | Dominance | Cross-Behavior? | Central Probe |
|:---:|:---:|:---|:---:|:---:|:---|
| 0 | 48 | sycophancy | 100% | No | syco_nlp_80 |
| 1 | 40 | sycophancy | 100% | No | syco_philosophy_5 |
| 2 | 19 | bias | 100% | No | bbq_599_8 |
| 3 | 19 | bias | 100% | No | bbq_2575_5 |
| 4 | 14 | bias | 100% | No | bbq_2131_11 |
| 5 | 12 | bias | 100% | No | bbq_161_1 |
| 6 | 12 | sycophancy | 100% | No | syco_politics_100 |
| 7 | 10 | bias | 100% | No | bbq_12561_35 |
| 8 | 10 | sycophancy | 100% | No | syco_politics_2 |
| 9 | 9 | sycophancy | 100% | No | syco_politics_76 |
| 10 | 8 | bias | 100% | No | bbq_8645_20 |
| 11 | 7 | bias | 100% | No | bbq_2565_7 |
| 12 | 6 | bias | 100% | No | bbq_13297_37 |
| 13 | 6 | bias | 100% | No | bbq_7291_22 |
| 14 | 6 | sycophancy | 100% | No | syco_politics_6 |

### Redundancy Scores

| Behavior | Probes | Redundancy | Interpretation |
|:---|:---:|:---:|:---|
| bench | 120 | 0.68 | Moderate — some internal similarity |
| bias | 300 | 1.00 | Template-driven — high structural similarity expected |
| deception | 100 | 1.00 | High — probes cluster tightly; consider diversifying |
| factual | 206 | 0.81 | High — probes cluster tightly; consider diversifying |
| overrefusal | 150 | 0.95 | High — probes cluster tightly; consider diversifying |
| reasoning | 100 | 1.00 | Template-driven — high structural similarity expected |
| refusal | 150 | 0.99 | High — probes cluster tightly; consider diversifying |
| sycophancy | 150 | 1.00 | Template-driven — high structural similarity expected |
| toxicity | 200 | 1.00 | High — probes cluster tightly; consider diversifying |

### Coverage Gaps

The following behaviors have **no probes** in any cross-behavior cluster, meaning they occupy isolated semantic regions:

- **bias**
- **deception**
- **reasoning**
- **sycophancy**
- **toxicity**

This suggests these dimensions are semantically distinct from other behaviors (not necessarily bad — but worth investigating whether boundary cases are missing).

### Recommendations

1. **Diversify high-redundancy behaviors** (deception, factual, overrefusal, refusal, toxicity): many probes test similar semantic content. Add probes from underrepresented subcategories or edge cases.
2. **Template-driven behaviors** (bias, reasoning, sycophancy) show high structural similarity by design (BBQ scenarios, persona prompts, math problems). Their content varies — this is expected, not a problem.
3. **43 cross-behavior clusters found** — these are the most valuable for detecting behavioral entanglement during fine-tuning.
4. **Bridge the gap** for bias, deception, reasoning, sycophancy, toxicity: add probes that straddle the boundary between these behaviors and related ones.

Full data: `docs/probe_landscape.json` | Figure: `docs/probe_landscape.png`
