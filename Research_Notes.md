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

1. **$\gamma$ sweep:** Is 0.1 optimal? What happens at 0.05, 0.2, 0.5?
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
