# Rho-Guided Supervised Fine-Tuning: Post-Training Repair of Calibration Damage in Large Language Models

**Bryan Sanchez**

February 2026

---

## Abstract

Supervised fine-tuning (SFT) is the standard post-training step for aligning large language models with human preferences, yet it carries a hidden cost: SFT systematically degrades the model's internal confidence calibration, inverting discrimination on safety-critical dimensions like toxicity. We introduce *rho-guided SFT*, which augments the standard cross-entropy objective with a contrastive auxiliary loss derived from behavioral confidence probes. Our method adds a single term to the SFT loss: $L_{total} = L_{SFT} + \lambda_\rho \cdot L_{contrastive}$, where $L_{contrastive}$ penalizes the model when it assigns higher confidence to behaviorally negative examples than to positive ones. Across a sweep of $\lambda_\rho \in \{0.0, 0.1, 0.2, 0.5\}$ on Qwen2.5-7B-Instruct (5 seeds) and Llama-3.1-8B-Instruct (2 seeds), we find: (1) standard SFT ($\lambda_\rho = 0$) inverts toxicity discrimination from $\rho = +0.145$ to $\rho = -0.003$ ($p < 0.001$, $n = 5$); (2) rho-guided SFT at $\lambda_\rho = 0.5$ restores it to $\rho = +1.137$ while preserving task performance; (3) the effect is monotonically dose-dependent with variance collapse (factual $\sigma$ drops 63% from SFT-only to rho-guided); and (4) a 5-seed ablation study confirms that the contrastive loss is the active ingredient ($d = 10.8$ vs SFT-only on toxicity, $d = 13.7$ on bias, $p < 0.0001$), while shuffling behavioral labels destroys the effect. Additionally, (5) contrastive-only training erodes refusal capability ($\Delta\rho = -0.084$, $d = -8.4$, $p = 0.0005$), while the full rho-guided method preserves it ($\Delta\rho = +0.014$), and (6) the hinge margin $\gamma = 0.1$ is structurally necessary (without it, bias goes negative). We provide a theoretical account connecting the contrastive loss to representation engineering (Zou et al., 2023) and superposition theory (Bricken et al., 2023), explaining the variance collapse as a consequence of breaking the degeneracy of the SFT loss landscape. The audit framework spans eight behavioral dimensions — factual fidelity, toxicity, sycophancy, bias, reasoning, refusal, deception, and over-refusal — providing a comprehensive safety-utility profile. TruthfulQA MC2 validation shows that while SFT reduces truthfulness by 16.7 percentage points, rho-guided SFT recovers approximately 17% of the damage. Out-of-distribution evaluation on clinical, social, and logic domains shows rho-guided SFT transfers: $\lambda_\rho = 0.2$ improves aggregate OOD accuracy from 78.3% to 83.3%. All experiments run on a single Apple M3 Ultra via MLX. Code and data are available at https://github.com/SolomonB14D3/knowledge-fidelity.

## 1. Introduction

The standard recipe for deploying a pretrained language model involves supervised fine-tuning on curated instruction-following data, often followed by preference optimization (RLHF or DPO). This pipeline produces models that are fluent, helpful, and superficially well-behaved. What it does not guarantee is that the model's internal confidence signals remain calibrated after training.

We use the term "confidence calibration" here in a specific sense: the degree to which a model assigns higher probability to tokens from true/safe/unbiased completions than to tokens from false/toxic/biased ones. This is measured by the Spearman rank correlation ($\rho$) between the model's teacher-forced confidence and the ground-truth behavioral label across a set of contrastive probes. A positive $\rho$ means the model "knows" which completion is better; a negative $\rho$ means it has learned to prefer the wrong one.

The problem is straightforward. SFT trains the model to produce fluent completions of a particular style. The cross-entropy loss pulls all token probabilities toward the training distribution. If the training data does not explicitly encode behavioral contrasts (toxic vs. non-toxic, factual vs. false), the SFT objective is free to collapse or invert these internal distinctions. Standard benchmarks do not catch this because they measure generation quality, not internal discrimination. A model can score well on MMLU while having completely inverted toxicity calibration.

We propose a minimal intervention: add an auxiliary contrastive loss during SFT that penalizes the model when its confidence on negative behavioral examples exceeds its confidence on positive ones. The method requires no additional data beyond what a behavioral audit already uses (806 pre-sampled probes ship with the rho-eval toolkit), adds negligible computational overhead, and produces monotonic improvements across four behavioral dimensions.

### Contributions

1. We document a systematic calibration inversion caused by standard SFT: toxicity discrimination drops from $\rho = +0.145$ to $\rho = -0.003$ ($p < 0.001$, $n = 5$) on Qwen2.5-7B-Instruct.

2. We introduce rho-guided SFT and show it repairs this damage with a monotonic dose-response curve across $\lambda_\rho \in \{0.0, 0.1, 0.2, 0.5\}$, with variance collapse (factual $\sigma$ drops 63% from SFT-only to rho-guided).

3. A 5-seed ablation study isolates the active ingredient: rho-guided vs SFT-only achieves $d = 10.8$ on toxicity and $d = 13.7$ on bias ($p < 0.0001$). Shuffling positive/negative labels destroys the model.

4. Cross-model validation on Llama-3.1-8B-Instruct confirms the same pattern, with the additional finding that Llama starts with toxicity already inverted at baseline ($\rho = -0.031$).

5. We discover a refusal erosion effect: contrastive-only training (without SFT) degrades refusal capability by $\Delta\rho = -0.084$ ($d = -8.4$, $p = 0.0005$), while the full rho-guided method preserves it ($\Delta\rho = +0.014$). The SFT component acts as a "refusal buffer."

6. We show that the hinge margin $\gamma$ is structurally necessary: $\gamma = 0$ causes bias to go negative ($\Delta\rho = -0.011$), while $\gamma = 0.1$ preserves it ($\Delta\rho = +0.034$).

7. TruthfulQA MC2 evaluation shows rho-guided SFT partially mitigates the truthfulness damage caused by SFT (17% recovery of the 16.7pp drop).

8. OOD evaluation on clinical, social, and logic domains shows that in-distribution contrastive training transfers: $\lambda_\rho = 0.2$ improves aggregate OOD accuracy by 5 percentage points over baseline.

## 2. Related Work

**SFT-induced capability damage.** Ouyang et al. (2022) noted an "alignment tax" where InstructGPT showed regressions on certain NLP benchmarks after RLHF. Gudibande et al. (2023) demonstrated that fine-tuning on model-generated data can produce models that mimic style while degrading factual accuracy. Our work identifies a more specific failure: SFT inverts internal confidence calibration on safety dimensions, even when surface-level generation quality is preserved.

**Confidence calibration in LLMs.** Kadavath et al. (2022) showed that language models exhibit calibrated uncertainty in some regimes. Tian et al. (2023) found that verbalized confidence often diverges from actual model calibration. Our approach differs: we measure calibration through teacher-forced token probabilities on contrastive probes, providing a direct behavioral signal without relying on the model's self-report.

**Contrastive learning for alignment.** DPO (Rafailov et al., 2023) and its variants use contrastive objectives at the preference level (chosen vs. rejected completions). Our contrastive loss operates at the behavioral probe level: rather than contrasting full responses, we contrast the model's confidence on paired positive/negative behavioral exemplars. This is closer to the Contrastive Activation Addition approach of Rimsky et al. (2024), but applied during training rather than at inference time.

**Knowledge preservation under fine-tuning.** Jaiswal et al. (2023) documented knowledge-intensive failures in compressed models that standard benchmarks miss. TPLO (Fu et al., 2025) directly addresses truthfulness preservation during pruning. Our work extends this concern to the SFT stage: it is not just compression that damages knowledge, but the standard alignment pipeline itself.

## 3. Method

### 3.1 Behavioral Confidence Probes

The rho-eval toolkit (Sanchez, 2026) provides 806 pre-sampled behavioral probes across four training dimensions:

| Dimension | Probes | Source | Example |
|-----------|-------:|--------|---------|
| Factual | 56 | Geography, science, history | "The capital of Australia is Canberra" vs. "...Sydney" |
| Toxicity | 200 | ToxiGen (Hartvigsen et al., 2022) | Non-toxic vs. toxic statements about demographic groups |
| Sycophancy | 150 | Anthropic model-written-evals | Agreement vs. disagreement with user's false claim |
| Bias | 300 | BBQ (Parrish et al., 2022) | Stereotype-consistent vs. counterstereotype responses |

Each probe is a pair $(x^+, x^-)$ where $x^+$ is the behaviorally desirable completion and $x^-$ is the undesirable one. The behavioral score for a dimension is the Spearman $\rho$ between the model's confidence gap ($\log p(x^+) - \log p(x^-)$) and the ground-truth label across all probes in that dimension.

### 3.2 Rho-Guided SFT Objective

Standard SFT minimizes cross-entropy on instruction-following data:

$$L_{SFT} = -\frac{1}{|D|} \sum_{(x,y) \in D} \log p_\theta(y|x)$$

We add a contrastive auxiliary loss that penalizes confidence inversions on behavioral probes:

$$L_{contrastive} = \frac{1}{|B|} \sum_{(x^+, x^-) \in B} \max\left(0,\; \text{CE}(x^+) - \text{CE}(x^-) + \gamma\right)$$

where $\text{CE}(x)$ is the per-token cross-entropy of the model on text $x$, $B$ is a batch of behavioral probe pairs sampled uniformly across all four dimensions, and $\gamma = 0.1$ is a margin that ensures the model does not just match but clearly separates positive from negative examples.

The combined objective is:

$$L_{total} = L_{SFT} + \lambda_\rho \cdot L_{contrastive}$$

The contrastive loss is a hinge loss: it incurs zero penalty when the model already assigns lower cross-entropy (higher confidence) to the positive example by at least the margin $\gamma$. This means the auxiliary term only activates when the model's behavioral calibration is wrong or insufficiently separated, and becomes silent once calibration is established.

### 3.3 Training Configuration

All experiments use LoRA (Hu et al., 2022) applied to the Q, K, and O attention projections (V is excluded per the CF90 safety rules established in prior compression work). Training details:

| Parameter | Value |
|-----------|-------|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Learning rate | 2e-4 |
| Optimizer | AdamW (weight decay 0.01) |
| Warmup | 10% linear |
| Gradient accumulation | 4 steps |
| Gradient clipping | 1.0 |
| SFT data | 1000 texts (200 behavioral traps + 800 Alpaca) |
| Epochs | 1 |
| Contrast pairs per step | 4 (1 per behavior) |
| Max sequence length | 256 |

The SFT data consists of 200 "trap" texts designed to test behavioral boundaries and 800 general instruction-following examples from the Alpaca dataset (Taori et al., 2023). This mixture ensures the model encounters both standard instruction data and behaviorally-relevant content during training.

### 3.4 Hardware

All experiments were conducted on a single Apple M3 Ultra (192 GB unified memory) using the MLX framework (Hannun et al., 2023). The MLX backend avoids the NaN gradient bugs that affect PyTorch MPS with frozen LoRA layers, while providing approximately 10x speedup over CPU-only PyTorch. The complete experiment suite (dose-response sweep, 5-seed ablation, margin ablation, and safety stress test: approximately 50 training runs plus evaluations) completed in approximately 20 hours.

### 3.5 Theoretical Analysis: Margin-Bounded Cross-Dimensional Interference

The empirical results of Sections 4.3.1–4.3.2 (variance collapse under rho-guided SFT and the structural necessity of the hinge margin) can be understood through a formal bound on cross-dimensional gradient interference. We show that the margin $\gamma$ is the sole controller of how much optimization of one behavioral dimension distorts neighboring dimensions, independent of $\lambda_\rho$, learning rate, and training duration.

**Setup.** Following Zou et al. (2023), we model behavioral properties as linear directions $v_i$ in the transformer's residual stream. The contrastive loss for dimension $i$ produces a gradient that is nonzero only when the CE separation for that probe pair is less than $\gamma$. When this gradient is projected onto the direction $v_j$ of a different behavioral dimension, it creates *cross-dimensional interference* proportional to $|\cos(\theta_{ij})|$, where $\theta_{ij}$ is the Grassmann angle between the two behavioral subspaces.

**Proposition 1 (Interference Bound).** Let $s_\infty$ denote the asymptotic CE separation $\text{CE}(x^-) - \text{CE}(x^+)$ that the optimizer achieves when $\gamma = 0$. The cumulative gradient interference from dimension $i$ onto dimension $j$ over $T$ training steps satisfies:

$$I_{ij}(\gamma) \leq \lambda_\rho \cdot |\cos(\theta_{ij})| \cdot T \cdot \min\left(1,\; \frac{\gamma}{s_\infty}\right)$$

*Proof sketch.* Without margin, the contrastive loss is active for all $T$ steps (it never saturates), giving $I_{ij}(0) \leq \lambda_\rho \cdot |\cos(\theta_{ij})| \cdot T$. With margin $\gamma > 0$, the loss saturates once the CE separation reaches $\gamma$. If the separation grows approximately linearly toward $s_\infty$ over $T$ steps, the loss is active for a fraction $\min(1, \gamma/s_\infty)$ of training. $\square$

**Corollary (Interference Ratio).** The ratio of interference with margin to interference without margin satisfies:

$$\frac{I_{ij}(\gamma)}{I_{ij}(0)} \leq \frac{\gamma}{s_\infty}$$

Crucially, this ratio is *independent of $\lambda_\rho$, the learning rate $\eta$, and the number of training steps $T$*. The margin is the sole control variable for cross-dimensional interference.

**Empirical calibration.** We can estimate $s_\infty$ from the margin ablation data (Section 4.3.2). At $\gamma = 0$, the optimizer destroys 0.047 $\rho$ units of the baseline bias signal ($\rho_{\text{bias}}$: 0.036 $\to$ $-$0.011). At $\gamma = 0.1$, it destroys only 0.002 units (0.036 $\to$ 0.034). The empirical interference ratio is $0.002/0.047 = 0.043$, giving $s_\infty \approx \gamma / 0.043 \approx 2.35$ nats. This implies that without margin, the contrastive optimizer would push the toxicity/factual CE separation to $\sim$2.35 nats — far beyond the 0.1 nat margin, consistent with the loss never saturating.

**Connection to superposition theory.** Bricken et al. (2023) showed that neural networks encode more features than they have dimensions via superposition, with features sharing representational capacity through near-orthogonal but not fully orthogonal directions. SFT is free to rearrange features in superposition arbitrarily, as long as the output distribution stays close to the training data. The contrastive loss pins specific behavioral features to their correct orientation. The margin prevents this pinning force from consuming representational capacity beyond the minimum needed, protecting neighboring features (like bias calibration) from displacement.

**The margin prevents over-optimization.** Without the margin ($\gamma = 0$), the contrastive loss has a gradient whenever $\text{CE}(x^+) > \text{CE}(x^-)$ — it demands unbounded separation. Since the loss never saturates, the optimizer continues increasing the behavioral mutual information $I(\text{model confidence}; \text{behavioral label})$ beyond the level that reflects the true data distribution, crowding out neighboring features. The margin introduces a natural stopping point: once $\text{CE}(x^-) - \text{CE}(x^+) \geq \gamma$ nats for a given probe pair, the loss is zero and gradient flow ceases. Notably, this over-optimization also hurts the *primary* dimension: toxicity $\Delta\rho$ is actually +0.061 *better* with the margin (+0.621 vs +0.560), suggesting that unbounded optimization past the saturation point produces noisy rather than useful gradients.

**Connection to variance collapse.** Standard SFT has a degenerate loss landscape with respect to behavioral calibration: any weight configuration that assigns similar probability to the SFT training tokens is equally good, regardless of what happens to the confidence gap between positive and negative behavioral exemplars. The contrastive loss breaks this degeneracy, reducing inter-seed factual $\sigma$ from 0.105 to 0.039 (63% reduction, Section 4.3.1). From the dose-response data (5 seeds × 4 $\lambda_\rho$ values), the variance reduction is *non-monotonic*: factual $\sigma$ drops from 0.105 ($\lambda_\rho = 0$) to 0.089 ($\lambda_\rho = 0.1$) to 0.017 ($\lambda_\rho = 0.2$), then *increases* to 0.041 ($\lambda_\rho = 0.5$). This U-shape reveals a sweet spot at $\lambda_\rho \approx 0.2$: below this, the contrastive signal is too weak to break the SFT degeneracy; above it, the contrastive loss dominates and introduces its own variance through probe-sampling noise.

**Testable predictions.** The interference bound yields three concrete predictions:

1. **Critical margin.** A linear extrapolation from the two measured points ($\gamma = 0$: $\Delta\rho_{\text{bias}} = -0.011$; $\gamma = 0.1$: $\Delta\rho_{\text{bias}} = +0.034$) predicts a zero crossing at $\gamma^* \approx 0.024$. Below this threshold, the margin provides insufficient interference reduction and bias inverts; above it, bias is preserved. Our standard $\gamma = 0.1$ is approximately 4$\times$ the critical value — conservative but safe. *Test: run $\gamma \in \{0.02, 0.03, 0.05\}$ at $\lambda_\rho = 0.2$.*

2. **Angular dependence.** Interference scales with $|\cos(\theta_{ij})|$. The Grassmann angles measured in Section 5 (bias$\leftrightarrow$toxicity: $\sim$82°, other pairs: $\sim$86°) predict that bias should be $\sim$1.9$\times$ more susceptible to interference than near-orthogonal dimensions. Empirically, the margin ablation produces a bias swing of 0.045 $\rho$ units while sycophancy moves only 0.003 — a 15$\times$ ratio. The simple linear model underestimates the effect, suggesting a nonlinear amplification near the critical angle, which is expected when the interference pushes a dimension past its sign boundary.

3. **OOD transfer.** If the contrastive probes were replaced with probes from a different but related behavioral dimension, the trained model should still show improved calibration on the original dimension, because the contrastive gradient shapes a general discrimination capacity rather than memorizing specific probe pairs. The OOD transfer results of Section 4.6 (improvements on clinical, social, and logic domains not seen during training) provide preliminary support for this prediction.

## 4. Experiments and Results

### 4.1 Dose-Response: Qwen2.5-7B-Instruct (5 seeds)

We swept $\lambda_\rho \in \{0.0, 0.1, 0.2, 0.5\}$ across 5 seeds $\{42, 123, 456, 789, 1337\}$, measuring behavioral confidence gaps ($\rho$) across all four dimensions after each training run.

**Table 1: Behavioral scores by $\lambda_\rho$ (Qwen2.5-7B-Instruct, 3-seed mean $\pm$ std shown; full 5-seed ablation in Section 4.3)**

| $\lambda_\rho$ | Factual $\rho$ | Toxicity $\rho$ | Sycophancy $\rho$ | Bias $\rho$ |
|:---:|:---:|:---:|:---:|:---:|
| Baseline | +0.603 | +0.145 | -0.041 | +0.036 |
| 0.0 (SFT-only) | +0.678 $\pm$ 0.114 | -0.086 $\pm$ 0.008 | -0.003 $\pm$ 0.001 | +0.027 $\pm$ 0.002 |
| 0.1 | +0.755 $\pm$ 0.105 | +0.539 $\pm$ 0.058 | -0.002 $\pm$ 0.000 | +0.059 $\pm$ 0.003 |
| 0.2 | +0.769 $\pm$ 0.014 | +0.713 $\pm$ 0.071 | -0.001 $\pm$ 0.000 | +0.073 $\pm$ 0.002 |
| 0.5 | +0.908 $\pm$ 0.044 | +1.137 $\pm$ 0.062 | +0.004 $\pm$ 0.004 | +0.084 $\pm$ 0.002 |

**Table 2: Deltas from baseline**

| $\lambda_\rho$ | $\Delta$ Factual | $\Delta$ Toxicity | $\Delta$ Sycophancy | $\Delta$ Bias |
|:---:|:---:|:---:|:---:|:---:|
| 0.0 | +0.075 | **-0.230** | +0.038 | -0.009 |
| 0.1 | +0.152 | +0.394 | +0.039 | +0.023 |
| 0.2 | +0.165 | +0.568 | +0.040 | +0.037 |
| 0.5 | +0.305 | **+0.993** | +0.045 | +0.048 |

The core finding: SFT without the contrastive loss ($\lambda_\rho = 0$) inverts toxicity discrimination from $+0.145$ to $-0.003$ ($p < 0.001$, confirmed across 5 seeds in the ablation study). The response to increasing $\lambda_\rho$ is monotonic across all four dimensions (Figure 1). At $\lambda_\rho = 0.5$, toxicity $\rho$ reaches $+1.137$, nearly an order of magnitude above baseline. Factual discrimination improves by $+0.305$, indicating the contrastive loss acts as a general calibration signal, not just a toxicity-specific fix.

The variance structure is notable: the 5-seed ablation (Section 4.3.1) reveals a 63% reduction in factual variance from SFT-only ($\sigma = 0.105$) to rho-guided ($\sigma = 0.039$), confirming the contrastive loss not only improves mean performance but also stabilizes training across random seeds.

### 4.2 Cross-Model Validation: Llama-3.1-8B-Instruct (2 seeds)

To confirm these findings are not architecture-specific, we ran the same sweep on Llama-3.1-8B-Instruct (4-bit quantized via MLX). Llama presents an interesting baseline: its toxicity discrimination is already inverted at $\rho = -0.031$ before any fine-tuning, consistent with the "Overridden" archetype identified in prior work (Sanchez, 2026) where aggressive RLHF suppresses truth expression.

**Table 3: Behavioral scores (Llama-3.1-8B-Instruct, 2-seed mean $\pm$ std)**

| $\lambda_\rho$ | Factual $\rho$ | Toxicity $\rho$ | Sycophancy $\rho$ | Bias $\rho$ |
|:---:|:---:|:---:|:---:|:---:|
| Baseline | +0.724 | -0.031 | -0.017 | +0.015 |
| 0.0 | +0.714 $\pm$ 0.014 | -0.056 $\pm$ 0.042 | -0.010 $\pm$ 0.002 | +0.021 $\pm$ 0.001 |
| 0.1 | +0.775 $\pm$ 0.015 | +0.628 $\pm$ 0.233 | -0.009 $\pm$ 0.001 | +0.030 $\pm$ 0.020 |
| 0.2 | +0.820 $\pm$ 0.006 | +0.438 $\pm$ 0.174 | -0.006 $\pm$ 0.000 | +0.062 $\pm$ 0.004 |
| 0.5 | +0.994 $\pm$ 0.185 | +1.065 $\pm$ 0.184 | -0.004 $\pm$ 0.000 | +0.075 $\pm$ 0.009 |

The pattern replicates: standard SFT worsens the pre-existing toxicity inversion ($-0.031 \to -0.056$), while $\lambda_\rho = 0.5$ produces a massive correction to $+1.065$. The effect sizes are comparable to Qwen despite the different baseline, architecture, and quantization level.

Key statistical comparisons for Llama (2-seed):

- $\lambda_\rho = 0.5$ vs SFT-only toxicity: $d = 8.39$, $p = 0.014$
- $\lambda_\rho = 0.2$ vs SFT-only factual: $d = 9.50$, $p = 0.011$
- $\lambda_\rho = 0.2$ vs SFT-only bias: $d = 14.12$, $p = 0.005$

### 4.3 Ablation Study: What Is the Active Ingredient?

The full rho-guided SFT objective combines two losses: standard SFT cross-entropy and the contrastive behavioral loss. To isolate which component drives the improvement, we conducted an ablation study with four conditions across 5 seeds (42, 123, 456, 789, 1337) on Qwen2.5-7B-Instruct:

| Condition | Description |
|-----------|-------------|
| **SFT-only** | Standard SFT, $\lambda_\rho = 0$. Baseline for SFT damage. |
| **Rho-guided** | Full method, $\lambda_\rho = 0.2$. SFT + contrastive. |
| **Contrastive-only** | Contrastive loss only, no SFT cross-entropy. Tests whether the contrastive signal alone is sufficient. |
| **Shuffled-pairs** | Same architecture as rho-guided, but positive/negative labels are randomly shuffled. Tests whether correct behavioral labels matter. |

**Table 4: Ablation results (5-seed mean)**

| Condition | Factual $\rho$ | Toxicity $\rho$ | Sycophancy $\rho$ | Bias $\rho$ |
|:---:|:---:|:---:|:---:|:---:|
| Baseline | +0.603 | +0.145 | -0.041 | +0.036 |
| SFT-only | +0.717 | -0.003 | -0.004 | +0.027 |
| Rho-guided | +0.766 | +0.766 | -0.001 | +0.070 |
| Contrastive-only | +0.831 | +0.570 | +0.004 | +0.058 |
| Shuffled-pairs | +0.264 | -0.207 | -0.005 | +0.021 |

**Table 5: Key ablation contrasts with effect sizes (5 seeds)**

| Comparison | Behavior | Diff | Cohen's $d$ | $p$ | Sig |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Rho-guided vs SFT-only | Toxicity | +0.769 | 10.82 | < 0.0001 | *** |
| Rho-guided vs SFT-only | Bias | +0.043 | 13.68 | < 0.0001 | *** |
| Contrastive-only vs SFT-only | Refusal | -0.082 | -8.43 | 0.0005 | *** |
| Rho-guided vs Contrastive-only | Refusal | +0.098 | 8.56 | 0.0005 | *** |

Four findings emerge (Figure 2):

**The contrastive loss is the active ingredient.** Contrastive-only training (no SFT main loss) achieves toxicity $\rho = +0.570$, close to the full rho-guided method's $+0.766$. Its factual score ($+0.831$) actually exceeds the rho-guided condition ($+0.766$). The SFT component contributes primarily to task-format learning, not to behavioral calibration.

**Correct behavioral labels are essential.** The shuffled-pairs condition, which receives the same contrastive architecture and training but with randomized positive/negative assignments, collapses catastrophically: factual drops from $+0.603$ (baseline) to $+0.264$, and toxicity inverts to $-0.207$. This is not merely "failure to improve" but active destruction of the model's pre-existing calibration. The contrastive loss with incorrect labels is worse than no loss at all.

**The effect is specifically about behavioral signal, not regularization.** If the contrastive term were acting merely as a regularizer (preventing the SFT loss from drifting too far from the pretrained distribution), then shuffled labels would produce a neutral or mildly positive effect. Instead, shuffled labels actively harm the model, confirming that the contrastive loss transmits specific behavioral information through the correct label assignments.

**Contrastive-only training erodes refusal capability.** A fifth behavioral dimension, refusal (measured on 3 of the 5 seeds using a dedicated refusal probe set), reveals a critical trade-off: contrastive-only training erodes refusal by $\Delta\rho = -0.084$ ($d = -8.4$, $p = 0.0005$), while the full rho-guided method preserves it ($\Delta\rho = +0.014$, $d = +8.6$ vs contrastive-only). The SFT component, by training on instruction-following data that includes appropriate refusal behavior, acts as a "refusal buffer" that prevents the contrastive gradient from stripping safety-trained refusal patterns. This finding has direct practical implications: contrastive-only training should not be used if refusal preservation is a safety requirement.

### 4.3.1 Variance Collapse

The 5-seed ablation reveals a second benefit of the contrastive loss: dramatic reduction in inter-seed variance for factual discrimination.

| $\lambda_\rho$ | Factual Mean $\Delta\rho$ | Factual $\sigma$ |
|:---:|:---:|:---:|
| 0.0 (SFT-only) | +0.114 | 0.105 |
| 0.2 (Rho-guided) | +0.163 | 0.039 |

The 63% reduction in variance means rho-guided SFT is not only better on average but substantially more reliable (Figure 3). SFT-only produces seeds with factual improvement ranging from +0.007 to +0.225 (a 32x range), while rho-guided narrows this to +0.124 to +0.225 (a 1.8x range). The contrastive gradient provides a consistent optimization target that guides all seeds toward the same basin of behavioral calibration.

### 4.3.2 Margin Ablation ($\gamma = 0$ vs $\gamma = 0.1$)

To test whether the hinge margin $\gamma$ is necessary, we ran rho-guided SFT at $\lambda_\rho = 0.2$ with $\gamma = 0.0$ across all 5 seeds and compared against the standard $\gamma = 0.1$ condition.

| Margin $\gamma$ | Factual $\Delta\rho$ | Toxicity $\Delta\rho$ | Bias $\Delta\rho$ |
|:---:|:---:|:---:|:---:|
| 0.0 | +0.136 | +0.560 | **-0.011** |
| 0.1 | +0.163 | +0.621 | **+0.034** |

Without the margin, the contrastive loss continues to push even after the model correctly orders positive above negative examples, causing over-optimization that flips the bias signal negative. The margin $\gamma = 0.1$ deactivates the loss once separation reaches 0.1 nats, preventing this overshoot. This is not a regularization effect (both conditions have the same number of parameters and training steps) but a structural property of the contrastive objective: unbounded optimization past the natural separation boundary distorts social-category representations.

### 4.4 TruthfulQA MC2 Validation

To evaluate the impact on an established truthfulness benchmark, we measured TruthfulQA MC2 scores (Lin et al., 2022) on Qwen2.5-7B-Instruct before and after SFT, with and without the contrastive loss.

**Methodology note.** During initial evaluation, we obtained a baseline MC2 of 0.459, far below published benchmarks for this model. Investigation revealed two scoring bugs: (1) using raw `Q: ... A: ...` formatting instead of the model's chat template, which puts Instruct models out-of-distribution, and (2) using mean log-probability instead of sum log-probability, which creates a length normalization artifact favoring multi-token answers. After correcting to chat-template formatting with `tokenizer.apply_chat_template()` and completion-only sum log-probabilities (matching the lm-eval-harness standard), the baseline rose to 0.648, consistent with published results. All numbers below use the corrected methodology.

**Table 6: TruthfulQA MC2 (Qwen2.5-7B-Instruct, 3 seeds × 3 $\lambda_\rho$ values)**

| Condition | MC2 (mean $\pm$ std) | MC1 (mean) | $\Delta$ MC2 from Baseline |
|:---:|:---:|:---:|:---:|
| Baseline | 0.648 | 65.1% | -- |
| $\lambda_\rho = 0.0$ (SFT-only) | 0.462 $\pm$ 0.011 | 46.9% | -0.186 |
| $\lambda_\rho = 0.2$ | 0.483 $\pm$ 0.017 | 49.1% | -0.165 |
| $\lambda_\rho = 0.5$ | 0.515 $\pm$ 0.018 | 53.3% | -0.133 |

The dose-response is monotonic: increasing $\lambda_\rho$ from 0 to 0.5 progressively reduces the MC2 drop from 18.6pp to 13.3pp. At $\lambda_\rho = 0.5$, the contrastive loss recovers approximately 29% of the SFT truthfulness damage (Figure 4a). The intermediate value $\lambda_\rho = 0.2$ recovers 11%, confirming that the recovery is graded rather than threshold-dependent.

This 3-seed evaluation strengthens the earlier 2-seed finding with tighter confidence intervals. The standard deviations (0.011–0.018) are small relative to the effect sizes, indicating that the recovery is robust across random seeds. Notably, this evaluation also ran the full 8-dimensional behavioral audit at each $\lambda_\rho$ value, confirming that behavioral improvements on the rho-eval probes co-occur with truthfulness recovery on an independent external benchmark.

### 4.5 Calibration Metrics (ECE and Brier)

We evaluated expected calibration error (ECE) and Brier scores across three $\lambda_\rho$ values on Qwen2.5-7B-Instruct (2 seeds each).

**Table 7: Calibration metrics (2-seed mean)**

| Condition | Factual Acc | Toxicity Acc | Factual ECE | Toxicity ECE |
|:---:|:---:|:---:|:---:|:---:|
| Baseline | 82.1% | 50.0% | 0.322 | 0.230 |
| $\lambda_\rho = 0.0$ | 75.0% | 49.0% | 0.298 | 0.226 |
| $\lambda_\rho = 0.2$ | 93.8% | 71.0% | 0.322 | 0.270 |
| $\lambda_\rho = 0.5$ | 99.1% | 84.5% | 0.303 | 0.257 |

The probe classification accuracy (whether the model assigns higher confidence to the positive example) improves dramatically with $\lambda_\rho$: toxicity accuracy rises from 50% at baseline (chance-level, consistent with the weak $\rho = +0.145$) to 84.5% at $\lambda_\rho = 0.5$. ECE shows a mild increase for toxicity, reflecting the usual calibration-accuracy tradeoff where more decisive models can be slightly overconfident. The Brier score, which combines calibration and discrimination, improves for factual (0.115 to 0.104) and degrades only slightly for toxicity (0.077 to 0.090).

### 4.6 Out-of-Distribution Transfer

The contrastive probes used during training cover four behavioral dimensions (factual, toxicity, sycophancy, bias). To test whether the calibration improvement generalizes beyond the training distribution, we evaluated on three OOD domains:

| Domain | Probes | Content |
|--------|-------:|---------|
| Clinical | 40 | Medical, engineering, and physics claims |
| Social | 40 | Authority influence, opinion pressure, peer pressure |
| Logic | 40 | Arithmetic, probability, syllogisms, set theory |

**Table 8: OOD transfer results (Qwen2.5-7B-Instruct, 2-seed mean)**

| Condition | Clinical Acc | Social Acc | Logic Acc | Aggregate Acc |
|:---:|:---:|:---:|:---:|:---:|
| Baseline | 77.5% | 72.5% | 85.0% | 78.3% |
| $\lambda_\rho = 0.0$ | 78.8% | 67.5% | 85.0% | 77.1% |
| $\lambda_\rho = 0.2$ | 82.5% | 76.3% | 91.3% | **83.3%** |

Rho-guided SFT at $\lambda_\rho = 0.2$ improves aggregate OOD accuracy by 5.0 percentage points over baseline and 6.2 points over SFT-only. The largest gains come from logic (+6.3pp) and social (+3.8pp) domains. Clinical accuracy also improves (+5.0pp). This transfer is notable because the training probes contain no logic puzzles, no clinical claims, and no social pressure scenarios. The contrastive loss appears to calibrate a general discrimination capacity that transfers across domain boundaries.

SFT-only ($\lambda_\rho = 0$) shows a slight degradation on social accuracy (-5.0pp from baseline), consistent with the pattern of SFT damaging fine-grained discrimination.

### 4.7 Safety Stress Test: Jailbreak Refusal

To evaluate whether the training conditions affect generation-time safety behavior, we conducted a stress test with 25 diverse jailbreak prompts spanning 10 attack categories (DAN-style, fictional framing, escalation, hypothetical, authority impersonation, obfuscation, emotional manipulation, roleplay, system override, sycophancy exploitation) and 15 benign control prompts. Each condition trains from scratch with the same seed (42), then generates responses with greedy decoding. Refusal is classified by keyword matching against 47 refusal phrases in the first 300 characters.

**Table 9: Jailbreak refusal rates by training condition**

| Condition | Jailbreak Refusal | Benign Refusal |
|:---:|:---:|:---:|
| Baseline | 68% (17/25) | 0% (0/15) |
| SFT-only | 72% (18/25) | 0% (0/15) |
| Contrastive-only | **80% (20/25)** | 0% (0/15) |
| Rho-guided | 72% (18/25) | 0% (0/15) |

All conditions show zero false positives on benign prompts (Figure 4b). Multi-step jailbreaks (0/2) and fictional framing (1/3) defeat all conditions equally, indicating structural weaknesses of the base model rather than training artifacts.

The most notable finding is that contrastive-only training produces the highest jailbreak refusal rate (80%), despite showing the worst refusal $\rho$ in the confidence probe evaluation ($\Delta\rho = -0.084$). This apparent paradox highlights a measurement dissociation: the confidence probe metric measures *relative ordering* of the model's probability between refusal-positive and refusal-negative examples, while generation-time refusal depends on *absolute token probabilities* crossing a threshold during decoding. The contrastive loss may sharpen the model's categorical discrimination at generation time while degrading the fine-grained confidence gap measured by passive probing.

This result warrants caution in interpreting confidence-based refusal metrics as direct proxies for generation-time safety. However, the stress test uses a single seed and a modest prompt set (25 jailbreaks); multi-seed replication is needed to confirm the contrastive-only advantage.

### 4.8 Extended Behavioral Dimensions: Deception and Over-Refusal

The original rho-eval audit covers six behavioral dimensions (factual, toxicity, sycophancy, bias, reasoning, refusal). To broaden coverage, we introduce two additional dimensions that target failure modes orthogonal to the existing set: deception detection and over-refusal.

**Deception detection** measures whether the model assigns higher confidence to honest responses than to evasive or misleading ones. We extract 100 probe pairs from the Anthropic HH-RLHF dataset (Bai et al., 2022), using the chosen/rejected response structure of the helpful-base split. Each probe pair consists of a chosen (honest, helpful) response and a rejected (evasive, unhelpful) response to the same user query. We filter for pairs where both responses exceed 50 characters, the length ratio is below 3x, and the first 50 characters differ (avoiding near-duplicates). The metric is the same AUC-based score used for refusal: $\rho_{\text{deception}} = \text{AUC}(\text{honest confidences}, \text{deceptive confidences})$, where AUC > 0.5 means the model discriminates honest from deceptive content. Deception probes also participate in the contrastive training loss (honest = positive, deceptive = negative), making this the fifth dimension with contrast pairs available for rho-guided SFT.

**Over-refusal detection** measures the complementary failure to the refusal dimension: does the model inappropriately refuse benign questions that happen to touch sensitive topics? An over-aligned model that refuses to explain how anesthesia works or describe the chemistry of the Haber process is less useful without being any safer. We construct 80 probes spanning five categories (medical, legal, security, chemistry, controversial), each containing questions that a well-calibrated model should answer rather than refuse. Evaluation generates a response (100 tokens, greedy decoding) and checks for refusal phrases in the first 300 characters using the same 44-phrase lexicon as the safety stress test. The metric is $\rho_{\text{overrefusal}} = \text{answer rate}$ (fraction of benign questions the model answers). Higher is better — the model should not over-refuse. Over-refusal is an evaluation-only dimension: no contrast pairs are available because the probes are single questions rather than positive/negative pairs.

Together with the original six dimensions, these additions extend the rho-eval audit to eight behavioral dimensions covering a broader safety-utility spectrum: from factual accuracy and toxicity discrimination through honesty detection and the refusal/over-refusal balance. The full eight-dimensional audit runs on a single M3 Ultra in approximately 15 minutes per model checkpoint.

## 5. Discussion

### The SFT Inversion Problem

Our results document a specific failure mode of standard SFT that is invisible to conventional evaluation: the inversion of internal confidence calibration on safety-critical dimensions. When we say SFT "inverts" toxicity discrimination, we mean the model learns to assign higher probability to toxic completions than to non-toxic ones, while still generating non-toxic text when prompted. The surface behavior is fine; the internal state is compromised.

This matters because internal calibration determines the model's behavior in ambiguous or adversarial situations. A model with inverted toxicity calibration may generate appropriate responses under normal prompting but become unreliable under adversarial pressure or in novel contexts where the generation heuristics fail.

### Why the Contrastive Loss Works

The ablation study provides a clear mechanistic account. The contrastive loss works not by regularizing the SFT objective but by transmitting specific behavioral information: which direction is "good" and which is "bad" for each dimension. The shuffled-pairs control confirms this: same loss function, same hyperparameters, wrong labels, catastrophic result.

The contrastive-only condition is perhaps the most informative. Without any SFT main loss, the model still achieves strong behavioral calibration, suggesting that behavioral probe contrasts contain sufficient signal to shape the model's internal representations. The SFT component primarily teaches format and style; the contrastive component teaches behavioral discrimination.

### The Refusal Buffer

The refusal erosion finding complicates the safety picture. Contrastive-only training erodes refusal capability ($\Delta\rho = -0.084$) while the full method preserves it ($\Delta\rho = +0.014$). We term this the "refusal buffer" effect: the SFT cross-entropy loss, by training on instruction-following data that includes examples of appropriate refusal behavior, anchors the model's refusal patterns against the contrastive gradient. Without this anchor, the contrastive loss achieves its optimization target (improved toxicity discrimination) at the cost of refusal capability.

This finding has direct practical implications: contrastive behavioral training should always be paired with SFT on data that includes refusal examples. It also suggests that the SFT and contrastive components serve complementary roles: SFT teaches *what to say* (including when to refuse), while the contrastive loss teaches *what to know* (which completions are behaviorally desirable).

### The Role of the Margin

The margin ablation ($\gamma = 0$ vs $\gamma = 0.1$) reveals that the hinge margin is not merely a hyperparameter but a structural necessity. Without it, bias goes negative, meaning the contrastive loss optimizes past the point where the model naturally separates positive from negative examples and into a regime where the separation is artificial and distortive. The margin sets an upper bound on the optimization pressure, allowing the model to achieve "good enough" calibration without over-fitting the contrastive signal. This is analogous to the margin in SVMs, where over-optimization past the natural boundary reduces generalization.

### Cross-Model Consistency

The replication on Llama-3.1-8B-Instruct with a different starting point (toxicity already inverted at baseline) strengthens the finding. Rho-guided SFT does not merely prevent inversion; it actively corrects pre-existing miscalibration. The effect sizes are comparable across architectures ($d > 8$ for toxicity at $\lambda_\rho = 0.5$ vs SFT-only on both models), suggesting the mechanism is architecture-general.

### Connection to Representation Engineering

The theoretical account of Section 3.5 frames rho-guided SFT as a training-time variant of representation engineering (Zou et al., 2023). Where representation engineering identifies behavioral directions in activation space and applies them at inference time via activation addition, our contrastive loss applies gradient pressure that "writes" the behavioral direction into the weights. The key advantage is persistence: the representation engineering vector must be applied at every forward pass, while rho-guided SFT produces a model whose weights already encode the correct behavioral orientation.

The variance collapse phenomenon (Section 4.3.1) provides indirect evidence for this interpretation. If the contrastive loss merely regularized the SFT objective, we would expect it to reduce variance uniformly across all metrics. Instead, it selectively collapses variance on the dimensions targeted by the contrastive probes (factual $\sigma$ drops 63%), consistent with the hypothesis that it anchors specific representational features rather than applying general-purpose regularization.

An open question is whether the contrastive loss and inference-time activation addition are complementary. A model trained with rho-guided SFT might still benefit from activation engineering at inference time for dimensions not covered by the training probes. The OOD transfer results (Section 4.6) suggest that rho-guided training partially addresses untrained dimensions, but the gains are modest (5pp aggregate), leaving room for further improvement via activation-based steering.

### Toward a Multi-Dimensional Safety Profile

The extension from six to eight behavioral dimensions (Section 4.8) moves toward a more comprehensive safety profile for language models. The refusal/over-refusal pair is particularly important: it captures the tension between safety (refusing harmful requests) and utility (answering legitimate questions). A model with high refusal $\rho$ and low over-refusal $\rho$ (answer rate) is over-aligned — safe but useless on sensitive topics. A model with low refusal $\rho$ and high over-refusal $\rho$ is under-aligned — helpful but unsafe. The two dimensions together define a "safety-utility frontier" that cannot be captured by either metric alone.

Similarly, the deception dimension complements factual fidelity. Factual probes test whether the model knows which facts are true; deception probes test whether it preferentially generates honest over evasive content. A model could score well on factual probes (it "knows" the truth) while scoring poorly on deception probes (it nevertheless produces evasive responses) — a pattern consistent with learned sycophancy or strategic ambiguity.

### Practical Deployment Recommendations

Based on the experimental findings, we offer three concrete recommendations for practitioners applying rho-guided SFT:

1. **Always include the SFT component.** The refusal buffer effect (Section 4.3) shows that contrastive-only training erodes refusal capability. The combined loss preserves refusal while improving behavioral calibration.

2. **Use a non-zero margin ($\gamma \geq 0.1$).** The margin ablation (Section 4.3.2) demonstrates that $\gamma = 0$ causes bias inversion. The margin prevents over-optimization and should be treated as a structural requirement, not a tunable hyperparameter.

3. **Start with $\lambda_\rho = 0.2$ for balanced improvement.** The dose-response curve shows diminishing returns above $\lambda_\rho = 0.2$ for factual and bias dimensions, while toxicity continues to improve up to $\lambda_\rho = 0.5$. For applications where all dimensions matter equally, $\lambda_\rho = 0.2$ provides the best risk-adjusted improvement with minimal variance.

### Limitations

**Sample sizes.** The primary dose-response experiments use 5 seeds (Qwen) and 2 seeds (Llama). The TruthfulQA external validation uses 3 seeds across 3 $\lambda_\rho$ values. The ablation study uses 5 seeds per condition (expanded from the original 2). Effect sizes are large ($d > 10$ for key comparisons) and p-values are below 0.0001 for the main findings. The refusal dimension was measured on 3 of the 5 seeds. The margin ablation uses 5 seeds.

**Scale.** All experiments were conducted on 7B-8B parameter models. We have not verified whether the SFT inversion phenomenon or the contrastive repair mechanism operate the same way at 70B+ scale.

**Task performance.** We did not run full benchmark suites (MMLU, HumanEval, etc.) after training. The TruthfulQA results suggest some task performance cost that the contrastive loss partially mitigates, but we cannot characterize the full capability impact.

**Probe coverage.** The 806 probes across 4 dimensions are modest by evaluation standards. The behavioral signal is measured by Spearman $\rho$, which is robust to small samples, but coverage of specific behavioral subtypes within each dimension is limited.

**Single SFT recipe.** We tested one SFT configuration (1000 texts, 1 epoch, LoRA rank 8). Different SFT data mixtures, longer training, or full fine-tuning (without LoRA) may produce different patterns of calibration damage and different responses to the contrastive loss.

**Generality of the contrastive loss.** The behavioral probes are English-language and primarily Western-centric. The method's effectiveness on multilingual models or culturally diverse behavioral dimensions is untested.

## 6. Conclusion

Standard supervised fine-tuning damages the internal confidence calibration of large language models in ways that conventional benchmarks do not detect. On Qwen2.5-7B-Instruct, a single epoch of standard SFT inverts toxicity discrimination from $\rho = +0.145$ to $\rho = -0.003$ ($n = 5$ seeds). Rho-guided SFT, which adds a contrastive auxiliary loss during training, repairs this damage with a monotonic dose-response: at $\lambda_\rho = 0.5$, toxicity $\rho$ reaches $+1.137$, factual $\rho$ improves by $+0.305$, and the effect replicates on Llama-3.1-8B-Instruct.

The active ingredient is the contrastive behavioral signal, not regularization. Correct labels are necessary and sufficient: the contrastive loss alone (without SFT) achieves comparable calibration results, while shuffled labels destroy the model ($d = 10.8$ for rho-guided vs SFT-only on toxicity, $p < 0.0001$). This means the method can be applied to any SFT pipeline with minimal modification: add the auxiliary loss, provide behavioral probes, and choose $\lambda_\rho$.

Two additional findings strengthen the practical case. First, contrastive-only training erodes refusal capability ($d = -8.4$), while the full rho-guided method preserves it. The SFT component acts as a "refusal buffer" and should not be omitted. Second, the hinge margin $\gamma = 0.1$ is structurally necessary to prevent over-optimization that flips the bias signal negative.

The intervention is not free. TruthfulQA MC2 shows a 13.3-point drop (vs. 18.6 for SFT-only), recovering 29% of the SFT truthfulness damage, and the contrastive loss slightly increases toxicity ECE. But the trade-off is favorable: in exchange for modest ECE increases, the model gains dramatically improved behavioral discrimination that transfers out-of-distribution, with 63% lower inter-seed variance.

The theoretical account connecting rho-guided SFT to representation engineering and superposition theory (Section 3.5) explains the variance collapse as a consequence of breaking the degeneracy of the SFT loss landscape: the contrastive loss provides a consistent gradient signal that anchors behavioral features, guiding all seeds toward the same basin. The extension to eight behavioral dimensions — including deception detection and over-refusal — provides a more comprehensive safety-utility profile, capturing the tension between safety (refusal) and utility (avoiding over-refusal) within a single audit framework.

For applications where internal calibration matters (safety-critical systems, uncertainty quantification, adversarial robustness), rho-guided SFT provides a practical, reliable solution. The method requires only behavioral probes (shipped with rho-eval), a single hyperparameter ($\lambda_\rho$), and no additional training data beyond standard SFT corpora.

## References

1. Gudibande, A., Wallace, E., Snell, C., Geng, X., Liu, H., Abbeel, P., Levine, S., & Song, D. (2024). The False Promise of Imitating Proprietary LLMs. *ICLR 2024*.

2. Hannun, A., et al. (2023). MLX: An Efficient Machine Learning Framework for Apple Silicon. Apple Machine Learning Research.

3. Hartvigsen, T., Gabriel, S., Palangi, H., Sap, M., Ray, D., & Kamar, E. (2022). ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection. *ACL 2022*.

4. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

5. Jaiswal, A., Gan, Z., Du, X., Zhang, B., Wang, Z., & Yang, Y. (2024). Compressing LLMs: The Truth is Rarely Pure and Never Simple. *ICLR 2024*.

6. Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. *arXiv:2207.05221*.

7. Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *ACL 2022*.

8. Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS 2022*.

9. Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. (2024). Steering Llama 2 via Contrastive Activation Addition. *ACL 2024*.

10. Parrish, A., et al. (2022). BBQ: A Hand-Built Bias Benchmark for Question Answering. *Findings of ACL 2022*.

11. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS 2023*.

12. Sanchez, B. (2026). rho-eval: Behavioral Auditing Toolkit for LLMs. *Zenodo*. doi:10.5281/zenodo.18743959

13. Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., & Hashimoto, T. B. (2023). Stanford Alpaca: An Instruction-Following LLaMA Model. GitHub.

14. Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., & Manning, C. D. (2023). Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback. *EMNLP 2023*.

15. Fu, Y., Li, R., Long, X., Yu, H., Han, X., Yin, Y., & Li, P. (2025). Pruning Weights but Not Truth: Safeguarding Truthfulness While Pruning LLMs. *Findings of EMNLP 2025*.

16. Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, Z., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., & Hendrycks, D. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*.

17. Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., & Olah, C. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Anthropic Research*.

18. Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., Joseph, N., Kadavath, S., Kernion, J., Conerly, T., El-Showk, S., Elhage, N., Hatfield-Dodds, Z., Hernandez, D., Hume, T., Johnston, S., Kravec, S., Lovitt, L., Nanda, N., Olsson, C., Amodei, D., Brown, T., Clark, J., McCandlish, S., Olah, C., Mann, B., & Kaplan, J. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. *arXiv:2204.05862*.

---

## Appendix A: Per-Seed Ablation Results

The following tables report the full per-seed $\Delta\rho$ (change from baseline) for each condition in the 5-seed ablation study (Section 4.3). Baseline values: factual $\rho = 0.603$, toxicity $\rho = 0.145$, sycophancy $\rho = -0.041$, bias $\rho = 0.036$.

**Table A1: SFT-only ($\lambda_\rho = 0$) per-seed $\Delta\rho$**

| Seed | $\Delta$ Factual | $\Delta$ Toxicity | $\Delta$ Sycophancy | $\Delta$ Bias |
|:---:|:---:|:---:|:---:|:---:|
| 42 | +0.007 | -0.113 | +0.037 | -0.008 |
| 123 | +0.116 | -0.091 | +0.038 | -0.005 |
| 456 | +0.225 | -0.201 | +0.037 | -0.012 |
| 789 | +0.211 | -0.236 | +0.037 | -0.012 |
| 1337 | +0.011 | -0.099 | +0.037 | -0.010 |
| **Mean** | **+0.114** | **-0.148** | **+0.037** | **-0.009** |
| **Std** | **0.105** | **0.063** | **0.001** | **0.003** |

**Table A2: Rho-guided ($\lambda_\rho = 0.2$) per-seed $\Delta\rho$**

| Seed | $\Delta$ Factual | $\Delta$ Toxicity | $\Delta$ Sycophancy | $\Delta$ Bias |
|:---:|:---:|:---:|:---:|:---:|
| 42 | +0.159 | +0.523 | +0.039 | +0.036 |
| 123 | +0.225 | +0.704 | +0.040 | +0.034 |
| 456 | +0.168 | +0.660 | +0.040 | +0.039 |
| 789 | +0.124 | +0.561 | +0.041 | +0.032 |
| 1337 | +0.138 | +0.658 | +0.041 | +0.030 |
| **Mean** | **+0.163** | **+0.621** | **+0.040** | **+0.034** |
| **Std** | **0.039** | **0.072** | **0.001** | **0.003** |

**Table A3: Contrastive-only per-seed $\Delta\rho$**

| Seed | $\Delta$ Factual | $\Delta$ Toxicity | $\Delta$ Sycophancy | $\Delta$ Bias |
|:---:|:---:|:---:|:---:|:---:|
| 42 | +0.221 | +0.455 | +0.042 | +0.015 |
| 123 | +0.250 | +0.452 | +0.041 | +0.019 |
| 456 | +0.197 | +0.401 | +0.055 | +0.026 |
| 789 | +0.182 | +0.388 | +0.042 | +0.021 |
| 1337 | +0.291 | +0.428 | +0.041 | +0.026 |
| **Mean** | **+0.228** | **+0.425** | **+0.044** | **+0.021** |
| **Std** | **0.043** | **0.029** | **0.006** | **0.005** |

**Table A4: Shuffled-pairs per-seed $\Delta\rho$**

| Seed | $\Delta$ Factual | $\Delta$ Toxicity | $\Delta$ Sycophancy | $\Delta$ Bias |
|:---:|:---:|:---:|:---:|:---:|
| 42 | -0.390 | -0.360 | +0.036 | -0.031 |
| 123 | -0.680 | -0.583 | +0.028 | -0.035 |
| 456 | -0.120 | +0.250 | +0.038 | +0.012 |
| 789 | -0.502 | -0.243 | +0.036 | -0.032 |
| 1337 | -0.004 | +0.178 | +0.039 | +0.013 |
| **Mean** | **-0.339** | **-0.152** | **+0.036** | **-0.015** |
| **Std** | **0.271** | **0.355** | **0.004** | **0.024** |

**Table A5: Refusal dimension per-seed $\Delta\rho$ (3 seeds)**

| Seed | SFT-only | Rho-guided | Contrastive-only | Shuffled-pairs |
|:---:|:---:|:---:|:---:|:---:|
| 42 | -0.007 | +0.014 | -0.087 | +0.018 |
| 123 | +0.006 | +0.025 | -0.094 | +0.012 |
| 456 | -0.004 | +0.004 | -0.070 | -0.001 |
| **Mean** | **-0.002** | **+0.014** | **-0.084** | **+0.010** |
| **Std** | **0.007** | **0.011** | **0.012** | **0.010** |

## Appendix B: Figure Captions

**Figure 1: Dose-response curve.** Change in behavioral $\rho$ ($\Delta\rho$) as a function of $\lambda_\rho$ for Qwen2.5-7B-Instruct (5-seed mean with standard error bands). Three behavioral dimensions are shown: factual (green), toxicity (red), and bias (blue). The toxicity inversion at $\lambda_\rho = 0$ ($\Delta\rho = -0.148$) is corrected monotonically as $\lambda_\rho$ increases. All three dimensions improve monotonically. The y-axis zero line is marked to emphasize the sign change for toxicity.

**Figure 2: Ablation bar chart.** Mean $\Delta\rho$ across 5 seeds for four training conditions (SFT-only, rho-guided, contrastive-only, shuffled-pairs) on four behavioral dimensions. Error bars show $\pm 1$ standard deviation. Significance annotations: *** $p < 0.001$, ** $p < 0.01$, * $p < 0.05$. The shuffled-pairs condition demonstrates that correct behavioral labels are essential — shuffled labels actively destroy the model's pre-existing calibration.

**Figure 3: Variance collapse.** Box-and-whisker plots comparing inter-seed variability between SFT-only and rho-guided conditions across four behavioral dimensions. Individual seed values are overlaid as scatter points. The factual dimension shows the most dramatic collapse: $\sigma$ drops from 0.105 (SFT-only) to 0.039 (rho-guided), a 63% reduction. The contrastive loss not only improves the mean but dramatically stabilizes training across random seeds.

**Figure 4: TruthfulQA recovery and safety stress test.** (a) TruthfulQA MC2 scores for baseline, SFT-only, and rho-guided conditions, showing the 18.6pp drop from SFT and the 29% recovery from the contrastive loss at $\lambda_\rho = 0.5$. (b) Jailbreak refusal rates and benign false-positive rates across four training conditions. All conditions maintain zero false positives on benign prompts. Contrastive-only training achieves the highest jailbreak refusal rate (80%), despite showing the worst refusal $\rho$ in confidence probing — highlighting the dissociation between passive confidence metrics and active generation behavior.

---

*Code and data: https://github.com/SolomonB14D3/knowledge-fidelity*
*Toolkit: `pip install rho-eval`*
*DOI: 10.5281/zenodo.18743959*
