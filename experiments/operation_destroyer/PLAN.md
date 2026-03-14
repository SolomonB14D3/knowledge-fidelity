# Operation Destroyer: Beat Qwen3-4B-Instruct on All Benchmarks

## Target Model

- **Base:** Qwen3-4B-Base (4.02B params, Apache 2.0, 15.3M downloads)
- **Opponent:** Qwen3-4B (instruct version, 25.6M downloads, 566 likes)
- **Architecture:** qwen3 (GQA, SwiGLU, RoPE, RMSNorm, QK-Norm)

## Qwen3-4B Instruct Benchmark Scores (from Qwen technical report + EvalScope)

Non-Thinking Mode (our direct competitor):

| Benchmark | Qwen3-4B Score | Category |
|-----------|---------------|----------|
| MMLU-Pro | 61.45% | Knowledge |
| MMLU-Redux | 83.13% | Knowledge |
| IFEval (strict inst) | 68.88% | Instruction Following |
| IFEval (strict prompt) | 61.45% | Instruction Following |
| GPQA | 41.54% | Expert Knowledge |
| MATH-500 | 43.55% | Math |
| LiveCodeBench | 28.57% | Code |

## Qwen3-4B-Base Scores (from tech report Table 7)

| Benchmark | Qwen3-4B-Base | Category |
|-----------|--------------|----------|
| MMLU | ~65% (estimated) | Knowledge |
| MMLU-Pro | ~35% (estimated) | Knowledge |
| BBH | ~60% (estimated) | Reasoning |
| GSM8K | ~70% (estimated) | Math |

## What We Need to Beat

To "destroy" Qwen3-4B-Instruct, we need to exceed it on:

1. MMLU / MMLU-Pro / MMLU-Redux (knowledge)
2. IFEval (instruction following)
3. Safety / refusal (behavioral)
4. TruthfulQA (truthfulness)
5. ARC-Challenge (science reasoning)
6. HellaSwag (commonsense)
7. GSM8K (math reasoning)
8. Our internal benchmarks (rho-eval bias, sycophancy, expression gap)

## Architecture

Same proven logit-level adapter:

```
logits = lm_head(h) + adapter(lm_head(h))
```

- Base model: Qwen3-4B-Base (frozen, never modified)
- Adapter: ~30-50M params logit-level MLP (SwiGLU)

### Key Vocab Check

**CRITICAL:** Verify Qwen3-4B-Base vocab size matches our adapter dimension. Qwen3 may use a different vocab than Qwen2.5. Run: `model.config.vocab_size` on Qwen3-4B-Base before training.

## Training Data Strategy

The v2 mix proved the concept. v3 needs to destroy.

### Phase 1: Best available SFT data (~50K examples)

| Source | Count | Purpose |
|--------|-------|---------|
| OpenHermes 2.5 (filtered) | 15K | Diverse high-quality instruction following |
| UltraChat (multi-turn) | 10K | Conversation, context tracking |
| MAGPIE-Pro (Qwen-generated) | 5K | In-distribution for Qwen architecture |
| Constraint/IFEval-style | 5K | Exact format compliance, word limits, counting |
| Safety/refusal (HH-RLHF + BeaverTails) | 5K | Refusal behavior |
| Concise responses | 3K | Counter verbosity bias |
| Math reasoning (GSM8K format) | 3K | Step-by-step math |
| Code instruction pairs | 2K | Basic code generation |
| TruthfulQA-style (factual accuracy) | 2K | Truthfulness |
| **Total** | **~50K** | |

Key improvements over v2 (10K):

1. 5x more data for more robust learning
2. Dedicated math and code slices (new capability targets)
3. MAGPIE data for Qwen-distribution alignment (GRAPE principle)
4. Much larger constraint-following slice (5K vs 1K)
5. Much larger safety slice (5K vs 1K)

### Phase 2: KL Distillation (if instruct logits available)

- Cache Qwen3-4B instruct's logits on all 50K training prompts
- Train adapter with mixed loss:
  - `0.5 * KL(adapter_output || instruct_logits)`
  - `0.3 * CE(adapter_output, instruct_tokens)`
  - `0.2 * contrastive_loss(correct_answer_logit - wrong_answer_logit)`
- This teaches the full distribution shape, not just top-1 token

### Phase 3: DPO Refinement (1-2 hours additional)

- Generate pairs from the v3 adapter: verbose vs concise, compliant vs non-compliant
- Train DPO to prefer:
  - Concise when concise is appropriate
  - Constraint-compliant over verbose
  - Refusal over harmful compliance
- This specifically targets remaining IFEval and safety gaps

## Training Configuration

- **Architecture:** Logit-level MLP (SwiGLU, d_inner=128 or 256)
- **Parameters:** ~30-50M (0.75-1.25% of base)
- **Learning rate:** 1e-5 (conservative, proven)
- **Epochs:** 3 over 50K examples
- **Base model:** Frozen Qwen3-4B-Base
- **Loss:** KL + CE + contrastive (Phase 2) or CE only (Phase 1)
- **Hardware:** Mac Studio M3 Ultra
- **Estimated training time:**
  - Phase 1 (SFT): 6-10 hours (5x more data than previous)
  - Phase 2 (KL): Same training, different loss
  - Phase 3 (DPO): 2-3 hours additional
  - Total: ~12-15 hours (overnight + morning)

## Evaluation Plan

### Standard Public Benchmarks

| Benchmark | Questions | What it tests | Target |
|-----------|-----------|---------------|--------|
| MMLU | 500+ | General knowledge | >= Qwen3-4B score |
| MMLU-Pro | 500+ | Harder knowledge | >= 61.45% |
| ARC-Challenge | 500+ | Science reasoning | > Qwen3-4B |
| HellaSwag | 500+ | Commonsense | > Qwen3-4B |
| TruthfulQA | 500+ | Truthfulness | > Qwen3-4B |
| GSM8K | 500+ | Math reasoning | > Qwen3-4B |
| IFEval | Full set (541) | Instruction following | > 68.88% |

### Behavioral Benchmarks (our unique advantage)

| Benchmark | What it tests | Target |
|-----------|---------------|--------|
| Safety refusal | Harmful prompt rejection | > 90% |
| rho-eval bias | Bias detection | > Qwen3-4B |
| rho-eval sycophancy | Sycophancy resistance | > Qwen3-4B |
| Expression gap | Knowledge vs expression | < 1% gap |
| Multi-turn coherence | Context tracking | >= Qwen3-4B |
| Qualitative generation | Response quality | Comparable or better |

### The Destruction Criteria

To claim we "destroyed" Qwen3-4B-Instruct:

- Win on >= 6 of 7 standard benchmarks
- Win on all behavioral benchmarks
- Zero MMLU regression from base (0.0% delta)
- Qualitative generation is comparable

## Risk Assessment

### Where we're likely to win

- **Safety:** v2 already beat SmolLM2-360M-Instruct. With 5K safety examples, we should beat Qwen3-4B.
- **Knowledge preservation:** Logit adapter is structurally unable to damage knowledge. 0.0% delta proven.
- **Bias/sycophancy:** Our rho-eval probes + training data should beat generic instruct tuning.
- **TruthfulQA:** The adapter preserves base model knowledge while instruct tuning may have degraded it (alignment tax).

### Where it's a fight

- **MMLU-Pro/MMLU-Redux:** The instruct model's training may have added genuine knowledge. We can only express what the base model already knows. If the base model scores lower than instruct on logit accuracy, we can't close that gap.
- **IFEval:** This is our weakest dimension. v2 hit 50% from 24%. We need 69%+. The 5K constraint examples and DPO should help but this is the hardest benchmark to beat.
- **GSM8K:** Math reasoning requires chain-of-thought, not just format. The adapter may not teach multi-step reasoning.

### Where we might lose

- **Code (LiveCodeBench):** Code generation is a genuine capability that requires training, not just expression. The base model may not have enough code knowledge for the adapter to unlock.
- **IFEval strict prompt level (61.45%):** Very specific constraint compliance. May need more targeted training.

## Contingency: What If Base Model Has Low Logit Accuracy?

Before training the adapter, run `rho-unlock diagnose` on Qwen3-4B-Base:

- If logit accuracy is close to instruct scores: adapter can match instruct (expression bottleneck)
- If logit accuracy is much lower: instruct tuning genuinely added knowledge, adapter ceiling is lower
- If large expression gap: contrastive decoding + adapter combination may close the gap

**This diagnostic MUST run first.** It determines whether "destroying" instruct is possible or whether we're aiming for "matching instruct with zero knowledge loss."

## Timeline

### Day 1 (Morning)

1. Download Qwen3-4B-Base and Qwen3-4B
2. Check vocab size compatibility
3. Run `rho-unlock diagnose` on base model (MMLU, ARC, TruthfulQA, HellaSwag, IFEval)
4. Run `rho-unlock diagnose` on instruct model (get ceiling numbers)
5. Decision point: Is destruction feasible based on expression gap?

### Day 1 (Afternoon)

1. Prepare training data (50K mixed dataset)
2. Cache instruct model logits for KL distillation (if using Phase 2)
3. Start Phase 1 SFT training (runs overnight)

### Day 2 (Morning)

1. Evaluate Phase 1 adapter on all benchmarks
2. Start Phase 3 DPO if needed (runs 2-3 hours)
3. Final evaluation on complete benchmark suite

### Day 2 (Afternoon)

1. Compare all numbers against Qwen3-4B-Instruct
2. If winning: upload to Hugging Face, write model card
3. If close: iterate on training data mix, retrain
4. If losing on knowledge: combine adapter + contrastive decoding

### Day 3

1. Release model on Hugging Face
2. Write comparison blog post / Reddit post
3. Paper 8 update with Qwen3-4B results

## Release Plan (if successful)

### Hugging Face Model Card

- **Name:** "Qwen3-4B-Base + rho-unlock v3 Communication Head"
- **Description:** "A logit-level snap-on adapter that matches or beats Qwen3-4B-Instruct on N/7 standard benchmarks with zero knowledge loss. The base model is never modified."
- **Include:** full benchmark comparison table, training details, adapter weights download
- **Link:** `pip install rho-eval` for the training pipeline

### The Post

> "I just released a 4B model that beats Qwen3-4B-Instruct on [N] out of [7] benchmarks.
>
> No instruct tuning. No RLHF. A 30M snap-on communication head trained in 12 hours on a Mac Studio.
>
> The base model's weights were never touched.
>
> [benchmark comparison table]
>
> Model: [HuggingFace link]
> Tool: pip install rho-eval
> Papers: [DOIs]"

## First Step

Download Qwen3-4B-Base, check vocab size, run diagnostics. Everything depends on what the base model already knows.

---

## Compatibility Verification

- [x] Vocab size: **151,936** — matches Qwen2.5-0.5B/1.5B/3B. Existing adapters compatible.
- [x] MLX support: `qwen3` model type present in mlx_lm 0.30.7
- [ ] Model download (Qwen3-4B-Base)
- [ ] Model download (Qwen3-4B instruct)
- [ ] rho-unlock diagnose on base
- [ ] rho-unlock diagnose on instruct
- [ ] Decision: feasible?

### Qwen3-4B-Base Architecture

| Property | Value |
|----------|-------|
| vocab_size | 151,936 |
| d_model | 2,560 |
| num_layers | 36 |
| num_heads | 32 |
| num_kv_heads | 8 (GQA) |
| intermediate_size | 9,728 |
| model_type | qwen3 |
| adapter params (d_inner=64) | ~29M |
| adapter params (d_inner=128) | ~58M |
| adapter params (d_inner=256) | ~117M |

## Results Log

*(Updated as experiments complete)*

### Phase 1.5/1.6: Vortex Domain Adapter (2026-03-13)

**Discovery**: Restricted 3-vortex near-invariant Q = r₁₂ + ε(r₁₃ + r₂₃), ε = Γ₃

| Metric | Baseline | With Adapter v4 |
|--------|----------|-----------------|
| Pass rate | 2/11 (18.2%) | 8/11 (72.7%) |
| Mean margin | -12.392 | +0.997 |
| Diagnostic | KNOWLEDGE_GAP | FIXABLE_BIAS |
| vp11 margin | -29.959 | +3.987 ✓ |

**Training**: lr=5e-7, 1000 steps, d_inner=64, vortex_aligned_30.json (30 examples)

### Discovery 5 — H-Lz Exact Invariant Repair (2026-03-13)
- Found numerically exact invariant (frac_var = exact)
- Oracle: -19.64 → +26.05 after 800-step targeted adapter
- Now have multi_domain_v2.npz (merged choreography + vortex + H-Lz)
- Workflow officially validated across 3 physics domains
