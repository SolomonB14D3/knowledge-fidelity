# rho-eval v2.6.0: Behavioral Auditing for LLMs

[![PyPI](https://img.shields.io/pypi/v/rho-eval)](https://pypi.org/project/rho-eval/)
[![Paper: SFT](https://zenodo.org/badge/DOI/10.5281/zenodo.18854943.svg)](https://doi.org/10.5281/zenodo.18854943)
[![Paper: Grassmann](https://zenodo.org/badge/DOI/10.5281/zenodo.18865861.svg)](https://doi.org/10.5281/zenodo.18865861)
[![Paper: Phase Transitions](https://zenodo.org/badge/DOI/10.5281/zenodo.18865198.svg)](https://doi.org/10.5281/zenodo.18865198)
[![Paper: Confidence Cartography](https://zenodo.org/badge/DOI/10.5281/zenodo.18703505.svg)](https://doi.org/10.5281/zenodo.18703505)
[![Paper: Contrastive Pretraining](https://zenodo.org/badge/DOI/10.5281/zenodo.18870555.svg)](https://doi.org/10.5281/zenodo.18870555)
[![Paper: Expression Bottleneck](https://zenodo.org/badge/DOI/10.5281/zenodo.18895248.svg)](https://doi.org/10.5281/zenodo.18895248)
[![Paper: Snap-On](https://zenodo.org/badge/DOI/10.5281/zenodo.18902617.svg)](https://doi.org/10.5281/zenodo.18902617)
[![Paper: CF90](https://zenodo.org/badge/DOI/10.5281/zenodo.18718545.svg)](https://doi.org/10.5281/zenodo.18718545)
[![Tests](https://img.shields.io/badge/tests-213%20passed-brightgreen)](tests/)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Demo-blue)](https://huggingface.co/spaces/bsanch52/knowledge-fidelity-demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLX](https://img.shields.io/badge/Apple%20Silicon-MLX%20Accelerated-black?logo=apple)](https://github.com/ml-explore/mlx)
[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink?logo=github)](https://github.com/sponsors/SolomonB14D3)

**Measure where language models get surprising truths wrong — then fix it.**

rho-eval measures 8 behavioral dimensions — factual accuracy, toxicity, bias, sycophancy, reasoning, refusal, deception, and over-refusal — using Spearman rank correlation over teacher-forced confidence probes. It ships 1,826 probes as JSON with no internet required.

> *Formerly `knowledge-fidelity`. All v1.x imports still work.*

## What It Does

| Module | Purpose |
|--------|---------|
| **`rho-audit`** | Behavioral auditing across 8 dimensions via confidence probes |
| **`rho-interpret`** | SVD subspace extraction and Grassmann angle analysis |
| **`rho-align`** | Rho-Guided SFT with contrastive auxiliary loss |
| **`rho-steer`** | SAE-based disentangled behavioral steering |
| **`rho-bench`** | Fidelity-Bench 2.0: adversarial pressure testing |
| **`rho-surgery`** | End-to-end behavioral repair: diagnose, compress, LoRA SFT, verify |
| **`rho-benchmark`** | Full benchmarking (8-dim audit + TruthfulQA MC2) with comparison |
| **`rho-unlock`** | Expression gap diagnostic + contrastive decoding to rescue hidden capability |
| **`snap-on`** | Train tiny logit-space adapters on frozen base models — zero knowledge damage |

## Install

```bash
pip install rho-eval                    # Core (auditing + SVD + probes)
pip install "rho-eval[full]"            # Everything including MLX
```

Or from source:

```bash
git clone https://github.com/SolomonB14D3/knowledge-fidelity
cd knowledge-fidelity
pip install -e ".[full]"
```

## Quick Start

### Python API

```python
import rho_eval

# Audit any model across all 8 behaviors
report = rho_eval.audit("Qwen/Qwen2.5-7B-Instruct")
print(report)
# <AuditReport model='Qwen/Qwen2.5-7B-Instruct' behaviors=8 status=WARN>

# Compare two models
baseline = rho_eval.audit("Qwen/Qwen2.5-7B-Instruct")
repaired = rho_eval.audit("./repaired-7b/model/")
delta = rho_eval.compare(repaired, baseline)
print(delta.to_table())  # Colored delta table
```

### CLI

```bash
# Full behavioral report card
rho-eval Qwen/Qwen2.5-7B-Instruct --behaviors all

# Specific behaviors, JSON output
rho-eval my-model/ --behaviors factual,bias,sycophancy --format json

# Compare against a baseline
rho-eval compressed-model/ --compare baseline.json

# Adversarial pressure test
rho-bench Qwen/Qwen2.5-7B-Instruct

# One-command behavioral repair
rho-surgery Qwen/Qwen2.5-7B-Instruct -o ./repaired-7b/

# Benchmark before vs after (8-dim audit + TruthfulQA MC2)
rho-benchmark ./repaired-7b/model/ --baseline Qwen/Qwen2.5-7B-Instruct

# Diagnose expression gaps (base models have knowledge they can't express)
rho-unlock diagnose Qwen/Qwen2.5-7B --behaviors mmlu,arc,truthfulqa

# Train a snap-on adapter (zero knowledge damage)
snap-on train --model Qwen/Qwen2.5-7B --mode logit --save_dir ./my_adapter
snap-on eval --model Qwen/Qwen2.5-7B --adapter ./my_adapter --mmlu_n 500
```

## Why This Exists

Language models fail where truth is surprising. Sycophancy picks the expected answer over the true one. Bias picks the stereotype over the individual. Standard SFT makes this worse in ways benchmarks don't catch.

rho-eval measures exactly where a model gets surprising truths wrong, and rho-guided SFT repairs it — without the alignment tax. See [our papers](#papers) for the full experimental story.

## Built-In Probes (1,826 total)

All probes ship as JSON. No internet download needed.

| Behavior | Probe Sets | Count |
|----------|-----------|------:|
| Factual | default, mandela, medical, commonsense, truthfulqa, expanded | 206 |
| Bias | BBQ 300 + bridge probes + biology-grounded | 357 |
| Sycophancy | Anthropic model-written-evals | 150 |
| Toxicity | ToxiGen balanced | 200 |
| Reasoning | GSM8K + adversarial flattery | 100 |
| Deception | HH-RLHF honest/deceptive pairs | 100 |
| Refusal | harmful/benign pairs + expanded | 150 |
| Over-refusal | benign-but-edgy + expanded | 150 |
| Bench (Fidelity-Bench) | logic, social, clinical | 120 |

Run `rho-eval --list-probes` to see all available sets.

### Custom Behaviors (Plugin System)

```python
from rho_eval.behaviors import ABCBehavior, register

@register
class MyDomainBehavior(ABCBehavior):
    name = "my_domain"
    description = "Domain-specific evaluation"
    probe_type = "confidence"
    default_n = 50

    def load_probes(self, n=None, seed=42, **kwargs):
        return self._load_json_probes("my_domain/probes.json", n=n, seed=seed)

    def evaluate(self, model, tokenizer, probes, device="cpu", **kwargs):
        # Your evaluation logic
        return BehaviorResult(behavior=self.name, rho=0.7, ...)

# Now available everywhere:
report = rho_eval.audit("my-model", behaviors=["factual", "my_domain"])
```

## Model Compatibility

Works on any HuggingFace causal LM with standard attention layouts.

**Validated:** Qwen2.5 (0.5B-32B), Mistral 7B, Llama 3.1 8B, GPT-2 (7M-210M scale ladder)

## Apple Silicon (MLX)

rho-eval auto-dispatches to MLX on Apple Silicon. No code changes needed.

```bash
pip install mlx mlx-lm  # or: pip install "rho-eval[full]"
```

```python
import mlx_lm
from rho_eval import audit

model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-7B-Instruct-4bit")
report = audit(model=model, tokenizer=tokenizer, behaviors="all")
# Same API — ~5-10x faster on Apple Silicon
```

| Component | MLX Speedup |
|-----------|:-----------:|
| `audit()` — 8-behavior probe suite | ~5x |
| `mlx_rho_guided_sft()` — alignment training | ~10x |
| `analyze_confidence()` — cartography | ~5x |

## Compression Safety Guide

| Layer Type | Safe to Compress | Notes |
|------------|:---:|-------|
| Q, K, O projections | Yes (70% rank) | Main target |
| V projection | 90-95% only | High risk below 90% |
| MLP layers | Never | Destroys model at any level |

## Limitations

- **Probe scale:** 1,826 probes across 37 sets. Spearman correlation is robust to small samples, but statistical power for subtle shifts is limited.
- **Western-centric:** Probes cover primarily English-language, U.S.-centric social categories.
- **7B scale.** Merge and steering results validated on 7B models. Larger scales (70B+) should not be extrapolated without verification.
- **Toxicity is unaffected** by weight edits — it relies on highly distributed lexical features that structural interventions cannot modulate.

## Key Findings

- **Behavioral emergence is a data problem, not a scale problem.** Injecting contrastive pairs during pretraining breaks behavioral emergence barriers at small scales, exceeding vanilla models 5x larger with zero inference cost.
- **Contrastive injection stabilizes behavioral scaling.** Vanilla training produces non-monotonic scaling anomalies where capabilities regress at larger scale. Contrastive injection eliminates these anomalies across four model scales tested.
- **Broad truth fixes propagate to narrow ones — above a capacity threshold.** Fixing sycophancy spontaneously improves bias, but not vice versa. Cross-transfer is absent at 3M parameters, emerges sharply at 5M, and persists at 7M. This asymmetric hierarchy operates across pretraining scales and post-hoc surgery, establishing it as a structural property of behavioral representations with a developmental onset.
- **Behavioral capabilities emerge through sharp phase transitions.** Training small language models from scratch reveals that behaviors like over-refusal appear in discrete jumps, not gradual improvement.
- **Geometry precedes emergence.** Effective dimensionality expansion in weight subspaces predicts behavioral phase transitions by hundreds of training steps — the geometry reorganizes before the behavior appears.
- **Surgery concentrates, not rotates.** Grassmann angle analysis of rho-guided SFT shows behavioral subspaces sharpen (effective dimension compresses) rather than rotating to new orientations.
- **Compression preserves behavioral structure when protecting the right singular values.** SVD at 70% rank on Q/K/O projections retains behavioral fidelity; V and MLP layers are fragile.
- **Expression gaps are universal in base models and vanish with instruction tuning, not scale.** Every base model from 0.5B to 7B has significant expression gaps (knowledge present at the logit level but absent in free generation). Instruction-tuned models at 1.5B+ have zero gap. The bottleneck is instruction tuning, not model size.
- **Logit-space adapters produce instruction-following with zero knowledge damage.** A 29M-parameter adapter operating on the frozen output logits achieves 0.0% MMLU degradation. Hidden-space adapters consistently destroy 5.0-8.5% of factual accuracy. A single adapter transfers across model scales (1.5B to 3B, 0.0% delta) and across model families (Qwen to Llama, -0.2% delta) without retraining.

Full experimental details, tables, and statistical analysis are in the papers below.

## Papers

1. **Rho-Guided Supervised Fine-Tuning: Post-Training Repair of Calibration Damage in Large Language Models** — [DOI: 10.5281/zenodo.18854943](https://doi.org/10.5281/zenodo.18854943)
2. **Behavioral Entanglement in Transformers: Grassmann Geometry of Rho-Guided SFT** — [DOI: 10.5281/zenodo.18865861](https://doi.org/10.5281/zenodo.18865861)
3. **Behavioral Phase Transitions in Small Language Models: Geometric Scaffolding Precedes Behavioral Emergence** — [DOI: 10.5281/zenodo.18865198](https://doi.org/10.5281/zenodo.18865198)
4. **Confidence Cartography: Teacher-Forced Probability as a False-Belief Sensor in Language Models** — [DOI: 10.5281/zenodo.18703505](https://doi.org/10.5281/zenodo.18703505) | [Repo](https://github.com/SolomonB14D3/confidence-cartography)
5. **CF90: Knowledge-Preserving SVD Compression for Large Language Models** — [DOI: 10.5281/zenodo.18718545](https://doi.org/10.5281/zenodo.18718545) | [Repo](https://github.com/SolomonB14D3/intelligent-svd)
6. **Contrastive Pretraining Teaches Format Generation, Not Behavioral Knowledge** — [DOI: 10.5281/zenodo.18870555](https://doi.org/10.5281/zenodo.18870555)
7. **Small Language Models Already Know More Than They Can Say** — [DOI: 10.5281/zenodo.18895248](https://doi.org/10.5281/zenodo.18895248)
8. **Snap-On Communication Modules: Instruction-Following Adapters That Preserve Base Model Knowledge** — [DOI: 10.5281/zenodo.18902617](https://doi.org/10.5281/zenodo.18902617)

## Citation

```bibtex
@article{sanchez2026rhoguided,
  author = {Sanchez, Bryan},
  title = {Rho-Guided Supervised Fine-Tuning: Post-Training Repair of
           Calibration Damage in Large Language Models},
  year = {2026},
  doi = {10.5281/zenodo.18854943},
  url = {https://doi.org/10.5281/zenodo.18854943}
}

@article{sanchez2026grassmann,
  author = {Sanchez, Bryan},
  title = {Behavioral Entanglement in Transformers: Grassmann Geometry
           of Rho-Guided Supervised Fine-Tuning},
  year = {2026},
  doi = {10.5281/zenodo.18865861},
  url = {https://doi.org/10.5281/zenodo.18865861}
}

@article{sanchez2026phasetransitions,
  author = {Sanchez, Bryan},
  title = {Behavioral Phase Transitions in Small Language Models:
           Geometric Scaffolding Precedes Behavioral Emergence},
  year = {2026},
  doi = {10.5281/zenodo.18865198},
  url = {https://doi.org/10.5281/zenodo.18865198}
}

@article{sanchez2026cartography,
  author = {Sanchez, Bryan},
  title = {Confidence Cartography: Teacher-Forced Probability as a
           False-Belief Sensor in Language Models},
  year = {2026},
  doi = {10.5281/zenodo.18703505},
  url = {https://doi.org/10.5281/zenodo.18703505}
}

@software{sanchez2026cf90,
  author = {Sanchez, Bryan},
  title = {CF90: Knowledge-Preserving SVD Compression for Large
           Language Models},
  year = {2026},
  doi = {10.5281/zenodo.18718545},
  url = {https://doi.org/10.5281/zenodo.18718545}
}

@article{sanchez2026contrastive,
  author = {Sanchez, Bryan},
  title = {Contrastive Pretraining Teaches Format Generation,
           Not Behavioral Knowledge},
  year = {2026},
  doi = {10.5281/zenodo.18870555},
  url = {https://doi.org/10.5281/zenodo.18870555}
}

@article{sanchez2026expression,
  author = {Sanchez, Bryan},
  title = {Small Language Models Already Know More Than They Can Say},
  year = {2026},
  doi = {10.5281/zenodo.18895248},
  url = {https://doi.org/10.5281/zenodo.18895248}
}

@article{sanchez2026snapon,
  author = {Sanchez, Bryan},
  title = {Snap-On Communication Modules: Instruction-Following Adapters
           That Preserve Base Model Knowledge},
  year = {2026},
  doi = {10.5281/zenodo.18902617},
  url = {https://doi.org/10.5281/zenodo.18902617}
}

@software{sanchez2026rhoeval,
  author = {Sanchez, Bryan},
  title = {rho-eval: Behavioral Auditing for Large Language Models},
  year = {2026},
  doi = {10.5281/zenodo.18743959},
  url = {https://doi.org/10.5281/zenodo.18743959}
}
```

## Contributing

PRs welcome for new probes, model support, or bug fixes. See [open issues](https://github.com/SolomonB14D3/knowledge-fidelity/issues).

## License

MIT
