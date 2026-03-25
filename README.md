# rho-eval: Behavioral Auditing for LLMs

[![PyPI](https://img.shields.io/pypi/v/rho-eval)](https://pypi.org/project/rho-eval/)
[![Tests](https://img.shields.io/badge/tests-213%20passed-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLX](https://img.shields.io/badge/Apple%20Silicon-MLX%20Accelerated-black?logo=apple)](https://github.com/ml-explore/mlx)

**Measure where language models get surprising truths wrong — then fix it.**

Measures 8 behavioral dimensions — factual accuracy, toxicity, bias, sycophancy, reasoning, refusal, deception, and over-refusal — using Spearman rank correlation over teacher-forced confidence probes. Ships 1,826 probes as JSON with no internet required.

## Install

```bash
pip install rho-eval                    # Core (auditing + SVD + probes)
pip install "rho-eval[full]"            # Everything including MLX
```

## Quick Start

```python
import rho_eval

# Audit any model across all 8 behaviors
report = rho_eval.audit("Qwen/Qwen2.5-7B-Instruct")
print(report)
```

```bash
# CLI
rho-eval Qwen/Qwen2.5-7B-Instruct --behaviors all

# One-command behavioral repair
rho-surgery Qwen/Qwen2.5-7B-Instruct -o ./repaired-7b/

# Diagnose expression gaps in base models
rho-unlock diagnose Qwen/Qwen2.5-1.5B --behaviors mmlu,arc,truthfulqa
```

## What It Does

| Module | Purpose |
|--------|---------|
| **`rho-eval`** | Behavioral auditing across 8 dimensions via confidence probes |
| **`rho-surgery`** | End-to-end behavioral repair: diagnose, compress, LoRA SFT, verify |
| **`rho-unlock`** | Expression gap diagnostic + contrastive decoding |
| **`snap-on`** | Train tiny logit-space adapters on frozen base models |
| **`rho-bench`** | Adversarial pressure testing |

## Built-In Probes (1,826 total)

| Behavior | Count |
|----------|------:|
| Factual | 206 |
| Bias | 357 |
| Sycophancy | 150 |
| Toxicity | 200 |
| Reasoning | 100 |
| Deception | 100 |
| Refusal | 150 |
| Over-refusal | 150 |

## Model Compatibility

Works on any HuggingFace causal LM. Auto-dispatches to MLX on Apple Silicon.

**Validated:** Qwen2.5 (0.5B–32B), Mistral 7B, Llama 3.1 8B, GPT-2 (7M–210M)

## Citation

```bibtex
@software{sanchez2026rhoeval,
  author = {Sanchez, Bryan},
  title = {rho-eval: Behavioral Auditing for Large Language Models},
  year = {2026},
  url = {https://github.com/SolomonB14D3/knowledge-fidelity}
}
```

## License

MIT
