# Knowledge Fidelity

[![PyPI](https://img.shields.io/pypi/v/knowledge-fidelity)](https://pypi.org/project/knowledge-fidelity/)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Demo-blue)](https://huggingface.co/spaces/bsanch52/knowledge-fidelity-demo)

**Compress an LLM while auditing whether it still knows truth vs popular myths.**

The first toolkit that uses the same factual probes for both structural importance scoring (SVD compression) and behavioral false-belief detection (confidence cartography). One call to compress and audit:

```python
from knowledge_fidelity import compress_and_audit

report = compress_and_audit("Qwen/Qwen2.5-7B-Instruct", ratio=0.7)
print(f"Retention: {report['retention']:.0%} | "
      f"False-belief signal: rho={report['rho_after']:.3f}")
# Retention: 100% | False-belief signal: rho=0.725
```

Or from the CLI:

```bash
# Auto-find the compression ratio that maximizes factual signal
knowledge-fidelity Qwen/Qwen2.5-0.5B --denoise
# DENOISING DETECTED: Mandela rho 0.257 → 0.771 (+0.514) at 60% ratio

# Benchmark across all probe categories
python experiments/fidelity_bench.py --model Qwen/Qwen2.5-0.5B
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

## Results (v0.2)

All results from the unified toolkit on Apple Silicon (M3 Ultra, CPU). Three model families validated.

### Multi-Seed CF90 Validation (70% rank, 3 seeds)

| Metric | Qwen2.5-0.5B | Qwen2.5-7B-Instruct | Mistral-7B-v0.1 |
|--------|:------------:|:-------------------:|:---------------:|
| Retention | **95%** ± 0% | **100%** ± 0% | **95%** ± 0% |
| rho before | 0.821 | 0.746 | 0.743 |
| rho after | 0.720 | 0.725 | 0.705 |
| rho drop | 0.101 ± 0.000 | **0.021** ± 0.000 | 0.038 ± 0.000 |
| Matrices compressed | 72 | 84 | 96 |
| Layers frozen | 18/24 | 21/28 | 24/32 |

CF90 generalizes across architectures: 95-100% retention with minimal rho loss at all scales.

### Joint Ablation: Compression Ratio vs Confidence (Qwen2.5-0.5B)

| Ratio | Default rho | Mandela rho | Medical rho |
|:-----:|:-----------:|:-----------:|:-----------:|
| 50% | 0.821 → 0.761 | 0.257 → 0.714 | 0.100 → 0.700 |
| 60% | 0.821 → 0.714 | 0.257 → 0.771 | 0.100 → 0.900 |
| 70% | 0.821 → 0.720 | 0.257 → 0.771 | 0.100 → 0.100 |
| 80% | 0.821 → 0.690 | 0.257 → 0.257 | 0.100 → 0.600 |
| 90% | 0.821 → 0.821 | 0.257 → 0.371 | 0.100 → 0.100 |
| 100% | 0.821 → 0.821 | 0.257 → 0.257 | 0.100 → 0.100 |

### Joint Ablation: Compression Ratio vs Confidence (Qwen2.5-7B-Instruct)

| Ratio | Default rho | Mandela rho | Medical rho |
|:-----:|:-----------:|:-----------:|:-----------:|
| 50% | 0.746 → 0.689 | 0.829 → 0.771 | −0.700 → 0.600 |
| 70% | 0.746 → 0.725 | 0.829 → **0.943** | −0.700 → −0.600 |
| 90% | 0.746 → 0.713 | 0.829 → **0.943** | −0.700 → −0.900 |
| 100% | 0.746 → 0.746 | 0.829 → 0.829 | −0.700 → −0.700 |

### Joint Ablation: Compression Ratio vs Confidence (Mistral-7B-v0.1)

| Ratio | Default rho | Mandela rho | Medical rho |
|:-----:|:-----------:|:-----------:|:-----------:|
| 50% | 0.743 → 0.686 | 0.771 → 0.771 | 0.300 → 0.300 |
| 60% | 0.743 → 0.723 | 0.771 → 0.771 | 0.300 → 0.400 |
| 70% | 0.743 → 0.705 | 0.771 → **0.829** | 0.300 → 0.400 |
| 80% | 0.743 → 0.729 | 0.771 → 0.771 | 0.300 → 0.300 |
| 90% | 0.743 → 0.743 | 0.771 → 0.771 | 0.300 → 0.300 |
| 100% | 0.743 → 0.743 | 0.771 → 0.771 | 0.300 → 0.300 |

### SVD as a Denoiser

**SVD compression can _improve_ the Mandela effect signal** — confirmed across two model families:

| Model | Baseline Mandela rho | Best compressed rho | Optimal ratio |
|-------|:-------------------:|:-------------------:|:-------------:|
| Qwen2.5-7B-Instruct | 0.829 | **0.943** (+0.114) | 70% |
| Mistral-7B-v0.1 | 0.771 | **0.829** (+0.057) | 70% |
| Qwen2.5-0.5B | 0.257 | **0.771** (+0.514) | 60% |

The denoising effect is consistent: at 70% rank, truncated SVD strips noise from attention projections while preserving the principal signal directions that encode factual knowledge. The `--denoise` flag auto-discovers this optimal ratio.

### Fidelity-Bench Baseline Comparison

| Category | Qwen-0.5B | Qwen-7B | Mistral-7B |
|----------|:---------:|:-------:|:----------:|
| default (20) | 0.821, 80% | 0.746, — | 0.743, 85% |
| mandela (6) | 0.257, 50% | 0.829, — | 0.771, 67% |
| medical (5) | 0.100, 80% | —, — | 0.300, 80% |
| commonsense (10) | 0.261, 70% | —, — | 0.503, 40% |
| truthfulqa (15) | 0.596, 40% | —, — | 0.586, 47% |

### Scale-Dependent Findings

| Finding | 0.5B | 7B (Qwen) | 7B (Mistral) |
|---------|:----:|:---------:|:------------:|
| Mandela baseline rho | 0.257 (weak) | **0.829** (strong) | 0.771 (strong) |
| CF90 rho drop | 0.101 (moderate) | **0.021** (minimal) | 0.038 (small) |
| CF90 retention | 95% | **100%** | 95% |
| SVD denoising on Mandela | +0.514 rho | **+0.114 rho** | +0.057 rho |

The Mandela effect signal strengthens with scale, and CF90 compression generalizes across Qwen and Mistral architectures with 95-100% retention.

### Prior Results (from Component Projects)

These findings come from the standalone [intelligent-svd](https://github.com/SolomonB14D3/intelligent-svd) and [confidence-cartography](https://github.com/SolomonB14D3/confidence-cartography) projects that this toolkit unifies:

| Finding | Result |
|---------|--------|
| Confidence correlates with human false-belief prevalence | rho=0.652, p=0.016 (Pythia 160M–12B) |
| Out-of-domain medical claims | 88% accuracy at 6.9B |
| Targeted resampling at low-confidence tokens | Outperforms uniform best-of-N |
| CF90 + INT8 stacking | 72–77% retention (Qwen-0.5B, Llama-7B) |
| Importance-guided SVD at 50% rank | 3× better retention than standard SVD |

### Compression Safety Guide

| Layer Type | Safe to Compress | Notes |
|------------|------------------|-------|
| **Q, K, O projections** | Yes at 70% rank | Main target |
| **V projection** | 90–95% only | Marginal gains, high risk below 90% |
| **MLP layers** | **Never** | Destroys model at any compression level |

## Install

```bash
pip install knowledge-fidelity                    # Core (SVD + probes)
pip install "knowledge-fidelity[cartography]"     # + confidence analysis + plots
pip install "knowledge-fidelity[demo]"            # + Gradio demo app
pip install "knowledge-fidelity[full]"            # Everything including MLX
```

Or from source:

```bash
git clone https://github.com/SolomonB14D3/knowledge-fidelity
cd knowledge-fidelity
pip install -e ".[full]"
```

## Quick Start

### One-Call Compress + Audit

```python
from knowledge_fidelity import compress_and_audit

report = compress_and_audit(
    "Qwen/Qwen2.5-7B-Instruct",
    ratio=0.7,           # Keep 70% of singular values
    freeze_ratio=0.75,   # Freeze bottom 75% of layers
)

print(report["summary"])
# Compressed Qwen/Qwen2.5-7B-Instruct at 70% rank | 84 matrices | 21/28 frozen | Retention: 100% | rho: 0.746 -> 0.725
```

### Step-by-Step (More Control)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from knowledge_fidelity.svd import compress_qko, freeze_layers
from knowledge_fidelity import audit_model

# Load
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Compress
compress_qko(model, ratio=0.7)     # SVD on Q, K, O projections
freeze_layers(model, ratio=0.75)   # Freeze bottom 75%

# Audit
audit = audit_model(model, tokenizer)
print(f"rho={audit['rho']:.3f}, {audit['n_positive_delta']}/{audit['n_probes']} probes positive")

# Fine-tune gently: 1 epoch, lr=1e-5
```

### Importance-Guided Compression (for Aggressive Ratios)

When compressing below 70%, standard SVD loses facts. The importance-guided variant uses gradient information to decide which singular values to keep:

```python
from knowledge_fidelity.svd import compress_qko_importance, compute_importance

importance = compute_importance(model, tokenizer)  # Uses shared probes
compress_qko_importance(model, importance, ratio=0.5)  # 3x better at 50%
```

### Confidence Analysis Only

```python
from knowledge_fidelity.cartography import analyze_confidence

# Teacher-forced: how confident is the model on each token?
record = analyze_confidence(
    "The capital of France is Paris.",
    model_name="EleutherAI/pythia-1.4b",
)
print(f"Mean confidence: {record.mean_top1_prob:.3f}")
print(f"Min confidence at: '{record.min_confidence_token}' "
      f"(prob={record.min_confidence_value:.3f})")
```

### Custom Probes

```python
from knowledge_fidelity import compress_and_audit, load_probes

# Use domain-specific probes
medical_probes = load_probes("data/probes/medical_claims.json")
report = compress_and_audit("my-model", probes=medical_probes)

# Or inline
custom = [
    {"text": "TCP uses a three-way handshake.",
     "false": "TCP uses a two-way handshake.",
     "domain": "networking", "id": "tcp_handshake"},
]
report = compress_and_audit("my-model", probes=custom)
```

## Built-In Probe Sets

| Set | Count | Purpose |
|-----|-------|---------|
| `get_default_probes()` | 20 | Geography, science, history, biology |
| `get_mandela_probes()` | 6 | Popular false memories (Berenstain Bears, Vader quote, etc.) |
| `get_medical_probes()` | 5 | Common medical misconceptions |
| `get_commonsense_probes()` | 10 | Commonsense myths (goldfish memory, sugar hyperactivity, etc.) |
| `get_truthfulqa_probes()` | 15 | TruthfulQA-derived misconceptions (evolution, Viking helmets, etc.) |
| `get_all_probes()` | 56 | All of the above |

Community contributions welcome — add probes for your domain and submit a PR.

## Denoise Mode (v0.2)

SVD compression can _improve_ factual discrimination by stripping noise from attention projections. The `--denoise` flag auto-finds the compression ratio that maximizes this effect:

```bash
knowledge-fidelity Qwen/Qwen2.5-0.5B --denoise
```

```
Baseline: rho=0.257
Testing ratio 0.50: rho=0.714 (IMPROVED by +0.457)
Testing ratio 0.60: rho=0.771 (IMPROVED by +0.514)  ← optimal
Testing ratio 0.70: rho=0.771 (IMPROVED by +0.514)
Testing ratio 0.80: rho=0.257 (no change)
Testing ratio 0.90: rho=0.371 (IMPROVED by +0.114)

DENOISING DETECTED: Mandela rho 0.257 → 0.771 (+0.514) at 60% ratio
```

Or from Python:

```python
from knowledge_fidelity import find_optimal_denoise_ratio

result = find_optimal_denoise_ratio("Qwen/Qwen2.5-0.5B", probe_set="mandela")
print(f"Optimal ratio: {result['optimal_ratio']}")
print(f"Improvement: {result['improvement']:+.3f}")
```

## Fidelity-Bench (v0.2)

Benchmark any model across all 56 probes organized by category:

```bash
python experiments/fidelity_bench.py --model Qwen/Qwen2.5-0.5B
```

```
Fidelity-Bench: Qwen/Qwen2.5-0.5B
| Category    | Probes | rho   | Correct | Accuracy | Mean Δ  |
|-------------|--------|-------|---------|----------|---------|
| default     |     20 | 0.821 |  16/20  |     80%  | +0.0837 |
| mandela     |      6 | 0.257 |   3/6   |     50%  | +0.0527 |
| medical     |      5 | 0.100 |   4/5   |     80%  | +0.0466 |
| commonsense |     10 | 0.261 |   7/10  |     70%  | -0.0226 |
| truthfulqa  |     15 | 0.596 |   6/15  |     40%  | -0.0392 |
```

Add `--json` for machine-readable output. Use `--output results.json` to save.

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

## CLI

```bash
# Compress + audit (default: 70% rank, CF90 protection)
knowledge-fidelity Qwen/Qwen2.5-0.5B

# Audit only (no compression, baseline measurement)
knowledge-fidelity Qwen/Qwen2.5-0.5B --audit-only

# Auto-find optimal denoising ratio
knowledge-fidelity Qwen/Qwen2.5-0.5B --denoise

# Denoise with specific probe set
knowledge-fidelity Qwen/Qwen2.5-0.5B --denoise --denoise-probe-set medical

# Use all 56 probes
knowledge-fidelity Qwen/Qwen2.5-0.5B --audit-only --probes all

# Save compressed model
knowledge-fidelity Qwen/Qwen2.5-0.5B --denoise --output ./denoised-model
```

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
```

## Deployment

```bash
# Export to GGUF for llama.cpp / Ollama
python deployment/export_gguf.py --input compressed_model/ --output model.gguf --quantize q4_k_m

# Benchmark with vLLM
python deployment/vllm_benchmark.py --baseline Qwen/Qwen2.5-7B-Instruct --compressed ./compressed_model
```

See [`deployment/mlx_recipe.md`](deployment/mlx_recipe.md) for Apple Silicon inference with MLX.

## Platform Notes (Apple Silicon)

- Use **CPU** for compression and fine-tuning (MPS has matmul errors with some architectures and NaN gradients with frozen layers)
- Use **MLX** for fast inference after compression
- Set `HF_HOME` to external storage for large models

## Model Compatibility

Works on any HuggingFace causal LM with `model.model.layers[i].self_attn.{q,k,o}_proj` (standard for Qwen, Llama, Mistral) or `model.transformer.h` (GPT-2 style).

Validated on:
- **Qwen2.5**: 0.5B, 1.5B, 7B, 32B
- **Mistral**: 7B-v0.1
- **Llama 2**: 7B
- Should work on Phi, Gemma (same layer layout) — PRs with test results welcome

## Built On

This toolkit unifies two standalone research projects:

- [**Intelligent SVD**](https://github.com/SolomonB14D3/intelligent-svd) — CF90 compression method and safety rules
- [**Confidence Cartography**](https://github.com/SolomonB14D3/confidence-cartography) — False-belief detection via teacher-forced confidence

Both remain available as independent repos. Knowledge Fidelity combines their core ideas into a single pipeline with a shared probe system.

## Citation

```bibtex
@software{knowledge_fidelity,
  author = {Bryan Sanchez},
  title = {Knowledge Fidelity: Compress LLMs While Auditing What They Still Know},
  year = {2026},
  url = {https://github.com/SolomonB14D3/knowledge-fidelity}
}
```

## License

MIT
