# Knowledge Fidelity

**Compress an LLM while auditing whether it still knows truth vs popular myths.**

The first toolkit that uses the same factual probes for both structural importance scoring (SVD compression) and behavioral false-belief detection (confidence cartography). One call to compress and audit:

```python
from knowledge_fidelity import compress_and_audit

report = compress_and_audit("meta-llama/Llama-3.1-8B-Instruct", ratio=0.7)
print(f"Retention: {report['retention']:.0%} | "
      f"False-belief signal: rho={report['rho_after']:.3f}")
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

## Key Results

### Cross-Architecture Validation

| Finding | Qwen (0.5B) | Llama 2 (7B) |
|---------|------------|-------------|
| CF90 fact retention | **79%** (p=0.0072, 5 seeds) | **78%** (3 seeds) |
| Unprotected forgetting | 4% retained | 7% retained |
| CF90 + INT8 | 72% retained | 77% retained |
| Repetition (CF90 vs baseline) | 33% vs 5% (bottleneck) | **25% vs 40%** (regularizer) |

### Confidence Cartography Findings

| Finding | Result |
|---------|--------|
| Confidence correlates with human false-belief prevalence | rho=0.652, p=0.016 (Pythia 160M-12B) |
| Out-of-domain medical claims | 88% accuracy at 6.9B |
| Mandela effects show lower confidence | Systematic across all model scales |
| Targeted resampling at low-confidence tokens | Outperforms uniform best-of-N |

### Compression Safety Guide

| Layer Type | Safe to Compress | Notes |
|------------|------------------|-------|
| **Q, K, O projections** | Yes at 70% rank | Main target |
| **V projection** | 90-95% only | Marginal gains, high risk below 90% |
| **MLP layers** | **Never** | Destroys model at any compression level |

## Install

```bash
pip install knowledge-fidelity                    # Core (SVD + probes)
pip install "knowledge-fidelity[cartography]"     # + confidence analysis + plots
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
    "Qwen/Qwen2.5-7B",
    ratio=0.7,           # Keep 70% of singular values
    freeze_ratio=0.75,   # Freeze bottom 75% of layers
)

print(report["summary"])
# Compressed Qwen/Qwen2.5-7B at 70% rank | 84 matrices | 21/28 frozen | Retention: 95% | rho: 0.812 -> 0.793
```

### Step-by-Step (More Control)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from knowledge_fidelity.svd import compress_qko, freeze_layers
from knowledge_fidelity import audit_model

# Load
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

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
| `get_all_probes()` | 31 | All of the above |

Community contributions welcome — add probes for your domain and submit a PR.

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
# Quick demo (~5 min on Qwen-0.5B)
python examples/quick_demo.py

# Joint ablation: compression ratio vs confidence preservation
python experiments/joint_ablation.py --model Qwen/Qwen2.5-7B

# Multi-seed CF90 validation
python experiments/run_cf90_multiseed.py --model meta-llama/Llama-3.1-8B-Instruct --seeds 5
```

## Deployment

```bash
# Export to GGUF for llama.cpp / Ollama
python deployment/export_gguf.py --input compressed_model/ --output model.gguf --quantize q4_k_m

# Benchmark with vLLM
python deployment/vllm_benchmark.py --baseline Qwen/Qwen2.5-7B --compressed ./compressed_model
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
- **Llama 2**: 7B
- Should work on Mistral, Phi, Gemma (same layer layout) — PRs with test results welcome

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
