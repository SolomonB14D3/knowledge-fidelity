#!/usr/bin/env python3
"""
Benchmark compressed model inference speed with vLLM.

Compares baseline vs CF90-compressed model on throughput and latency.

Requires: pip install vllm

Usage:
    python deployment/vllm_benchmark.py --baseline Qwen/Qwen2.5-7B --compressed ./compressed_model
"""

import sys
import time
import argparse
from pathlib import Path


def benchmark_vllm(model_path: str, prompts: list[str], max_tokens: int = 128):
    """Run vLLM benchmark on a model.

    Returns dict with throughput and latency stats.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed. Install with: pip install vllm")
        print("Alternatively, use the model with transformers directly.")
        return None

    print(f"Loading {model_path} with vLLM...")
    llm = LLM(model=model_path, trust_remote_code=True)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    # Warmup
    llm.generate(prompts[:1], params)

    # Benchmark
    start = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed

    return {
        "model": model_path,
        "n_prompts": len(prompts),
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "throughput_tok_s": throughput,
        "avg_latency_s": elapsed / len(prompts),
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM Benchmark")
    parser.add_argument("--baseline", required=True, help="Baseline model path")
    parser.add_argument("--compressed", required=True, help="Compressed model path")
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    prompts = [
        "Explain the theory of relativity in simple terms:",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis:",
        "What is machine learning and how does it work?",
        "Explain quantum computing to a beginner:",
    ] * (args.n_prompts // 5 + 1)
    prompts = prompts[:args.n_prompts]

    print("=" * 60)
    print("vLLM Inference Benchmark: Baseline vs CF90")
    print("=" * 60)

    baseline = benchmark_vllm(args.baseline, prompts, args.max_tokens)
    compressed = benchmark_vllm(args.compressed, prompts, args.max_tokens)

    if baseline and compressed:
        speedup = compressed["throughput_tok_s"] / baseline["throughput_tok_s"]
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Baseline:   {baseline['throughput_tok_s']:.1f} tok/s")
        print(f"  Compressed: {compressed['throughput_tok_s']:.1f} tok/s")
        print(f"  Speedup:    {speedup:.2f}x")


if __name__ == "__main__":
    main()
