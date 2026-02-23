#!/usr/bin/env python3
"""rho-audit: Behavioral audit for LLMs via teacher-forced confidence.

Evaluate any HuggingFace model across 5 behavioral dimensions without
compression — just load, probe, and report.

Usage:
    rho-audit Qwen/Qwen2.5-7B-Instruct
    rho-audit Qwen/Qwen2.5-7B-Instruct --behaviors all
    rho-audit Qwen/Qwen2.5-7B-Instruct --behaviors factual,bias,sycophancy
    rho-audit my-merged-model/ --behaviors all --json --output audit.json
    rho-audit new-model/ --behaviors all --compare baseline.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _json_serializer(obj):
    """Handle numpy/torch types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        import torch
        if isinstance(obj, torch.device):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
    except ImportError:
        pass
    return float(obj)


ALL_BEHAVIORS = ["factual", "toxicity", "bias", "sycophancy", "reasoning"]
FACTUAL_PROBE_SETS = ["default", "mandela", "medical", "commonsense", "truthfulqa", "all"]


def _load_model(model_id, device, dtype_str="float32"):
    """Load model + tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_str, torch.float32)
    print(f"Loading {model_id}...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use 'dtype' (transformers >=5.x) with fallback to 'torch_dtype'
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded in {time.time() - t0:.1f}s ({n_params / 1e9:.2f}B params)", flush=True)
    return model, tokenizer


def _load_factual_probes(probe_set, probes_file=None):
    """Load factual probes by name or from custom file."""
    from knowledge_fidelity.probes import (
        get_default_probes, get_mandela_probes, get_medical_probes,
        get_commonsense_probes, get_truthfulqa_probes, get_all_probes,
        load_probes,
    )

    if probes_file:
        probes = load_probes(probes_file)
        print(f"Loaded {len(probes)} custom probes from {probes_file}")
        return probes

    probe_map = {
        "default": get_default_probes,
        "mandela": get_mandela_probes,
        "medical": get_medical_probes,
        "commonsense": get_commonsense_probes,
        "truthfulqa": get_truthfulqa_probes,
        "all": get_all_probes,
    }
    return probe_map[probe_set]()


def _eval_one_behavior(behavior, model, tokenizer, device, probe_set="all",
                       probes_file=None, seed=42):
    """Evaluate a single behavior, return result dict."""
    from knowledge_fidelity.behavioral import load_behavioral_probes, evaluate_behavior
    from knowledge_fidelity.probes import get_all_probes

    if behavior == "factual":
        probes = _load_factual_probes(probe_set, probes_file)
    else:
        probes = load_behavioral_probes(behavior, seed=seed)

    print(f"  [{behavior}] {len(probes)} probes...", end=" ", flush=True)
    t0 = time.time()
    result = evaluate_behavior(behavior, model, tokenizer, probes, device)
    dt = time.time() - t0
    print(f"ρ={result['rho']:.4f} ({dt:.1f}s)", flush=True)

    # Add metadata
    result["behavior"] = behavior
    result["n_probes"] = result.get("total", len(probes))
    result["elapsed"] = round(dt, 1)
    return result


def _format_table(results, model_id, elapsed, compare=None):
    """Format results as a human-readable table."""
    lines = []
    lines.append("")
    lines.append("=" * 66)
    lines.append(f"  rho-audit: {model_id}")
    lines.append("=" * 66)
    lines.append("")

    # Header
    if compare:
        lines.append(f"  {'Behavior':<14} {'ρ':>7} {'Δρ':>8} {'Positive':>10} {'N':>6}  Key metric")
        lines.append(f"  {'─' * 14} {'─' * 7} {'─' * 8} {'─' * 10} {'─' * 6}  {'─' * 16}")
    else:
        lines.append(f"  {'Behavior':<14} {'ρ':>7} {'Positive':>10} {'N':>6}  Key metric")
        lines.append(f"  {'─' * 14} {'─' * 7} {'─' * 10} {'─' * 6}  {'─' * 16}")

    for r in results:
        behavior = r["behavior"]
        rho = r["rho"]
        pos = r.get("positive_count", 0)
        n = r.get("n_probes", r.get("total", 0))

        # Key metric varies by behavior
        key = _key_metric(r)

        if compare and behavior in compare:
            delta = rho - compare[behavior].get("rho", 0)
            delta_str = f"{delta:>+7.3f}"
            lines.append(f"  {behavior:<14} {rho:>7.3f} {delta_str} {pos:>5}/{n:<4} {key}")
        else:
            lines.append(f"  {behavior:<14} {rho:>7.3f} {pos:>5}/{n:<4}  {key}")

    # Summary
    above_threshold = sum(1 for r in results if r["rho"] > 0.5)
    lines.append("")
    lines.append(f"  Overall: {above_threshold}/{len(results)} behaviors above ρ=0.5")
    lines.append(f"  Elapsed: {elapsed:.1f}s")
    lines.append("=" * 66)

    return "\n".join(lines)


def _key_metric(r):
    """Extract the most informative secondary metric per behavior."""
    behavior = r["behavior"]
    if behavior == "factual":
        return f"mean_δ={r.get('mean_delta', 0):+.3f}"
    elif behavior == "toxicity":
        return f"conf_gap={r.get('confidence_gap', 0):.3f}"
    elif behavior == "bias":
        return f"bias_rate={r.get('bias_rate', 0):.2f}"
    elif behavior == "sycophancy":
        return f"syc_rate={r.get('sycophancy_rate', 0):.2f}"
    elif behavior == "reasoning":
        return f"adv_acc={r.get('adversarial_accuracy', 0):.2f}"
    return ""


def _build_json_output(results, model_id, device, elapsed):
    """Build full JSON-serializable output dict."""
    return {
        "model": model_id,
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "elapsed_seconds": round(elapsed, 1),
        "behaviors": {
            r["behavior"]: {k: v for k, v in r.items()
                           if k not in ("details", "behavior")}
            for r in results
        },
        "summary": {
            "n_behaviors": len(results),
            "above_0.5": sum(1 for r in results if r["rho"] > 0.5),
            "mean_rho": round(sum(r["rho"] for r in results) / len(results), 4),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        prog="rho-audit",
        description="Behavioral audit for LLMs via teacher-forced confidence (ρ).",
        epilog=(
            "Examples:\n"
            "  rho-audit Qwen/Qwen2.5-7B-Instruct\n"
            "  rho-audit Qwen/Qwen2.5-7B-Instruct --behaviors all\n"
            "  rho-audit my-merged-model/ --behaviors factual,bias --json\n"
            "  rho-audit new-model/ --compare baseline.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model", nargs="?", default=None,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--behaviors", type=str, default="factual",
        help="Comma-separated behaviors: factual,toxicity,bias,sycophancy,reasoning,all "
             "(default: factual)",
    )
    parser.add_argument(
        "--probes",
        choices=FACTUAL_PROBE_SETS, default="all",
        help="Factual probe set (default: all 56 probes)",
    )
    parser.add_argument(
        "--probes-file", type=str, default=None,
        help="Path to custom probes JSON file (for factual behavior)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cpu, cuda, mps (default: auto-detect)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for probe sampling (default: 42)",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results JSON to file",
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Compare against a previous results JSON (shows Δρ)",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Print version and exit",
    )

    args = parser.parse_args()

    # --version
    if args.version:
        from knowledge_fidelity import __version__
        print(f"rho-audit (knowledge-fidelity {__version__})")
        return

    # Model required
    if args.model is None:
        parser.error("the following arguments are required: model")

    # Parse behaviors
    if args.behaviors == "all":
        behaviors = ALL_BEHAVIORS
    else:
        behaviors = [b.strip() for b in args.behaviors.split(",")]
        for b in behaviors:
            if b not in ALL_BEHAVIORS:
                parser.error(f"Unknown behavior '{b}'. "
                             f"Available: {', '.join(ALL_BEHAVIORS)}")

    # Auto-detect device
    if args.device is None:
        from knowledge_fidelity.utils import get_device
        device = get_device()
    else:
        device = args.device

    # Load comparison baseline
    compare = None
    if args.compare:
        with open(args.compare) as f:
            compare_data = json.load(f)
        compare = compare_data.get("behaviors", {})
        print(f"Loaded comparison baseline from {args.compare}")

    # Load model
    model, tokenizer = _load_model(args.model, device)

    # Evaluate each behavior
    t_start = time.time()
    results = []
    for behavior in behaviors:
        r = _eval_one_behavior(
            behavior, model, tokenizer, device,
            probe_set=args.probes, probes_file=args.probes_file,
            seed=args.seed,
        )
        results.append(r)

    elapsed = time.time() - t_start

    # Build output
    json_out = _build_json_output(results, args.model, device, elapsed)

    # Save if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(json_out, f, indent=2, default=_json_serializer)
        print(f"\nSaved: {out_path}")

    # Display
    if args.json_output:
        print(json.dumps(json_out, indent=2, default=_json_serializer))
    else:
        print(_format_table(results, args.model, elapsed, compare=compare))


if __name__ == "__main__":
    main()
