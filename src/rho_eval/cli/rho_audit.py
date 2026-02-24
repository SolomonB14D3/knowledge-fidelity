#!/usr/bin/env python3
"""rho-audit / rho-eval: Behavioral audit for LLMs via teacher-forced confidence.

Evaluate any HuggingFace model across 5 behavioral dimensions without
compression — just load, probe, and report.

Usage:
    rho-audit Qwen/Qwen2.5-7B-Instruct
    rho-audit Qwen/Qwen2.5-7B-Instruct --behaviors all
    rho-audit Qwen/Qwen2.5-7B-Instruct --behaviors factual,bias,sycophancy
    rho-audit model/ --format json --output audit.json
    rho-audit new-model/ --compare baseline.json
    rho-audit --list-behaviors
    rho-audit --list-probes
"""

import argparse
import json
import sys
import time
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


def _load_model(model_id, device, dtype_str="float32"):
    """Load model + tokenizer with progress reporting."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_str, torch.float32)
    print(f"\033[1mLoading {model_id}...\033[0m", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    print(f"  Loaded in {time.time() - t0:.1f}s ({n_params / 1e9:.2f}B params)", flush=True)
    return model, tokenizer


def _status_color(status: str) -> str:
    """Return ANSI-colored status string."""
    colors = {"PASS": "\033[92m", "WARN": "\033[93m", "FAIL": "\033[91m"}
    reset = "\033[0m"
    c = colors.get(status, "")
    return f"{c}{status}{reset}"


def _run_audit(args):
    """Run the behavioral audit."""
    from rho_eval.audit import audit
    from rho_eval.output.exporters import to_json, to_markdown, to_csv, to_table
    from rho_eval.output import AuditReport, compare

    # Resolve device
    if args.device is None:
        from rho_eval.utils import get_device
        device = str(get_device())
    else:
        device = args.device

    # Parse behaviors
    if args.behaviors == "all":
        from rho_eval.behaviors import list_behaviors
        behavior_names = list_behaviors()
    else:
        behavior_names = [b.strip() for b in args.behaviors.split(",")]

    # Load model
    model, tokenizer = _load_model(args.model, device)

    # Run audit using the new API
    print(f"\n\033[1mRunning behavioral audit ({len(behavior_names)} behaviors)...\033[0m\n", flush=True)

    report = audit(
        model=model,
        tokenizer=tokenizer,
        behaviors=behavior_names,
        n=args.n_probes,
        seed=args.seed,
        device=device,
    )
    report.model = args.model  # Use the original model ID string

    # Print per-behavior progress
    for name, result in sorted(report.behaviors.items()):
        status = _status_color(result.status)
        print(
            f"  {name:<12s}  "
            f"ρ={result.rho:+.4f}  "
            f"{result.positive_count:>3d}/{result.total:<4d}  "
            f"[{status}]  "
            f"{result.elapsed:.1f}s",
            flush=True,
        )

    print()

    # Comparison
    if args.compare:
        baseline = AuditReport.load(args.compare)
        delta = compare(report, baseline)
        if args.format == "json":
            print(json.dumps(delta.to_dict(), indent=2, default=_json_serializer))
        elif args.format == "markdown":
            print(delta.to_markdown())
        else:
            print(delta.to_table(color=not args.no_color))
    else:
        # Output in requested format
        if args.format == "json":
            print(to_json(report, include_details=not args.no_details))
        elif args.format == "markdown":
            print(to_markdown(report))
        elif args.format == "csv":
            print(to_csv(report))
        else:  # table (default)
            print(to_table(report, color=not args.no_color))

    # Save
    if args.output:
        out_path = report.save(args.output, include_details=not args.no_details)
        print(f"\nSaved: {out_path}")

    # Exit code: 0 if all PASS, 1 if any FAIL
    if report.overall_status == "FAIL":
        sys.exit(1)


def _list_behaviors():
    """Print all registered behaviors with details."""
    from rho_eval.behaviors import get_all_behaviors

    print("\n\033[1mRegistered Behaviors\033[0m\n")
    print(f"  {'Name':<12s}  {'Type':<12s}  {'Default N':>9s}  Description")
    print(f"  {'─' * 12}  {'─' * 12}  {'─' * 9}  {'─' * 40}")

    for name, behavior in get_all_behaviors().items():
        print(
            f"  {name:<12s}  {behavior.probe_type:<12s}  "
            f"{behavior.default_n:>9d}  {behavior.description}"
        )
    print()


def _list_probes():
    """Print all available probe sets with counts."""
    from rho_eval.probes import get_probe_counts

    print("\n\033[1mAvailable Probe Sets\033[0m\n")
    print(f"  {'Probe Set':<30s}  {'Count':>6s}")
    print(f"  {'─' * 30}  {'─' * 6}")

    counts = get_probe_counts()
    total = 0
    for name, count in sorted(counts.items()):
        print(f"  {name:<30s}  {count:>6d}")
        total += count

    print(f"  {'─' * 30}  {'─' * 6}")
    print(f"  {'TOTAL':<30s}  {total:>6d}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="rho-audit",
        description="Behavioral audit for LLMs via teacher-forced confidence (ρ).",
        epilog=(
            "Examples:\n"
            "  rho-audit Qwen/Qwen2.5-7B-Instruct\n"
            "  rho-audit Qwen/Qwen2.5-7B-Instruct --behaviors all\n"
            "  rho-audit my-model/ --behaviors factual,bias --format json\n"
            "  rho-audit new-model/ --compare baseline.json\n"
            "  rho-audit --list-behaviors\n"
            "  rho-audit --list-probes\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional
    parser.add_argument(
        "model", nargs="?", default=None,
        help="HuggingFace model name or local path",
    )

    # Behavior selection
    parser.add_argument(
        "--behaviors", type=str, default="all",
        help="Comma-separated behaviors: factual,toxicity,bias,sycophancy,reasoning,all "
             "(default: all)",
    )
    parser.add_argument(
        "-n", "--n-probes", type=int, default=None,
        help="Number of probes per behavior (default: behavior-specific)",
    )

    # Output format
    parser.add_argument(
        "--format", choices=["table", "json", "markdown", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save results JSON to file",
    )
    parser.add_argument(
        "--no-details", action="store_true",
        help="Omit per-probe details in JSON output (smaller file)",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI color codes in output",
    )

    # Comparison
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Compare against a previous audit JSON (shows Δρ)",
    )

    # Device & seed
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cpu, cuda, mps (default: auto-detect)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for probe sampling (default: 42)",
    )

    # Info commands
    parser.add_argument(
        "--list-behaviors", action="store_true",
        help="List all registered behaviors and exit",
    )
    parser.add_argument(
        "--list-probes", action="store_true",
        help="List all available probe sets and exit",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Print version and exit",
    )

    args = parser.parse_args()

    # Info commands (no model needed)
    if args.version:
        from rho_eval import __version__
        print(f"rho-eval {__version__}")
        return

    if args.list_behaviors:
        _list_behaviors()
        return

    if args.list_probes:
        _list_probes()
        return

    # Model required for audit
    if args.model is None:
        parser.error("the following arguments are required: model")

    _run_audit(args)


if __name__ == "__main__":
    main()
