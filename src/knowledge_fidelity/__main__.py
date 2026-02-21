"""CLI entry point: python -m knowledge_fidelity [model] [options]

Examples:
    python -m knowledge_fidelity Qwen/Qwen2.5-0.5B
    python -m knowledge_fidelity Qwen/Qwen2.5-7B-Instruct --ratio 0.5 --probes mandela
    python -m knowledge_fidelity my-model --importance --output ./compressed
    python -m knowledge_fidelity Qwen/Qwen2.5-0.5B --audit-only
    python -m knowledge_fidelity Qwen/Qwen2.5-7B-Instruct --denoise
    python -m knowledge_fidelity --version
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="knowledge_fidelity",
        description="Compress an LLM while auditing what it still knows.",
    )
    parser.add_argument(
        "model", nargs="?", default=None,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Print version and exit",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.7,
        help="SVD compression ratio (default: 0.7 = keep 70%%)",
    )
    parser.add_argument(
        "--freeze-ratio", type=float, default=0.75,
        help="Fraction of layers to freeze (default: 0.75)",
    )
    parser.add_argument(
        "--importance", action="store_true",
        help="Use gradient-based importance scoring (slower, better below 70%%)",
    )
    parser.add_argument(
        "--probes",
        choices=["default", "mandela", "medical", "commonsense", "truthfulqa", "all"],
        default="default",
        help="Which probe set to use (default: default)",
    )
    parser.add_argument(
        "--probes-file", type=str, default=None,
        help="Path to custom probes JSON file",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Directory to save compressed model",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Only audit (no compression). Useful for baselining a model.",
    )
    parser.add_argument(
        "--denoise", action="store_true",
        help="Auto-find the compression ratio that maximizes factual signal.",
    )
    parser.add_argument(
        "--denoise-probe-set",
        choices=["mandela", "medical", "default"],
        default="mandela",
        help="Which probe set to optimize for denoising (default: mandela)",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Output results as JSON (for scripting)",
    )

    args = parser.parse_args()

    # --version
    if args.version:
        from . import __version__
        print(f"knowledge-fidelity {__version__}")
        return

    # Model is required for everything else
    if args.model is None:
        parser.error("the following arguments are required: model")

    # Import here so --help and --version are fast
    from .probes import (
        get_default_probes, get_mandela_probes,
        get_medical_probes, get_all_probes, load_probes,
    )

    # Select probes
    if args.probes_file:
        probes = load_probes(args.probes_file)
        print(f"Loaded {len(probes)} probes from {args.probes_file}")
    else:
        probe_map = {
            "default": get_default_probes,
            "mandela": get_mandela_probes,
            "medical": get_medical_probes,
            "all": get_all_probes,
        }
        # Add new probe sets if available
        try:
            from .probes import get_commonsense_probes, get_truthfulqa_probes
            probe_map["commonsense"] = get_commonsense_probes
            probe_map["truthfulqa"] = get_truthfulqa_probes
        except ImportError:
            pass

        if args.probes in probe_map:
            probes = probe_map[args.probes]()
        else:
            parser.error(f"Unknown probe set: {args.probes}")
            return

    # --- Denoise mode ---
    if args.denoise:
        from .denoise import find_optimal_denoise_ratio

        result = find_optimal_denoise_ratio(
            args.model,
            probe_set=args.denoise_probe_set,
            device=args.device,
        )

        if args.json_output:
            out = {
                "model": args.model,
                "mode": "denoise",
                "probe_set": result["probe_set"],
                "optimal_ratio": result["optimal_ratio"],
                "optimal_rho": result["optimal_rho"],
                "baseline_rho": result["baseline_rho"],
                "improvement": result["improvement"],
                "denoising_detected": result["denoising_detected"],
                "all_results": result["all_results"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
            print(json.dumps(out, indent=2))
        else:
            print(f"\n{'='*60}")
            if result["denoising_detected"]:
                print(f"  DENOISING DETECTED on {result['probe_set']} probes!")
                print(f"  Optimal ratio: {result['optimal_ratio']:.0%}")
                print(f"  rho: {result['baseline_rho']:.3f} -> "
                      f"{result['optimal_rho']:.3f} "
                      f"(+{result['improvement']:.3f})")
            else:
                print(f"  No denoising found on {result['probe_set']} probes.")
                print(f"  Best ratio: {result['optimal_ratio']:.0%} "
                      f"(rho={result['optimal_rho']:.3f})")
            print(f"  Elapsed: {result['elapsed_seconds']:.1f}s")
            print(f"{'='*60}")

            # Show all ratios
            print(f"\n{'Ratio':>6} {'rho':>8} {'Improved':>10}")
            print("-" * 28)
            for r in result["all_results"]:
                marker = " **" if r["improved_over_baseline"] else ""
                print(f"{r['ratio']:>6.0%} {r['rho']:>8.3f} "
                      f"{'Yes' if r['improved_over_baseline'] else 'No':>10}{marker}")

        # Save compressed model if --output
        if args.output:
            from pathlib import Path
            out_path = Path(args.output)
            out_path.mkdir(parents=True, exist_ok=True)
            result["model"].save_pretrained(out_path)
            result["tokenizer"].save_pretrained(out_path)
            print(f"\nSaved denoised model to {out_path}")

        return

    # --- Audit-only mode ---
    if args.audit_only:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .core import audit_model

        print(f"Loading {args.model} for audit-only...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, trust_remote_code=True
        ).to(args.device)
        model.eval()

        audit = audit_model(model, tokenizer, probes=probes, device=args.device)

        if args.json_output:
            out = {
                "model": args.model,
                "mode": "audit_only",
                "probe_set": args.probes,
                "rho": audit["rho"],
                "rho_p": audit["rho_p"],
                "mean_delta": audit["mean_delta"],
                "n_positive": audit["n_positive_delta"],
                "n_probes": audit["n_probes"],
            }
            print(json.dumps(out, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"Audit: {args.model}")
            print(f"  Probes: {args.probes} ({len(probes)})")
            print(f"  Spearman rho: {audit['rho']:.3f} (p={audit['rho_p']:.4f})")
            print(f"  Mean delta:   {audit['mean_delta']:.4f}")
            print(f"  Positive:     {audit['n_positive_delta']}/{audit['n_probes']}")
            print(f"{'='*60}")
        return

    # --- Full compress + audit ---
    from .core import compress_and_audit

    report = compress_and_audit(
        args.model,
        ratio=args.ratio,
        freeze_ratio=args.freeze_ratio,
        use_importance=args.importance,
        probes=probes,
        output_dir=args.output,
        device=args.device,
    )

    if args.json_output:
        out = {
            "model": args.model,
            "ratio": args.ratio,
            "freeze_ratio": args.freeze_ratio,
            "retention": report["retention"],
            "rho_before": report["rho_before"],
            "rho_after": report["rho_after"],
            "rho_drop": report["rho_before"] - report["rho_after"],
            "n_compressed": report["compression"]["n_compressed"],
            "n_frozen": report["freeze"]["n_frozen"],
            "n_layers": report["freeze"]["n_layers"],
            "elapsed_seconds": report["elapsed_seconds"],
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"\n{'='*60}")
        print(report["summary"])
        print(f"{'='*60}")

        # Per-probe breakdown
        print(f"\n{'Probe':<25} {'Before':>8} {'After':>8} {'Status':>8}")
        print("-" * 52)
        for i, p in enumerate(probes):
            db = report["audit_before"]["deltas"][i]
            da = report["audit_after"]["deltas"][i]
            status = "OK" if da > 0 else ("FLIP" if db > 0 else "was neg")
            print(f"{p.get('id', f'probe_{i}'):<25} {db:>+8.4f} {da:>+8.4f} {status:>8}")


if __name__ == "__main__":
    main()
