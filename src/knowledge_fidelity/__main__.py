"""CLI entry point: python -m knowledge_fidelity [model] [options]

Examples:
    python -m knowledge_fidelity Qwen/Qwen2.5-0.5B
    python -m knowledge_fidelity Qwen/Qwen2.5-7B-Instruct --ratio 0.5 --probes mandela
    python -m knowledge_fidelity my-model --importance --output ./compressed
    python -m knowledge_fidelity Qwen/Qwen2.5-0.5B --audit-only
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
        "model",
        help="HuggingFace model name or local path",
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
        "--probes", choices=["default", "mandela", "medical", "all"],
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
        "--json", dest="json_output", action="store_true",
        help="Output results as JSON (for scripting)",
    )

    args = parser.parse_args()

    # Import here so --help is fast
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
        probes = probe_map[args.probes]()

    if args.audit_only:
        # Audit without compression
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
    else:
        # Full compress + audit
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
