"""CLI entry point for SAE-based behavioral steering.

Usage:
    rho-steer train  Qwen/Qwen2.5-0.5B --layer 17
    rho-steer analyze sae_layer17.pt --activations act_layer17.pt
    rho-steer eval   Qwen/Qwen2.5-0.5B --sae sae_layer17.pt --target sycophancy
"""

import argparse
import json
import sys
import time
from pathlib import Path


RESULTS_DIR = Path("results") / "steering"
DEFAULT_BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias"]
EVAL_BEHAVIORS = ["factual", "toxicity", "sycophancy", "bias", "reasoning"]


def cmd_train(args):
    """Train a Gated SAE on behavioral activations."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rho_eval.utils import get_device
    from rho_eval.steering import train_behavioral_sae, SAEConfig

    device = args.device or str(get_device())
    behaviors = [b.strip() for b in args.behaviors.split(",")]

    print(f"\n{'='*60}")
    print(f"  rho-steer train: Gated SAE Training")
    print(f"  Model:      {args.model}")
    print(f"  Layer:      {args.layer}")
    print(f"  Behaviors:  {behaviors}")
    print(f"  Expansion:  {args.expansion}")
    print(f"  Device:     {device}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    # Configure SAE
    from rho_eval.utils import get_layers
    n_layers = len(get_layers(model))
    hidden_dim = model.config.hidden_size

    config = SAEConfig(
        hidden_dim=hidden_dim,
        expansion_factor=args.expansion,
        sparsity_lambda=args.sparsity_lambda,
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        device=device,
    )

    print(f"\n  SAE config: {config.n_features} features "
          f"({hidden_dim} x {args.expansion})")

    # Train
    t0 = time.time()
    sae, act_data, stats = train_behavioral_sae(
        model, tokenizer, behaviors, args.layer,
        config=config, device=device,
        max_probes=args.max_probes,
        verbose=True,
    )
    elapsed = time.time() - t0

    # Save outputs
    out_dir = Path(args.output) if args.output else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    sae_path = out_dir / f"sae_layer{args.layer}.pt"
    act_path = out_dir / f"activations_layer{args.layer}.pt"
    meta_path = out_dir / f"sae_meta_layer{args.layer}.json"

    torch.save(sae.state_dict(), sae_path)
    act_data.save(act_path)

    meta = {
        "model": args.model,
        "layer_idx": args.layer,
        "config": config.to_dict(),
        "train_stats": {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in stats.items() if k != "sae"
        },
        "behaviors": behaviors,
        "n_samples": act_data.n_samples,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved:")
    print(f"    SAE weights: {sae_path}")
    print(f"    Activations: {act_path}")
    print(f"    Metadata:    {meta_path}")
    print(f"\n  Total time: {elapsed:.1f}s")


def cmd_analyze(args):
    """Analyze a trained SAE to identify behavioral features."""
    import torch

    from rho_eval.steering import (
        GatedSAE, ActivationData,
        identify_behavioral_features, feature_overlap_matrix,
        SAESteeringReport,
    )

    print(f"\n{'='*60}")
    print(f"  rho-steer analyze: Feature Identification")
    print(f"{'='*60}\n")

    # Load metadata
    meta_path = Path(args.sae).with_name(
        args.sae.replace("sae_", "sae_meta_").replace(".pt", ".json")
    ) if not args.meta else Path(args.meta)

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        hidden_dim = meta["config"]["hidden_dim"]
        expansion = meta["config"]["expansion_factor"]
        layer_idx = meta["layer_idx"]
        model_name = meta["model"]
        print(f"  Model: {model_name}, Layer: {layer_idx}")
    else:
        print("  WARNING: No metadata found, using defaults")
        hidden_dim = args.hidden_dim or 896
        expansion = args.expansion or 8
        layer_idx = args.layer or 17
        model_name = "unknown"

    # Load SAE
    print("  Loading SAE...")
    sae = GatedSAE(hidden_dim, expansion)
    sae.load_state_dict(torch.load(args.sae, weights_only=True))
    sae.eval()

    # Load activations
    act_path = args.activations or str(
        Path(args.sae).with_name(f"activations_layer{layer_idx}.pt")
    )
    print(f"  Loading activations from {act_path}...")
    act_data = ActivationData.load(act_path)

    # Identify features
    print(f"\n  Analyzing features (threshold={args.threshold})...")
    reports, behavioral_features = identify_behavioral_features(
        sae, act_data, threshold=args.threshold,
    )

    # Compute overlap
    behaviors = act_data.behaviors
    overlap = feature_overlap_matrix(reports, behaviors)

    # Print results
    print(f"\n  {'='*50}")
    print(f"  BEHAVIORAL FEATURE SUMMARY")
    print(f"  {'='*50}")

    for behavior in behaviors:
        feats = behavioral_features.get(behavior, [])
        print(f"\n  {behavior}: {len(feats)} features")
        if feats:
            # Show top 5 features
            for idx in feats[:5]:
                for r in reports:
                    if r.feature_idx == idx:
                        print(f"    Feature {idx}: selectivity={r.selectivity:.2f}, "
                              f"scores={r.behavior_scores}")
                        break

    print(f"\n  Feature Overlap Matrix (Jaccard):")
    print(f"  {'':12s}", end="")
    for b in behaviors:
        print(f"  {b:>10s}", end="")
    print()
    for b1 in behaviors:
        print(f"  {b1:12s}", end="")
        for b2 in behaviors:
            print(f"  {overlap.get(b1, {}).get(b2, 0.0):10.4f}", end="")
        print()

    # Save report
    if args.output:
        report = SAESteeringReport(
            model=model_name,
            layer_idx=layer_idx,
            feature_reports=reports,
            behavioral_features={b: list(v) for b, v in behavioral_features.items()},
            overlap_before=overlap,
        )
        out_path = Path(args.output)
        report.save(out_path)
        print(f"\n  Report saved to: {out_path}")


def cmd_eval(args):
    """Evaluate SAE steering vs SVD steering."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rho_eval.utils import get_device
    from rho_eval.steering import (
        GatedSAE, ActivationData,
        identify_behavioral_features,
        evaluate_sae_steering,
        SAESteeringReport, SAEConfig,
    )

    device = args.device or str(get_device())
    eval_behaviors = [b.strip() for b in args.eval_behaviors.split(",")]

    print(f"\n{'='*60}")
    print(f"  rho-steer eval: SAE vs SVD Steering Comparison")
    print(f"  Model:   {args.model}")
    print(f"  Target:  {args.target}")
    print(f"  Device:  {device}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    # Load SAE and activations
    meta_path = Path(args.sae).with_name(
        Path(args.sae).name.replace("sae_", "sae_meta_").replace(".pt", ".json")
    )
    with open(meta_path) as f:
        meta = json.load(f)

    hidden_dim = meta["config"]["hidden_dim"]
    expansion = meta["config"]["expansion_factor"]
    layer_idx = meta["layer_idx"]

    sae = GatedSAE(hidden_dim, expansion)
    sae.load_state_dict(torch.load(args.sae, weights_only=True))
    sae.eval()

    act_path = args.activations or str(
        Path(args.sae).with_name(f"activations_layer{layer_idx}.pt")
    )
    act_data = ActivationData.load(act_path)

    # Identify features
    print("\nIdentifying behavioral features...")
    reports, behavioral_features = identify_behavioral_features(
        sae, act_data, threshold=args.threshold,
    )

    # Parse scales
    scales = [float(s) for s in args.scales.split(",")]

    # SAE steering evaluation
    print("\n" + "="*50)
    print("  SAE STEERING")
    print("="*50)
    sae_results = evaluate_sae_steering(
        model, tokenizer, sae, behavioral_features,
        target_behavior=args.target,
        layer_idx=layer_idx,
        eval_behaviors=eval_behaviors,
        scales=scales,
        device=device,
        verbose=True,
    )

    # SVD comparison (if requested)
    svd_results = []
    if args.compare_svd:
        print("\n" + "="*50)
        print("  SVD STEERING (comparison)")
        print("="*50)

        from rho_eval.interpretability import extract_subspaces
        from rho_eval.interpretability.surgical import evaluate_surgical, evaluate_baseline

        subspaces = extract_subspaces(
            model, tokenizer,
            behaviors=act_data.behaviors,
            layers=[layer_idx],
            device=device,
            verbose=True,
        )

        baselines = evaluate_baseline(model, tokenizer, eval_behaviors, device)

        for alpha in [2.0, 4.0, 6.0]:
            result = evaluate_surgical(
                model, tokenizer, subspaces,
                target_behavior=args.target,
                layer_idx=layer_idx,
                eval_behaviors=eval_behaviors,
                alpha=alpha,
                device=device,
            )
            result_dict = result.to_dict()
            result_dict["baseline_rho_scores"] = baselines
            svd_results.append(result_dict)

    # Save results
    report = SAESteeringReport(
        model=args.model,
        layer_idx=layer_idx,
        sae_config=SAEConfig(hidden_dim=hidden_dim, expansion_factor=expansion),
        feature_reports=reports,
        behavioral_features={b: list(v) for b, v in behavioral_features.items()},
        steering_results=sae_results + [{"method": "svd", **r} for r in svd_results],
    )

    out_path = Path(args.output) if args.output else (
        RESULTS_DIR / f"eval_{args.target}_layer{layer_idx}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(out_path)
    print(f"\n  Results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="rho-steer",
        description="SAE-based behavioral steering for LLMs.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # ── train ────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Train a Gated SAE")
    p_train.add_argument("model", help="HuggingFace model name or path")
    p_train.add_argument("--layer", type=int, default=17, help="Layer index (default: 17)")
    p_train.add_argument("--behaviors", default=",".join(DEFAULT_BEHAVIORS),
                         help="Comma-separated behaviors")
    p_train.add_argument("--expansion", type=int, default=8, help="SAE expansion factor")
    p_train.add_argument("--sparsity-lambda", type=float, default=1e-3, help="L1 weight")
    p_train.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p_train.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p_train.add_argument("--epochs", type=int, default=5, help="Training epochs")
    p_train.add_argument("--max-probes", type=int, default=None, help="Max probes per behavior")
    p_train.add_argument("--output", "-o", default=None, help="Output directory")
    p_train.add_argument("--device", default=None, help="Torch device")
    p_train.set_defaults(func=cmd_train)

    # ── analyze ──────────────────────────────────────────────────
    p_analyze = subparsers.add_parser("analyze", help="Analyze trained SAE features")
    p_analyze.add_argument("sae", help="Path to saved SAE weights (.pt)")
    p_analyze.add_argument("--activations", default=None, help="Path to activation data")
    p_analyze.add_argument("--meta", default=None, help="Path to metadata JSON")
    p_analyze.add_argument("--threshold", type=float, default=2.0,
                           help="Selectivity threshold (default: 2.0)")
    p_analyze.add_argument("--hidden-dim", type=int, default=None, help="Hidden dim (fallback)")
    p_analyze.add_argument("--expansion", type=int, default=None, help="Expansion (fallback)")
    p_analyze.add_argument("--layer", type=int, default=None, help="Layer idx (fallback)")
    p_analyze.add_argument("--output", "-o", default=None, help="Output JSON path")
    p_analyze.set_defaults(func=cmd_analyze)

    # ── eval ─────────────────────────────────────────────────────
    p_eval = subparsers.add_parser("eval", help="Evaluate SAE steering")
    p_eval.add_argument("model", help="HuggingFace model name or path")
    p_eval.add_argument("--sae", required=True, help="Path to SAE weights")
    p_eval.add_argument("--activations", default=None, help="Path to activation data")
    p_eval.add_argument("--target", default="sycophancy", help="Behavior to steer")
    p_eval.add_argument("--eval-behaviors", default=",".join(EVAL_BEHAVIORS),
                         help="Behaviors to evaluate")
    p_eval.add_argument("--scales", default="0.0,1.5,2.0,3.0,4.0",
                         help="Comma-separated scale factors")
    p_eval.add_argument("--threshold", type=float, default=2.0,
                         help="Feature selectivity threshold")
    p_eval.add_argument("--compare-svd", action="store_true",
                         help="Also run SVD steering for comparison")
    p_eval.add_argument("--output", "-o", default=None, help="Output JSON path")
    p_eval.add_argument("--device", default=None, help="Torch device")
    p_eval.set_defaults(func=cmd_eval)

    # ── version ──────────────────────────────────────────────────
    parser.add_argument("--version", action="store_true", help="Show version")

    args = parser.parse_args()

    if args.version:
        try:
            from rho_eval import __version__
            print(f"rho-steer (rho-eval {__version__})")
        except ImportError:
            print("rho-steer (rho-eval)")
        return

    if args.command is None:
        parser.print_help()
        return

    if hasattr(args, 'func') and callable(args.func):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
