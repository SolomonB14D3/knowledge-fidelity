#!/usr/bin/env python3
"""rho-benchmark: Comprehensive behavioral benchmarking for LLMs.

Runs the full rho-eval audit (8 behavioral dimensions) plus external
benchmarks (TruthfulQA MC2), with optional baseline comparison.

Usage:
    rho-benchmark ./repaired-7b/model/ --baseline Qwen/Qwen2.5-7B-Instruct
    rho-benchmark Qwen/Qwen2.5-7B-Instruct
    rho-benchmark ./repaired-7b/model/ --baseline Qwen/Qwen2.5-7B-Instruct --quick
    rho-benchmark Qwen/Qwen2.5-7B-Instruct --external none --format json
"""

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="rho-benchmark",
        description="Comprehensive behavioral benchmarking for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark a repaired model against its baseline
  rho-benchmark ./repaired-7b/model/ --baseline Qwen/Qwen2.5-7B-Instruct

  # Quick mode (fewer probes, faster)
  rho-benchmark ./repaired-7b/model/ --baseline Qwen/Qwen2.5-7B-Instruct --quick

  # Standalone benchmark (no comparison)
  rho-benchmark Qwen/Qwen2.5-7B-Instruct

  # Internal audit only (skip TruthfulQA)
  rho-benchmark Qwen/Qwen2.5-7B-Instruct --external none

  # JSON output for CI pipelines
  rho-benchmark ./repaired-7b/model/ --baseline Qwen/Qwen2.5-7B-Instruct --format json -o report.json

  # Specific behaviors only
  rho-benchmark ./repaired-7b/model/ --behaviors sycophancy,bias,factual
""",
    )

    # ── Positional ──────────────────────────────────────────────────
    parser.add_argument(
        "model",
        help="Model to benchmark: HuggingFace ID or local path",
    )

    # ── Comparison ──────────────────────────────────────────────────
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Baseline model for comparison (HuggingFace ID or local path)",
    )

    # ── Benchmark config ────────────────────────────────────────────
    bench_group = parser.add_argument_group("Benchmark configuration")
    bench_group.add_argument(
        "--behaviors", type=str, default="all",
        help="Comma-separated behaviors to evaluate (default: all)",
    )
    bench_group.add_argument(
        "--external", type=str, default="truthfulqa",
        choices=["none", "truthfulqa", "all"],
        help="External benchmarks to run (default: truthfulqa)",
    )
    bench_group.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 50 probes/behavior, 200 TruthfulQA questions",
    )
    bench_group.add_argument(
        "--n-probes", type=int, default=None,
        help="Override probes per behavior (default: all, quick=50)",
    )
    bench_group.add_argument(
        "--tqa-questions", type=int, default=None,
        help="Override TruthfulQA question count (default: all 817, quick=200)",
    )
    bench_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    # ── Output ──────────────────────────────────────────────────────
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--format", choices=["table", "json", "markdown", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    output_group.add_argument(
        "-o", "--output", type=str, default=None,
        help="Save report to file or directory",
    )

    args = parser.parse_args(argv)

    # ── Resolve quick mode defaults ────────────────────────────────
    n_probes = args.n_probes
    tqa_n = args.tqa_questions
    if args.quick:
        if n_probes is None:
            n_probes = 50
        if tqa_n is None:
            tqa_n = 200

    # Parse behaviors
    if args.behaviors == "all":
        behavior_list = "all"
    else:
        behavior_list = [b.strip() for b in args.behaviors.split(",")]

    run_tqa = args.external in ("truthfulqa", "all")

    # ── Banner ─────────────────────────────────────────────────────
    t_global = time.time()
    print(f"\n{'='*60}")
    print(f"  rho-benchmark: Comprehensive Behavioral Benchmarking")
    print(f"  Model:      {args.model}")
    if args.baseline:
        print(f"  Baseline:   {args.baseline}")
    print(f"  Behaviors:  {args.behaviors}")
    print(f"  External:   {args.external}")
    if args.quick:
        print(f"  Mode:       quick ({n_probes} probes, {tqa_n} TruthfulQA)")
    print(f"{'='*60}\n")

    # ── Step 1: Load model ─────────────────────────────────────────
    print("  Step 1: Loading model...", flush=True)
    import mlx_lm
    model, tokenizer = mlx_lm.load(args.model)
    print(f"  Model loaded: {args.model}", flush=True)

    # ── Step 2: Internal audit ─────────────────────────────────────
    print("\n  Step 2: Internal audit (rho-eval)...", flush=True)
    from rho_eval import audit

    model_report = audit(
        model=model, tokenizer=tokenizer,
        behaviors=behavior_list,
        n=n_probes,
        seed=args.seed,
    )
    print(f"  Audit: mean_rho={model_report.mean_rho:.4f}, "
          f"status={model_report.overall_status}", flush=True)

    # ── Step 3: TruthfulQA MC2 ────────────────────────────────────
    model_tqa = None
    if run_tqa:
        print("\n  Step 3: TruthfulQA MC2...", flush=True)
        from rho_eval.benchmarking.truthfulqa import load_truthfulqa_mc2, score_mc2

        questions = load_truthfulqa_mc2(n=tqa_n, seed=args.seed)
        model_tqa = score_mc2(model, tokenizer, questions)
        print(f"  TruthfulQA MC2: {model_tqa['mc2_score']:.4f} "
              f"(MC1={model_tqa['mc1_accuracy']:.1%})", flush=True)
    else:
        print("\n  Step 3: TruthfulQA skipped (--external none)", flush=True)

    # ── Step 4: Baseline comparison (optional) ─────────────────────
    baseline_report = None
    baseline_tqa = None
    if args.baseline:
        print(f"\n  Step 4: Baseline evaluation ({args.baseline})...", flush=True)

        # Free model memory before loading baseline
        import gc
        del model
        gc.collect()

        print("  Loading baseline model...", flush=True)
        base_model, base_tokenizer = mlx_lm.load(args.baseline)
        print("  Baseline loaded.", flush=True)

        # Baseline audit
        print("  Running baseline audit...", flush=True)
        baseline_report = audit(
            model=base_model, tokenizer=base_tokenizer,
            behaviors=behavior_list,
            n=n_probes,
            seed=args.seed,
        )
        print(f"  Baseline: mean_rho={baseline_report.mean_rho:.4f}, "
              f"status={baseline_report.overall_status}", flush=True)

        # Baseline TruthfulQA
        if run_tqa:
            print("  Running baseline TruthfulQA...", flush=True)
            baseline_tqa = score_mc2(base_model, base_tokenizer, questions)
            print(f"  Baseline TruthfulQA MC2: {baseline_tqa['mc2_score']:.4f}", flush=True)

        del base_model, base_tokenizer
        gc.collect()
    else:
        print("\n  Step 4: No baseline (standalone benchmark)", flush=True)

    # ── Step 5: Results ────────────────────────────────────────────
    print(f"\n  Step 5: Results", flush=True)
    elapsed = time.time() - t_global

    # Build structured result
    result = _build_result(
        args, model_report, model_tqa,
        baseline_report, baseline_tqa,
        elapsed,
    )

    # Format output
    output_text = _format_output(
        args, result, model_report, model_tqa,
        baseline_report, baseline_tqa,
    )
    print(output_text)

    # ── Save ───────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        if out_path.suffix == "":
            # Directory mode
            out_path.mkdir(parents=True, exist_ok=True)
            json_path = out_path / "benchmark_report.json"
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            json_path = out_path

        json_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Saved: {json_path}")

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    return 0


def _build_result(args, model_report, model_tqa,
                  baseline_report, baseline_tqa, elapsed):
    """Build structured JSON result."""
    result = {
        "model": args.model,
        "baseline": args.baseline,
        "config": {
            "behaviors": args.behaviors,
            "external": args.external,
            "quick": args.quick,
            "seed": args.seed,
        },
        "audit": model_report.to_dict(),
    }

    if model_tqa:
        result["truthfulqa"] = {
            "mc2_score": model_tqa["mc2_score"],
            "mc1_accuracy": model_tqa["mc1_accuracy"],
            "n_questions": model_tqa["n_questions"],
            "elapsed": model_tqa["elapsed"],
        }

    if baseline_report:
        result["baseline_audit"] = baseline_report.to_dict()
        # Compute deltas
        deltas = {}
        for name in model_report.behaviors:
            model_rho = model_report.behaviors[name].rho
            base_rho = baseline_report.behaviors.get(name)
            if base_rho:
                delta = model_rho - base_rho.rho
                deltas[name] = {
                    "model_rho": round(model_rho, 4),
                    "baseline_rho": round(base_rho.rho, 4),
                    "delta": round(delta, 4),
                }
        result["deltas"] = deltas

    if baseline_tqa and model_tqa:
        result["truthfulqa_delta"] = {
            "model_mc2": model_tqa["mc2_score"],
            "baseline_mc2": baseline_tqa["mc2_score"],
            "delta": round(model_tqa["mc2_score"] - baseline_tqa["mc2_score"], 4),
        }

    result["elapsed_seconds"] = round(elapsed, 1)
    return result


def _format_output(args, result, model_report, model_tqa,
                   baseline_report, baseline_tqa):
    """Format output based on --format flag."""
    if args.format == "json":
        return json.dumps(result, indent=2, default=str)

    if args.format == "markdown":
        return _format_markdown(
            args, model_report, model_tqa,
            baseline_report, baseline_tqa,
        )

    if args.format == "csv":
        return _format_csv(result)

    # Default: table
    return _format_table(
        args, model_report, model_tqa,
        baseline_report, baseline_tqa,
    )


def _format_table(args, model_report, model_tqa,
                  baseline_report, baseline_tqa):
    """Format as colored terminal table."""
    lines = []

    if baseline_report:
        # Comparison table
        lines.append(f"\n  {'Behavior':<14s} {'Baseline':>9s} {'Model':>9s} {'Delta':>9s}  Status")
        lines.append(f"  {'─'*14}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}")

        for name in sorted(model_report.behaviors.keys()):
            model_rho = model_report.behaviors[name].rho
            base_result = baseline_report.behaviors.get(name)
            if base_result:
                base_rho = base_result.rho
                delta = model_rho - base_rho
                if delta > 0.01:
                    marker = "\033[92m IMPROVED\033[0m"
                elif delta < -0.01:
                    marker = "\033[91m DEGRADED\033[0m"
                else:
                    marker = " unchanged"
                lines.append(
                    f"  {name:<14s} {base_rho:>+9.4f} {model_rho:>+9.4f} "
                    f"{delta:>+9.4f} {marker}"
                )

        # Mean row
        base_mean = baseline_report.mean_rho
        model_mean = model_report.mean_rho
        delta_mean = model_mean - base_mean
        lines.append(f"  {'─'*14}  {'─'*9}  {'─'*9}  {'─'*9}")
        lines.append(f"  {'MEAN':<14s} {base_mean:>+9.4f} {model_mean:>+9.4f} "
                      f"{delta_mean:>+9.4f}")

        # TruthfulQA comparison
        if model_tqa and baseline_tqa:
            lines.append("")
            mc2_delta = model_tqa["mc2_score"] - baseline_tqa["mc2_score"]
            mc1_delta = model_tqa["mc1_accuracy"] - baseline_tqa["mc1_accuracy"]
            lines.append(f"  TruthfulQA MC2:  {baseline_tqa['mc2_score']:.4f} → "
                         f"{model_tqa['mc2_score']:.4f} (Δ={mc2_delta:+.4f})")
            lines.append(f"  TruthfulQA MC1:  {baseline_tqa['mc1_accuracy']:.1%} → "
                         f"{model_tqa['mc1_accuracy']:.1%} (Δ={mc1_delta:+.1%})")
    else:
        # Standalone table
        lines.append(f"\n  {'Behavior':<14s} {'ρ':>9s}  {'Status':>8s}")
        lines.append(f"  {'─'*14}  {'─'*9}  {'─'*8}")

        for name in sorted(model_report.behaviors.keys()):
            result = model_report.behaviors[name]
            lines.append(
                f"  {name:<14s} {result.rho:>+9.4f}  {result.status:>8s}"
            )

        lines.append(f"  {'─'*14}  {'─'*9}")
        lines.append(f"  {'MEAN':<14s} {model_report.mean_rho:>+9.4f}  "
                      f"{model_report.overall_status}")

        # TruthfulQA
        if model_tqa:
            lines.append("")
            lines.append(f"  TruthfulQA MC2:  {model_tqa['mc2_score']:.4f}")
            lines.append(f"  TruthfulQA MC1:  {model_tqa['mc1_accuracy']:.1%}")

    return "\n".join(lines)


def _format_markdown(args, model_report, model_tqa,
                     baseline_report, baseline_tqa):
    """Format as Markdown."""
    lines = [f"# Benchmark Report: {args.model}", ""]

    if baseline_report:
        lines.append(f"**Baseline:** {args.baseline}")
        lines.append("")
        lines.append("## Behavioral Audit Comparison")
        lines.append("")
        lines.append("| Behavior | Baseline ρ | Model ρ | Delta |")
        lines.append("|----------|----------:|--------:|------:|")

        for name in sorted(model_report.behaviors.keys()):
            model_rho = model_report.behaviors[name].rho
            base_result = baseline_report.behaviors.get(name)
            if base_result:
                delta = model_rho - base_result.rho
                lines.append(
                    f"| {name} | {base_result.rho:+.4f} | "
                    f"{model_rho:+.4f} | {delta:+.4f} |"
                )

        base_mean = baseline_report.mean_rho
        model_mean = model_report.mean_rho
        lines.append(f"| **MEAN** | **{base_mean:+.4f}** | "
                      f"**{model_mean:+.4f}** | **{model_mean - base_mean:+.4f}** |")
    else:
        lines.append("## Behavioral Audit")
        lines.append("")
        lines.append("| Behavior | ρ | Status |")
        lines.append("|----------|---:|:------:|")

        for name in sorted(model_report.behaviors.keys()):
            result = model_report.behaviors[name]
            lines.append(f"| {name} | {result.rho:+.4f} | {result.status} |")

        lines.append(f"| **MEAN** | **{model_report.mean_rho:+.4f}** | "
                      f"**{model_report.overall_status}** |")

    # TruthfulQA section
    if model_tqa:
        lines.append("")
        lines.append("## TruthfulQA MC2")
        lines.append("")
        if baseline_tqa:
            mc2_delta = model_tqa["mc2_score"] - baseline_tqa["mc2_score"]
            lines.append(f"| Metric | Baseline | Model | Delta |")
            lines.append(f"|--------|----------|-------|-------|")
            lines.append(f"| MC2 | {baseline_tqa['mc2_score']:.4f} | "
                          f"{model_tqa['mc2_score']:.4f} | {mc2_delta:+.4f} |")
            lines.append(f"| MC1 | {baseline_tqa['mc1_accuracy']:.1%} | "
                          f"{model_tqa['mc1_accuracy']:.1%} | "
                          f"{model_tqa['mc1_accuracy'] - baseline_tqa['mc1_accuracy']:+.1%} |")
        else:
            lines.append(f"- **MC2 Score:** {model_tqa['mc2_score']:.4f}")
            lines.append(f"- **MC1 Accuracy:** {model_tqa['mc1_accuracy']:.1%}")
            lines.append(f"- **Questions:** {model_tqa['n_questions']}")

    return "\n".join(lines)


def _format_csv(result):
    """Format as CSV rows."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    header = ["behavior", "model_rho", "status"]
    has_baseline = "deltas" in result
    if has_baseline:
        header = ["behavior", "baseline_rho", "model_rho", "delta"]
    writer.writerow(header)

    audit_data = result["audit"]
    behaviors = audit_data.get("behaviors", {})

    for name in sorted(behaviors.keys()):
        bdata = behaviors[name]
        if has_baseline and name in result["deltas"]:
            d = result["deltas"][name]
            writer.writerow([name, d["baseline_rho"], d["model_rho"], d["delta"]])
        else:
            writer.writerow([name, bdata.get("rho", 0), bdata.get("status", "")])

    # TruthfulQA row
    if "truthfulqa" in result:
        if has_baseline and "truthfulqa_delta" in result:
            d = result["truthfulqa_delta"]
            writer.writerow(["truthfulqa_mc2", d["baseline_mc2"], d["model_mc2"], d["delta"]])
        else:
            writer.writerow(["truthfulqa_mc2", result["truthfulqa"]["mc2_score"], "", ""])

    return output.getvalue()


if __name__ == "__main__":
    sys.exit(main() or 0)
