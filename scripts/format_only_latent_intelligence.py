#!/usr/bin/env python3
"""Format-Only Latent Intelligence Test.

Hypothesis: the 3M model has latent intelligence — it KNOWS the unbiased
answer but can't EXPRESS it. If giving it the contrastive FORMAT (via NLI
pairs, which share the A-correct/B-incorrect structure but have ZERO bias
content) reduces the "Neither" rate, we've proven the model was already
intelligent enough to be unbiased — it just didn't know how to talk.

Experiment:
1. Run vanilla 3M on bias probes → measure correct/biased/neither rates
2. Run NLI-trained 3M on bias probes → measure the same
3. Run bias-only 3M (positive control) → measure the same
4. Run other format-only models (Calculator, Subitizing, Primitive) → same
5. Compare: does FORMAT alone unlock latent intelligence?

If "Neither" drops while correct rate rises (or biased rate drops):
→ Model was "mute, not dumb" — it had the geometry, needed the vocabulary.

Data: results/scale_ladder/ checkpoints with audit_report.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rho_eval.behaviors.bias import BiasBehavior

ROOT = Path(__file__).resolve().parents[1]
SCALE_DIR = ROOT / "results" / "scale_ladder"
OUT_DIR = ROOT / "results" / "developmental_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All 3M conditions
CONDITIONS = [
    ("vanilla", "3M_seed42"),
    ("bias_only", "3M_seed42_contr_bia_r20"),
    ("syco_only", "3M_seed42_contr_syc_r20"),
    ("nli_format", "3M_seed42_contr_nli_r20"),
    ("calculator_format", "3M_seed42_contr_cal_r20"),
    ("subitizing_format", "3M_seed42_contr_sub_r20"),
    ("primitive_format", "3M_seed42_contr_pri_r20"),
]

# Also test at 5M and 7M for comparison
CONDITIONS_5M = [
    ("vanilla_5M", "5M_seed42"),
    ("nli_format_5M", "5M_seed42_contr_nli_r20"),
    ("bias_only_5M", "5M_seed42_contr_bia_r20"),
    ("syco_only_5M", "5M_seed42_contr_syc_r20"),
]

CONDITIONS_7M = [
    ("vanilla_7M", "7M_seed42"),
    ("nli_format_7M", "7M_seed42_contr_nli_r20"),
    ("bias_only_7M", "7M_seed42_contr_bia_r20"),
    ("syco_only_7M", "7M_seed42_contr_syc_r20"),
]


def run_bias_evaluation(model_dir, condition_name):
    """Run bias probes and return detailed per-probe results."""
    model_path = SCALE_DIR / model_dir / "model"
    if not model_path.exists():
        print(f"  SKIP {condition_name}: {model_path} not found")
        return None

    print(f"\n  [{condition_name}] Loading {model_dir}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Run bias evaluation
    behavior = BiasBehavior()
    probes = behavior.load_probes(n=300, seed=42)

    print(f"    Evaluating {len(probes)} probes...", flush=True)
    with torch.no_grad():
        result = behavior.evaluate(model, tokenizer, probes)

    # Extract detailed breakdown
    details = result.details or []
    n_total = result.total
    n_correct = result.positive_count
    n_biased = 0
    n_neither = 0
    n_unparsed = 0

    for d in details:
        if d.get("is_biased", False):
            n_biased += 1
        elif not d.get("is_correct", False):
            # Not correct and not biased = neither/unknown
            if d.get("model_answer") is None:
                n_unparsed += 1
            else:
                n_neither += 1

    # Category breakdown
    cat_metrics = result.metadata.get("category_metrics", {}) if result.metadata else {}

    # Answer distribution
    answer_dist = {}
    for d in details:
        ans = d.get("model_answer", "None")
        if ans is None:
            ans = "UNPARSED"
        answer_dist[ans] = answer_dist.get(ans, 0) + 1

    summary = {
        "condition": condition_name,
        "model_dir": model_dir,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_biased": n_biased,
        "n_neither": n_total - n_correct - n_biased,
        "n_unparsed": n_unparsed,
        "correct_rate": round(n_correct / n_total, 4) if n_total > 0 else 0,
        "biased_rate": round(n_biased / n_total, 4) if n_total > 0 else 0,
        "neither_rate": round((n_total - n_correct - n_biased) / n_total, 4) if n_total > 0 else 0,
        "rho": round(result.rho, 4),
        "answer_distribution": answer_dist,
        "category_metrics": {
            cat: {
                "accuracy": round(m["accuracy"], 4),
                "n": m["n"],
            }
            for cat, m in cat_metrics.items()
        },
    }

    print(f"    ρ={result.rho:.4f}  "
          f"correct={n_correct}/{n_total} ({n_correct/n_total:.1%})  "
          f"biased={n_biased} ({n_biased/n_total:.1%})  "
          f"neither={n_total-n_correct-n_biased} ({(n_total-n_correct-n_biased)/n_total:.1%})",
          flush=True)
    print(f"    Answer distribution: {answer_dist}", flush=True)

    del model, tokenizer
    return summary


def main():
    all_results = {"3M": [], "5M": [], "7M": []}

    # ── 3M conditions (main test) ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  3M LATENT INTELLIGENCE TEST (d=64, 2 layers)")
    print(f"{'='*70}")

    for name, dirname in CONDITIONS:
        result = run_bias_evaluation(dirname, name)
        if result:
            all_results["3M"].append(result)

    # ── 5M conditions (comparison) ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  5M COMPARISON (d=96, 2 layers)")
    print(f"{'='*70}")

    for name, dirname in CONDITIONS_5M:
        result = run_bias_evaluation(dirname, name)
        if result:
            all_results["5M"].append(result)

    # ── 7M conditions (comparison) ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  7M COMPARISON (d=128, 4 layers)")
    print(f"{'='*70}")

    for name, dirname in CONDITIONS_7M:
        result = run_bias_evaluation(dirname, name)
        if result:
            all_results["7M"].append(result)

    # ── Summary Table ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  LATENT INTELLIGENCE — SUMMARY")
    print(f"{'='*70}\n")

    print(f"  {'Condition':<22s} {'ρ':>6s} {'Correct':>8s} {'Biased':>8s} "
          f"{'Neither':>8s} {'Unparsed':>8s}")
    print(f"  {'-'*62}")

    for scale in ["3M", "5M", "7M"]:
        if not all_results[scale]:
            continue
        print(f"\n  --- {scale} ---")
        for r in all_results[scale]:
            print(f"  {r['condition']:<22s} "
                  f"{r['rho']:>6.3f} "
                  f"{r['correct_rate']:>7.1%} "
                  f"{r['biased_rate']:>7.1%} "
                  f"{r['neither_rate']:>7.1%} "
                  f"{r['n_unparsed']:>8d}")

    # ── The Key Question ──────────────────────────────────────────────
    print(f"\n  KEY QUESTION: Does format-only training reduce 'Neither'?")
    print(f"  {'-'*60}")

    if all_results["3M"]:
        vanilla = next((r for r in all_results["3M"] if r["condition"] == "vanilla"), None)
        if vanilla:
            v_neither = vanilla["neither_rate"]
            v_correct = vanilla["correct_rate"]
            v_biased = vanilla["biased_rate"]
            print(f"  Vanilla 3M:     neither={v_neither:.1%}, "
                  f"correct={v_correct:.1%}, biased={v_biased:.1%}")
            print()
            for r in all_results["3M"]:
                if r["condition"] == "vanilla":
                    continue
                d_neither = r["neither_rate"] - v_neither
                d_correct = r["correct_rate"] - v_correct
                d_biased = r["biased_rate"] - v_biased
                marker = "✓" if d_neither < -0.05 and d_correct > 0 else " "
                print(f"  {r['condition']:<22s} "
                      f"Δneither={d_neither:>+6.1%}, "
                      f"Δcorrect={d_correct:>+6.1%}, "
                      f"Δbiased={d_biased:>+6.1%} {marker}")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = OUT_DIR / "format_only_latent_intelligence.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
