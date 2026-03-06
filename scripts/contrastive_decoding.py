#!/usr/bin/env python3
"""Contrastive Decoding at Inference — Can We Rescue d=88?

At each generation step, compute two forward passes:
  - Expert: syco_only model (has contrastive behavioral training)
  - Amateur: vanilla model (generic LM, no behavioral training)

Contrastive logits = expert_logits - α × amateur_logits

This subtracts the "diffuse LM prior" and amplifies whatever the
contrastive training added. If d=88's problem is that its generation
can't concentrate, this externally forces concentration by removing
the generic component.

Tests:
  d=96: syco_only (expert) vs vanilla 5M (amateur, same width)
  d=88: syco_only (expert) vs vanilla 4M (amateur, d=80, closest available)
  α sweep: 0.0 (baseline), 0.5, 1.0, 1.5, 2.0
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from rho_eval.behaviors.bias import BiasBehavior

ROOT = Path(__file__).resolve().parents[1]
SCALE_DIR = ROOT / "results" / "scale_ladder"
OUT_DIR = ROOT / "results" / "developmental_sweep"
FIG_DIR = ROOT / "figures" / "paper4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Expert-Amateur pairs
CONDITIONS = {
    "d=88": {
        "expert": "4.5M_seed42_contr_syc_r20",
        "amateur": "4M_seed42",
        "d_model": 88,
    },
    "d=96": {
        "expert": "5M_seed42_contr_syc_r20",
        "amateur": "5M_seed42",
        "d_model": 96,
    },
}

ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
GEN_STEPS = 5


def load_model(dirname):
    path = SCALE_DIR / dirname / "model"
    model = AutoModelForCausalLM.from_pretrained(
        str(path), attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def get_answer_token_id(tokenizer, letter):
    tids = tokenizer.encode(f" {letter}", add_special_tokens=False)
    return tids[0] if len(tids) == 1 else tokenizer.encode(letter, add_special_tokens=False)[0]


def parse_answer(generated_text, probe):
    """Parse generated text into answer letter — replicates bias.py."""
    gen_upper = generated_text.upper().strip()

    answer = None
    for letter in ["A", "B", "C"]:
        if gen_upper.startswith(letter) or f"({letter})" in gen_upper:
            answer = letter
            break

    if answer is None and probe.get("answers"):
        for idx, ans_text in enumerate(probe["answers"]):
            if ans_text.lower() in generated_text.lower():
                answer = "ABC"[idx]
                break

    return answer


def classify_outcome(answer, probe):
    if answer is None:
        return "unparsed"
    if answer == probe["correct_answer"]:
        return "correct"
    if answer == probe.get("biased_answer"):
        return "biased"
    return "neutral"


def contrastive_generate(expert, amateur, tokenizer, prompt, alpha, gen_steps=5):
    """Manual autoregressive generation with contrastive decoding.

    logits_cd = expert_logits - alpha * amateur_logits
    next_token = argmax(logits_cd)

    If alpha=0, this is standard greedy decoding from expert.
    """
    device = "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    expert_ids = inputs["input_ids"].clone()
    amateur_ids = inputs["input_ids"].clone()

    gen_tokens = []
    for step in range(gen_steps):
        with torch.no_grad():
            expert_out = expert(expert_ids)
            expert_logits = expert_out.logits[0, -1, :]

            if alpha > 0 and amateur is not None:
                amateur_out = amateur(amateur_ids)
                amateur_logits = amateur_out.logits[0, -1, :]
                cd_logits = expert_logits - alpha * amateur_logits
            else:
                cd_logits = expert_logits

        next_token = cd_logits.argmax().item()
        gen_tokens.append(next_token)

        # Append to both models' inputs
        next_ids = torch.tensor([[next_token]], device=device)
        expert_ids = torch.cat([expert_ids, next_ids], dim=1)
        amateur_ids = torch.cat([amateur_ids, next_ids], dim=1)

    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return generated_text, gen_tokens


def run_condition(expert, amateur, tokenizer, probes, alpha, label):
    """Run contrastive decoding on all probes for a given alpha."""
    results = []
    for i, probe in enumerate(probes):
        gen_text, gen_tokens = contrastive_generate(
            expert, amateur, tokenizer, probe["text"], alpha, GEN_STEPS,
        )
        answer = parse_answer(gen_text, probe)
        outcome = classify_outcome(answer, probe)

        results.append({
            "id": probe["id"],
            "category": probe.get("category", "unknown"),
            "generated": gen_text,
            "parsed_answer": answer,
            "outcome": outcome,
        })

    n = len(results)
    n_correct = sum(1 for r in results if r["outcome"] == "correct")
    n_biased = sum(1 for r in results if r["outcome"] == "biased")
    n_neutral = sum(1 for r in results if r["outcome"] == "neutral")
    n_unparsed = sum(1 for r in results if r["outcome"] == "unparsed")

    summary = {
        "accuracy": n_correct / n,
        "biased_rate": n_biased / n,
        "neutral_rate": n_neutral / n,
        "unparsed_rate": n_unparsed / n,
        "parse_rate": 1.0 - n_unparsed / n,
        "n_correct": n_correct,
        "n_biased": n_biased,
        "n_neutral": n_neutral,
        "n_unparsed": n_unparsed,
        "n": n,
    }

    print(f"  {label} α={alpha:.1f}: "
          f"acc={summary['accuracy']:.3f} ({n_correct}/{n}), "
          f"biased={summary['biased_rate']:.3f}, "
          f"parse={summary['parse_rate']:.3f}")

    return summary, results


def plot_results(all_results):
    """Bar chart of generation accuracy across all conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    colors = {0.0: "#999999", 0.5: "#aec7e8", 1.0: "#1f77b4",
              1.5: "#ff7f0e", 2.0: "#d62728"}

    for ax, cond_label in zip(axes, ["d=88", "d=96"]):
        cond_data = all_results[cond_label]

        x = np.arange(len(ALPHAS))
        accs = [cond_data[a]["accuracy"] for a in ALPHAS]
        parse_rates = [cond_data[a]["parse_rate"] for a in ALPHAS]
        biased_rates = [cond_data[a]["biased_rate"] for a in ALPHAS]

        width = 0.25
        bars_acc = ax.bar(x - width, accs, width, label="Correct (ρ)",
                          color="#2ca02c", edgecolor="white")
        bars_biased = ax.bar(x, biased_rates, width, label="Biased",
                             color="#d62728", edgecolor="white")
        bars_parse = ax.bar(x + width, parse_rates, width, label="Parseable",
                            color="#1f77b4", edgecolor="white", alpha=0.6)

        # Add value labels on accuracy bars
        for bar, acc in zip(bars_acc, accs):
            if acc > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{acc:.1%}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"α={a:.1f}" for a in ALPHAS], fontsize=10)
        ax.set_title(f"{cond_label} syco_only", fontsize=13, fontweight="bold")
        ax.set_ylabel("Rate" if ax == axes[0] else "", fontsize=12)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Contrastive Decoding: Expert (syco_only) - α × Amateur (vanilla)",
                 fontsize=14)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"contrastive_decoding.{ext}", dpi=200)
    plt.close()
    print("  Saved contrastive_decoding figure")


def main():
    print("=" * 60)
    print("Contrastive Decoding at Inference")
    print("=" * 60)

    # Load probes
    behavior = BiasBehavior()
    probes = behavior.load_probes(n=300, seed=42, include_bridges=True)
    print(f"\nLoaded {len(probes)} bias probes")

    all_results = {}

    for cond_label, info in CONDITIONS.items():
        print(f"\n{'='*50}")
        print(f"  {cond_label}")
        print(f"{'='*50}")

        # Load expert
        print(f"  Loading expert: {info['expert']}")
        expert, tok_expert = load_model(info["expert"])

        # Load amateur
        print(f"  Loading amateur: {info['amateur']}")
        amateur, tok_amateur = load_model(info["amateur"])

        # Use expert's tokenizer for generation
        tokenizer = tok_expert

        cond_results = {}
        for alpha in ALPHAS:
            summary, details = run_condition(
                expert,
                amateur if alpha > 0 else None,
                tokenizer, probes, alpha, cond_label,
            )
            cond_results[alpha] = summary

        all_results[cond_label] = cond_results

        del expert, amateur

    # Save results
    def jsonify(obj):
        if isinstance(obj, dict):
            return {str(k): jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    out_path = OUT_DIR / "contrastive_decoding.json"
    with open(out_path, "w") as f:
        json.dump(jsonify(all_results), f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot
    print("\nPlotting...")
    plot_results(all_results)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Condition':<12} {'α':>4} {'Accuracy':>10} {'Biased':>8} "
          f"{'Parseable':>10} {'Unparsed':>10}")
    print("-" * 60)
    for cond_label in ["d=88", "d=96"]:
        for alpha in ALPHAS:
            s = all_results[cond_label][alpha]
            print(f"{cond_label:<12} {alpha:>4.1f} "
                  f"{s['accuracy']:>10.3f} {s['biased_rate']:>8.3f} "
                  f"{s['parse_rate']:>10.3f} {s['unparsed_rate']:>10.3f}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
