#!/usr/bin/env python3
"""Format-Forced Decoding — Full Scale Verification + Coherence Test.

Two questions:
1. UNIVERSALITY: Does the ~40% logit accuracy hold at ALL scales (3M→64M)?
   If vanilla models at d=64 through d=512 all get ~40%, the behavioral
   discrimination is a universal property of LM pretraining, not something
   contrastive injection teaches.

2. COHERENCE: When we force the correct answer token, does the continuation
   remain coherent? If yes → the mouth was the only thing paralyzed.
   If gibberish → the aphasia goes deeper into syntax generation.

Tests logit-only classification (argmax A/B/C) across all vanilla models,
plus extended generation (20 tokens) after forcing the correct answer.
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

# ALL vanilla models across the full scale ladder
VANILLA_CONDITIONS = {
    "3M (d=64)":  {"dirname": "3M_seed42"},
    "4M (d=80)":  {"dirname": "4M_seed42"},
    "5M (d=96)":  {"dirname": "5M_seed42"},
    "7M (d=128)": {"dirname": "7M_seed42"},
    "12M":        {"dirname": "12M_seed42"},
    "18M":        {"dirname": "18M_seed42"},
    "34M":        {"dirname": "34M_seed42"},
    "64M":        {"dirname": "64M_seed42"},
}

# Also test trained models for comparison
TRAINED_CONDITIONS = {
    "4.5M syco (d=88)":  {"dirname": "4.5M_seed42_contr_syc_r20"},
    "5M syco (d=96)":    {"dirname": "5M_seed42_contr_syc_r20"},
    "7M syco":           {"dirname": "7M_seed42_contr_syc_r20"},
    "7M bias+syco":      {"dirname": "7M_seed42_contr_bia_syc_r20_clean"},
    "12M bias+syco":     {"dirname": "12M_seed42_contr_bia_syc_r20_clean"},
}

GEN_STEPS_FREE = 5
GEN_STEPS_COHERENCE = 20  # Extended for coherence check
N_COHERENCE_EXAMPLES = 30  # Number of probes to show full continuations for


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


def logit_only_classify(model, tokenizer, prompt, abc_ids, probe):
    """Pure logit classification: pick highest-logit A/B/C token. No generation."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model(inputs["input_ids"])
        logits = out.logits[0, -1, :]

    abc_logits = [logits[tid].item() for tid in abc_ids]
    best_idx = int(np.argmax(abc_logits))
    answer = "ABC"[best_idx]
    return answer, abc_logits


def free_generate(model, tokenizer, prompt, gen_steps=5):
    """Standard greedy generation (baseline)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]

    gen_tokens = []
    for step in range(gen_steps):
        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits[0, -1, :]

        next_token = logits.argmax().item()
        gen_tokens.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

    return tokenizer.decode(gen_tokens, skip_special_tokens=True), gen_tokens


def forced_correct_generate(model, tokenizer, prompt, correct_token_id, gen_steps=20):
    """Force the CORRECT answer as token 0, then free-generate the rest.

    This tests whether the model can produce coherent continuations
    after being forced to emit the correct answer letter.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]

    # Force the correct answer token
    gen_tokens = [correct_token_id]
    input_ids = torch.cat([input_ids, torch.tensor([[correct_token_id]])], dim=1)

    # Free-generate the rest
    for step in range(gen_steps - 1):
        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits[0, -1, :]

        next_token = logits.argmax().item()
        gen_tokens.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

    return tokenizer.decode(gen_tokens, skip_special_tokens=True), gen_tokens


def assess_coherence(gen_text, gen_tokens, tokenizer):
    """Heuristic coherence assessment of generated text.

    Returns a dict with coherence metrics:
    - has_repetition: True if any 3-gram repeats
    - unique_token_ratio: unique tokens / total tokens
    - has_eos: True if EOS token appears (model chose to stop)
    - text_length: character length of generated text
    """
    # Check for 3-gram repetition
    trigrams = []
    for i in range(len(gen_tokens) - 2):
        trigrams.append(tuple(gen_tokens[i:i+3]))
    has_repetition = len(trigrams) != len(set(trigrams))

    # Unique token ratio
    unique_ratio = len(set(gen_tokens)) / max(len(gen_tokens), 1)

    # EOS check
    eos_id = tokenizer.eos_token_id
    has_eos = eos_id in gen_tokens

    return {
        "has_repetition": has_repetition,
        "unique_token_ratio": round(unique_ratio, 3),
        "has_eos": has_eos,
        "text_length": len(gen_text.strip()),
    }


def run_logit_classification(model, tokenizer, probes, label):
    """Run logit-only classification on all probes."""
    tok_A = get_answer_token_id(tokenizer, "A")
    tok_B = get_answer_token_id(tokenizer, "B")
    tok_C = get_answer_token_id(tokenizer, "C")
    abc_ids = [tok_A, tok_B, tok_C]

    outcomes = []
    for probe in probes:
        answer, _ = logit_only_classify(model, tokenizer, probe["text"], abc_ids, probe)
        outcome = classify_outcome(answer, probe)
        outcomes.append(outcome)

    n = len(outcomes)
    n_correct = sum(1 for r in outcomes if r == "correct")
    n_biased = sum(1 for r in outcomes if r == "biased")
    n_neutral = sum(1 for r in outcomes if r == "neutral")

    summary = {
        "logit_accuracy": n_correct / n,
        "biased_rate": n_biased / n,
        "neutral_rate": n_neutral / n,
        "n_correct": n_correct,
        "n_biased": n_biased,
        "n_neutral": n_neutral,
        "n": n,
    }

    print(f"  {label}: logit_acc={summary['logit_accuracy']:.3f} ({n_correct}/{n}), "
          f"biased={summary['biased_rate']:.3f}, neutral={summary['neutral_rate']:.3f}")

    return summary


def run_free_generation(model, tokenizer, probes, label):
    """Run free generation on all probes (for comparison)."""
    outcomes = []
    for probe in probes:
        gen_text, _ = free_generate(model, tokenizer, probe["text"], GEN_STEPS_FREE)
        answer = parse_answer(gen_text, probe)
        outcome = classify_outcome(answer, probe)
        outcomes.append(outcome)

    n = len(outcomes)
    n_correct = sum(1 for r in outcomes if r == "correct")
    n_unparsed = sum(1 for r in outcomes if r == "unparsed")

    summary = {
        "free_accuracy": n_correct / n,
        "unparsed_rate": n_unparsed / n,
        "parse_rate": 1.0 - n_unparsed / n,
        "n_correct": n_correct,
        "n_unparsed": n_unparsed,
        "n": n,
    }

    print(f"  {label} [free]: acc={summary['free_accuracy']:.3f} ({n_correct}/{n}), "
          f"parse={summary['parse_rate']:.3f}")

    return summary


def run_coherence_test(model, tokenizer, probes, label, n_examples=30):
    """Force correct answer and check if continuation is coherent.

    Tests the "paralyzed mouth" hypothesis:
    - If continuations are coherent → only the format token was broken
    - If gibberish → deeper aphasia in syntax generation
    """
    tok_A = get_answer_token_id(tokenizer, "A")
    tok_B = get_answer_token_id(tokenizer, "B")
    tok_C = get_answer_token_id(tokenizer, "C")
    letter_to_tok = {"A": tok_A, "B": tok_B, "C": tok_C}

    examples = []
    coherence_stats = []

    for i, probe in enumerate(probes[:n_examples]):
        correct_tok = letter_to_tok[probe["correct_answer"]]
        gen_text, gen_tokens = forced_correct_generate(
            model, tokenizer, probe["text"], correct_tok, GEN_STEPS_COHERENCE
        )
        coherence = assess_coherence(gen_text, gen_tokens, tokenizer)
        coherence_stats.append(coherence)

        examples.append({
            "probe_idx": i,
            "correct_answer": probe["correct_answer"],
            "forced_continuation": gen_text,
            "coherence": coherence,
        })

    # Also run with WRONG (biased) answer forced
    wrong_examples = []
    for i, probe in enumerate(probes[:n_examples]):
        biased_answer = probe.get("biased_answer")
        if biased_answer and biased_answer in letter_to_tok:
            biased_tok = letter_to_tok[biased_answer]
            gen_text, gen_tokens = forced_correct_generate(
                model, tokenizer, probe["text"], biased_tok, GEN_STEPS_COHERENCE
            )
            coherence = assess_coherence(gen_text, gen_tokens, tokenizer)
            wrong_examples.append({
                "probe_idx": i,
                "biased_answer": biased_answer,
                "forced_continuation": gen_text,
                "coherence": coherence,
            })

    # Aggregate coherence
    n_coherent = len(coherence_stats)
    avg_unique_ratio = np.mean([c["unique_token_ratio"] for c in coherence_stats])
    n_with_repetition = sum(1 for c in coherence_stats if c["has_repetition"])
    n_with_eos = sum(1 for c in coherence_stats if c["has_eos"])
    avg_text_length = np.mean([c["text_length"] for c in coherence_stats])

    summary = {
        "n_tested": n_coherent,
        "avg_unique_token_ratio": round(float(avg_unique_ratio), 3),
        "pct_with_repetition": round(n_with_repetition / n_coherent, 3),
        "pct_with_eos": round(n_with_eos / n_coherent, 3),
        "avg_text_length": round(float(avg_text_length), 1),
    }

    print(f"  {label} [coherence]: "
          f"unique_tok_ratio={summary['avg_unique_token_ratio']:.3f}, "
          f"repetition={summary['pct_with_repetition']:.1%}, "
          f"eos={summary['pct_with_eos']:.1%}, "
          f"avg_len={summary['avg_text_length']:.0f} chars")

    return summary, examples, wrong_examples


def plot_universality(all_results):
    """Bar chart showing logit accuracy across all vanilla scales."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Separate vanilla and trained
    vanilla_labels = [k for k in all_results if "vanilla" in all_results[k].get("type", "")]
    trained_labels = [k for k in all_results if "trained" in all_results[k].get("type", "")]

    # Plot vanilla
    v_accs = [all_results[k]["logit"]["logit_accuracy"] for k in vanilla_labels]
    v_free = [all_results[k].get("free", {}).get("free_accuracy", 0) for k in vanilla_labels]

    x = np.arange(len(vanilla_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, v_accs, width, label="Logit-Only (A/B/C argmax)",
                   color="#1f77b4", edgecolor="white")
    bars2 = ax.bar(x + width/2, v_free, width, label="Free Generation",
                   color="#d62728", edgecolor="white", alpha=0.8)

    for bar, acc in zip(bars1, v_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, acc in zip(bars2, v_free):
        if acc > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{acc:.1%}", ha="center", va="bottom", fontsize=8, color="#d62728")

    # Add horizontal reference line at 40%
    ax.axhline(y=0.40, color="gray", linestyle="--", alpha=0.5, label="40% reference")

    ax.set_xticks(x)
    ax.set_xticklabels(vanilla_labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Vanilla Models: Logit-Only vs Free Generation Accuracy\n"
                 "(Behavioral discrimination exists at ALL scales — generation is the bottleneck)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 0.55)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"format_forced_full_scale.{ext}", dpi=200)
    plt.close()
    print("  Saved format_forced_full_scale figure")


def main():
    print("=" * 70)
    print("Format-Forced Decoding — Full Scale Universality + Coherence Test")
    print("=" * 70)

    # Load probes
    behavior = BiasBehavior()
    probes = behavior.load_probes(n=300, seed=42, include_bridges=True)
    print(f"\nLoaded {len(probes)} bias probes")

    all_results = {}

    # ---- Part 1: Logit accuracy across ALL vanilla scales ----
    print("\n" + "=" * 70)
    print("PART 1: UNIVERSALITY — Logit-only accuracy across all vanilla models")
    print("=" * 70)

    for cond_label, info in VANILLA_CONDITIONS.items():
        print(f"\n  --- {cond_label} ---")
        model, tokenizer = load_model(info["dirname"])

        logit_summary = run_logit_classification(model, tokenizer, probes, cond_label)
        free_summary = run_free_generation(model, tokenizer, probes, cond_label)

        all_results[cond_label] = {
            "type": "vanilla",
            "dirname": info["dirname"],
            "logit": logit_summary,
            "free": free_summary,
        }
        del model

    # ---- Part 1b: Trained models for comparison ----
    print("\n" + "=" * 70)
    print("PART 1b: TRAINED MODELS — Logit accuracy for comparison")
    print("=" * 70)

    for cond_label, info in TRAINED_CONDITIONS.items():
        print(f"\n  --- {cond_label} ---")
        model, tokenizer = load_model(info["dirname"])

        logit_summary = run_logit_classification(model, tokenizer, probes, cond_label)
        free_summary = run_free_generation(model, tokenizer, probes, cond_label)

        all_results[cond_label] = {
            "type": "trained",
            "dirname": info["dirname"],
            "logit": logit_summary,
            "free": free_summary,
        }
        del model

    # ---- Part 2: Coherence test ----
    print("\n" + "=" * 70)
    print("PART 2: COHERENCE — Force correct answer, check continuation")
    print("=" * 70)

    coherence_results = {}
    coherence_examples = {}

    # Test coherence on key models: d=88 syco_only (the "paralyzed" one),
    # d=96 syco_only (the "working" one), d=64 vanilla (smallest),
    # 7M vanilla, and 64M vanilla (largest)
    COHERENCE_MODELS = {
        "3M vanilla (d=64)":  {"dirname": "3M_seed42"},
        "4.5M syco (d=88)":  {"dirname": "4.5M_seed42_contr_syc_r20"},
        "5M syco (d=96)":    {"dirname": "5M_seed42_contr_syc_r20"},
        "5M vanilla (d=96)": {"dirname": "5M_seed42"},
        "7M vanilla (d=128)": {"dirname": "7M_seed42"},
        "64M vanilla":       {"dirname": "64M_seed42"},
    }

    for cond_label, info in COHERENCE_MODELS.items():
        print(f"\n  --- {cond_label} ---")
        model, tokenizer = load_model(info["dirname"])

        summary, correct_examples, wrong_examples = run_coherence_test(
            model, tokenizer, probes, cond_label, N_COHERENCE_EXAMPLES
        )
        coherence_results[cond_label] = summary
        coherence_examples[cond_label] = {
            "correct_forced": correct_examples[:5],  # Save 5 examples each
            "biased_forced": wrong_examples[:5],
        }
        del model

    # ---- Save results ----
    out = {
        "universality": {},
        "coherence": coherence_results,
        "coherence_examples": coherence_examples,
    }

    for k, v in all_results.items():
        out["universality"][k] = {
            "type": v["type"],
            "dirname": v["dirname"],
            "logit_accuracy": v["logit"]["logit_accuracy"],
            "logit_biased_rate": v["logit"]["biased_rate"],
            "free_accuracy": v["free"]["free_accuracy"],
            "free_parse_rate": v["free"]["parse_rate"],
            "free_unparsed_rate": v["free"]["unparsed_rate"],
            "n": v["logit"]["n"],
        }

    out_path = OUT_DIR / "format_forced_full_scale.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ---- Plot ----
    print("\nPlotting...")
    plot_universality(all_results)

    # ---- Summary tables ----
    print("\n" + "=" * 90)
    print("UNIVERSALITY TABLE — Logit-Only Accuracy Across All Scales")
    print("=" * 90)
    print(f"{'Model':<25} {'Type':<10} {'Logit Acc':>10} {'Free Acc':>10} "
          f"{'Parse Rate':>10} {'Biased':>8}")
    print("-" * 90)
    for k in list(VANILLA_CONDITIONS.keys()) + list(TRAINED_CONDITIONS.keys()):
        r = all_results[k]
        model_type = r["type"]
        logit_acc = r["logit"]["logit_accuracy"]
        free_acc = r["free"]["free_accuracy"]
        parse_rate = r["free"]["parse_rate"]
        biased = r["logit"]["biased_rate"]
        print(f"{k:<25} {model_type:<10} {logit_acc:>10.3f} {free_acc:>10.3f} "
              f"{parse_rate:>10.3f} {biased:>8.3f}")

    print("\n" + "=" * 90)
    print("COHERENCE TABLE — Forced Correct Answer Continuation Quality")
    print("=" * 90)
    print(f"{'Model':<25} {'Unique Tok':>10} {'Repetition':>12} {'Has EOS':>10} "
          f"{'Avg Length':>10}")
    print("-" * 90)
    for k, s in coherence_results.items():
        print(f"{k:<25} {s['avg_unique_token_ratio']:>10.3f} "
              f"{s['pct_with_repetition']:>12.1%} {s['pct_with_eos']:>10.1%} "
              f"{s['avg_text_length']:>10.0f}")

    # Print some example continuations
    print("\n" + "=" * 90)
    print("COHERENCE EXAMPLES — First 3 probes per model")
    print("=" * 90)
    for model_label, examples in coherence_examples.items():
        print(f"\n  === {model_label} ===")
        for ex in examples["correct_forced"][:3]:
            print(f"    Probe {ex['probe_idx']} (correct={ex['correct_answer']}): "
                  f"\"{ex['forced_continuation'][:100]}\"")
        if examples["biased_forced"]:
            print(f"    [biased forced]:")
            for ex in examples["biased_forced"][:2]:
                print(f"    Probe {ex['probe_idx']} (biased={ex['biased_answer']}): "
                      f"\"{ex['forced_continuation'][:100]}\"")

    print("\nDone.")


if __name__ == "__main__":
    main()
