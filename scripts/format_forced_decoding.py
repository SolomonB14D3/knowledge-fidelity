#!/usr/bin/env python3
"""Format-Forced Decoding — Bypassing the Broken Generation Circuit.

At inference time, mask all logits except {"A", "B", "C"} for the first
generation token. This bypasses the format generation bottleneck entirely
and directly tests whether the model's internal preferences (which we know
are identical at d=88 and d=96 from confidence_shift.py) can produce
correct answers when the "paralyzed mouth" is forced open.

Tests:
  d=88 syco_only: baseline (free generation) vs format-forced (A/B/C mask)
  d=96 syco_only: baseline vs format-forced
  d=88 vanilla: format-forced (does vanilla have the signal too?)
  d=96 vanilla: format-forced

If format-forcing rescues d=88 to match d=96, it proves the bottleneck
is 100% in the generation circuit's inability to emit format tokens.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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

# Models to test
# Note: no vanilla 4.5M exists; closest is 4M_seed42 (d=80)
CONDITIONS = {
    "d=88 syco_only": {"dirname": "4.5M_seed42_contr_syc_r20", "d_model": 88},
    "d=96 syco_only": {"dirname": "5M_seed42_contr_syc_r20", "d_model": 96},
    "d=80 vanilla":   {"dirname": "4M_seed42",                "d_model": 80},
    "d=96 vanilla":   {"dirname": "5M_seed42",                "d_model": 96},
}

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


def forced_generate(model, tokenizer, prompt, abc_ids, gen_steps=5):
    """Format-forced generation: first token masked to A/B/C only.

    After the first (forced) token, generation proceeds freely.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]

    gen_tokens = []
    for step in range(gen_steps):
        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits[0, -1, :]

        if step == 0:
            # Mask everything except A, B, C tokens
            mask = torch.full_like(logits, float("-inf"))
            for tid in abc_ids:
                mask[tid] = logits[tid]
            next_token = mask.argmax().item()
        else:
            next_token = logits.argmax().item()

        gen_tokens.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

    return tokenizer.decode(gen_tokens, skip_special_tokens=True), gen_tokens


def logit_only_classify(model, tokenizer, prompt, abc_ids, probe):
    """Pure logit classification: pick highest-logit A/B/C token. No generation."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model(inputs["input_ids"])
        logits = out.logits[0, -1, :]

    abc_logits = [logits[tid].item() for tid in abc_ids]
    best_idx = int(np.argmax(abc_logits))
    answer = "ABC"[best_idx]
    return answer


def run_condition(model, tokenizer, probes, label):
    """Run all three modes (free, forced, logit-only) on all probes."""
    tok_A = get_answer_token_id(tokenizer, "A")
    tok_B = get_answer_token_id(tokenizer, "B")
    tok_C = get_answer_token_id(tokenizer, "C")
    abc_ids = [tok_A, tok_B, tok_C]

    results = {"free": [], "forced": [], "logit": []}

    for i, probe in enumerate(probes):
        # Free generation
        free_text, free_tokens = free_generate(model, tokenizer, probe["text"], GEN_STEPS)
        free_answer = parse_answer(free_text, probe)
        free_outcome = classify_outcome(free_answer, probe)

        # Format-forced generation
        forced_text, forced_tokens = forced_generate(model, tokenizer, probe["text"], abc_ids, GEN_STEPS)
        forced_answer = parse_answer(forced_text, probe)
        forced_outcome = classify_outcome(forced_answer, probe)

        # Pure logit classification (no generation at all)
        logit_answer = logit_only_classify(model, tokenizer, probe["text"], abc_ids, probe)
        logit_outcome = classify_outcome(logit_answer, probe)

        results["free"].append(free_outcome)
        results["forced"].append(forced_outcome)
        results["logit"].append(logit_outcome)

    # Summarize
    summaries = {}
    for mode in ["free", "forced", "logit"]:
        n = len(results[mode])
        n_correct = sum(1 for r in results[mode] if r == "correct")
        n_biased = sum(1 for r in results[mode] if r == "biased")
        n_neutral = sum(1 for r in results[mode] if r == "neutral")
        n_unparsed = sum(1 for r in results[mode] if r == "unparsed")

        summaries[mode] = {
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

        print(f"  {label} [{mode}]: "
              f"acc={summaries[mode]['accuracy']:.3f} ({n_correct}/{n}), "
              f"biased={summaries[mode]['biased_rate']:.3f}, "
              f"parse={summaries[mode]['parse_rate']:.3f}")

    return summaries


def plot_results(all_results):
    """Grouped bar chart: free vs forced vs logit across all conditions."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    conditions = list(all_results.keys())
    modes = ["free", "forced", "logit"]
    mode_labels = ["Free Generation", "Format-Forced", "Logit-Only"]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]

    x = np.arange(len(conditions))
    width = 0.25

    for j, (mode, mlabel, color) in enumerate(zip(modes, mode_labels, colors)):
        accs = [all_results[c][mode]["accuracy"] for c in conditions]
        bars = ax.bar(x + j * width - width, accs, width, label=mlabel,
                      color=color, edgecolor="white")
        for bar, acc in zip(bars, accs):
            if acc > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{acc:.1%}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (correct / total)", fontsize=12)
    ax.set_title("Format-Forced Decoding: Bypassing the Generation Bottleneck",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 0.55)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"format_forced_decoding.{ext}", dpi=200)
    plt.close()
    print("  Saved format_forced_decoding figure")


def main():
    print("=" * 60)
    print("Format-Forced Decoding — Bypass the Paralyzed Mouth")
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

        model, tokenizer = load_model(info["dirname"])
        summaries = run_condition(model, tokenizer, probes, cond_label)
        all_results[cond_label] = summaries

        del model

    # Save results
    out_path = OUT_DIR / "format_forced_decoding.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot
    print("\nPlotting...")
    plot_results(all_results)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Condition':<20} {'Mode':<12} {'Accuracy':>10} {'Biased':>8} "
          f"{'Parseable':>10} {'Unparsed':>10}")
    print("-" * 80)
    for cond_label in CONDITIONS:
        for mode in ["free", "forced", "logit"]:
            s = all_results[cond_label][mode]
            print(f"{cond_label:<20} {mode:<12} "
                  f"{s['accuracy']:>10.3f} {s['biased_rate']:>8.3f} "
                  f"{s['parse_rate']:>10.3f} {s['unparsed_rate']:>10.3f}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
