#!/usr/bin/env python3
"""Activation Steering — Injecting Concentration Externally via Hooks.

Extract bias/syco activation-difference vectors from the trained d=96 (5M)
syco_only model, then add a scaled version to the residual stream of the
d=88 model during autoregressive generation. This injects the "slack"
direction externally.

The steering vector is computed as:
  steering = mean(h[correct_probes]) - mean(h[biased_probes])
where h are the hidden states at the last token position from the
d=96 syco_only model on probes where it gets the answer right.

For d=88 (d_model=88), the d=96 steering vector (d_model=96) needs
dimension matching. We extract vectors from d=88's OWN activation space
using the same method, plus cross-width projection via PCA alignment.

Tests (all on 300 bias probes):
  1. d=88 syco_only + self-steering (vector from d=88's own activations)
  2. d=96 syco_only + self-steering (sanity check)
  Strengths: 0.0 (baseline), 0.3, 0.5, 0.8, 1.0, 1.5, 2.0
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

CONDITIONS = {
    "d=88": {
        "dirname": "4.5M_seed42_contr_syc_r20",
        "d_model": 88,
    },
    "d=96": {
        "dirname": "5M_seed42_contr_syc_r20",
        "d_model": 96,
    },
}

STRENGTHS = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
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


def extract_steering_vector(model, tokenizer, probes, d_model):
    """Extract steering vector from model's own activations.

    Run all probes, get hidden states at last token (last layer output).
    Classify based on logit preference (correct vs biased).
    Steering vector = mean(correct_preferred) - mean(biased_preferred).
    """
    tok_A = get_answer_token_id(tokenizer, "A")
    tok_B = get_answer_token_id(tokenizer, "B")
    tok_C = get_answer_token_id(tokenizer, "C")
    abc_toks = [tok_A, tok_B, tok_C]

    correct_hidden = []
    biased_hidden = []

    for probe in probes:
        inputs = tokenizer(
            probe["text"], return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            out = model(inputs["input_ids"], output_hidden_states=True)

        # Get last hidden layer, last token
        h = out.hidden_states[-1][0, -1, :].cpu().numpy()

        # Classify by logit preference
        logits = out.logits[0, -1, :]
        letter_to_idx = {"A": 0, "B": 1, "C": 2}
        correct_tok = abc_toks[letter_to_idx[probe["correct_answer"]]]
        biased_answer = probe.get("biased_answer", "")
        if biased_answer in letter_to_idx:
            biased_tok = abc_toks[letter_to_idx[biased_answer]]
        else:
            biased_tok = abc_toks[0]  # fallback

        if logits[correct_tok].item() > logits[biased_tok].item():
            correct_hidden.append(h)
        else:
            biased_hidden.append(h)

    correct_mean = np.mean(correct_hidden, axis=0)
    biased_mean = np.mean(biased_hidden, axis=0)
    steering = correct_mean - biased_mean

    # Normalize to unit length
    norm = np.linalg.norm(steering)
    if norm > 1e-12:
        steering = steering / norm

    print(f"    Steering vector: {len(correct_hidden)} correct-preferred, "
          f"{len(biased_hidden)} biased-preferred, ||v||={norm:.4f}")

    return steering


def steered_generate(model, tokenizer, prompt, steering_vector, strength,
                     layer_idx, gen_steps=5):
    """Autoregressive generation with activation steering via hooks.

    At each generation step, add steering_vector * strength to the
    residual stream at the specified layer.
    """
    # Register hook
    hooks = []
    steering_tensor = torch.tensor(steering_vector, dtype=torch.float32)

    def hook_fn(module, input, output):
        # output is a tuple; first element is the hidden states
        if isinstance(output, tuple):
            hidden = output[0]
            # Add steering to last token position
            hidden[:, -1, :] = hidden[:, -1, :] + strength * steering_tensor
            return (hidden,) + output[1:]
        else:
            output[:, -1, :] = output[:, -1, :] + strength * steering_tensor
            return output

    # Hook into the last transformer block's output
    # GPT-2 structure: model.transformer.h[layer_idx]
    target_layer = model.transformer.h[layer_idx]
    hook = target_layer.register_forward_hook(hook_fn)
    hooks.append(hook)

    try:
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

        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()

    return generated_text, gen_tokens


def run_condition(model, tokenizer, probes, steering_vector, strength, label,
                  layer_idx):
    """Run activation steering on all probes for a given strength."""
    results = []

    for i, probe in enumerate(probes):
        if strength == 0.0:
            # Baseline: free generation without hooks
            inputs = tokenizer(
                probe["text"], return_tensors="pt", truncation=True, max_length=512
            )
            input_ids = inputs["input_ids"]
            gen_tokens = []
            for step in range(GEN_STEPS):
                with torch.no_grad():
                    out = model(input_ids)
                next_token = out.logits[0, -1, :].argmax().item()
                gen_tokens.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        else:
            gen_text, gen_tokens = steered_generate(
                model, tokenizer, probe["text"], steering_vector, strength,
                layer_idx, GEN_STEPS,
            )

        answer = parse_answer(gen_text, probe)
        outcome = classify_outcome(answer, probe)
        results.append(outcome)

    n = len(results)
    n_correct = sum(1 for r in results if r == "correct")
    n_biased = sum(1 for r in results if r == "biased")
    n_neutral = sum(1 for r in results if r == "neutral")
    n_unparsed = sum(1 for r in results if r == "unparsed")

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

    print(f"  {label} s={strength:.1f}: "
          f"acc={summary['accuracy']:.3f} ({n_correct}/{n}), "
          f"biased={summary['biased_rate']:.3f}, "
          f"parse={summary['parse_rate']:.3f}")

    return summary


def plot_results(all_results):
    """Line plot of accuracy vs steering strength for both conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    for ax, cond_label in zip(axes, ["d=88", "d=96"]):
        cond_data = all_results[cond_label]

        accs = [cond_data[s]["accuracy"] for s in STRENGTHS]
        parse_rates = [cond_data[s]["parse_rate"] for s in STRENGTHS]
        biased_rates = [cond_data[s]["biased_rate"] for s in STRENGTHS]

        ax.plot(STRENGTHS, accs, "o-", color="#2ca02c", linewidth=2,
                markersize=8, label="Correct (ρ)", zorder=3)
        ax.plot(STRENGTHS, parse_rates, "s--", color="#1f77b4", linewidth=1.5,
                markersize=6, label="Parseable", alpha=0.7)
        ax.plot(STRENGTHS, biased_rates, "^--", color="#d62728", linewidth=1.5,
                markersize=6, label="Biased", alpha=0.7)

        # Annotate accuracy values
        for s, acc in zip(STRENGTHS, accs):
            if acc > 0.005:
                ax.annotate(f"{acc:.1%}", (s, acc), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")

        ax.set_xlabel("Steering Strength", fontsize=12)
        ax.set_ylabel("Rate" if ax == axes[0] else "", fontsize=12)
        ax.set_title(f"{cond_label} syco_only + Self-Steering",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)

    fig.suptitle("Activation Steering: Injecting Concentration Direction into Residual Stream",
                 fontsize=14)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"activation_steering.{ext}", dpi=200)
    plt.close()
    print("  Saved activation_steering figure")


def main():
    print("=" * 60)
    print("Activation Steering — Inject Concentration Externally")
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

        # Extract steering vector from this model's own activations
        print(f"  Extracting steering vector...")
        steering = extract_steering_vector(model, tokenizer, probes, info["d_model"])

        # Hook into last transformer layer (layer 1 for 2-layer models)
        layer_idx = 1  # Last transformer block

        cond_results = {}
        for strength in STRENGTHS:
            summary = run_condition(
                model, tokenizer, probes, steering, strength,
                cond_label, layer_idx,
            )
            cond_results[strength] = summary

        all_results[cond_label] = cond_results
        del model

    # Save results
    def jsonify(obj):
        if isinstance(obj, dict):
            return {str(k): jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    out_path = OUT_DIR / "activation_steering.json"
    with open(out_path, "w") as f:
        json.dump(jsonify(all_results), f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot
    print("\nPlotting...")
    plot_results(all_results)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Condition':<12} {'Strength':>8} {'Accuracy':>10} {'Biased':>8} "
          f"{'Parseable':>10} {'Unparsed':>10}")
    print("-" * 70)
    for cond_label in ["d=88", "d=96"]:
        for strength in STRENGTHS:
            s = all_results[cond_label][strength]
            print(f"{cond_label:<12} {strength:>8.1f} "
                  f"{s['accuracy']:>10.3f} {s['biased_rate']:>8.3f} "
                  f"{s['parse_rate']:>10.3f} {s['unparsed_rate']:>10.3f}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
