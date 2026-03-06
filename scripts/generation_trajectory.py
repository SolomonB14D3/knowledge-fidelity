#!/usr/bin/env python3
"""Generation-Time Hidden State Trajectory — d=88 vs d=96.

The confidence shift experiment proved the d=88→d=96 phase transition is
purely FORMAT/EXPRESSION: internal preferences are identical (r=0.997),
but d=88 can't output A/B/C tokens (96% unparsed) while d=96 can (86%
parseable). The divergence MUST happen during autoregressive generation.

This script captures hidden states at each generation step to find
WHERE and HOW generation differs:

1. Manual autoregressive generation (5 steps) with hidden state capture
2. Effective rank of hidden states across 300 probes at each (step, layer)
3. Drift from prompt-end hidden state at each generation step
4. Logit gap evolution during generation (correct - biased)

Produces 4 figures showing the generation-time divergence.
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

MODELS = {
    "d=88": {
        "dirname": "4.5M_seed42_contr_syc_r20",
        "d_model": 88, "n_layers": 2, "n_heads": 2,
    },
    "d=96": {
        "dirname": "5M_seed42_contr_syc_r20",
        "d_model": 96, "n_layers": 2, "n_heads": 2,
    },
}

GEN_STEPS = 5
STEP_LABELS = ["prompt"] + [f"gen_{i}" for i in range(GEN_STEPS)]


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


def parse_answer(generated_tokens, tokenizer, probe):
    """Parse generated tokens into answer letter."""
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    gen_upper = text.upper().strip()

    answer = None
    for letter in ["A", "B", "C"]:
        if gen_upper.startswith(letter) or f"({letter})" in gen_upper:
            answer = letter
            break

    if answer is None and probe.get("answers"):
        for idx, ans_text in enumerate(probe["answers"]):
            if ans_text.lower() in text.lower():
                answer = "ABC"[idx]
                break

    return answer


def effective_rank(H):
    """Compute effective rank of matrix H (n_probes × d_model).

    eff_rank = exp(entropy of normalized singular values).
    """
    S = np.linalg.svd(H, compute_uv=False)
    S = S[S > 1e-12]  # Filter near-zero
    p = S / S.sum()
    return np.exp(-np.sum(p * np.log(p + 1e-30)))


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def collect_generation_trajectory(model, tokenizer, probes, model_info, label):
    """Manual autoregressive generation with hidden state capture."""
    device = "cpu"
    d_model = model_info["d_model"]
    n_layers = model_info["n_layers"]
    n_hidden_layers = n_layers + 1  # embedding + n_layers

    tok_A = get_answer_token_id(tokenizer, "A")
    tok_B = get_answer_token_id(tokenizer, "B")
    tok_C = get_answer_token_id(tokenizer, "C")

    # Storage: hidden_states[step][layer] = list of d_model vectors (one per probe)
    # step 0 = prompt, steps 1..5 = generation
    all_hidden = {step: {layer: [] for layer in range(n_hidden_layers)}
                  for step in range(GEN_STEPS + 1)}

    # Per-probe logit gap at each step (for answer alignment tracking)
    all_logit_gaps = {step: [] for step in range(GEN_STEPS + 1)}

    # Per-probe drift (cosine distance from prompt to each gen step)
    all_drift = {step: {layer: [] for layer in range(n_hidden_layers)}
                 for step in range(1, GEN_STEPS + 1)}

    # Per-probe outcome
    outcomes = []
    generated_tokens_all = []

    for i, probe in enumerate(probes):
        inputs = tokenizer(
            probe["text"], return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        input_ids = inputs["input_ids"]

        # Map correct/biased answers
        letter_to_idx = {"A": 0, "B": 1, "C": 2}
        abc_toks = [tok_A, tok_B, tok_C]
        correct_tok = abc_toks[letter_to_idx[probe["correct_answer"]]]
        biased_tok = abc_toks[letter_to_idx.get(probe.get("biased_answer", ""), 0)]

        # Step 0: prompt-end hidden states
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)

        prompt_hidden = {}
        for layer in range(n_hidden_layers):
            h = out.hidden_states[layer][0, -1, :].cpu().numpy()
            all_hidden[0][layer].append(h)
            prompt_hidden[layer] = h

        # Logit gap at prompt end
        logits = out.logits[0, -1, :]
        gap = (logits[correct_tok] - logits[biased_tok]).item()
        all_logit_gaps[0].append(gap)

        # Steps 1..5: autoregressive generation
        gen_tokens = []
        for step in range(1, GEN_STEPS + 1):
            # Greedy: pick next token from last logits
            next_token = out.logits[0, -1, :].argmax().item()
            gen_tokens.append(next_token)

            # Append to input
            next_ids = torch.tensor([[next_token]], device=device)
            input_ids = torch.cat([input_ids, next_ids], dim=1)

            # Forward pass with hidden states
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)

            # Capture hidden states at last token
            for layer in range(n_hidden_layers):
                h = out.hidden_states[layer][0, -1, :].cpu().numpy()
                all_hidden[step][layer].append(h)

                # Drift from prompt
                drift = 1.0 - cosine_sim(prompt_hidden[layer], h)
                all_drift[step][layer].append(drift)

            # Logit gap at this step
            logits = out.logits[0, -1, :]
            gap = (logits[correct_tok] - logits[biased_tok]).item()
            all_logit_gaps[step].append(gap)

        # Parse final answer
        answer = parse_answer(gen_tokens, tokenizer, probe)
        if answer is None:
            outcome = "unparsed"
        elif answer == probe["correct_answer"]:
            outcome = "correct"
        elif answer == probe.get("biased_answer"):
            outcome = "biased"
        else:
            outcome = "neutral"

        outcomes.append(outcome)
        generated_tokens_all.append(gen_tokens)

        if (i + 1) % 50 == 0:
            n_correct = sum(1 for o in outcomes if o == "correct")
            n_unparsed = sum(1 for o in outcomes if o == "unparsed")
            print(f"  [{label}] {i+1}/{len(probes)} — "
                  f"correct={n_correct}, unparsed={n_unparsed}")

    # ── Compute metrics across all probes ────────────────────────────

    # Effective rank at each (step, layer)
    eff_ranks = {}
    for step in range(GEN_STEPS + 1):
        eff_ranks[step] = {}
        for layer in range(n_hidden_layers):
            H = np.stack(all_hidden[step][layer])  # (300, d_model)
            # Center the matrix (subtract mean) for meaningful SVD
            H_centered = H - H.mean(axis=0, keepdims=True)
            er = effective_rank(H_centered)
            eff_ranks[step][layer] = er

    # Mean drift at each (step, layer)
    mean_drift = {}
    for step in range(1, GEN_STEPS + 1):
        mean_drift[step] = {}
        for layer in range(n_hidden_layers):
            mean_drift[step][layer] = float(np.mean(all_drift[step][layer]))

    # Mean logit gap at each step
    mean_logit_gap = {}
    std_logit_gap = {}
    for step in range(GEN_STEPS + 1):
        mean_logit_gap[step] = float(np.mean(all_logit_gaps[step]))
        std_logit_gap[step] = float(np.std(all_logit_gaps[step]))

    # Summary
    n = len(probes)
    n_correct = sum(1 for o in outcomes if o == "correct")
    n_biased = sum(1 for o in outcomes if o == "biased")
    n_unparsed = sum(1 for o in outcomes if o == "unparsed")

    print(f"\n  Summary ({label}):")
    print(f"    correct={n_correct}/{n} ({n_correct/n:.3f}), "
          f"biased={n_biased}/{n}, unparsed={n_unparsed}/{n}")
    print(f"    Eff rank at prompt (last layer): {eff_ranks[0][n_layers]:.2f} "
          f"({eff_ranks[0][n_layers]/d_model:.3f} normalized)")
    print(f"    Eff rank at gen_0 (last layer): {eff_ranks[1][n_layers]:.2f} "
          f"({eff_ranks[1][n_layers]/d_model:.3f} normalized)")

    return {
        "eff_ranks": eff_ranks,
        "mean_drift": mean_drift,
        "mean_logit_gap": mean_logit_gap,
        "std_logit_gap": std_logit_gap,
        "outcomes": outcomes,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_hidden_layers": n_hidden_layers,
    }


def plot_effective_rank(results):
    """Figure A: Effective rank trajectory — 2 panels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (label, data) in zip(axes, results.items()):
        d = data["d_model"]
        n_hl = data["n_hidden_layers"]
        steps = range(GEN_STEPS + 1)

        layer_names = ["embed"] + [f"L{i}_out" for i in range(data["n_layers"])]

        for layer in range(n_hl):
            ranks = [data["eff_ranks"][s][layer] / d for s in steps]
            ax.plot(steps, ranks, "o-", label=layer_names[layer],
                    linewidth=2, markersize=5)

        ax.set_xticks(list(steps))
        ax.set_xticklabels(STEP_LABELS, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Normalized eff. rank (eff_rank / d_model)" if ax == axes[0] else "",
                       fontsize=11)
        ax.set_title(f"{label} (d_model={d})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Effective Rank of Hidden States During Generation", fontsize=14)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"gen_traj_effective_rank.{ext}", dpi=200)
    plt.close()
    print("  Saved effective_rank")


def plot_rank_ratio(results):
    """Figure B: Rank ratio (d=96 / d=88, normalized)."""
    data_88 = results["d=88"]
    data_96 = results["d=96"]
    d_88 = data_88["d_model"]
    d_96 = data_96["d_model"]
    n_hl = data_88["n_hidden_layers"]

    fig, ax = plt.subplots(figsize=(8, 5))

    steps = range(GEN_STEPS + 1)
    layer_names = ["embed"] + [f"L{i}_out" for i in range(data_88["n_layers"])]

    for layer in range(n_hl):
        ratios = []
        for s in steps:
            norm_88 = data_88["eff_ranks"][s][layer] / d_88
            norm_96 = data_96["eff_ranks"][s][layer] / d_96
            ratios.append(norm_96 / norm_88 if norm_88 > 0 else 1.0)
        ax.plot(steps, ratios, "o-", label=layer_names[layer],
                linewidth=2, markersize=6)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(list(steps))
    ax.set_xticklabels(STEP_LABELS, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Normalized rank ratio: (d=96 / d=88)", fontsize=12)
    ax.set_title("Proportional Dimensional Utilization: d=96 vs d=88", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"gen_traj_rank_ratio.{ext}", dpi=200)
    plt.close()
    print("  Saved rank_ratio")


def plot_drift(results):
    """Figure C: Drift from prompt — 2 panels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (label, data) in zip(axes, results.items()):
        n_hl = data["n_hidden_layers"]
        steps = range(1, GEN_STEPS + 1)

        layer_names = ["embed"] + [f"L{i}_out" for i in range(data["n_layers"])]

        for layer in range(n_hl):
            drift = [data["mean_drift"][s][layer] for s in steps]
            ax.plot(list(steps), drift, "o-", label=layer_names[layer],
                    linewidth=2, markersize=5)

        ax.set_xticks(list(steps))
        ax.set_xticklabels([f"gen_{i}" for i in range(GEN_STEPS)],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean drift (1 - cosine sim to prompt)" if ax == axes[0] else "",
                       fontsize=11)
        ax.set_title(f"{label}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Hidden State Drift from Prompt During Generation", fontsize=14)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"gen_traj_drift.{ext}", dpi=200)
    plt.close()
    print("  Saved drift")


def plot_logit_evolution(results):
    """Figure D: Logit gap evolution during generation — 2 panels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (label, data) in zip(axes, results.items()):
        steps = range(GEN_STEPS + 1)
        means = [data["mean_logit_gap"][s] for s in steps]
        stds = [data["std_logit_gap"][s] for s in steps]

        ax.errorbar(list(steps), means, yerr=stds,
                     fmt="o-", linewidth=2, markersize=6, capsize=4,
                     color="#1f77b4", ecolor="#aaaaaa")

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xticks(list(steps))
        ax.set_xticklabels(STEP_LABELS, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean logit gap (correct - biased)" if ax == axes[0] else "",
                       fontsize=11)
        ax.set_title(f"{label}", fontsize=12)
        ax.grid(alpha=0.3)

    fig.suptitle("Logit Gap Evolution During Autoregressive Generation", fontsize=14)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"gen_traj_logit_evolution.{ext}", dpi=200)
    plt.close()
    print("  Saved logit_evolution")


def main():
    print("=" * 60)
    print("Generation-Time Hidden State Trajectory — d=88 vs d=96")
    print("=" * 60)

    # Load probes
    behavior = BiasBehavior()
    probes = behavior.load_probes(n=300, seed=42, include_bridges=True)
    print(f"\nLoaded {len(probes)} bias probes")

    # Collect data for each model
    results = {}
    for label, info in MODELS.items():
        print(f"\n{'='*40}")
        print(f"  {label} (d_model={info['d_model']})")
        print(f"{'='*40}")

        model, tokenizer = load_model(info["dirname"])
        data = collect_generation_trajectory(model, tokenizer, probes, info, label)
        results[label] = data
        del model

    # Save raw metrics (convert int keys to strings for JSON)
    def jsonify(obj):
        if isinstance(obj, dict):
            return {str(k): jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = OUT_DIR / "generation_trajectory.json"
    with open(out_path, "w") as f:
        json.dump(jsonify(results), f, indent=2)
    print(f"\nSaved metrics: {out_path}")

    # ── Figures ──────────────────────────────────────────────────────
    print("\nPlotting...")
    plot_effective_rank(results)
    plot_rank_ratio(results)
    plot_drift(results)
    plot_logit_evolution(results)

    # ── Summary comparison ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EFFECTIVE RANK COMPARISON")
    print("=" * 60)

    for step in range(GEN_STEPS + 1):
        print(f"\n  Step: {STEP_LABELS[step]}")
        for label, data in results.items():
            d = data["d_model"]
            for layer in range(data["n_hidden_layers"]):
                er = data["eff_ranks"][step][layer]
                layer_name = "embed" if layer == 0 else f"L{layer-1}_out"
                print(f"    {label} {layer_name}: eff_rank={er:.2f} "
                      f"(norm={er/d:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
