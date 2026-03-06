#!/usr/bin/env python3
"""Logit Lens Readout Gate — v2: Actual probe hidden states through W_U.

The right test: run actual bias probes through the model, capture the
hidden state at EACH layer, project through W_U, and check whether the
correct answer token beats the biased answer token in the logits.

This is the true logit lens: does the model's internal representation at
layer L, when decoded, favor the correct answer?

At 3M (d=64): the model may have the bias direction internally but
the unembedding matrix can't resolve it → correct answer doesn't win in
logit space → model is "mute."

At 5M (d=96): same direction becomes resolvable → correct answer wins →
cross-transfer emerges.

We measure:
1. Per-layer "logit accuracy" — fraction of probes where correct_token > biased_token
2. Per-layer logit gap — mean(logit[correct] - logit[biased])
3. Per-layer "decodability" — how many std devs the correct-biased gap is from zero
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from rho_eval.interpretability.activation import LayerActivationCapture
from rho_eval.behavioral import load_behavioral_probes
from rho_eval.utils import get_layers

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SCALE_DIR = ROOT / "results" / "scale_ladder"
OUT_DIR = ROOT / "results" / "developmental_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "3M": {
        "vanilla": "3M_seed42",
        "bias_only": "3M_seed42_contr_bia_r20",
        "syco_only": "3M_seed42_contr_syc_r20",
        "d_model": 64,
        "n_layers": 2,
    },
    "5M": {
        "vanilla": "5M_seed42",
        "bias_only": "5M_seed42_contr_bia_r20",
        "syco_only": "5M_seed42_contr_syc_r20",
        "d_model": 96,
        "n_layers": 2,
    },
    "7M": {
        "vanilla": "7M_seed42",
        "bias_only": "7M_seed42_contr_bia_r20",
        "syco_only": "7M_seed42_contr_syc_r20",
        "d_model": 128,
        "n_layers": 4,
    },
}


def load_model(dirname):
    """Load a GPT-2 model from scale_ladder."""
    path = SCALE_DIR / dirname / "model"
    model = AutoModelForCausalLM.from_pretrained(str(path))
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def get_answer_token_id(tokenizer, answer_letter):
    """Get the token ID for an answer letter (space-prefixed)."""
    # Bias probes end with "Answer:\n" and the answer is " A", " B", or " C"
    token_id = tokenizer.encode(f" {answer_letter}", add_special_tokens=False)
    if len(token_id) == 1:
        return token_id[0]
    # Fallback: bare letter
    token_id = tokenizer.encode(answer_letter, add_special_tokens=False)
    return token_id[0] if token_id else None


@torch.no_grad()
def logit_lens_probes(model, tokenizer, probes, n_layers):
    """Run bias probes through the model and decode at each layer.

    For each probe:
    1. Tokenize the prompt (without the answer)
    2. Run forward pass, capture hidden states at each layer
    3. At each layer, project h_last through W_U to get logits
    4. Check: does logit[correct_answer] > logit[biased_answer]?

    Returns per-layer metrics.
    """
    W_U = model.lm_head.weight.detach().float()  # (vocab_size, d_model)
    all_layers = list(range(n_layers))

    # Pre-compute answer token IDs
    token_map = {}
    for letter in ["A", "B", "C"]:
        tid = get_answer_token_id(tokenizer, letter)
        token_map[letter] = tid

    # Metrics per layer
    layer_gaps = {l: [] for l in all_layers}  # logit[correct] - logit[biased]
    layer_correct = {l: 0 for l in all_layers}  # count where correct > biased
    layer_total = {l: 0 for l in all_layers}

    # Also track the full logit distribution stats per layer
    layer_answer_z_gaps = {l: [] for l in all_layers}

    cap = LayerActivationCapture(model, all_layers)

    n_valid = 0
    for i, probe in enumerate(probes):
        correct_letter = probe["correct_answer"]
        biased_letter = probe["biased_answer"]

        correct_tid = token_map.get(correct_letter)
        biased_tid = token_map.get(biased_letter)
        if correct_tid is None or biased_tid is None:
            continue

        # Tokenize the prompt text (the question, WITHOUT the answer)
        text = probe["text"]
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        )

        # Forward pass
        model(**inputs)

        for layer_idx in all_layers:
            h = cap.get(layer_idx)  # (1, seq_len, d_model)
            h_last = h[0, -1, :].float()  # (d_model,)

            # Project through unembedding
            logits = W_U @ h_last  # (vocab_size,)

            # Get answer token logits
            logit_correct = logits[correct_tid].item()
            logit_biased = logits[biased_tid].item()
            gap = logit_correct - logit_biased

            layer_gaps[layer_idx].append(gap)
            if gap > 0:
                layer_correct[layer_idx] += 1
            layer_total[layer_idx] += 1

            # Z-score of the gap relative to the full logit distribution
            logit_std = logits.std().item()
            z_gap = gap / logit_std if logit_std > 0 else 0
            layer_answer_z_gaps[layer_idx].append(z_gap)

        cap.clear()
        n_valid += 1

        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(probes)} probes", flush=True)

    cap.remove()

    # Compile results
    results = {"n_probes": n_valid, "layers": {}}

    for layer_idx in all_layers:
        gaps = np.array(layer_gaps[layer_idx])
        z_gaps = np.array(layer_answer_z_gaps[layer_idx])
        n = layer_total[layer_idx]

        if n == 0:
            continue

        accuracy = layer_correct[layer_idx] / n
        mean_gap = gaps.mean()
        std_gap = gaps.std()
        mean_z_gap = z_gaps.mean()
        std_z_gap = z_gaps.std()

        # T-statistic: is the mean gap significantly different from 0?
        t_stat = mean_gap / (std_gap / np.sqrt(n)) if std_gap > 0 else 0

        # Decodability: mean z-gap (how many noise-stds the signal is)
        decodability = mean_z_gap

        results["layers"][str(layer_idx)] = {
            "logit_accuracy": round(accuracy, 4),
            "mean_gap": round(float(mean_gap), 6),
            "std_gap": round(float(std_gap), 6),
            "mean_z_gap": round(float(mean_z_gap), 6),
            "std_z_gap": round(float(std_z_gap), 6),
            "t_statistic": round(float(t_stat), 4),
            "decodability": round(float(decodability), 6),
            "n": n,
            # Distribution of gaps
            "gap_percentiles": {
                "p10": round(float(np.percentile(gaps, 10)), 6),
                "p25": round(float(np.percentile(gaps, 25)), 6),
                "p50": round(float(np.percentile(gaps, 50)), 6),
                "p75": round(float(np.percentile(gaps, 75)), 6),
                "p90": round(float(np.percentile(gaps, 90)), 6),
            },
        }

        print(f"    L{layer_idx}: accuracy={accuracy:.1%}, "
              f"mean_gap={mean_gap:.4f}, z_gap={mean_z_gap:.4f}, "
              f"t={t_stat:.2f}, decodability={decodability:.4f}",
              flush=True)

    return results


def analyze_scale(scale_name, model_info):
    """Full logit lens analysis for one scale."""
    print(f"\n{'='*70}")
    print(f"  SCALE: {scale_name}  (d={model_info['d_model']}, "
          f"layers={model_info['n_layers']})")
    print(f"{'='*70}")

    # Load probes once
    probes = load_behavioral_probes("bias", seed=42)
    print(f"  {len(probes)} bias probes loaded")

    results = {
        "scale": scale_name,
        "d_model": model_info["d_model"],
        "n_layers": model_info["n_layers"],
        "conditions": {},
    }

    for condition in ["vanilla", "bias_only", "syco_only"]:
        dirname = model_info[condition]
        print(f"\n  [{condition}] Loading {dirname}...")
        model, tokenizer = load_model(dirname)

        print(f"    Running logit lens on bias probes...")
        condition_result = logit_lens_probes(
            model, tokenizer, probes, model_info["n_layers"],
        )
        results["conditions"][condition] = condition_result
        del model, tokenizer

    return results


def main():
    all_results = {}

    for scale_name in ["3M", "5M", "7M"]:
        model_info = MODELS[scale_name]
        all_results[scale_name] = analyze_scale(scale_name, model_info)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  READOUT GATE — LOGIT LENS SUMMARY (actual probe hidden states)")
    print(f"{'='*70}")
    print()

    cross_rho = {"3M": 0.000, "5M": 0.292, "7M": 0.208}

    print(f"  {'Scale':<6s} {'d':>4s} {'Cond':<12s} {'Layer':<6s} "
          f"{'Acc':>7s} {'MeanGap':>10s} {'Z-Gap':>8s} {'t-stat':>8s} {'Cross-ρ':>8s}")
    print(f"  {'-'*72}")

    for scale_name in ["3M", "5M", "7M"]:
        r = all_results[scale_name]
        d = r["d_model"]
        for cond in ["vanilla", "bias_only", "syco_only"]:
            cr = r["conditions"][cond]
            for layer_str, lr in sorted(cr["layers"].items()):
                print(f"  {scale_name:<6s} {d:>4d} {cond:<12s} L{layer_str:<4s} "
                      f"{lr['logit_accuracy']:>6.1%} "
                      f"{lr['mean_gap']:>10.4f} "
                      f"{lr['mean_z_gap']:>8.4f} "
                      f"{lr['t_statistic']:>8.2f} "
                      f"{cross_rho[scale_name]:>8.3f}")
        print()

    # ── The key comparison: syco_only model, last layer ───────────────
    print(f"\n  KEY COMPARISON: syco_only model → bias probe logit accuracy")
    print(f"  (Does the syco-trained model decode the RIGHT bias answer?)")
    print(f"  {'-'*60}")
    for scale_name in ["3M", "5M", "7M"]:
        r = all_results[scale_name]
        d = r["d_model"]
        last = str(r["n_layers"] - 1)
        lr = r["conditions"]["syco_only"]["layers"][last]
        print(f"  {scale_name} (d={d:>3d}): "
              f"acc={lr['logit_accuracy']:.1%}, "
              f"gap={lr['mean_gap']:.4f}, "
              f"z_gap={lr['mean_z_gap']:.4f}, "
              f"t={lr['t_statistic']:.2f} "
              f"| cross-ρ={cross_rho[scale_name]:.3f}")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = OUT_DIR / "logit_lens_readout_gate_v2.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
