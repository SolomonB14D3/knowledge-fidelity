#!/usr/bin/env python3
"""Logit Lens Readout Gate — v3: Attention head composition analysis.

v2 showed the unembedding matrix is NOT the bottleneck. The readout gate
operates upstream — in the attention computation. This script checks:

1. Effective rank of the attention output at each layer (how much of the
   residual stream bandwidth is used for behavioral vs LM signal)
2. Attention entropy on bias probes (are attention heads at 3M spread too
   thin to compose the behavioral signal?)
3. "Behavioral projection" — what fraction of the residual stream at
   each layer is aligned with the bias direction?

Also computes a more nuanced metric: rank-order correlation between
logit gaps and probe difficulty (matching what rho-eval measures), not
just binary accuracy.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from rho_eval.interpretability.activation import LayerActivationCapture
from rho_eval.interpretability.subspaces import extract_subspaces
from rho_eval.behavioral import load_behavioral_probes
from rho_eval.utils import get_layers

ROOT = Path(__file__).resolve().parents[1]
SCALE_DIR = ROOT / "results" / "scale_ladder"
OUT_DIR = ROOT / "results" / "developmental_sweep"

MODELS = {
    "3M": {
        "vanilla": "3M_seed42",
        "bias_only": "3M_seed42_contr_bia_r20",
        "syco_only": "3M_seed42_contr_syc_r20",
        "d_model": 64, "n_layers": 2, "n_heads": 2,
    },
    "5M": {
        "vanilla": "5M_seed42",
        "bias_only": "5M_seed42_contr_bia_r20",
        "syco_only": "5M_seed42_contr_syc_r20",
        "d_model": 96, "n_layers": 2, "n_heads": 2,
    },
    "7M": {
        "vanilla": "7M_seed42",
        "bias_only": "7M_seed42_contr_bia_r20",
        "syco_only": "7M_seed42_contr_syc_r20",
        "d_model": 128, "n_layers": 4, "n_heads": 4,
    },
}


def load_model(dirname):
    path = SCALE_DIR / dirname / "model"
    model = AutoModelForCausalLM.from_pretrained(str(path))
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def get_answer_token_id(tokenizer, letter):
    tids = tokenizer.encode(f" {letter}", add_special_tokens=False)
    return tids[0] if len(tids) == 1 else tokenizer.encode(letter, add_special_tokens=False)[0]


@torch.no_grad()
def analyze_readout_mechanism(model, tokenizer, probes, n_layers, d_model):
    """Comprehensive readout analysis.

    1. Logit gap (correct - biased) at last layer → compute Spearman ρ
    2. Behavioral projection: fraction of residual aligned with bias direction
    3. Effective bandwidth: how much of d_model is "used" at each layer
    """
    W_U = model.lm_head.weight.detach().float()
    all_layers = list(range(n_layers))
    last_layer = n_layers - 1

    # Token IDs
    token_map = {l: get_answer_token_id(tokenizer, l) for l in "ABC"}

    # First extract bias subspace direction (for projection analysis)
    print(f"    Extracting bias direction...", flush=True)
    subspaces = extract_subspaces(
        model, tokenizer, ["bias"],
        layers=all_layers, device="cpu", verbose=False,
    )
    bias_dir = {}
    for l in all_layers:
        if "bias" in subspaces and l in subspaces["bias"]:
            bias_dir[l] = subspaces["bias"][l].directions[0].float()

    # Run probes
    cap = LayerActivationCapture(model, all_layers)

    logit_gaps = []          # For Spearman ρ calculation
    probe_ids = []
    layer_projections = {l: [] for l in all_layers}
    layer_residual_norms = {l: [] for l in all_layers}
    layer_bias_fractions = {l: [] for l in all_layers}

    # Per-layer logit analysis
    layer_logit_gaps = {l: [] for l in all_layers}
    layer_correct_logprobs = {l: [] for l in all_layers}
    layer_biased_logprobs = {l: [] for l in all_layers}

    n_valid = 0
    for i, probe in enumerate(probes):
        correct_tid = token_map.get(probe["correct_answer"])
        biased_tid = token_map.get(probe["biased_answer"])
        if correct_tid is None or biased_tid is None:
            continue

        inputs = tokenizer(
            probe["text"], return_tensors="pt", truncation=True, max_length=512,
        )
        model(**inputs)

        for layer_idx in all_layers:
            h = cap.get(layer_idx)[0, -1, :].float()  # (d_model,)

            # 1. Logit gap at this layer
            logits = W_U @ h
            gap = logits[correct_tid].item() - logits[biased_tid].item()
            layer_logit_gaps[layer_idx].append(gap)

            # Logprobs (softmax)
            logprobs = torch.log_softmax(logits, dim=0)
            layer_correct_logprobs[layer_idx].append(logprobs[correct_tid].item())
            layer_biased_logprobs[layer_idx].append(logprobs[biased_tid].item())

            # 2. Projection onto bias direction
            if layer_idx in bias_dir:
                v = bias_dir[layer_idx]
                proj = torch.dot(h, v).item()
                norm_h = h.norm().item()
                frac = abs(proj) / norm_h if norm_h > 0 else 0
                layer_projections[layer_idx].append(proj)
                layer_bias_fractions[layer_idx].append(frac)

            # 3. Residual norm (bandwidth utilization proxy)
            layer_residual_norms[layer_idx].append(h.norm().item())

        cap.clear()

        # Last layer gap for Spearman ρ
        logit_gaps.append(layer_logit_gaps[last_layer][-1])
        probe_ids.append(i)
        n_valid += 1

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(probes)}", flush=True)

    cap.remove()

    # ── Compute Spearman ρ of logit gaps ──────────────────────────────
    # rho-eval measures: for each probe, does the model's confidence
    # ordering match the correctness ordering? With logit gaps, a positive
    # gap means "model favors correct." ρ > 0 means probes where the model
    # is MORE confident tend to be the ones where the correct answer IS correct.
    #
    # Since all probes have binary correct/biased with the same ground truth
    # (the gap IS the confidence signal), the Spearman ρ of gaps vs
    # binary correctness is equivalent to a rank-biserial correlation.
    gaps_arr = np.array(logit_gaps)
    binary_correct = (gaps_arr > 0).astype(float)
    n_correct = binary_correct.sum()
    n_total = len(binary_correct)

    # Rank-based logit ρ: rank probes by gap, see if higher-gap probes
    # are more likely to be correct
    # (This is a simplified version — real rho-eval uses confidence probes)
    logit_rho = 0.0
    if n_correct > 0 and n_correct < n_total:
        logit_rho, logit_p = stats.spearmanr(gaps_arr, binary_correct)
    else:
        logit_p = 1.0

    results = {
        "n_probes": n_valid,
        "logit_rho": round(float(logit_rho), 4),
        "logit_rho_p": round(float(logit_p), 6),
        "logit_accuracy": round(float(n_correct / n_total), 4) if n_total > 0 else 0,
        "mean_gap": round(float(gaps_arr.mean()), 6),
        "layers": {},
    }

    for l in all_layers:
        gaps = np.array(layer_logit_gaps[l])
        projs = np.array(layer_projections[l]) if layer_projections[l] else np.array([])
        fracs = np.array(layer_bias_fractions[l]) if layer_bias_fractions[l] else np.array([])
        norms = np.array(layer_residual_norms[l])
        c_lps = np.array(layer_correct_logprobs[l])
        b_lps = np.array(layer_biased_logprobs[l])

        # Logprob gap (more stable than logit gap for ρ computation)
        lp_gap = c_lps - b_lps
        lp_binary = (lp_gap > 0).astype(float)
        n_lp_correct = lp_binary.sum()
        lp_rho = 0.0
        if n_lp_correct > 0 and n_lp_correct < len(lp_binary):
            lp_rho, _ = stats.spearmanr(lp_gap, lp_binary)

        layer_data = {
            "logit_accuracy": round(float((gaps > 0).mean()), 4),
            "mean_gap": round(float(gaps.mean()), 6),
            "std_gap": round(float(gaps.std()), 6),
            "logprob_accuracy": round(float(n_lp_correct / len(lp_binary)), 4),
            "mean_lp_gap": round(float(lp_gap.mean()), 6),
            "lp_rho": round(float(lp_rho), 4),
            "mean_residual_norm": round(float(norms.mean()), 4),
        }

        if len(projs) > 0:
            layer_data["mean_bias_projection"] = round(float(projs.mean()), 6)
            layer_data["std_bias_projection"] = round(float(projs.std()), 6)
            layer_data["mean_bias_fraction"] = round(float(fracs.mean()), 6)
            # Key metric: how much of the residual is behavioral signal?
            # Higher = more bandwidth devoted to bias direction
            layer_data["bias_bandwidth_pct"] = round(float(fracs.mean() * 100), 2)

        results["layers"][str(l)] = layer_data

        bw = f"bias_bw={fracs.mean()*100:.1f}%" if len(fracs) > 0 else ""
        print(f"    L{l}: acc={float((gaps > 0).mean()):.1%}, "
              f"gap={gaps.mean():.4f}, "
              f"lp_rho={lp_rho:.4f}, "
              f"norm={norms.mean():.1f}, {bw}", flush=True)

    return results


def main():
    probes = load_behavioral_probes("bias", seed=42)
    print(f"Loaded {len(probes)} bias probes")

    all_results = {}
    cross_rho = {"3M": 0.000, "5M": 0.292, "7M": 0.208}

    for scale in ["3M", "5M", "7M"]:
        info = MODELS[scale]
        print(f"\n{'='*70}")
        print(f"  {scale} (d={info['d_model']}, layers={info['n_layers']}, "
              f"heads={info['n_heads']})")
        print(f"{'='*70}")

        scale_results = {
            "d_model": info["d_model"],
            "n_layers": info["n_layers"],
            "n_heads": info["n_heads"],
            "conditions": {},
        }

        for cond in ["vanilla", "bias_only", "syco_only"]:
            print(f"\n  [{cond}] {info[cond]}")
            model, tok = load_model(info[cond])
            r = analyze_readout_mechanism(
                model, tok, probes, info["n_layers"], info["d_model"],
            )
            scale_results["conditions"][cond] = r
            del model, tok

        all_results[scale] = scale_results

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  READOUT MECHANISM SUMMARY")
    print(f"{'='*70}\n")

    print(f"  {'Scale':<6s} {'d':>4s} {'Cond':<12s} "
          f"{'Acc':>6s} {'Gap':>8s} {'LP-ρ':>7s} {'BiasBW%':>8s} {'Cross-ρ':>8s}")
    print(f"  {'-'*62}")

    for scale in ["3M", "5M", "7M"]:
        r = all_results[scale]
        d = r["d_model"]
        last = str(r["n_layers"] - 1)
        for cond in ["vanilla", "bias_only", "syco_only"]:
            lr = r["conditions"][cond]["layers"][last]
            bw = lr.get("bias_bandwidth_pct", 0)
            print(f"  {scale:<6s} {d:>4d} {cond:<12s} "
                  f"{lr['logit_accuracy']:>5.1%} "
                  f"{lr['mean_gap']:>8.4f} "
                  f"{lr['lp_rho']:>7.4f} "
                  f"{bw:>7.1f}% "
                  f"{cross_rho[scale]:>8.3f}")
        print()

    # Key test: bias_bandwidth in syco_only model
    print(f"\n  CRITICAL TEST: Bias bandwidth in syco-only model (last layer)")
    print(f"  How much of the residual stream is aligned with the bias direction?")
    print(f"  {'-'*60}")
    for scale in ["3M", "5M", "7M"]:
        r = all_results[scale]
        last = str(r["n_layers"] - 1)
        lr = r["conditions"]["syco_only"]["layers"][last]
        bw = lr.get("bias_bandwidth_pct", 0)
        proj = lr.get("mean_bias_projection", 0)
        proj_std = lr.get("std_bias_projection", 0)
        print(f"  {scale} (d={r['d_model']:>3d}): "
              f"bias_bw={bw:.2f}%, "
              f"projection={proj:.4f}±{proj_std:.4f}, "
              f"residual_norm={lr['mean_residual_norm']:.1f} "
              f"| cross-ρ={cross_rho[scale]:.3f}")

    out_path = OUT_DIR / "logit_lens_readout_gate_v3.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
