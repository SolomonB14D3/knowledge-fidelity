#!/usr/bin/env python3
"""Logit Lens / Unembedding Audit — Physical proof of the readout gate.

The readout threshold hypothesis: 3M models (d=64) have the internal geometry
for cross-transfer (bias direction cos 0.76-0.78 with sycophancy direction)
but cannot EXPRESS it through the unembedding matrix. 5M models (d=96) can.

This experiment:
1. Extract the bias/sycophancy direction vectors at each layer
2. Project them through W_U (lm_head) to get logit-space vectors
3. Measure the signal-to-noise ratio for answer tokens (A/B/C)
4. Compare d=64 (3M) vs d=96 (5M) vs d=128 (7M)

If answer-token logits are aliased (indistinguishable from noise) at d=64
but clean at d=96, we have the physical proof of the readout gate:
the 3M model is "mute, not dumb."

Data source: trained model checkpoints in results/scale_ladder/
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from rho_eval.interpretability.subspaces import extract_subspaces
from rho_eval.utils import get_layers

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SCALE_DIR = ROOT / "results" / "scale_ladder"
OUT_DIR = ROOT / "results" / "developmental_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Models to analyze ─────────────────────────────────────────────────
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

# Answer tokens in GPT-2 tokenizer
# " A" = 317, " B" = 347, " C" = 327 (space-prefixed, as in "Answer: A")
# "A" = 32, "B" = 33, "C" = 34 (bare)
ANSWER_TOKENS = {
    " A": 317, " B": 347, " C": 327,
    "A": 32, "B": 33, "C": 34,
}


def load_model(dirname):
    """Load a GPT-2 model from scale_ladder."""
    path = SCALE_DIR / dirname / "model"
    model = AutoModelForCausalLM.from_pretrained(str(path))
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model.eval()
    return model, tokenizer


def get_unembedding(model):
    """Get the unembedding matrix W_U (vocab_size × d_model)."""
    return model.lm_head.weight.detach().float()


def project_direction_to_logits(direction, W_U):
    """Project a hidden-state direction through the unembedding matrix.

    direction: (d_model,) — the behavioral direction vector
    W_U: (vocab_size, d_model) — unembedding matrix

    Returns: (vocab_size,) — logit-space projection
    """
    # logits = W_U @ direction
    return W_U @ direction.float()


def compute_answer_snr(logits, correct_token_id, biased_token_id):
    """Compute signal-to-noise ratio for answer tokens.

    Signal: |logit[correct] - logit[biased]|
    Noise: std(logits) over full vocabulary

    Returns: SNR, signal, noise, logit_correct, logit_biased
    """
    logit_correct = logits[correct_token_id].item()
    logit_biased = logits[biased_token_id].item()
    signal = abs(logit_correct - logit_biased)
    noise = logits.std().item()
    snr = signal / noise if noise > 0 else float('inf')
    return snr, signal, noise, logit_correct, logit_biased


def compute_answer_rank(logits, token_ids):
    """Compute rank of answer tokens in the logit distribution.

    Lower rank = more distinguishable from noise.
    """
    sorted_indices = logits.argsort(descending=True)
    ranks = {}
    for name, tid in token_ids.items():
        rank = (sorted_indices == tid).nonzero(as_tuple=True)[0].item()
        ranks[name] = rank
    return ranks


def analyze_model(scale_name, model_info):
    """Run full logit lens analysis for one scale."""
    print(f"\n{'='*70}")
    print(f"  SCALE: {scale_name}  (d_model={model_info['d_model']}, "
          f"n_layers={model_info['n_layers']})")
    print(f"{'='*70}")

    results = {
        "scale": scale_name,
        "d_model": model_info["d_model"],
        "n_layers": model_info["n_layers"],
        "conditions": {},
    }

    all_layers = list(range(model_info["n_layers"]))

    for condition_name in ["bias_only", "syco_only"]:
        dirname = model_info[condition_name]
        print(f"\n  [{condition_name}] Loading {dirname}...")
        model, tokenizer = load_model(dirname)
        W_U = get_unembedding(model)
        print(f"    W_U shape: {W_U.shape}")  # (vocab_size, d_model)

        # Extract bias subspace directions at all layers
        print(f"    Extracting bias subspace...")
        subspaces = extract_subspaces(
            model, tokenizer, ["bias"],
            layers=all_layers, device="cpu", verbose=False,
        )

        # Also extract sycophancy subspace
        print(f"    Extracting sycophancy subspace...")
        syco_subspaces = extract_subspaces(
            model, tokenizer, ["sycophancy"],
            layers=all_layers, device="cpu", verbose=False,
        )

        condition_results = {"layers": {}}

        for layer_idx in all_layers:
            layer_result = {}

            for behavior_name, subs in [("bias", subspaces), ("sycophancy", syco_subspaces)]:
                if behavior_name not in subs or layer_idx not in subs[behavior_name]:
                    continue

                sub = subs[behavior_name][layer_idx]
                v1 = sub.directions[0]  # Top-1 direction (d_model,)
                sv1 = sub.singular_values[0]
                eff_dim = sub.effective_dim

                # Project direction through unembedding
                logits = project_direction_to_logits(v1, W_U)

                # Compute metrics for all answer token pairs
                answer_logits = {}
                for name, tid in ANSWER_TOKENS.items():
                    answer_logits[name] = logits[tid].item()

                # Answer token ranks
                ranks = compute_answer_rank(logits, ANSWER_TOKENS)

                # SNR for each pair of answer tokens
                snr_pairs = {}
                for pair_name, (t1, t2) in [
                    ("A_vs_B", (317, 347)),
                    ("A_vs_C", (317, 327)),
                    ("B_vs_C", (347, 327)),
                    ("bare_A_vs_B", (32, 33)),
                    ("bare_A_vs_C", (32, 34)),
                ]:
                    snr, sig, noise, lc, lb = compute_answer_snr(logits, t1, t2)
                    snr_pairs[pair_name] = {
                        "snr": round(snr, 4),
                        "signal": round(sig, 6),
                        "noise": round(noise, 6),
                    }

                # Percentile of answer tokens in the full logit distribution
                logits_np = logits.numpy()
                percentiles = {}
                for name, tid in ANSWER_TOKENS.items():
                    pct = (logits_np < logits_np[tid]).sum() / len(logits_np) * 100
                    percentiles[name] = round(pct, 1)

                # Full distribution stats
                logit_mean = logits.mean().item()
                logit_std = logits.std().item()

                # How many std devs are the answer tokens from the mean?
                z_scores = {}
                for name, tid in ANSWER_TOKENS.items():
                    z = (logits[tid].item() - logit_mean) / logit_std if logit_std > 0 else 0
                    z_scores[name] = round(z, 4)

                # Max answer token spread (max - min of answer logits)
                answer_vals = [logits[tid].item() for tid in ANSWER_TOKENS.values()]
                answer_spread = max(answer_vals) - min(answer_vals)

                layer_result[behavior_name] = {
                    "sv1": round(sv1, 4),
                    "eff_dim": eff_dim,
                    "n_pairs": sub.n_pairs,
                    "answer_logits": {k: round(v, 6) for k, v in answer_logits.items()},
                    "answer_ranks": ranks,
                    "answer_percentiles": percentiles,
                    "answer_z_scores": z_scores,
                    "answer_spread": round(answer_spread, 6),
                    "snr_pairs": snr_pairs,
                    "logit_mean": round(logit_mean, 6),
                    "logit_std": round(logit_std, 6),
                    "vocab_size": len(logits_np),
                }

                # Print summary
                best_snr = max(p["snr"] for p in snr_pairs.values())
                print(f"    L{layer_idx} [{behavior_name}]: "
                      f"SV1={sv1:.2f}, eff_dim={eff_dim}, "
                      f"best_SNR={best_snr:.4f}, "
                      f"spread={answer_spread:.6f}, "
                      f"z_scores=[{', '.join(f'{k}:{v:.2f}' for k, v in sorted(z_scores.items()))}]")

            condition_results["layers"][str(layer_idx)] = layer_result

        results["conditions"][condition_name] = condition_results
        del model, tokenizer  # Free memory

    # ── Cross-model analysis: project syco-only's bias direction ──────
    # This is the key test: at 3M, the syco-only model has a bias direction
    # (cos ~0.76 with bias-only's direction). Can it be READ through W_U?
    print(f"\n  [cross-model] Loading syco-only and extracting bias direction...")
    model_syco, tok_syco = load_model(model_info["syco_only"])
    W_U_syco = get_unembedding(model_syco)

    cross_subspaces = extract_subspaces(
        model_syco, tok_syco, ["bias"],
        layers=all_layers, device="cpu", verbose=False,
    )

    cross_results = {"layers": {}}
    for layer_idx in all_layers:
        if "bias" not in cross_subspaces or layer_idx not in cross_subspaces["bias"]:
            continue
        sub = cross_subspaces["bias"][layer_idx]
        v1 = sub.directions[0]
        sv1 = sub.singular_values[0]

        logits = project_direction_to_logits(v1, W_U_syco)
        logits_np = logits.numpy()

        answer_logits = {n: logits[tid].item() for n, tid in ANSWER_TOKENS.items()}
        logit_std = logits.std().item()
        logit_mean = logits.mean().item()

        z_scores = {}
        for name, tid in ANSWER_TOKENS.items():
            z = (logits[tid].item() - logit_mean) / logit_std if logit_std > 0 else 0
            z_scores[name] = round(z, 4)

        answer_vals = [logits[tid].item() for tid in ANSWER_TOKENS.values()]
        answer_spread = max(answer_vals) - min(answer_vals)

        snr_pairs = {}
        for pair_name, (t1, t2) in [
            ("A_vs_B", (317, 347)),
            ("A_vs_C", (317, 327)),
            ("B_vs_C", (347, 327)),
        ]:
            snr, sig, noise, _, _ = compute_answer_snr(logits, t1, t2)
            snr_pairs[pair_name] = {
                "snr": round(snr, 4),
                "signal": round(sig, 6),
                "noise": round(noise, 6),
            }

        cross_results["layers"][str(layer_idx)] = {
            "bias_direction_from_syco_model": {
                "sv1": round(sv1, 4),
                "eff_dim": sub.effective_dim,
                "n_pairs": sub.n_pairs,
                "answer_logits": {k: round(v, 6) for k, v in answer_logits.items()},
                "answer_z_scores": z_scores,
                "answer_spread": round(answer_spread, 6),
                "snr_pairs": snr_pairs,
                "logit_std": round(logit_std, 6),
            }
        }

        best_snr = max(p["snr"] for p in snr_pairs.values())
        print(f"    L{layer_idx} [syco_model→bias_direction]: "
              f"SV1={sv1:.2f}, "
              f"best_SNR={best_snr:.4f}, "
              f"spread={answer_spread:.6f}, "
              f"z_scores=[{', '.join(f'{k}:{v:.2f}' for k, v in sorted(z_scores.items()))}]")

    results["cross_model_readout"] = cross_results
    del model_syco, tok_syco

    return results


# ── Main ──────────────────────────────────────────────────────────────
def main():
    all_results = {}

    for scale_name, model_info in MODELS.items():
        # Verify model dirs exist
        for key in ["vanilla", "bias_only", "syco_only"]:
            model_dir = SCALE_DIR / model_info[key] / "model"
            if not model_dir.exists():
                print(f"  SKIP {scale_name}: {model_dir} not found")
                continue
        all_results[scale_name] = analyze_model(scale_name, model_info)

    # ── Summary comparison ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  READOUT GATE SUMMARY")
    print(f"{'='*70}")
    print()

    # For each scale, show the last-layer bias direction SNR
    print(f"  {'Scale':<6s} {'d':>4s} {'Condition':<12s} {'Layer':<6s} "
          f"{'Best SNR':>10s} {'Spread':>10s} {'Cross-ρ':>8s}")
    print(f"  {'-'*62}")

    cross_rho = {"3M": 0.000, "5M": 0.292, "7M": 0.208}

    for scale_name in ["3M", "5M", "7M"]:
        if scale_name not in all_results:
            continue
        r = all_results[scale_name]
        d = r["d_model"]
        last_layer = str(r["n_layers"] - 1)

        for cond in ["bias_only", "syco_only"]:
            if cond not in r["conditions"]:
                continue
            layer_data = r["conditions"][cond]["layers"].get(last_layer, {})
            for beh in ["bias", "sycophancy"]:
                if beh not in layer_data:
                    continue
                bd = layer_data[beh]
                best_snr = max(p["snr"] for p in bd["snr_pairs"].values())
                spread = bd["answer_spread"]
                print(f"  {scale_name:<6s} {d:>4d} {cond+'→'+beh:<20s} L{last_layer:<4s} "
                      f"{best_snr:>10.4f} {spread:>10.6f} {cross_rho[scale_name]:>8.3f}")

        # Cross-model readout
        if "cross_model_readout" in r:
            cr = r["cross_model_readout"]["layers"].get(last_layer, {})
            if "bias_direction_from_syco_model" in cr:
                bd = cr["bias_direction_from_syco_model"]
                best_snr = max(p["snr"] for p in bd["snr_pairs"].values())
                spread = bd["answer_spread"]
                print(f"  {scale_name:<6s} {d:>4d} {'syco→bias(cross)':<20s} L{last_layer:<4s} "
                      f"{best_snr:>10.4f} {spread:>10.6f} {cross_rho[scale_name]:>8.3f}")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = OUT_DIR / "logit_lens_readout_gate.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
