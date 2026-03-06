#!/usr/bin/env python3
"""OV Circuit Direct Logit Attribution — The Compositional Gate.

The attention routing analysis (QK circuit) showed that ALL models route
attention toward the correct answer tokens equally well. The bottleneck
isn't "where the model looks" — it's "what it does with what it sees."

This script decomposes the actual contribution of each attention head's
OV circuit to the answer token logits:

For each head h at the last token position:
1. Capture z_h: the attention-weighted value output (head_dim,)
2. Project through W_O_h: the output projection for this head
   → head_contrib = z_h @ W_O_h → (d_model,) contribution to residual
3. Project through W_U (lm_head): logit contribution
   → logit_correct = W_U[correct_token] @ head_contrib
   → logit_biased  = W_U[biased_token]  @ head_contrib
4. OV logit gap = logit_correct - logit_biased

PREDICTION:
At 3M (d_head=32): The OV path produces vectors orthogonal to the
answer token embeddings → near-zero logit gap per head.
At 5M (d_head=48): The OV path "turns toward" answer tokens →
positive logit gap for correct answer in trained models.

Also computes the STATIC OV analysis (no probe data needed):
- OV_h = W_V_h @ W_O_h (d_model, d_model) — the head's embedding-to-residual map
- W_U @ OV_h @ W_E[token] — "if this head attends 100% to token X,
  what logit does it contribute for token Y?"
- Effective rank of each head's OV matrix at each scale

GPT-2 Conv1D gotcha:
- c_attn.weight: (n_embd, 3*n_embd) — V portion: [:, 2*n_embd:3*n_embd]
- c_proj.weight: (n_embd, n_embd) — head h: [h*head_dim:(h+1)*head_dim, :]
- Forward: output = input @ weight (NOT weight.T!)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

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
    "3M": {
        "vanilla": "3M_seed42",
        "bias_only": "3M_seed42_contr_bia_r20",
        "syco_only": "3M_seed42_contr_syc_r20",
        "d_model": 64, "n_layers": 2, "n_heads": 2,
    },
    "4.5M": {
        "syco_only": "4.5M_seed42_contr_syc_r20",
        "d_model": 88, "n_layers": 2, "n_heads": 2,
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
    model = AutoModelForCausalLM.from_pretrained(
        str(path), attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def get_answer_token_id(tokenizer, letter):
    """Get token ID for answer letter (space-prefixed as in 'Answer: A')."""
    tids = tokenizer.encode(f" {letter}", add_special_tokens=False)
    return tids[0] if len(tids) == 1 else tokenizer.encode(letter, add_special_tokens=False)[0]


# ── Hook to capture c_proj input (per-head attention outputs) ──────────

class CprojInputCapture:
    """Capture the input to c_proj at a specific layer.

    In GPT-2, the input to c_proj is the merged attention output:
    shape (batch, seq_len, n_embd), which is the concatenation of
    per-head outputs [z_0 | z_1 | ... | z_{n_head-1}].
    """

    def __init__(self, model, layer_idx):
        attn = model.transformer.h[layer_idx].attn
        self._data = None
        self._hook = attn.c_proj.register_forward_hook(self._capture)

    def _capture(self, module, input, output):
        # input[0] is the merged attention output: (batch, seq_len, n_embd)
        self._data = input[0].detach()

    def get(self):
        return self._data

    def clear(self):
        self._data = None

    def remove(self):
        self._hook.remove()
        self._data = None


# ── Static OV matrix analysis ──────────────────────────────────────────

def analyze_static_ov(model, tokenizer, n_layers, n_heads, d_model):
    """Analyze the OV matrices without running any probes.

    For each head, compute:
    1. OV_h = W_V_h @ W_O_h  (d_model, d_model)
    2. The effective rank of OV_h
    3. W_U @ OV_h — how OV maps to logit space
    4. For answer tokens (A, B, C): the "answer column" of W_U @ OV_h @ W_E
       i.e., if this head attends 100% to token " A", what logit does it
       produce for token " A", " B", " C"?
    """
    W_U = model.lm_head.weight.detach().float()  # (vocab, d_model)
    W_E = model.transformer.wte.weight.detach().float()  # (vocab, d_model)

    token_ids = {l: get_answer_token_id(tokenizer, l) for l in "ABC"}
    head_dim = d_model // n_heads

    results = {"layers": {}}

    for l in range(n_layers):
        attn = model.transformer.h[l].attn
        c_attn_w = attn.c_attn.weight.detach().float()  # (n_embd, 3*n_embd)
        c_proj_w = attn.c_proj.weight.detach().float()  # (n_embd, n_embd)

        layer_data = {"heads": {}}

        for h in range(n_heads):
            # Extract W_V_h: (n_embd, head_dim) — Conv1D format
            W_V_h = c_attn_w[:, 2 * d_model + h * head_dim:2 * d_model + (h + 1) * head_dim]

            # Extract W_O_h: (head_dim, n_embd) — Conv1D format
            W_O_h = c_proj_w[h * head_dim:(h + 1) * head_dim, :]

            # OV_h = W_V_h @ W_O_h: (n_embd, head_dim) @ (head_dim, n_embd) = (n_embd, n_embd)
            OV_h = W_V_h @ W_O_h

            # Effective rank via singular values
            svs = torch.linalg.svdvals(OV_h).numpy()
            svs_norm = svs / svs.sum() if svs.sum() > 0 else svs
            eff_rank = float(np.exp(-np.sum(svs_norm * np.log(svs_norm + 1e-10))))

            # Frobenius norm
            frob_norm = float(OV_h.norm().item())

            # Answer token analysis:
            # "If head h attends 100% to the embedding of token X,
            #  what logit does it produce for token Y?"
            # answer_logit_matrix[X][Y] = W_U[Y] @ OV_h @ W_E[X]
            answer_logit_matrix = {}
            for src_letter, src_tid in token_ids.items():
                src_emb = W_E[src_tid]  # (d_model,)
                ov_output = OV_h @ src_emb  # (d_model,)

                tgt_logits = {}
                for tgt_letter, tgt_tid in token_ids.items():
                    logit = float((W_U[tgt_tid] @ ov_output).item())
                    tgt_logits[tgt_letter] = round(logit, 6)

                # Also measure alignment with answer direction
                # (how much of OV output is in the answer-token subspace)
                answer_vecs = torch.stack([W_U[tid] for tid in token_ids.values()])  # (3, d_model)
                ov_norm = ov_output.norm().item()
                if ov_norm > 0:
                    # Project OV output onto span of answer token embeddings
                    # Use SVD of answer_vecs to get orthonormal basis
                    U, S, Vh = torch.linalg.svd(answer_vecs, full_matrices=False)
                    proj = Vh @ ov_output  # project onto answer subspace
                    answer_alignment = float(proj.norm().item() / ov_norm)
                else:
                    answer_alignment = 0.0

                answer_logit_matrix[src_letter] = {
                    "logits": tgt_logits,
                    "answer_alignment": round(answer_alignment, 6),
                }

            layer_data["heads"][str(h)] = {
                "eff_rank": round(eff_rank, 2),
                "frob_norm": round(frob_norm, 4),
                "sv1": round(float(svs[0]), 4),
                "sv_ratio_1_2": round(float(svs[0] / svs[1]) if len(svs) > 1 and svs[1] > 0 else 0, 4),
                "answer_logit_matrix": answer_logit_matrix,
            }

            # Print summary
            # Key metric: if attending to correct answer token, does it boost correct logit?
            # Average the diagonal of the answer logit matrix
            diag_mean = np.mean([
                answer_logit_matrix[l_]["logits"][l_]
                for l_ in "ABC"
            ])
            align_mean = np.mean([
                answer_logit_matrix[l_]["answer_alignment"]
                for l_ in "ABC"
            ])
            print(f"      L{l}H{h}: eff_rank={eff_rank:.1f}, "
                  f"frob={frob_norm:.3f}, sv1={svs[0]:.3f}, "
                  f"diag_logit={diag_mean:.4f}, "
                  f"answer_align={align_mean:.4f}")

        results["layers"][str(l)] = layer_data

    return results


# ── Dynamic OV logit attribution (with actual probes) ──────────────────

@torch.no_grad()
def ov_logit_attribution(model, tokenizer, probes, n_layers, n_heads, d_model):
    """Per-head OV contribution to correct vs biased answer logits.

    For each probe, at the last token:
    1. Capture z_h (per-head attention output before c_proj)
    2. Project through W_O_h → head's residual contribution
    3. Project through W_U → logit contribution for correct/biased answer
    4. OV logit gap = logit_correct - logit_biased per head
    """
    W_U = model.lm_head.weight.detach().float()  # (vocab, d_model)
    head_dim = d_model // n_heads
    token_map = {l: get_answer_token_id(tokenizer, l) for l in "ABC"}

    # Set up hooks for all layers
    captures = {}
    for l in range(n_layers):
        captures[l] = CprojInputCapture(model, l)

    # Extract W_O for each layer/head (Conv1D format)
    W_O = {}
    for l in range(n_layers):
        c_proj_w = model.transformer.h[l].attn.c_proj.weight.detach().float()
        for h in range(n_heads):
            W_O[(l, h)] = c_proj_w[h * head_dim:(h + 1) * head_dim, :]  # (head_dim, n_embd)

    # Accumulators
    ov_logit_gaps = {(l, h): [] for l in range(n_layers) for h in range(n_heads)}
    ov_correct_logits = {(l, h): [] for l in range(n_layers) for h in range(n_heads)}
    ov_biased_logits = {(l, h): [] for l in range(n_layers) for h in range(n_heads)}
    ov_contrib_norms = {(l, h): [] for l in range(n_layers) for h in range(n_heads)}
    ov_answer_cosines = {(l, h): [] for l in range(n_layers) for h in range(n_heads)}

    n_valid = 0
    for i, probe in enumerate(probes):
        correct_tid = token_map.get(probe["correct_answer"])
        biased_tid = token_map.get(probe["biased_answer"])
        if correct_tid is None or biased_tid is None:
            continue

        inputs = tokenizer(probe["text"], return_tensors="pt", truncation=True, max_length=512)
        model(**inputs)

        for l in range(n_layers):
            merged_attn = captures[l].get()  # (1, seq_len, n_embd)
            last_merged = merged_attn[0, -1, :].float()  # (n_embd,)

            for h in range(n_heads):
                # Extract per-head attention output
                z_h = last_merged[h * head_dim:(h + 1) * head_dim]  # (head_dim,)

                # Project through W_O_h to get contribution to residual
                head_contrib = z_h @ W_O[(l, h)]  # (n_embd,)

                # Project through W_U for answer tokens
                logit_correct = float((W_U[correct_tid] @ head_contrib).item())
                logit_biased = float((W_U[biased_tid] @ head_contrib).item())
                gap = logit_correct - logit_biased

                ov_logit_gaps[(l, h)].append(gap)
                ov_correct_logits[(l, h)].append(logit_correct)
                ov_biased_logits[(l, h)].append(logit_biased)
                ov_contrib_norms[(l, h)].append(float(head_contrib.norm().item()))

                # Cosine similarity of head contribution with answer direction
                answer_dir = (W_U[correct_tid] - W_U[biased_tid]).float()
                cos_sim = float(torch.nn.functional.cosine_similarity(
                    head_contrib.unsqueeze(0), answer_dir.unsqueeze(0),
                ).item())
                ov_answer_cosines[(l, h)].append(cos_sim)

        for l in range(n_layers):
            captures[l].clear()

        n_valid += 1
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{len(probes)}", flush=True)

    # Clean up hooks
    for l in range(n_layers):
        captures[l].remove()

    # ── Compile ──────────────────────────────────────────────────────
    results = {"n_valid": n_valid, "layers": {}}

    for l in range(n_layers):
        layer_data = {"heads": {}}
        for h in range(n_heads):
            gaps = np.array(ov_logit_gaps[(l, h)])
            c_logits = np.array(ov_correct_logits[(l, h)])
            b_logits = np.array(ov_biased_logits[(l, h)])
            norms = np.array(ov_contrib_norms[(l, h)])
            cosines = np.array(ov_answer_cosines[(l, h)])

            if len(gaps) == 0:
                continue

            # T-test: is the mean gap significantly different from 0?
            t_stat, p_val = stats.ttest_1samp(gaps, 0.0) if len(gaps) > 1 else (0, 1)

            head_data = {
                "mean_ov_gap": round(float(gaps.mean()), 6),
                "std_ov_gap": round(float(gaps.std()), 6),
                "median_ov_gap": round(float(np.median(gaps)), 6),
                "frac_correct": round(float((gaps > 0).mean()), 4),
                "mean_correct_logit": round(float(c_logits.mean()), 6),
                "mean_biased_logit": round(float(b_logits.mean()), 6),
                "mean_contrib_norm": round(float(norms.mean()), 4),
                "mean_answer_cosine": round(float(cosines.mean()), 6),
                "std_answer_cosine": round(float(cosines.std()), 6),
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val), 6),
            }
            layer_data["heads"][str(h)] = head_data

            sig = "★" if p_val < 0.01 else ("*" if p_val < 0.05 else " ")
            direction = "→correct" if gaps.mean() > 0 else "→biased "
            print(f"      L{l}H{h}: OV_gap={gaps.mean():+.5f} "
                  f"({(gaps > 0).mean():.0%} correct) "
                  f"t={t_stat:.2f} p={p_val:.4f} {sig} "
                  f"‖contrib‖={norms.mean():.3f} "
                  f"cos={cosines.mean():+.4f} {direction}",
                  flush=True)

        results["layers"][str(l)] = layer_data

    return results


def main():
    behavior = BiasBehavior()
    probes = behavior.load_probes(n=300, seed=42)
    print(f"Loaded {len(probes)} bias probes\n")

    all_results = {}
    cross_rho = {"3M": 0.000, "4.5M": 0.010, "5M": 0.292, "7M": 0.208}

    for scale in ["3M", "4.5M", "5M", "7M"]:
        info = MODELS[scale]
        d_head = info["d_model"] // info["n_heads"]
        print(f"\n{'='*70}")
        print(f"  {scale} (d={info['d_model']}, {info['n_layers']}L, "
              f"{info['n_heads']}H, d_head={d_head})")
        print(f"{'='*70}")

        scale_results = {
            "d_model": info["d_model"],
            "n_layers": info["n_layers"],
            "n_heads": info["n_heads"],
            "d_head": d_head,
            "conditions": {},
        }

        for cond in ["vanilla", "bias_only", "syco_only"]:
            if cond not in info:
                continue
            dirname = info[cond]
            model_path = SCALE_DIR / dirname / "model"
            if not model_path.exists():
                print(f"  SKIP {cond}: {model_path} not found")
                continue

            print(f"\n  [{cond}] Loading {dirname}...")
            model, tokenizer = load_model(dirname)

            # 1. Static OV analysis (no probes needed)
            print(f"    Static OV analysis:")
            static = analyze_static_ov(
                model, tokenizer,
                info["n_layers"], info["n_heads"], info["d_model"],
            )

            # 2. Dynamic OV logit attribution (with probes)
            print(f"    Dynamic OV logit attribution:")
            dynamic = ov_logit_attribution(
                model, tokenizer, probes,
                info["n_layers"], info["n_heads"], info["d_model"],
            )

            scale_results["conditions"][cond] = {
                "static": static,
                "dynamic": dynamic,
            }
            del model, tokenizer

        all_results[scale] = scale_results

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  OV CIRCUIT DIRECT LOGIT ATTRIBUTION — SUMMARY")
    print(f"{'='*70}\n")

    print(f"  {'Scale':<5s} {'d_h':>4s} {'Cond':<12s} {'L':>2s} {'H':>2s} "
          f"{'OV Gap':>9s} {'%Corr':>6s} {'t':>7s} {'p':>8s} "
          f"{'‖c‖':>6s} {'cos':>7s} {'Cross-ρ':>8s}")
    print(f"  {'-'*85}")

    for scale in ["3M", "4.5M", "5M", "7M"]:
        if scale not in all_results:
            continue
        r = all_results[scale]
        d_head = r["d_head"]
        last = str(r["n_layers"] - 1)
        for cond in ["vanilla", "bias_only", "syco_only"]:
            if cond not in r["conditions"]:
                continue
            dyn = r["conditions"][cond]["dynamic"]
            if last not in dyn["layers"]:
                continue
            lr = dyn["layers"][last]
            for h_str, hr in sorted(lr["heads"].items()):
                sig = "★" if hr["p_value"] < 0.01 else ("*" if hr["p_value"] < 0.05 else " ")
                print(f"  {scale:<5s} {d_head:>4d} {cond:<12s} "
                      f"{last:>2s} {h_str:>2s} "
                      f"{hr['mean_ov_gap']:>+8.5f} "
                      f"{hr['frac_correct']:>5.0%} "
                      f"{hr['t_statistic']:>7.2f} "
                      f"{hr['p_value']:>8.4f}{sig} "
                      f"{hr['mean_contrib_norm']:>6.3f} "
                      f"{hr['mean_answer_cosine']:>+6.4f} "
                      f"{cross_rho[scale]:>8.3f}")
        print()

    # ── KEY COMPARISON: does OV gap increase at 5M? ──────────────────
    print(f"\n  KEY: OV circuit alignment with answer tokens across scales")
    print(f"  (Sum of all last-layer heads' mean OV gaps)")
    print(f"  {'-'*65}")

    for scale in ["3M", "4.5M", "5M", "7M"]:
        if scale not in all_results:
            continue
        r = all_results[scale]
        last = str(r["n_layers"] - 1)
        d_head = r["d_head"]
        for cond in ["vanilla", "bias_only", "syco_only"]:
            if cond not in r["conditions"]:
                continue
            dyn = r["conditions"][cond]["dynamic"]
            if last not in dyn["layers"]:
                continue
            lr = dyn["layers"][last]

            total_gap = sum(hr["mean_ov_gap"] for hr in lr["heads"].values())
            total_cos = np.mean([hr["mean_answer_cosine"] for hr in lr["heads"].values()])
            total_norm = sum(hr["mean_contrib_norm"] for hr in lr["heads"].values())

            print(f"  {scale} (d_h={d_head:>2d}) {cond:<12s}: "
                  f"Σgap={total_gap:+.5f}, "
                  f"mean_cos={total_cos:+.4f}, "
                  f"Σ‖contrib‖={total_norm:.3f} "
                  f"| cross-ρ={cross_rho[scale]:.3f}")

    # ── Static OV summary ────────────────────────────────────────────
    print(f"\n  STATIC OV: Effective rank and answer alignment")
    print(f"  {'-'*65}")

    for scale in ["3M", "4.5M", "5M", "7M"]:
        if scale not in all_results:
            continue
        r = all_results[scale]
        last = str(r["n_layers"] - 1)
        d_head = r["d_head"]
        for cond in ["vanilla", "bias_only", "syco_only"]:
            if cond not in r["conditions"]:
                continue
            stat = r["conditions"][cond]["static"]
            if last not in stat["layers"]:
                continue
            lr = stat["layers"][last]
            for h_str, hr in sorted(lr["heads"].items()):
                # Diagonal mean: if attending to token X, does it boost logit for X?
                diag = np.mean([
                    hr["answer_logit_matrix"][l]["logits"][l] for l in "ABC"
                ])
                align = np.mean([
                    hr["answer_logit_matrix"][l]["answer_alignment"] for l in "ABC"
                ])
                print(f"  {scale} (d_h={d_head:>2d}) {cond:<12s} L{last}H{h_str}: "
                      f"eff_rank={hr['eff_rank']:.1f}, "
                      f"frob={hr['frob_norm']:.3f}, "
                      f"diag_logit={diag:.4f}, "
                      f"answer_align={align:.4f}")

    # ── Save ─────────────────────────────────────────────────────────
    out_path = OUT_DIR / "ov_circuit_logit_attribution.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    # ── Visualization ────────────────────────────────────────────────
    plot_ov_gap_comparison(all_results, cross_rho)
    plot_ov_decomposition(all_results, cross_rho)


def plot_ov_gap_comparison(all_results, cross_rho):
    """Bar chart: OV logit gap per head across scales/conditions."""
    scales = [s for s in ["3M", "4.5M", "5M", "7M"] if s in all_results]
    n_scales = len(scales)
    fig, axes = plt.subplots(1, n_scales, figsize=(5 * n_scales, 5), sharey=True)
    if n_scales == 1:
        axes = [axes]
    colors = {"vanilla": "#999999", "bias_only": "#2196F3", "syco_only": "#FF5722"}

    for ax_idx, scale in enumerate(scales):
        ax = axes[ax_idx]
        r = all_results[scale]
        last = str(r["n_layers"] - 1)
        n_heads = r["n_heads"]
        d_head = r["d_head"]

        conditions = [c for c in ["vanilla", "bias_only", "syco_only"]
                      if c in r["conditions"]]
        n_conds = len(conditions)
        x = np.arange(n_heads)
        width = 0.25
        offsets = np.linspace(-(n_conds - 1) * width / 2,
                              (n_conds - 1) * width / 2, n_conds)

        for c_idx, cond in enumerate(conditions):
            dyn = r["conditions"][cond]["dynamic"]
            if last not in dyn["layers"]:
                continue
            lr = dyn["layers"][last]

            values = []
            errors = []
            for h in range(n_heads):
                hr = lr["heads"].get(str(h), {})
                values.append(hr.get("mean_ov_gap", 0))
                std = hr.get("std_ov_gap", 0)
                n = dyn.get("n_valid", 1)
                errors.append(std / np.sqrt(n) if n > 0 else 0)

            ax.bar(x + offsets[c_idx], values, width,
                   yerr=errors, label=cond,
                   color=colors[cond], alpha=0.85,
                   capsize=2, edgecolor="black", linewidth=0.5)

        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
        ax.set_title(f"{scale} (d_head={d_head}, {r['n_layers']}L)\n"
                     f"cross-ρ = {cross_rho[scale]:.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Head (last layer)")
        if ax_idx == 0:
            ax.set_ylabel("OV Logit Gap\n(correct − biased)")
            ax.legend(fontsize=8, loc="best")

    fig.suptitle("OV Circuit Direct Logit Attribution\n"
                 "Per-head contribution to correct vs biased answer logit",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout()
    path = FIG_DIR / "ov_logit_attribution_bars.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_ov_decomposition(all_results, cross_rho):
    """Decomposition plot: contrib norm × answer cosine = OV gap."""
    scales = [s for s in ["3M", "4.5M", "5M", "7M"] if s in all_results]
    n_scales = len(scales)
    fig, axes = plt.subplots(1, n_scales, figsize=(5 * n_scales, 5))
    if n_scales == 1:
        axes = [axes]

    for ax_idx, scale in enumerate(scales):
        ax = axes[ax_idx]
        r = all_results[scale]
        last = str(r["n_layers"] - 1)
        d_head = r["d_head"]

        markers = {"vanilla": "o", "bias_only": "s", "syco_only": "^"}
        colors_cond = {"vanilla": "#999999", "bias_only": "#2196F3", "syco_only": "#FF5722"}

        for cond in ["vanilla", "bias_only", "syco_only"]:
            if cond not in r["conditions"]:
                continue
            dyn = r["conditions"][cond]["dynamic"]
            if last not in dyn["layers"]:
                continue
            lr = dyn["layers"][last]

            for h_str, hr in lr["heads"].items():
                norm = hr["mean_contrib_norm"]
                cos = hr["mean_answer_cosine"]
                ax.scatter(norm, cos,
                           marker=markers[cond], color=colors_cond[cond],
                           s=100, edgecolors="black", linewidth=0.5,
                           label=f"{cond} H{h_str}" if ax_idx == 0 else "",
                           zorder=5)
                ax.annotate(f"H{h_str}", (norm, cos),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7)

        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("‖OV contribution‖ (magnitude)")
        if ax_idx == 0:
            ax.set_ylabel("cos(OV contribution, answer direction)")
        ax.set_title(f"{scale} (d_head={d_head})\n"
                     f"cross-ρ = {cross_rho[scale]:.3f}",
                     fontsize=10, fontweight="bold")

    # Add legend to first panel
    handles = [
        plt.Line2D([0], [0], marker="o", color="#999999", linestyle="", markersize=8, label="vanilla"),
        plt.Line2D([0], [0], marker="s", color="#2196F3", linestyle="", markersize=8, label="bias_only"),
        plt.Line2D([0], [0], marker="^", color="#FF5722", linestyle="", markersize=8, label="syco_only"),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="best")

    fig.suptitle("OV Decomposition: Magnitude × Alignment = Logit Contribution\n"
                 "OV gap ≈ ‖contribution‖ × cos(contribution, answer_direction)",
                 fontsize=11, fontweight="bold", y=1.04)
    plt.tight_layout()
    path = FIG_DIR / "ov_decomposition_scatter.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
