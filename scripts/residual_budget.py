#!/usr/bin/env python3
"""Residual Stream Budget — Where Does the Behavioral Signal Actually Live?

The attention routing (QK) and OV circuit analyses showed:
- QK: ALL models route correctly (3M routes BETTER than 5M)
- OV: ALL models are anti-correct (d=96 is MORE anti-correct than d=88)
- Yet cross-ρ jumps 29x from d=88 to d=96

Something other than attention heads must provide the positive delta.
In a 2-layer GPT-2, the residual stream at the last token is:

  h = emb + pos_emb                        (embedding)
  h = h + attn_out_L0(LN(h))               (attention layer 0)
  h = h + mlp_out_L0(LN(h))                (MLP layer 0)
  h = h + attn_out_L1(LN(h))               (attention layer 1)
  h = h + mlp_out_L1(LN(h))                (MLP layer 1)
  h = LN_f(h)                              (final layernorm)
  logits = h @ W_U.T                        (unembedding)

This script decomposes the logit for correct vs biased answer tokens
into contributions from each component:
1. Embedding (token + position)
2. Attention output per layer
3. MLP output per layer
4. LayerNorm scaling effect (final LN)

For each component c, we compute:
  logit_correct_c = W_U[correct] @ c
  logit_biased_c  = W_U[biased]  @ c
  gap_c = logit_correct_c - logit_biased_c

The sum of all gap_c should approximately equal the total logit gap
(exactly if we account for LayerNorm properly).

KEY QUESTION: Does the MLP provide the positive gap that overcomes
the anti-correct attention heads? And does this only happen at d≥96?
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
    tids = tokenizer.encode(f" {letter}", add_special_tokens=False)
    return tids[0] if len(tids) == 1 else tokenizer.encode(letter, add_special_tokens=False)[0]


class ComponentCapture:
    """Hook-based capture of each residual stream component.

    Captures at the last token position:
    - attn_out[layer]: output of attention sublayer (before adding to residual)
    - mlp_out[layer]: output of MLP sublayer (before adding to residual)
    """

    def __init__(self, model, n_layers):
        self.attn_out = {}
        self.mlp_out = {}
        self._hooks = []

        for l in range(n_layers):
            block = model.transformer.h[l]

            # Hook after attention: captures attn output before residual add
            # GPT-2 block forward: h = x + attn(ln1(x)); h = h + mlp(ln2(h))
            # We hook attn.c_proj (output projection) to get attn contribution
            h = block.attn.c_proj.register_forward_hook(
                self._make_capture("attn", l)
            )
            self._hooks.append(h)

            # Hook after MLP: captures MLP output before residual add
            h = block.mlp.c_proj.register_forward_hook(
                self._make_capture("mlp", l)
            )
            self._hooks.append(h)

    def _make_capture(self, component_type, layer):
        def hook(module, input, output):
            # output shape: (batch, seq_len, d_model)
            if component_type == "attn":
                self.attn_out[layer] = output.detach()
            else:
                self.mlp_out[layer] = output.detach()
        return hook

    def clear(self):
        self.attn_out = {}
        self.mlp_out = {}

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.clear()


@torch.no_grad()
def residual_budget(model, tokenizer, probes, n_layers, d_model):
    """Decompose logit gap into per-component contributions.

    Returns dict with per-component statistics averaged over all probes.
    """
    W_U = model.lm_head.weight.detach().float()  # (vocab, d_model)
    W_E = model.transformer.wte.weight.detach().float()  # (vocab, d_model)
    W_P = model.transformer.wpe.weight.detach().float()  # (max_pos, d_model)

    token_map = {l: get_answer_token_id(tokenizer, l) for l in "ABC"}

    capture = ComponentCapture(model, n_layers)

    # Component names for the budget
    components = ["embedding"]
    for l in range(n_layers):
        components.append(f"attn_L{l}")
        components.append(f"mlp_L{l}")
    components.append("ln_final")

    # Accumulators: gap per component per probe
    gaps = {c: [] for c in components}
    # Also track raw correct/biased logit per component
    correct_logits = {c: [] for c in components}
    biased_logits = {c: [] for c in components}
    # Track total model logit gap for sanity check
    total_gaps = []
    # Track which answer the model actually picks
    model_correct_count = 0

    n_valid = 0
    for i, probe in enumerate(probes):
        correct_tid = token_map.get(probe["correct_answer"])
        biased_tid = token_map.get(probe["biased_answer"])
        if correct_tid is None or biased_tid is None:
            continue

        inputs = tokenizer(probe["text"], return_tensors="pt",
                           truncation=True, max_length=512)
        outputs = model(**inputs)

        # Get actual model logits at last position
        logits_last = outputs.logits[0, -1, :].float()
        total_gap = float((logits_last[correct_tid] - logits_last[biased_tid]).item())
        total_gaps.append(total_gap)

        # Check if model picks correct answer
        answer_logits = {l: float(logits_last[tid].item()) for l, tid in token_map.items()}
        predicted = max(answer_logits, key=answer_logits.get)
        if predicted == probe["correct_answer"]:
            model_correct_count += 1

        seq_len = inputs["input_ids"].shape[1]

        # 1. Embedding contribution (token + position)
        input_ids = inputs["input_ids"][0]
        tok_emb = W_E[input_ids[-1]]  # last token embedding
        pos_emb = W_P[seq_len - 1]  # last position embedding
        emb = tok_emb + pos_emb  # (d_model,)

        emb_correct = float((W_U[correct_tid] @ emb).item())
        emb_biased = float((W_U[biased_tid] @ emb).item())
        gaps["embedding"].append(emb_correct - emb_biased)
        correct_logits["embedding"].append(emb_correct)
        biased_logits["embedding"].append(emb_biased)

        # 2. Per-layer attention and MLP contributions
        for l in range(n_layers):
            # Attention output at last token
            attn_out = capture.attn_out[l][0, -1, :].float()
            a_correct = float((W_U[correct_tid] @ attn_out).item())
            a_biased = float((W_U[biased_tid] @ attn_out).item())
            gaps[f"attn_L{l}"].append(a_correct - a_biased)
            correct_logits[f"attn_L{l}"].append(a_correct)
            biased_logits[f"attn_L{l}"].append(a_biased)

            # MLP output at last token
            mlp_out = capture.mlp_out[l][0, -1, :].float()
            m_correct = float((W_U[correct_tid] @ mlp_out).item())
            m_biased = float((W_U[biased_tid] @ mlp_out).item())
            gaps[f"mlp_L{l}"].append(m_correct - m_biased)
            correct_logits[f"mlp_L{l}"].append(m_correct)
            biased_logits[f"mlp_L{l}"].append(m_biased)

        # 3. LayerNorm final effect
        # The residual before final LN is sum of all above.
        # The final LN rescales this, creating an interaction term.
        # We compute it as: total_logit_gap - sum(component gaps)
        sum_component_gaps = (
            (emb_correct - emb_biased)
            + sum(gaps[f"attn_L{l}"][-1] for l in range(n_layers))
            + sum(gaps[f"mlp_L{l}"][-1] for l in range(n_layers))
        )
        ln_gap = total_gap - sum_component_gaps
        gaps["ln_final"].append(ln_gap)
        correct_logits["ln_final"].append(ln_gap / 2)  # approximate split
        biased_logits["ln_final"].append(-ln_gap / 2)

        capture.clear()
        n_valid += 1
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(probes)}", flush=True)

    capture.remove()

    # Compile statistics
    results = {
        "n_valid": n_valid,
        "model_accuracy": round(model_correct_count / n_valid, 4) if n_valid > 0 else 0,
        "total_gap_mean": round(float(np.mean(total_gaps)), 6),
        "total_gap_std": round(float(np.std(total_gaps)), 6),
        "components": {},
    }

    for c in components:
        g = np.array(gaps[c])
        cl = np.array(correct_logits[c])
        bl = np.array(biased_logits[c])
        results["components"][c] = {
            "mean_gap": round(float(g.mean()), 6),
            "std_gap": round(float(g.std()), 6),
            "median_gap": round(float(np.median(g)), 6),
            "frac_positive": round(float((g > 0).mean()), 4),
            "mean_correct": round(float(cl.mean()), 6),
            "mean_biased": round(float(bl.mean()), 6),
            "abs_mean_gap": round(float(np.abs(g).mean()), 6),
            # Fraction of total |gap| budget this component accounts for
            "budget_fraction": round(float(np.abs(g).mean() / np.abs(np.array(total_gaps)).mean()), 4)
                if np.abs(np.array(total_gaps)).mean() > 0 else 0,
        }

    return results


def main():
    behavior = BiasBehavior()
    probes = behavior.load_probes(n=300, seed=42)
    print(f"Loaded {len(probes)} bias probes\n")

    all_results = {}
    cross_rho = {"3M": 0.000, "4.5M": 0.010, "5M": 0.292, "7M": 0.208}

    for scale in ["3M", "4.5M", "5M", "7M"]:
        info = MODELS[scale]
        n_layers = info["n_layers"]
        d_model = info["d_model"]
        print(f"\n{'='*70}")
        print(f"  {scale} (d={d_model}, {n_layers}L)")
        print(f"{'='*70}")

        scale_results = {
            "d_model": d_model,
            "n_layers": n_layers,
            "conditions": {},
        }

        for cond in ["vanilla", "bias_only", "syco_only"]:
            if cond not in info:
                continue
            dirname = info[cond]
            model_path = SCALE_DIR / dirname / "model"
            if not model_path.exists():
                print(f"  SKIP {cond}: not found")
                continue

            print(f"\n  [{cond}] Loading {dirname}...")
            model, tokenizer = load_model(dirname)

            print(f"    Residual budget decomposition:")
            budget = residual_budget(
                model, tokenizer, probes, n_layers, d_model,
            )

            # Print component budget
            print(f"\n    Component Budget (mean logit gap: correct - biased):")
            print(f"    {'Component':<15s} {'Gap':>10s} {'%Pos':>6s} {'|Gap|':>10s} {'Budget%':>8s}")
            print(f"    {'-'*52}")
            for c, data in budget["components"].items():
                direction = "→corr" if data["mean_gap"] > 0 else "→bias"
                print(f"    {c:<15s} {data['mean_gap']:>+9.5f} {data['frac_positive']:>5.0%}"
                      f"  {data['abs_mean_gap']:>9.5f} {data['budget_fraction']:>7.1%} {direction}")
            print(f"    {'─'*52}")
            print(f"    {'TOTAL':<15s} {budget['total_gap_mean']:>+9.5f}")
            print(f"    Model accuracy: {budget['model_accuracy']:.1%}")
            print(f"    Cross-ρ: {cross_rho[scale]:.3f}")

            scale_results["conditions"][cond] = budget
            del model, tokenizer

        all_results[scale] = scale_results

    # ── Cross-scale comparison ────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  RESIDUAL BUDGET — CROSS-SCALE COMPARISON (syco_only)")
    print(f"{'='*70}\n")

    # Focus on syco_only since that's where cross-transfer happens
    print(f"  {'Scale':<6s} {'d':>4s} {'Embed':>9s} {'Attn':>9s} {'MLP':>9s} {'LN':>9s}"
          f" {'Total':>9s} {'Cross-ρ':>8s}")
    print(f"  {'-'*62}")

    for scale in ["3M", "4.5M", "5M", "7M"]:
        if scale not in all_results:
            continue
        r = all_results[scale]
        if "syco_only" not in r["conditions"]:
            continue
        b = r["conditions"]["syco_only"]
        n_layers = r["n_layers"]

        emb_gap = b["components"]["embedding"]["mean_gap"]
        attn_gap = sum(b["components"][f"attn_L{l}"]["mean_gap"] for l in range(n_layers))
        mlp_gap = sum(b["components"][f"mlp_L{l}"]["mean_gap"] for l in range(n_layers))
        ln_gap = b["components"]["ln_final"]["mean_gap"]
        total = b["total_gap_mean"]

        print(f"  {scale:<6s} {r['d_model']:>4d} {emb_gap:>+8.5f} {attn_gap:>+8.5f}"
              f" {mlp_gap:>+8.5f} {ln_gap:>+8.5f} {total:>+8.5f}"
              f" {cross_rho[scale]:>8.3f}")

    # Per-layer breakdown for the phase transition pair
    print(f"\n\n  PER-LAYER DETAIL (syco_only conditions)")
    print(f"  {'-'*70}")

    for scale in ["3M", "4.5M", "5M", "7M"]:
        if scale not in all_results:
            continue
        r = all_results[scale]
        if "syco_only" not in r["conditions"]:
            continue
        b = r["conditions"]["syco_only"]
        n_layers = r["n_layers"]

        print(f"\n  {scale} (d={r['d_model']}, cross-ρ={cross_rho[scale]:.3f}):")
        for l in range(n_layers):
            a = b["components"][f"attn_L{l}"]
            m = b["components"][f"mlp_L{l}"]
            a_dir = "→corr" if a["mean_gap"] > 0 else "→bias"
            m_dir = "→corr" if m["mean_gap"] > 0 else "→bias"
            print(f"    L{l}: attn={a['mean_gap']:>+8.5f} ({a['frac_positive']:>4.0%} pos) {a_dir}"
                  f"  |  mlp={m['mean_gap']:>+8.5f} ({m['frac_positive']:>4.0%} pos) {m_dir}")

    # ── The key question: MLP gap difference at d=88 vs d=96 ──────────
    if "4.5M" in all_results and "5M" in all_results:
        r88 = all_results["4.5M"]["conditions"].get("syco_only", {})
        r96 = all_results["5M"]["conditions"].get("syco_only", {})

        if r88 and r96:
            n88 = all_results["4.5M"]["n_layers"]
            n96 = all_results["5M"]["n_layers"]

            print(f"\n\n{'='*70}")
            print(f"  THE PHASE TRANSITION: d=88 vs d=96")
            print(f"{'='*70}\n")

            for c in ["embedding"] + [f"attn_L{l}" for l in range(max(n88, n96))] + \
                     [f"mlp_L{l}" for l in range(max(n88, n96))] + ["ln_final"]:
                g88 = r88["components"].get(c, {}).get("mean_gap", 0)
                g96 = r96["components"].get(c, {}).get("mean_gap", 0)
                delta = g96 - g88
                arrow = "▲" if delta > 0 else "▼" if delta < 0 else "─"
                print(f"  {c:<15s}  d=88: {g88:>+9.5f}  d=96: {g96:>+9.5f}"
                      f"  Δ={delta:>+9.5f} {arrow}")

            # Totals
            t88 = r88["total_gap_mean"]
            t96 = r96["total_gap_mean"]
            print(f"  {'TOTAL':<15s}  d=88: {t88:>+9.5f}  d=96: {t96:>+9.5f}"
                  f"  Δ={t96-t88:>+9.5f}")
            print(f"\n  Cross-ρ:         d=88: 0.010       d=96: 0.292")

    # ── Save results ──────────────────────────────────────────────────
    out_path = OUT_DIR / "residual_budget.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    # ── Visualization ─────────────────────────────────────────────────
    plot_residual_budget(all_results, cross_rho)


def plot_residual_budget(all_results, cross_rho):
    """Stacked bar chart: component contributions to logit gap."""
    scales = [s for s in ["3M", "4.5M", "5M", "7M"] if s in all_results]

    # We'll plot syco_only for all, plus vanilla/bias_only where available
    conds_to_plot = []
    for scale in scales:
        for cond in ["vanilla", "bias_only", "syco_only"]:
            if cond in all_results[scale]["conditions"]:
                conds_to_plot.append((scale, cond))

    n_bars = len(conds_to_plot)
    fig, ax = plt.subplots(figsize=(max(12, n_bars * 1.2), 6))

    # Color scheme per component type
    component_colors = {
        "embedding": "#78909C",   # blue-grey
        "attn": "#EF5350",        # red (anti-correct expected)
        "mlp": "#66BB6A",         # green (hero?)
        "ln_final": "#FFA726",    # orange
    }

    x = np.arange(n_bars)
    bar_data = {}  # component -> list of gaps

    # Collect all component names across all models
    all_components = set()
    for scale, cond in conds_to_plot:
        all_components.update(all_results[scale]["conditions"][cond]["components"].keys())
    # Sort: embedding first, then attn/mlp interleaved by layer, then ln_final
    sorted_components = sorted(all_components, key=lambda c: (
        0 if c == "embedding" else
        (1 + int(c.split("_L")[1]) * 2 + (0 if "attn" in c else 1)) if "_L" in c else
        100
    ))

    for comp in sorted_components:
        bar_data[comp] = []
        for scale, cond in conds_to_plot:
            g = all_results[scale]["conditions"][cond]["components"].get(comp, {}).get("mean_gap", 0)
            bar_data[comp].append(g)

    # Plot as grouped bars — positive and negative stacked separately
    width = 0.7
    for comp in sorted_components:
        values = bar_data[comp]
        if "embedding" in comp:
            color = component_colors["embedding"]
        elif "attn" in comp:
            color = component_colors["attn"]
        elif "mlp" in comp:
            color = component_colors["mlp"]
        else:
            color = component_colors["ln_final"]

        ax.bar(x, values, width, label=comp, color=color,
               edgecolor="black", linewidth=0.3, alpha=0.85,
               bottom=[0] * n_bars)  # Simple non-stacked for clarity

    # Actually, stacking makes the chart hard to read. Let's do grouped bars per component.
    # Reset and do it properly.
    ax.clear()

    # Group by: for each scale/cond, show component gaps as a mini bar chart
    n_comps = len(sorted_components)
    total_width = 0.8
    comp_width = total_width / n_comps

    for ci, comp in enumerate(sorted_components):
        values = bar_data[comp]
        offset = (ci - n_comps / 2 + 0.5) * comp_width

        if "embedding" in comp:
            color = component_colors["embedding"]
        elif "attn" in comp:
            color = component_colors["attn"]
        elif "mlp" in comp:
            color = component_colors["mlp"]
        else:
            color = component_colors["ln_final"]

        ax.bar(x + offset, values, comp_width * 0.9,
               label=comp, color=color, edgecolor="black", linewidth=0.3, alpha=0.85)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{c}" for s, c in conds_to_plot],
                       fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Mean Logit Gap (correct − biased)")
    ax.set_title("Residual Stream Budget: Where Does the Behavioral Signal Live?\n"
                 "Positive = pushes toward correct answer",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="best", ncol=2)

    # Add cross-ρ annotations
    for i, (scale, cond) in enumerate(conds_to_plot):
        if cond == "syco_only":
            ax.annotate(f"ρ={cross_rho[scale]:.3f}",
                        xy=(i, 0), xytext=(0, -25),
                        textcoords="offset points", ha="center",
                        fontsize=7, fontweight="bold", color="purple")

    plt.tight_layout()
    path = FIG_DIR / "residual_budget.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
