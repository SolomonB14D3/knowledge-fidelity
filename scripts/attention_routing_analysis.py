#!/usr/bin/env python3
"""Attention Head Routing Analysis — The Compositional Gate.

The logit lens experiments (v1-v3) showed:
- The readout gate is NOT in the unembedding matrix (W_U)
- 3M has MORE bias bandwidth than 5M, yet zero cross-transfer
- The gate operates upstream in attention head composition

This script directly visualizes the compositional gate:
1. For each bias probe, extract attention weights at every layer/head
2. At the LAST token (where the model generates its answer), measure
   how much each head attends to:
   - Tokens in the CORRECT answer option span
   - Tokens in the BIASED answer option span
   - Tokens in the NEUTRAL answer option span
3. Routing score = attn_to_correct - attn_to_biased per head
4. A positive routing score means the head preferentially routes
   information from the correct answer

If bias_only 5M shows a "routing head" that's absent in 3M (both have
2 heads!), that's the smoking gun for the compositional gate: the same
number of heads, but d_head=48 (5M) vs d_head=32 (3M) allows the head
to multiplex behavioral routing alongside LM routing.

Cross-scale comparison:
- 3M: 2 layers, 2 heads, d_head=32  (no cross-transfer)
- 5M: 2 layers, 2 heads, d_head=48  (cross-transfer emerges)
- 7M: 4 layers, 4 heads, d_head=32  (cross-transfer present)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from rho_eval.behaviors.bias import BiasBehavior

ROOT = Path(__file__).resolve().parents[1]
SCALE_DIR = ROOT / "results" / "scale_ladder"
OUT_DIR = ROOT / "results" / "developmental_sweep"
FIG_DIR = ROOT / "figures" / "paper4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Model conditions ─────────────────────────────────────────────────
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
    # Force eager attention implementation to get attention weight matrices
    model = AutoModelForCausalLM.from_pretrained(
        str(path), attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def find_option_token_spans(text, tokenizer):
    """Find token index spans for each answer option (A, B, C).

    Returns dict: {"A": (start, end), "B": (start, end), "C": (start, end)}
    where [start, end) are token indices covering that option's text.

    Strategy: find "\nA) ", "\nB) ", "\nC) " character positions, then
    map to token indices. Each option runs until the next option or "Answer:".
    """
    # Find character positions of each option
    markers = {}
    for letter in ["A", "B", "C"]:
        # Look for "\nA) " pattern
        idx = text.find(f"\n{letter}) ")
        if idx == -1:
            idx = text.find(f"{letter}) ")
        markers[letter] = idx

    # Find "Answer:" position (end of last option)
    answer_pos = text.find("\nAnswer:")
    if answer_pos == -1:
        answer_pos = text.find("Answer:")

    if any(v == -1 for v in markers.values()) or answer_pos == -1:
        return None

    # Sort options by position
    sorted_letters = sorted(markers.keys(), key=lambda l: markers[l])

    # Define character spans
    char_spans = {}
    for i, letter in enumerate(sorted_letters):
        start_char = markers[letter]
        if i + 1 < len(sorted_letters):
            end_char = markers[sorted_letters[i + 1]]
        else:
            end_char = answer_pos
        char_spans[letter] = (start_char, end_char)

    # Convert character spans to token spans
    # Tokenize the full text and get offset mapping
    encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoding["offset_mapping"][0]  # (seq_len, 2)

    token_spans = {}
    for letter, (cs, ce) in char_spans.items():
        # Find first token that starts at or after cs
        start_tok = None
        end_tok = None
        for t_idx, (t_start, t_end) in enumerate(offsets.tolist()):
            if t_end == 0 and t_start == 0:
                continue  # special token
            if start_tok is None and t_end > cs:
                start_tok = t_idx
            if t_start < ce:
                end_tok = t_idx + 1

        if start_tok is not None and end_tok is not None:
            token_spans[letter] = (start_tok, end_tok)

    if len(token_spans) != 3:
        return None

    return token_spans


def find_option_spans_simple(text, tokenizer):
    """Simpler approach: tokenize prefix up to each marker to get token position."""
    spans = {}
    full_ids = tokenizer.encode(text)
    n_tokens = len(full_ids)

    for letter in ["A", "B", "C"]:
        marker = f"\n{letter}) "
        idx = text.find(marker)
        if idx == -1:
            marker = f"{letter}) "
            idx = text.find(marker)
        if idx == -1:
            return None

        # Token position: length of tokens for text[:idx]
        prefix_ids = tokenizer.encode(text[:idx])
        start_tok = len(prefix_ids)
        spans[letter] = start_tok

    # Get end positions (next option start or "Answer:")
    answer_marker = text.find("\nAnswer:")
    if answer_marker == -1:
        answer_marker = text.find("Answer:")
    if answer_marker == -1:
        return None

    answer_tok = len(tokenizer.encode(text[:answer_marker]))

    sorted_letters = sorted(spans.keys(), key=lambda l: spans[l])
    token_spans = {}
    for i, letter in enumerate(sorted_letters):
        start = spans[letter]
        if i + 1 < len(sorted_letters):
            end = spans[sorted_letters[i + 1]]
        else:
            end = answer_tok
        token_spans[letter] = (start, end)

    return token_spans


@torch.no_grad()
def analyze_attention_routing(model, tokenizer, probes, n_layers, n_heads):
    """Extract attention routing scores for bias probes.

    For each probe, at the LAST token position:
    - Measure how much each head attends to correct vs biased option tokens
    - Routing score = sum(attn over correct tokens) - sum(attn over biased tokens)

    Returns per-layer, per-head routing scores aggregated across probes.
    """
    # Per-layer, per-head accumulators
    routing_scores = {l: {h: [] for h in range(n_heads)} for l in range(n_layers)}
    attn_to_correct = {l: {h: [] for h in range(n_heads)} for l in range(n_layers)}
    attn_to_biased = {l: {h: [] for h in range(n_heads)} for l in range(n_layers)}
    attn_to_neutral = {l: {h: [] for h in range(n_heads)} for l in range(n_layers)}

    # Per-head "behavioral entropy" — how focused is attention on answer options?
    head_option_attn_frac = {l: {h: [] for h in range(n_heads)} for l in range(n_layers)}

    # Track per-probe routing for detailed analysis
    per_probe_routing = []

    n_valid = 0
    n_skipped = 0

    for i, probe in enumerate(probes):
        text = probe["text"]
        correct_letter = probe["correct_answer"]
        biased_letter = probe["biased_answer"]
        neutral_letter = [l for l in "ABC" if l != correct_letter and l != biased_letter][0]

        # Find token spans for each answer option
        token_spans = find_option_spans_simple(text, tokenizer)
        if token_spans is None:
            n_skipped += 1
            continue

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        seq_len = inputs["input_ids"].shape[1]

        # Validate spans are within sequence
        max_span_end = max(end for _, end in token_spans.values())
        if max_span_end > seq_len:
            n_skipped += 1
            continue

        # Forward pass with attention weights
        outputs = model(**inputs, output_attentions=True)

        # outputs.attentions: tuple of (batch, n_heads, seq_len, seq_len) per layer
        last_pos = seq_len - 1  # Last token position

        probe_data = {
            "id": probe.get("id", ""),
            "correct": correct_letter,
            "biased": biased_letter,
            "layers": {},
        }

        for layer_idx in range(n_layers):
            attn_weights = outputs.attentions[layer_idx]  # (1, n_heads, seq_len, seq_len)
            # Attention at the last position: (n_heads, seq_len)
            last_attn = attn_weights[0, :, last_pos, :]  # (n_heads, seq_len)

            layer_data = {}
            for head_idx in range(n_heads):
                head_attn = last_attn[head_idx]  # (seq_len,)

                # Sum attention over each option's token span
                c_start, c_end = token_spans[correct_letter]
                b_start, b_end = token_spans[biased_letter]
                n_start, n_end = token_spans[neutral_letter]

                attn_c = head_attn[c_start:c_end].sum().item()
                attn_b = head_attn[b_start:b_end].sum().item()
                attn_n = head_attn[n_start:n_end].sum().item()

                # Routing score: positive = routes toward correct
                routing = attn_c - attn_b

                routing_scores[layer_idx][head_idx].append(routing)
                attn_to_correct[layer_idx][head_idx].append(attn_c)
                attn_to_biased[layer_idx][head_idx].append(attn_b)
                attn_to_neutral[layer_idx][head_idx].append(attn_n)

                # Fraction of total attention going to ANY option
                total_option_attn = attn_c + attn_b + attn_n
                head_option_attn_frac[layer_idx][head_idx].append(total_option_attn)

                layer_data[f"h{head_idx}"] = {
                    "routing": round(routing, 6),
                    "attn_correct": round(attn_c, 6),
                    "attn_biased": round(attn_b, 6),
                    "attn_neutral": round(attn_n, 6),
                }

            probe_data["layers"][str(layer_idx)] = layer_data

        per_probe_routing.append(probe_data)
        n_valid += 1

        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{len(probes)} probes", flush=True)

    # ── Aggregate ────────────────────────────────────────────────────
    results = {
        "n_valid": n_valid,
        "n_skipped": n_skipped,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "layers": {},
    }

    for l in range(n_layers):
        layer_result = {"heads": {}}
        for h in range(n_heads):
            scores = np.array(routing_scores[l][h])
            a_c = np.array(attn_to_correct[l][h])
            a_b = np.array(attn_to_biased[l][h])
            a_n = np.array(attn_to_neutral[l][h])
            opt_frac = np.array(head_option_attn_frac[l][h])

            if len(scores) == 0:
                continue

            # Fraction of probes where this head routes toward correct
            frac_correct_routing = float((scores > 0).mean())

            # Is this head a statistically significant router?
            from scipy import stats
            t_stat, p_val = stats.ttest_1samp(scores, 0.0) if len(scores) > 1 else (0, 1)

            head_result = {
                "mean_routing": round(float(scores.mean()), 6),
                "std_routing": round(float(scores.std()), 6),
                "median_routing": round(float(np.median(scores)), 6),
                "frac_correct_routing": round(frac_correct_routing, 4),
                "mean_attn_correct": round(float(a_c.mean()), 6),
                "mean_attn_biased": round(float(a_b.mean()), 6),
                "mean_attn_neutral": round(float(a_n.mean()), 6),
                "mean_option_attn_frac": round(float(opt_frac.mean()), 6),
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val), 6),
            }
            layer_result["heads"][str(h)] = head_result

            sig = "*" if p_val < 0.05 else " "
            direction = "→correct" if scores.mean() > 0 else "→biased "
            print(f"      L{l}H{h}: routing={scores.mean():+.4f} "
                  f"({frac_correct_routing:.0%} correct) "
                  f"t={t_stat:.2f} p={p_val:.4f} {sig} "
                  f"attn_c={a_c.mean():.4f} attn_b={a_b.mean():.4f} "
                  f"opt_frac={opt_frac.mean():.3f} {direction}",
                  flush=True)

        results["layers"][str(l)] = layer_result

    return results


def main():
    # Load probes once
    behavior = BiasBehavior()
    probes = behavior.load_probes(n=300, seed=42)
    print(f"Loaded {len(probes)} bias probes\n")

    all_results = {}

    for scale in ["3M", "5M", "7M"]:
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
            dirname = info[cond]
            model_path = SCALE_DIR / dirname / "model"
            if not model_path.exists():
                print(f"  SKIP {cond}: {model_path} not found")
                continue

            print(f"\n  [{cond}] Loading {dirname}...")
            model, tokenizer = load_model(dirname)

            print(f"    Analyzing attention routing...")
            result = analyze_attention_routing(
                model, tokenizer, probes,
                info["n_layers"], info["n_heads"],
            )
            scale_results["conditions"][cond] = result
            del model, tokenizer

        all_results[scale] = scale_results

    # ── Summary Table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ATTENTION ROUTING — COMPOSITIONAL GATE SUMMARY")
    print(f"{'='*70}\n")

    cross_rho = {"3M": 0.000, "5M": 0.292, "7M": 0.208}

    print(f"  {'Scale':<5s} {'d_h':>4s} {'Cond':<12s} {'Layer':>5s} {'Head':>5s} "
          f"{'Routing':>9s} {'%Correct':>9s} {'t-stat':>8s} {'p':>8s} {'Cross-ρ':>8s}")
    print(f"  {'-'*80}")

    for scale in ["3M", "5M", "7M"]:
        r = all_results[scale]
        d_head = r["d_head"]
        for cond in ["vanilla", "bias_only", "syco_only"]:
            if cond not in r["conditions"]:
                continue
            cr = r["conditions"][cond]
            for l_str, lr in sorted(cr["layers"].items()):
                for h_str, hr in sorted(lr["heads"].items()):
                    sig = "*" if hr["p_value"] < 0.05 else " "
                    print(f"  {scale:<5s} {d_head:>4d} {cond:<12s} "
                          f"L{l_str:>4s} H{h_str:>4s} "
                          f"{hr['mean_routing']:>+8.4f} "
                          f"{hr['frac_correct_routing']:>8.0%} "
                          f"{hr['t_statistic']:>8.2f} "
                          f"{hr['p_value']:>8.4f}{sig} "
                          f"{cross_rho[scale]:>8.3f}")
        print()

    # ── Key Comparison ───────────────────────────────────────────────
    print(f"\n  KEY: Does bias_only training create a 'routing head'?")
    print(f"  (A head that significantly routes attention toward the correct answer)")
    print(f"  {'-'*65}")

    for scale in ["3M", "5M", "7M"]:
        r = all_results[scale]
        d_head = r["d_head"]
        last_layer = str(r["n_layers"] - 1)
        n_heads = r["n_heads"]

        for cond in ["vanilla", "bias_only", "syco_only"]:
            if cond not in r["conditions"]:
                continue
            cr = r["conditions"][cond]
            if last_layer not in cr["layers"]:
                continue
            lr = cr["layers"][last_layer]

            # Find head with strongest positive routing
            best_head = None
            best_routing = -999
            for h_str, hr in lr["heads"].items():
                if hr["mean_routing"] > best_routing:
                    best_routing = hr["mean_routing"]
                    best_head = h_str

            if best_head:
                hr = lr["heads"][best_head]
                sig = "★" if hr["p_value"] < 0.01 else ("*" if hr["p_value"] < 0.05 else " ")
                print(f"  {scale} (d_h={d_head:>2d}) {cond:<12s}: "
                      f"best=H{best_head} routing={hr['mean_routing']:+.4f} "
                      f"({hr['frac_correct_routing']:.0%} correct) "
                      f"p={hr['p_value']:.4f} {sig} "
                      f"| cross-ρ={cross_rho[scale]:.3f}")

    # ── Save ─────────────────────────────────────────────────────────
    out_path = OUT_DIR / "attention_routing_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    # ── Visualization ────────────────────────────────────────────────
    plot_routing_heatmap(all_results, cross_rho)
    plot_routing_bars(all_results, cross_rho)


def plot_routing_heatmap(all_results, cross_rho):
    """Heatmap: routing score per head across all conditions."""
    # Collect data for heatmap
    rows = []
    row_labels = []
    for scale in ["3M", "5M", "7M"]:
        r = all_results[scale]
        last_layer = str(r["n_layers"] - 1)
        d_head = r["d_head"]
        n_heads = r["n_heads"]

        for cond in ["vanilla", "bias_only", "syco_only"]:
            if cond not in r["conditions"]:
                continue
            cr = r["conditions"][cond]
            if last_layer not in cr["layers"]:
                continue
            lr = cr["layers"][last_layer]

            row = []
            for h in range(n_heads):
                h_str = str(h)
                if h_str in lr["heads"]:
                    row.append(lr["heads"][h_str]["mean_routing"])
                else:
                    row.append(0.0)
            # Pad to max heads (4 for 7M)
            while len(row) < 4:
                row.append(np.nan)
            rows.append(row)

            rho_str = f"ρ={cross_rho[scale]:.3f}"
            row_labels.append(f"{scale} {cond}\n(d_h={d_head}) [{rho_str}]")

    data = np.array(rows)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use diverging colormap centered at 0
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    if vmax < 0.001:
        vmax = 0.01
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Head 0", "Head 1", "Head 2", "Head 3"])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Annotate with values
    for i in range(len(rows)):
        for j in range(4):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:+.4f}", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_title("Attention Routing Score (Last Layer)\n"
                 "Positive = routes toward correct answer",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Attention Head")

    plt.colorbar(im, ax=ax, label="Routing Score (attn_correct − attn_biased)")
    plt.tight_layout()

    path = FIG_DIR / "attention_routing_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_routing_bars(all_results, cross_rho):
    """Bar chart: per-head routing score, grouped by scale."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    colors = {"vanilla": "#999999", "bias_only": "#2196F3", "syco_only": "#FF5722"}

    for ax_idx, scale in enumerate(["3M", "5M", "7M"]):
        ax = axes[ax_idx]
        r = all_results[scale]
        last_layer = str(r["n_layers"] - 1)
        n_heads = r["n_heads"]
        d_head = r["d_head"]

        conditions = [c for c in ["vanilla", "bias_only", "syco_only"]
                      if c in r["conditions"]]
        n_conds = len(conditions)

        # Group bar positions
        x = np.arange(n_heads)
        width = 0.25
        offsets = np.linspace(-(n_conds - 1) * width / 2,
                              (n_conds - 1) * width / 2, n_conds)

        for c_idx, cond in enumerate(conditions):
            cr = r["conditions"][cond]
            if last_layer not in cr["layers"]:
                continue
            lr = cr["layers"][last_layer]

            values = []
            errors = []
            sigs = []
            for h in range(n_heads):
                h_str = str(h)
                if h_str in lr["heads"]:
                    hr = lr["heads"][h_str]
                    values.append(hr["mean_routing"])
                    errors.append(hr["std_routing"] / np.sqrt(cr["n_valid"]))
                    sigs.append(hr["p_value"] < 0.05)
                else:
                    values.append(0)
                    errors.append(0)
                    sigs.append(False)

            bars = ax.bar(x + offsets[c_idx], values, width,
                          yerr=errors, label=cond,
                          color=colors[cond], alpha=0.85,
                          capsize=2, edgecolor="black", linewidth=0.5)

            # Mark significant bars
            for b_idx, (bar, sig) in enumerate(zip(bars, sigs)):
                if sig:
                    y = bar.get_height()
                    sign = 1 if y >= 0 else -1
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            y + sign * 0.001, "*",
                            ha="center", va="bottom" if y >= 0 else "top",
                            fontsize=12, fontweight="bold", color=colors[cond])

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
        ax.set_title(f"{scale} (d_head={d_head}, {r['n_layers']}L)\n"
                     f"cross-ρ = {cross_rho[scale]:.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Head (last layer)")

        if ax_idx == 0:
            ax.set_ylabel("Routing Score\n(attn_correct − attn_biased)")
            ax.legend(fontsize=8, loc="best")

    fig.suptitle("Compositional Gate: Attention Head Routing on Bias Probes",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = FIG_DIR / "attention_routing_bars.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
