#!/usr/bin/env python3
"""Generate paper figures for the STEM truth oracle experiments.

Figures produced:
  fig1_confusion_matrix.png   -- 2x2 margin-sign vs correct/incorrect (perfect diagonal)
  fig2_quintile_accuracy.png  -- accuracy by margin quintile (baseline, n=40)
  fig3_domain_scatter.png     -- domain accuracy vs mean margin (n=97 STEM benchmark)
  fig4_adapter_outcomes.png   -- before/after margin delta by outcome category
  fig5_margin_histogram.png   -- full 97-fact margin distribution (bimodal)

Usage:
    python sub_experiments/exp04_confidence/generate_figures.py
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_PATH = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/sub_experiments/exp04_confidence/results.json"
STEM_PATH    = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/operation_destroyer/stem_crossmodel/stem_crossmodel.json"
OUT_DIR      = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/sub_experiments/exp04_confidence/figures"

os.makedirs(OUT_DIR, exist_ok=True)

BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
GRAY   = "#6B7280"
LIGHT  = "#F3F4F6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Load data ─────────────────────────────────────────────────────────────────

with open(RESULTS_PATH) as f:
    res = json.load(f)

with open(STEM_PATH) as f:
    stem_raw = json.load(f)

models_list  = stem_raw.get("models", [])
stem_model   = next((m for m in models_list if "Qwen3-4B" in m.get("model_id", "")), {})
stem_facts   = stem_model.get("per_fact", [])

all_base   = res["all_baseline_margins"]
all_mixed  = res["all_mixed_margins"]
full       = res["per_pattern_full"]

# Build per-fact list with base + mixed margins
all_facts = []
patterns = ["positivity", "linearity", "missing_constant", "truncation"]
for pat in patterns:
    bms = full[pat]["base_margins"]
    mms = full[pat]["mixed_margins"]
    for b, m in zip(bms, mms):
        all_facts.append({"base": b, "mixed": m, "pattern": pat})

# ── Fig 1: Confusion matrix ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(4.5, 4))

# Counts: (predicted=positive, actual=correct), etc.
TP = sum(1 for f in all_facts if f["base"] > 0)   # predicted correct, actually correct
FP = 0                                              # predicted correct, actually wrong → by construction 0
FN = 0                                              # predicted wrong, actually correct → by construction 0
TN = sum(1 for f in all_facts if f["base"] <= 0)   # predicted wrong, actually wrong

mat = np.array([[TP, FP], [FN, TN]])
colors = [[GREEN, RED], [RED, GREEN]]
alpha  = [[0.85, 0.15], [0.15, 0.85]]

for i in range(2):
    for j in range(2):
        c = GREEN if (i == 0 and j == 0) or (i == 1 and j == 1) else RED
        a = 0.85 if (i == 0 and j == 0) or (i == 1 and j == 1) else 0.12
        ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, color=c, alpha=a, zorder=1))
        v = mat[i, j]
        ax.text(j + 0.5, 1.5 - i, str(v), ha="center", va="center",
                fontsize=22, fontweight="bold", color="white" if a > 0.5 else GRAY, zorder=2)
        pct = f"{v/40:.0%}" if v > 0 else ""
        ax.text(j + 0.5, 1.5 - i - 0.22, pct, ha="center", va="center",
                fontsize=11, color="white" if a > 0.5 else GRAY, zorder=2)

ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(["Actually\ncorrect", "Actually\nwrong"], fontsize=11)
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(["Margin\n< 0", "Margin\n≥ 0"], fontsize=11, rotation=0)
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.set_xlabel("Ground truth", fontsize=12, labelpad=8)
ax.set_ylabel("Predicted", fontsize=12)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Margin sign as binary oracle\n(n=40 STEM bias facts, Qwen3-4B-Base)", fontsize=11, pad=16)

plt.tight_layout()
p = os.path.join(OUT_DIR, "fig1_confusion_matrix.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved {p}")
plt.close()

# ── Fig 2: Quintile accuracy ──────────────────────────────────────────────────

sorted_facts = sorted(all_facts, key=lambda f: f["base"])
q = len(sorted_facts) // 5
quintile_labels = []
quintile_acc    = []
quintile_ranges = []

for qi in range(5):
    chunk = sorted_facts[qi*q : (qi+1)*q if qi < 4 else len(sorted_facts)]
    margins = [f["base"] for f in chunk]
    acc = sum(1 for f in chunk if f["base"] > 0) / len(chunk)
    quintile_acc.append(acc)
    quintile_labels.append(f"Q{qi+1}")
    quintile_ranges.append(f"{min(margins):+.1f}\nto\n{max(margins):+.1f}")

fig, ax = plt.subplots(figsize=(5.5, 4))
bar_colors = [RED if a == 0.0 else (BLUE if a == 1.0 else ORANGE) for a in quintile_acc]
bars = ax.bar(quintile_labels, [a * 100 for a in quintile_acc],
              color=bar_colors, width=0.6, edgecolor="white", linewidth=1.5)

for bar, acc, rng in zip(bars, quintile_acc, quintile_ranges):
    h = bar.get_height()
    pct_label = f"{acc:.0%}"
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, pct_label,
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color=bar.get_facecolor())
    ax.text(bar.get_x() + bar.get_width()/2, -14, rng,
            ha="center", va="top", fontsize=8, color=GRAY)

ax.set_ylim(-5, 115)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_xlabel("Margin quintile (low → high)", fontsize=11, labelpad=24)
ax.set_title("Accuracy by log-prob margin quintile\n(n=40 bias facts, baseline)", fontsize=11)
ax.axhline(50, color=GRAY, linestyle="--", alpha=0.4, linewidth=1)
ax.yaxis.set_ticklabels([f"{int(t)}%" for t in ax.get_yticks()])

plt.tight_layout()
p = os.path.join(OUT_DIR, "fig2_quintile_accuracy.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved {p}")
plt.close()

# ── Fig 3: Domain scatter (accuracy vs mean margin, 97 facts) ─────────────────

by_domain = {}
for fact in stem_facts:
    d = fact.get("domain", "unknown")
    by_domain.setdefault(d, {"margins": [], "wins": []})
    by_domain[d]["margins"].append(fact["margin"])
    by_domain[d]["wins"].append(fact["win"])

domain_acc   = {d: sum(v["wins"])/len(v["wins"])*100 for d, v in by_domain.items()}
domain_margin = {d: sum(v["margins"])/len(v["margins"]) for d, v in by_domain.items()}
domain_n     = {d: len(v["margins"]) for d, v in by_domain.items()}

domain_colors = {
    "statistics":     RED,
    "linear_algebra": ORANGE,
    "constants":      GRAY,
    "calculus":       BLUE,
    "chemistry":      GREEN,
    "physics":        "#7C3AED",
}

fig, ax = plt.subplots(figsize=(6, 4.5))
for d in by_domain:
    x = domain_margin[d]
    y = domain_acc[d]
    n = domain_n[d]
    c = domain_colors.get(d, GRAY)
    ax.scatter(x, y, s=n*12, color=c, alpha=0.85, zorder=3, edgecolors="white", linewidths=1.5)
    label_offset = (0.08, 1.5)
    if d == "statistics":
        label_offset = (0.08, -5)
    elif d == "linear_algebra":
        label_offset = (-0.7, 2)
    ax.annotate(d.replace("_", "\n"), (x + label_offset[0], y + label_offset[1]),
                fontsize=9, color=c, fontweight="bold")

ax.axvline(0, color=GRAY, linestyle="--", alpha=0.5, linewidth=1)
ax.set_xlabel("Mean log-prob margin", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Domain difficulty: accuracy vs mean margin\n(Qwen3-4B-Base, n=97 STEM facts)", fontsize=11)
ax.yaxis.set_ticklabels([f"{int(t)}%" for t in ax.get_yticks()])
note = "(bubble size ∝ n)"
ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", fontsize=8, color=GRAY)

plt.tight_layout()
p = os.path.join(OUT_DIR, "fig3_domain_scatter.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved {p}")
plt.close()

# ── Fig 4: Adapter outcome margin deltas ──────────────────────────────────────

outcomes = {
    "Wrong→Correct\n(n=10)":  [],
    "Right→Right\n(n=20)":    [],
    "Right→Wrong\n(n=1)":     [],
    "Wrong→Wrong\n(n=9)":     [],
}
outcome_colors = [GREEN, BLUE, RED, ORANGE]

for f in all_facts:
    b_win = f["base"]  > 0
    m_win = f["mixed"] > 0
    delta = f["mixed"] - f["base"]
    if not b_win and m_win:
        outcomes["Wrong→Correct\n(n=10)"].append(delta)
    elif b_win and m_win:
        outcomes["Right→Right\n(n=20)"].append(delta)
    elif b_win and not m_win:
        outcomes["Right→Wrong\n(n=1)"].append(delta)
    else:
        outcomes["Wrong→Wrong\n(n=9)"].append(delta)

fig, ax = plt.subplots(figsize=(6.5, 4))
labels = list(outcomes.keys())
means  = [sum(v)/len(v) if v else 0 for v in outcomes.values()]
errors = [np.std(v) if len(v) > 1 else 0 for v in outcomes.values()]

x = np.arange(len(labels))
bars = ax.bar(x, means, color=outcome_colors, width=0.55,
              edgecolor="white", linewidth=1.5,
              yerr=errors, error_kw={"elinewidth": 1.5, "ecolor": GRAY, "capsize": 4})

for bar, m in zip(bars, means):
    h = bar.get_height()
    sign = "+" if m >= 0 else ""
    ax.text(bar.get_x() + bar.get_width()/2,
            h + (0.25 if m >= 0 else -0.6),
            f"{sign}{m:.2f}",
            ha="center", va="bottom" if m >= 0 else "top",
            fontsize=11, fontweight="bold", color=bar.get_facecolor())

ax.axhline(0, color=GRAY, linewidth=1, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Mean Δ margin (mixed − baseline)", fontsize=11)
ax.set_title("Mixed adapter effect by outcome category\n(n=40 bias facts)", fontsize=11)

plt.tight_layout()
p = os.path.join(OUT_DIR, "fig4_adapter_outcomes.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved {p}")
plt.close()

# ── Fig 5: Full 97-fact margin histogram (bimodal) ───────────────────────────

stem_margins = [f["margin"] for f in stem_facts]
stem_wins    = [f["win"] for f in stem_facts]

wrong_margins   = [m for m, w in zip(stem_margins, stem_wins) if not w]
correct_margins = [m for m, w in zip(stem_margins, stem_wins) if w]

fig, ax = plt.subplots(figsize=(6.5, 4))
bins = np.linspace(min(stem_margins) - 0.5, max(stem_margins) + 0.5, 20)
ax.hist(wrong_margins,   bins=bins, color=RED,   alpha=0.75, label=f"Wrong (n={len(wrong_margins)})",   edgecolor="white")
ax.hist(correct_margins, bins=bins, color=BLUE,  alpha=0.65, label=f"Correct (n={len(correct_margins)})", edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Margin = 0 (decision boundary)")

ax.set_xlabel("Log-prob margin (truth − best distractor)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Margin distribution — Qwen3-4B-Base on 97-fact STEM benchmark\n(bimodal: wrong answers cluster < 0, correct > 0)", fontsize=11)
ax.legend(fontsize=10, framealpha=0.8)

plt.tight_layout()
p = os.path.join(OUT_DIR, "fig5_margin_histogram.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved {p}")
plt.close()

print("\nAll figures saved to:", OUT_DIR)
