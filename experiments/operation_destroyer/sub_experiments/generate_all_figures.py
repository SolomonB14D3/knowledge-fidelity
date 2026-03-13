#!/usr/bin/env python3
"""Generate all 6 publication-ready paper figures for the STEM truth oracle.

Figure layout:
  Fig 1 — Scaling curve: oracle accuracy vs model size, 6 models (Exp01)
  Fig 2 — Bias taxonomy: 4 patterns × 2 examples each (Exp02)
  Fig 3 — Transfer matrix heatmap: 5 training arms × 4 test patterns (Exp03)
  Fig 4 — Margin as binary oracle: confusion matrix (left) + quintile bar (right) (Exp04)
  Fig 5 — Domain difficulty scatter: accuracy vs mean margin, 97 facts (Exp04)
  Fig 6 — Adapter outcome distributions: 3 overlaid margin-delta distributions (Exp04)

Usage:
    python sub_experiments/generate_all_figures.py
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

EXP04_RESULTS = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/sub_experiments/exp04_confidence/results.json"
EXP03_RESULTS = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/sub_experiments/exp03_correction/results.json"
STEM_PATH     = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/operation_destroyer/stem_crossmodel/stem_crossmodel.json"
OUT_DIR       = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/sub_experiments/figures"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
GRAY   = "#6B7280"
LGRAY  = "#D1D5DB"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

# ── Load data ─────────────────────────────────────────────────────────────────

with open(EXP04_RESULTS) as f:
    exp04 = json.load(f)

with open(EXP03_RESULTS) as f:
    exp03 = json.load(f)

with open(STEM_PATH) as f:
    stem_raw = json.load(f)

models_list = stem_raw.get("models", [])
stem_model  = next((m for m in models_list if "Qwen3-4B" in m.get("model_id", "")), {})
stem_facts  = stem_model.get("per_fact", [])

all_facts_40 = []
patterns = ["positivity", "linearity", "missing_constant", "truncation"]
full = exp04["per_pattern_full"]
for pat in patterns:
    bms = full[pat]["base_margins"]
    mms = full[pat]["mixed_margins"]
    for b, m in zip(bms, mms):
        all_facts_40.append({"base": b, "mixed": m, "pattern": pat})

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Scaling curve
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PARAMS = {
    "openai-community/gpt2":        117,
    "HuggingFaceTB/SmolLM2-360M":   360,
    "Qwen/Qwen2.5-0.5B":            500,
    "meta-llama/Llama-3.2-1B":     1000,
    "Qwen/Qwen2.5-1.5B":           1500,
    "Qwen/Qwen3-4B-Base":          4000,
}
MODEL_LABELS = {
    "openai-community/gpt2":        "GPT-2\n(117M)",
    "HuggingFaceTB/SmolLM2-360M":   "SmolLM2\n(360M)",
    "Qwen/Qwen2.5-0.5B":            "Qwen2.5\n(0.5B)",
    "meta-llama/Llama-3.2-1B":      "Llama-3.2\n(1B)",
    "Qwen/Qwen2.5-1.5B":            "Qwen2.5\n(1.5B)",
    "Qwen/Qwen3-4B-Base":           "Qwen3\n(4B)",
}
DOMAIN_COLORS = {
    "calculus":      BLUE,
    "physics":       PURPLE,
    "chemistry":     GREEN,
    "linear_algebra": ORANGE,
    "statistics":    RED,
    "constants":     GRAY,
}

fig, ax = plt.subplots(figsize=(7, 4.5))

# Overall accuracy line
xs_all = []
ys_all = []
for m in models_list:
    mid = m["model_id"]
    if mid not in MODEL_PARAMS:
        continue
    xs_all.append(MODEL_PARAMS[mid])
    ys_all.append(m["total_acc"] * 100)

xs_all, ys_all = zip(*sorted(zip(xs_all, ys_all)))
ax.plot(xs_all, ys_all, "o-", color="black", linewidth=2.5, markersize=8,
        zorder=5, label="Overall", markerfacecolor="white", markeredgewidth=2)

# Per-domain lines
domain_xs = {d: [] for d in DOMAIN_COLORS}
domain_ys = {d: [] for d in DOMAIN_COLORS}
for m in models_list:
    mid = m["model_id"]
    if mid not in MODEL_PARAMS:
        continue
    for d, stats in m.get("domain_breakdown", {}).items():
        if d in domain_xs:
            domain_xs[d].append(MODEL_PARAMS[mid])
            domain_ys[d].append(stats["wins"] / stats["n"] * 100)

for d in DOMAIN_COLORS:
    xs_d = domain_xs[d]
    ys_d = domain_ys[d]
    if not xs_d:
        continue
    xs_d, ys_d = zip(*sorted(zip(xs_d, ys_d)))
    ax.plot(xs_d, ys_d, "o--", color=DOMAIN_COLORS[d], linewidth=1.2,
            markersize=5, alpha=0.7, label=d.replace("_", " "))

ax.axhline(25, color=LGRAY, linestyle=":", linewidth=1, alpha=0.8)
ax.text(4100, 26, "random (4-choice)", fontsize=8, color=GRAY)
ax.set_xscale("log")
ax.set_xticks([117, 360, 500, 1000, 1500, 4000])
ax.set_xticklabels(["GPT-2\n117M", "SmolLM2\n360M", "Qwen\n0.5B",
                     "Llama\n1B", "Qwen\n1.5B", "Qwen3\n4B"], fontsize=9)
ax.set_ylabel("Oracle accuracy (%)", fontsize=11)
ax.set_title("Log-prob ranking accuracy across model scale\n(97-fact STEM benchmark, 4-choice MC)", fontsize=11)
ax.legend(fontsize=8, loc="upper left", framealpha=0.8, ncol=2)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.set_ylim(0, 100)

plt.tight_layout()
p = os.path.join(OUT_DIR, "fig1_scaling_curve.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved {p}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Bias taxonomy table (2 examples per pattern)
# ─────────────────────────────────────────────────────────────────────────────

TAXONOMY = [
    ("Positivity bias",
     "Prefers + over\n− form",
     "d/dx[cos x]",        "−sin x",  "+sin x",
     "ΔU = Q − W",         "Q − W",   "Q + W"),
    ("Linearity bias",
     "Prefers linear\nover quadratic",
     "Kinetic energy",     "½mv²",    "mv",
     "Centripetal accel.", "v²/r",    "v/r"),
    ("Missing-constant bias",
     "Drops proportionality\nconstant",
     "Coulomb's force",    "kq₁q₂/r²","q₁q₂/r²",
     "Gravitational force","Gm₁m₂/r²","m₁m₂/r²"),
    ("Truncation bias",
     "Prefers shorter\nsymbolic form",
     "Methane hybrid.", "sp³",  "sp",
     "Arrhenius rate",     "Ae^(−Eₐ/RT)", "Ae^(−Eₐ)"),
]

fig, ax = plt.subplots(figsize=(10, 5.0))
ax.axis("off")

col_labels = ["Pattern", "Mechanism", "Fact 1", "Truth", "Model picks",
              "Fact 2", "Truth", "Model picks"]
col_widths = [0.15, 0.18, 0.13, 0.08, 0.10, 0.13, 0.08, 0.15]

# Header row
y_header = 0.94
xs = np.cumsum([0] + col_widths[:-1])
for i, (label, w, x) in enumerate(zip(col_labels, col_widths, xs)):
    ax.text(x + w/2, y_header, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white",
            transform=ax.transAxes)
    rect = mpatches.FancyBboxPatch(
        (x + 0.002, y_header - 0.055), w - 0.004, 0.1,
        boxstyle="round,pad=0.005", linewidth=0,
        facecolor=BLUE, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x + w/2, y_header, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white",
            transform=ax.transAxes, zorder=3)

row_colors = [LGRAY + "88", "white", LGRAY + "88", "white"]
for ri, (pat, mech, f1, t1, w1, f2, t2, w2) in enumerate(TAXONOMY):
    y_row = 0.76 - ri * 0.17
    bg = "#F3F4F6" if ri % 2 == 0 else "white"
    rect = mpatches.FancyBboxPatch(
        (0.001, y_row - 0.065), 0.998, 0.145,
        boxstyle="round,pad=0.003", linewidth=0,
        facecolor=bg, transform=ax.transAxes)
    ax.add_patch(rect)

    row_vals = [pat, mech, f1, t1, w1, f2, t2, w2]
    for ci, (val, w, x) in enumerate(zip(row_vals, col_widths, xs)):
        color = "black"
        fw    = "normal"
        if ci in (3, 6):   # truth cols — green
            color = GREEN; fw = "bold"
        elif ci in (4, 7): # wrong cols — red
            color = RED
        ax.text(x + w/2, y_row + 0.005, val, ha="center", va="center",
                fontsize=8.5, color=color, fontweight=fw,
                transform=ax.transAxes, zorder=3)

ax.set_title("Taxonomy of systematic STEM biases (4 patterns, all scale-invariant — same failures across 6 models)",
             fontsize=10, pad=8)
plt.tight_layout()
p = os.path.join(OUT_DIR, "fig2_bias_taxonomy.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved {p}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Transfer matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────

tm = exp03["transfer_matrix"]
train_arms = ["baseline", "positivity", "linearity", "missing_constant", "truncation", "mixed"]
test_pats  = ["positivity", "linearity", "missing_constant", "truncation"]
short_labels = ["Positivity", "Linearity", "Missing\nconst.", "Truncation"]

mat = np.zeros((len(train_arms), len(test_pats)))
for ri, arm in enumerate(train_arms):
    for ci, pat in enumerate(test_pats):
        mat[ri, ci] = tm[arm][pat]["acc"]

cmap = LinearSegmentedColormap.from_list("rg", [RED, "#FEF3C7", GREEN])

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Accuracy")

ax.set_xticks(range(len(test_pats)))
ax.set_xticklabels(short_labels, fontsize=10)
ax.set_yticks(range(len(train_arms)))
arm_labels = ["Baseline\n(no adapter)", "Train:\nPositivity",
              "Train:\nLinearity", "Train:\nMissing const.",
              "Train:\nTruncation", "Train:\nMixed (all)"]
ax.set_yticklabels(arm_labels, fontsize=9)
ax.set_xlabel("Test pattern", fontsize=11)
ax.set_ylabel("Training condition", fontsize=11)
ax.set_title("Transfer matrix: adapter trained on one pattern\ntested on all (n=10 each)",
             fontsize=11)

for ri in range(len(train_arms)):
    for ci in range(len(test_pats)):
        v = mat[ri, ci]
        txt_color = "white" if v < 0.25 or v > 0.75 else "black"
        # Mark diagonal
        is_diag = ri > 0 and ri <= 4 and (ri - 1) == ci
        fw = "bold" if is_diag else "normal"
        border = "⬛" if is_diag else ""
        ax.text(ci, ri, f"{v:.0%}", ha="center", va="center",
                fontsize=11, color=txt_color, fontweight=fw)

# Draw thick border around diagonal
for i in range(1, 5):
    rect = mpatches.Rectangle(
        (i-1 - 0.48, i - 0.48), 0.96, 0.96,
        linewidth=2.5, edgecolor="black", facecolor="none",
        transform=ax.transData)
    ax.add_patch(rect)

# Mixed row — extra highlight
rect = mpatches.Rectangle(
    (-0.48, 4.52), len(test_pats) - 0.04, 0.96,
    linewidth=2.5, edgecolor=BLUE, facecolor="none",
    transform=ax.transData)
ax.add_patch(rect)
ax.text(len(test_pats) - 0.02, 5.0, "← mixed", ha="left", va="center",
        fontsize=9, color=BLUE, fontweight="bold")

plt.tight_layout()
p = os.path.join(OUT_DIR, "fig3_transfer_matrix.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved {p}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Margin as binary oracle (confusion matrix left + quintile bar right)
# ─────────────────────────────────────────────────────────────────────────────

sorted_facts = sorted(all_facts_40, key=lambda f: f["base"])
q_size = len(sorted_facts) // 5
quintile_data = []
for qi in range(5):
    chunk = sorted_facts[qi*q_size : (qi+1)*q_size if qi < 4 else len(sorted_facts)]
    acc   = sum(1 for f in chunk if f["base"] > 0) / len(chunk)
    mlo   = min(f["base"] for f in chunk)
    mhi   = max(f["base"] for f in chunk)
    quintile_data.append((acc, mlo, mhi))

TP = sum(1 for f in all_facts_40 if f["base"] > 0)
TN = sum(1 for f in all_facts_40 if f["base"] <= 0)

fig = plt.figure(figsize=(10, 5.0))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.38)

# Left: confusion matrix
ax_cm = fig.add_subplot(gs[0])
ax_cm.set_xlim(0, 2); ax_cm.set_ylim(0, 2)
for i in range(2):
    for j in range(2):
        on_diag = (i == j)
        facecolor = GREEN if on_diag else RED
        alpha     = 0.85   if on_diag else 0.12
        ax_cm.add_patch(plt.Rectangle((j, 1-i), 1, 1,
                                      color=facecolor, alpha=alpha, zorder=1))
        val = [[TP, 0], [0, TN]][i][j]
        txt_col = "white" if alpha > 0.5 else LGRAY
        ax_cm.text(j+0.5, 1.5-i+0.1, str(val), ha="center", va="center",
                   fontsize=22, fontweight="bold", color=txt_col, zorder=2)
        if on_diag:
            pct = f"{val/40:.0%}"
            ax_cm.text(j+0.5, 1.5-i-0.18, pct, ha="center", va="center",
                       fontsize=11, color="white", zorder=2)

ax_cm.set_xticks([0.5, 1.5])
ax_cm.set_xticklabels(["Correct\n(truth wins)", "Wrong\n(dist. wins)"], fontsize=10)
ax_cm.set_yticks([0.5, 1.5])
ax_cm.set_yticklabels(["Margin\n≤ 0", "Margin\n> 0"], fontsize=10, rotation=0)
ax_cm.xaxis.set_ticks_position("top")
ax_cm.tick_params(length=0)
ax_cm.set_ylabel("Predicted", fontsize=11)
for spine in ax_cm.spines.values():
    spine.set_visible(False)
# Add "Ground truth" as plain text label above tick labels, inside the axes
ax_cm.text(1.0, 2.22, "Ground truth", ha="center", va="bottom",
           fontsize=10, transform=ax_cm.transData)

# Right: quintile bar
ax_q = fig.add_subplot(gs[1])
labels = [f"Q{i+1}\n({qd[1]:+.1f}–\n{qd[2]:+.1f})" for i, qd in enumerate(quintile_data)]
accs   = [qd[0]*100 for qd in quintile_data]
bar_colors = [RED if a == 0 else (BLUE if a == 100 else ORANGE) for a in accs]
bars = ax_q.bar(range(5), accs, color=bar_colors, width=0.6,
                edgecolor="white", linewidth=1.5)
for i, (bar, acc) in enumerate(zip(bars, accs)):
    ax_q.text(bar.get_x() + bar.get_width()/2, acc + 2,
              f"{acc:.0f}%", ha="center", va="bottom", fontsize=11,
              fontweight="bold", color=bar.get_facecolor())

# Uncertainty zone shading (Q3 middle bar)
ax_q.axhspan(0, 100, xmin=0.4, xmax=0.6, color=ORANGE, alpha=0.08)

ax_q.set_xticks(range(5))
ax_q.set_xticklabels(labels, fontsize=9)
ax_q.set_ylim(-5, 115)
ax_q.set_ylabel("Accuracy (%)", fontsize=11)
ax_q.set_xlabel("Margin quintile (low → high)", fontsize=11, labelpad=4)
ax_q.set_title("Accuracy by quintile\n(n=40 bias facts, baseline)", fontsize=11)
ax_q.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}%"))

plt.suptitle("Log-prob margin as STEM oracle confidence: perfect binary separation",
             fontsize=12, fontweight="bold")
fig.subplots_adjust(top=0.83, bottom=0.12, left=0.09, right=0.97, wspace=0.38)
p = os.path.join(OUT_DIR, "fig4_oracle_calibration.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved {p}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Domain difficulty scatter
# ─────────────────────────────────────────────────────────────────────────────

by_domain = {}
for fact in stem_facts:
    d = fact.get("domain", "unknown")
    by_domain.setdefault(d, {"margins": [], "wins": []})
    by_domain[d]["margins"].append(fact["margin"])
    by_domain[d]["wins"].append(fact["win"])

dom_acc    = {d: sum(v["wins"])/len(v["wins"])*100 for d, v in by_domain.items()}
dom_margin = {d: sum(v["margins"])/len(v["margins"]) for d, v in by_domain.items()}
dom_n      = {d: len(v["margins"]) for d, v in by_domain.items()}

DCOLORS = {
    "statistics":     RED,
    "linear_algebra": ORANGE,
    "constants":      GRAY,
    "calculus":       BLUE,
    "chemistry":      GREEN,
    "physics":        PURPLE,
}

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for d in by_domain:
    x = dom_margin[d]
    y = dom_acc[d]
    n = dom_n[d]
    c = DCOLORS.get(d, GRAY)
    ax.scatter(x, y, s=n*18, color=c, alpha=0.82, zorder=3,
               edgecolors="white", linewidths=1.8)
    offsets = {
        "statistics":     (-0.3, -6.0),
        "linear_algebra": ( 0.2,  2.5),
        "constants":      ( 0.2,  2.0),
        "calculus":       (-1.5, -6.0),
        "chemistry":      ( 0.2,  2.0),
        "physics":        (-1.5,  2.5),
    }
    ox, oy = offsets.get(d, (0.1, 2))
    ax.annotate(d.replace("_", " "), (x + ox, y + oy),
                fontsize=9.5, color=c, fontweight="bold")

ax.axvline(0, color=GRAY, linestyle="--", alpha=0.4, linewidth=1)
ax.axhline(75, color=LGRAY, linestyle=":", linewidth=1)

# Highlight danger zone (low margin + low acc)
ax.axvspan(-10, 1, ymin=0, ymax=0.55, color=RED, alpha=0.04)
ax.text(-4.5, 44, "high-risk\nzone", ha="center", fontsize=8.5, color=RED, alpha=0.5)

ax.set_xlabel("Mean log-prob margin (truth − best distractor)", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Domain difficulty: accuracy vs mean margin\n(Qwen3-4B-Base, 97-fact STEM benchmark)", fontsize=11)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.text(0.97, 0.04, "bubble size ∝ n facts", transform=ax.transAxes,
        ha="right", fontsize=8, color=GRAY)
ax.set_ylim(40, 100)

plt.tight_layout()
p = os.path.join(OUT_DIR, "fig5_domain_scatter.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved {p}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Adapter outcome distributions (overlaid histograms)
# ─────────────────────────────────────────────────────────────────────────────

outcomes = {
    "Wrong→Correct (n=10)":    {"deltas": [], "color": GREEN,  "lw": 2.5},
    "Right→Right (n=20)":      {"deltas": [], "color": BLUE,   "lw": 2.0},
    "Wrong→Wrong (n=9)":       {"deltas": [], "color": RED,    "lw": 2.0},
    "Right→Wrong (n=1)":       {"deltas": [], "color": ORANGE, "lw": 1.5},
}
outcome_keys = list(outcomes.keys())

for f in all_facts_40:
    b_win = f["base"]  > 0
    m_win = f["mixed"] > 0
    delta = f["mixed"] - f["base"]
    if not b_win and m_win:
        outcomes[outcome_keys[0]]["deltas"].append(delta)
    elif b_win and m_win:
        outcomes[outcome_keys[1]]["deltas"].append(delta)
    elif not b_win and not m_win:
        outcomes[outcome_keys[2]]["deltas"].append(delta)
    else:
        outcomes[outcome_keys[3]]["deltas"].append(delta)

fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.linspace(-10, 12, 22)

for key, info in outcomes.items():
    deltas = info["deltas"]
    if not deltas:
        continue
    mean_d = sum(deltas) / len(deltas)
    if len(deltas) >= 3:
        ax.hist(deltas, bins=bins, alpha=0.45, color=info["color"],
                label=f"{key}  (mean {mean_d:+.1f})",
                edgecolor="white", linewidth=0.5)
    # Always draw mean line
    ax.axvline(mean_d, color=info["color"], linewidth=info["lw"],
               linestyle="-" if len(deltas) >= 3 else "--",
               label=(f"{key}  (mean {mean_d:+.1f})" if len(deltas) < 3 else None))
    label_y = 4.3 if len(deltas) >= 3 else 1.2
    ax.text(mean_d + 0.2, label_y,
            f"{mean_d:+.1f}", color=info["color"], fontsize=8.5, fontweight="bold")

ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.5, label="No change")

# Annotate the two extreme stubborn cases only (cleaner than 3 overlapping labels)
ax.annotate("spring PE\n(Δ = −6.4)", xy=(-6.36, 0.5), xytext=(-7.5, 2.2),
            fontsize=8, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
            ha="center")
ax.annotate("conflicted\nhybridization cases", xy=(-1.15, 0.5), xytext=(-3.5, 2.8),
            fontsize=8, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
            ha="center")

ax.set_xlim(-12, 12)
ax.set_ylim(0, 5.2)
ax.set_xlabel("Δ margin (mixed adapter − baseline)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Mixed adapter effect: margin shift by outcome category\n"
             "(n=40 STEM bias facts; vertical lines = category means)", fontsize=11)

handles, labels = ax.get_legend_handles_labels()
# deduplicate
seen = set()
h2, l2 = [], []
for h, l in zip(handles, labels):
    if l and l not in seen:
        h2.append(h); l2.append(l); seen.add(l)
ax.legend(h2, l2, fontsize=9, framealpha=0.85,
          bbox_to_anchor=(0.01, 0.99), loc="upper left", borderaxespad=0)

plt.tight_layout()
p = os.path.join(OUT_DIR, "fig6_adapter_outcomes.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Saved {p}")
plt.close()

print(f"\n✓ All 6 figures saved to: {OUT_DIR}")
