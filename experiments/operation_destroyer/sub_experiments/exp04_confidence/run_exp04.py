#!/usr/bin/env python3
"""Exp04: Margin as oracle confidence score.

Questions:
1. Does margin reliably predict wrong answers? (negative margin = wrong)
2. What is the margin distribution by bias pattern (baseline)?
3. Does the mixed adapter uniformly improve margins, or concentrate?
4. Can margin magnitude predict *how hard* a fact is to correct?
5. Distribution across the full 97-fact STEM benchmark.

Method:
- Score all 40 exp03 training examples (4 patterns × 10 each) with no adapter
- Score same 40 with mixed adapter (from exp03)
- Score full 97-fact STEM benchmark baseline
- Compute: per-pattern margin histograms, correct/wrong separation,
  adapter improvement distribution, margin-as-calibration analysis

Usage:
    python sub_experiments/exp04_confidence/run_exp04.py
"""

import json, os, sys, time
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/snap_on")

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
from mlx.utils import tree_unflatten

from experiments.operation_destroyer.eval_mc import get_completion_logprob, score_fact_mc
from experiments.snap_on.module import SnapOnConfig, create_adapter
import experiments.operation_destroyer.train_v3 as t3

MODEL_ID  = "Qwen/Qwen3-4B-Base"
EXP03_DIR = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/sub_experiments/exp03_correction"
OUT_DIR   = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/sub_experiments/exp04_confidence"
STEM_PATH = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/operation_destroyer/stem_crossmodel/stem_crossmodel.json"

# ── Reuse exp03 bias data ─────────────────────────────────────────────────────
POSITIVITY = [
    ("derivative of cos(x)",         "-sin(x)",        ["sin(x)", "-cos(x)", "cos(x)", "tan(x)"],           "calculus"),
    ("second derivative of sin(x)",  "-sin(x)",        ["sin(x)", "cos(x)", "-cos(x)", "0"],                "calculus"),
    ("integral of sin(x)",           "-cos(x)+C",      ["cos(x)+C", "-sin(x)+C", "sin^2(x)+C", "sin(x)+C"], "calculus"),
    ("derivative of cos(2*x)",       "-2*sin(2*x)",    ["2*sin(2*x)", "-sin(2*x)", "2*cos(2*x)", "-2*cos(2*x)"], "calculus"),
    ("second derivative of cos(x)",  "-cos(x)",        ["cos(x)", "-sin(x)", "sin(x)", "0"],                 "calculus"),
    ("integral of sin(2*x)",         "-(1/2)*cos(2*x)+C", ["(1/2)*cos(2*x)+C", "-cos(2*x)+C", "cos(2*x)+C", "(1/2)*sin(2*x)+C"], "calculus"),
    ("first law of thermodynamics",  "delta(U)=Q-W",   ["delta(U)=Q+W", "delta(U)=W-Q", "Q=delta(U)+W", "delta(U)=Q*W"], "physics"),
    ("Gibbs free energy formula",    "G=H-T*S",        ["G=H+T*S", "G=H*T*S", "G=T*S-H", "G=H/(T*S)"],    "chemistry"),
    ("entropy change when heat flows out", "-Q/T",     ["Q/T", "-T/Q", "T/Q", "0"],                         "physics"),
    ("work done by gas in isothermal compression", "-P*delta(V)", ["P*delta(V)", "-delta(V)/P", "P/delta(V)", "delta(V)*T"], "physics"),
]
LINEARITY = [
    ("kinetic energy formula",       "(1/2)*m*v^2",    ["m*v^2", "m*v", "(1/2)*m*v", "2*m*v^2"],            "physics"),
    ("energy stored in capacitor",   "(1/2)*C*V^2",    ["C*V^2", "(1/2)*Q*V^2", "C*V^2/(2*Q)", "Q^2/(2*C)"], "physics"),
    ("centripetal acceleration",     "v^2/r",          ["v*r", "v/r", "v^2*r", "2*v/r"],                    "physics"),
    ("electric potential energy",    "k*q1*q2/r",      ["k*q1*q2/r^2", "k*q1*q2*r", "q1*q2/r", "k/r"],     "physics"),
    ("gravitational potential energy formula", "G*m1*m2/r", ["G*m1*m2/r^2", "G*m1*m2*r", "m1*m2/r", "G*r"], "physics"),
    ("spring potential energy",      "(1/2)*k*x^2",    ["k*x^2", "k*x", "(1/2)*k*x", "2*k*x^2"],            "physics"),
    ("integral of x",                "x^2/2+C",        ["x+C", "x^2+C", "2*x+C", "x^3/3+C"],               "calculus"),
    ("integral of x^2",              "x^3/3+C",        ["x^2/2+C", "x^2+C", "x^3+C", "3*x^2+C"],           "calculus"),
    ("area under parabola from 0 to a", "a^3/3",       ["a^2/2", "a^2", "a^3", "a/3"],                      "calculus"),
    ("variance of uniform distribution from 0 to 1", "(b-a)^2/12", ["(b-a)/12", "(b-a)^2/6", "(b+a)^2/12", "(b-a)/6"], "statistics"),
]
MISSING_CONSTANT = [
    ("Coulomb's law force",          "k*q1*q2/r^2",    ["k*q1*q2/r", "k*q1*q2*r^2", "q1*q2/r^2", "k/(q1*q2*r^2)"], "physics"),
    ("gravitational force",          "G*m1*m2/r^2",    ["m1*m2/r^2", "G*m1/r^2", "G*m1*m2/r", "G*m2/r^2"], "physics"),
    ("photon energy formula",        "E=h*f",          ["E=h*lambda", "E=h/f", "E=h*f^2", "E=f/h"],         "physics"),
    ("de Broglie wavelength",        "h/(m*v)",        ["m*v/h", "h*m*v", "h/(m^2*v)", "m*v/h^2"],          "physics"),
    ("Stefan-Boltzmann power",       "sigma*A*T^4",    ["A*T^4", "sigma*T^4", "A*T^4/sigma", "sigma*A/T^4"], "physics"),
    ("eigenvalue equation",          "A*v=lambda*v",   ["A*v=lambda", "A=lambda*v", "A*v=v/lambda", "A*lambda=v"], "linear_algebra"),
    ("scalar factor in inverse of 2x2 matrix [[a,b],[c,d]]", "1/(a*d-b*c)", ["1/(a*d+b*c)", "a*d-b*c", "1/(a*b-c*d)", "(a*d-b*c)"], "linear_algebra"),
    ("determinant of 2x2 matrix [[a,b],[c,d]]", "a*d-b*c", ["a*b-c*d", "a*d+b*c", "a*c-b*d", "a*b+c*d"],   "linear_algebra"),
    ("Bayes theorem formula",        "P(A|B)=P(B|A)*P(A)/P(B)", ["P(A|B)=P(A)*P(B|A)", "P(A|B)=P(B|A)/P(A)", "P(A|B)=P(A and B)/P(A)", "P(A|B)=P(B|A)*P(B)/P(A)"], "statistics"),
    ("standard error of the mean",  "sigma/sqrt(n)",  ["sigma/n", "sigma^2/n", "sigma*sqrt(n)", "sigma/n^2"], "statistics"),
]
TRUNCATION = [
    ("hybridization of carbon in methane",   "sp3",   ["sp2", "sp", "sp3d", "sp2d"],                        "chemistry"),
    ("hybridization of carbon in ethylene",  "sp2",   ["sp3", "sp", "sp3d", "sp2d"],                        "chemistry"),
    ("hybridization of carbon in acetylene", "sp",    ["sp2", "sp3", "sp3d", "sp2d"],                       "chemistry"),
    ("Arrhenius equation for rate constant", "k=A*e^(-Ea/(R*T))", ["k=A*e^(Ea/(R*T))", "k=Ea/(R*T)", "k=A/e^(Ea/(R*T))", "k=R*T/Ea"], "chemistry"),
    ("Henderson-Hasselbalch equation",       "pH=pKa+log([A-]/[HA])", ["pH=pKa-log([A-]/[HA])", "pH=pKa+log([HA]/[A-])", "pH=Ka+log([A-]/[HA])", "pH=pKa*log([A-]/[HA])"], "chemistry"),
    ("characteristic polynomial (definition)", "det(A-lambda*I)", ["det(A+lambda*I)", "tr(A-lambda*I)", "det(A)-lambda", "det(lambda*I-A)"], "linear_algebra"),
    ("Cauchy-Schwarz inequality",            "|u*v| <= |u|*|v|", ["|u*v| >= |u|*|v|", "|u*v| = |u|*|v|", "|u*v| < |u|+|v|", "|u*v| <= |u|+|v|"], "linear_algebra"),
    ("Taylor series first two terms of e^x around 0", "1+x", ["x+x^2", "e+e*x", "1+x+x^2/2", "1-x"],      "calculus"),
    ("period of a simple pendulum",          "2*pi*sqrt(L/g)", ["2*pi*sqrt(g/L)", "2*pi*L/g", "pi*sqrt(L/g)", "2*pi*sqrt(m/k)"], "physics"),
    ("escape velocity formula",              "sqrt(2*G*M/r)", ["sqrt(G*M/r)", "2*G*M/r^2", "sqrt(G*M/(2*r))", "G*M/r^2"], "physics"),
]

BIAS_DATA = {
    "positivity":       POSITIVITY,
    "linearity":        LINEARITY,
    "missing_constant": MISSING_CONSTANT,
    "truncation":       TRUNCATION,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_lm_head(model):
    try:
        return lambda h: model.model.embed_tokens.as_linear(h)
    except Exception:
        return model.lm_head


def compute_lp_with_adapter(adapter, lm_head, model, prompt, completion, tokenizer):
    prompt_ids = tokenizer.encode(prompt)
    comp_ids   = tokenizer.encode(completion)
    full_ids   = prompt_ids + comp_ids
    if not comp_ids:
        return -1e9
    tokens = mx.array(full_ids)[None, :]
    h = model.model(tokens)
    base_logits = lm_head(h)
    shifts = adapter(base_logits)
    shifts = shifts - shifts.mean(axis=-1, keepdims=True)
    combined = base_logits + shifts
    logits = t3.LOGIT_SOFTCAP * mx.tanh(combined / t3.LOGIT_SOFTCAP)
    total_lp = 0.0
    for i, tok_id in enumerate(comp_ids):
        pos = n_prompt = len(prompt_ids)
        pos = n_prompt - 1 + i
        lv = np.array(logits[0, pos].astype(mx.float32))
        lse = np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max()
        total_lp += float(lv[tok_id] - lse)
    return total_lp


def score_with_adapter(adapter, lm_head, model, tokenizer, context, truth, distractors):
    prompt = f"context: {context}\nanswer:"
    truth_lp  = compute_lp_with_adapter(adapter, lm_head, model, prompt, " " + truth, tokenizer)
    dist_lps  = [compute_lp_with_adapter(adapter, lm_head, model, prompt, " " + d, tokenizer)
                 for d in distractors]
    best_dist = max(dist_lps)
    margin    = truth_lp - best_dist
    win       = margin > 0
    return win, margin, truth_lp, best_dist


def load_adapter(path):
    raw     = dict(np.load(path))
    weights = [(k, mx.array(v)) for k, v in raw.items()]
    cfg = SnapOnConfig(
        vocab_size = 151936,
        d_inner    = 64,
        n_heads    = 4,
        mode       = "logit",
    )
    adapter = create_adapter(cfg)
    adapter.load_weights(weights)
    mx.eval(adapter.parameters())
    return adapter


def ascii_hist(values, bins=10, width=40, label=""):
    """Print a simple ASCII histogram."""
    if not values:
        return
    lo, hi = min(values), max(values)
    if lo == hi:
        hi = lo + 1
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        b = min(int((v - lo) / step), bins - 1)
        counts[b] += 1
    max_count = max(counts) or 1
    if label:
        print(f"\n  {label}")
    for i, c in enumerate(counts):
        bar_lo = lo + i * step
        bar = "█" * int(c / max_count * width)
        print(f"  {bar_lo:+7.2f} │{bar:<{width}} {c}")
    print(f"  {'':7s} └{'─'*width}")
    print(f"  n={len(values)}  mean={sum(values)/len(values):+.3f}  "
          f"min={min(values):+.3f}  max={max(values):+.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(MODEL_ID)
    lm_head = get_lm_head(model)
    mx.eval(model.parameters())
    print(f"Loaded in {time.time()-t0:.1f}s")

    # ── 1. Load mixed adapter ──────────────────────────────────────────────────
    mixed_path = os.path.join(EXP03_DIR, "adapter_mixed.npz")
    mixed_adapter = load_adapter(mixed_path)
    print(f"Loaded mixed adapter from {mixed_path}")

    # ── 2. Score all 40 bias examples: baseline vs mixed ──────────────────────
    print("\n" + "="*60)
    print("PART 1: Bias pattern margin analysis (40 facts)")
    print("="*60)

    per_pattern = {}
    all_baseline = []
    all_mixed    = []

    for pattern, examples in BIAS_DATA.items():
        base_margins  = []
        mixed_margins = []
        base_wins     = 0
        mixed_wins    = 0

        for context, truth, distractors, domain in examples:
            # Baseline (no adapter)
            w_b, m_b, _, _ = score_fact_mc(model, tokenizer, context, truth, distractors)
            base_margins.append(m_b)
            base_wins += int(w_b)

            # Mixed adapter
            w_m, m_m, _, _ = score_with_adapter(
                mixed_adapter, lm_head, model, tokenizer, context, truth, distractors)
            mixed_margins.append(m_m)
            mixed_wins += int(w_m)

        delta = [m - b for m, b in zip(mixed_margins, base_margins)]
        per_pattern[pattern] = {
            "base_margins":  base_margins,
            "mixed_margins": mixed_margins,
            "deltas":        delta,
            "base_acc":      base_wins / len(examples),
            "mixed_acc":     mixed_wins / len(examples),
            "base_mean":     sum(base_margins) / len(base_margins),
            "mixed_mean":    sum(mixed_margins) / len(mixed_margins),
            "delta_mean":    sum(delta) / len(delta),
            "n":             len(examples),
        }
        all_baseline.extend(base_margins)
        all_mixed.extend(mixed_margins)

        print(f"\n{'─'*55}")
        print(f"Pattern: {pattern.upper()}")
        print(f"  Baseline:  {base_wins}/{len(examples)} correct, mean margin {sum(base_margins)/len(base_margins):+.3f}")
        print(f"  Mixed:     {mixed_wins}/{len(examples)} correct, mean margin {sum(mixed_margins)/len(mixed_margins):+.3f}")
        print(f"  Δ margin:  {sum(delta)/len(delta):+.3f} avg")
        print(f"\n  Per-example (context | baseline→mixed | win_b→win_m):")
        for i, (context, truth, distractors, domain) in enumerate(examples):
            b = base_margins[i]
            m = mixed_margins[i]
            sym = "✓" if m > 0 else "✗"
            chg = "↑" if m > b else ("→" if abs(m-b) < 0.01 else "↓")
            print(f"    [{domain:14s}] {context[:35]:35s} │ {b:+6.2f} → {m:+6.2f} {chg} {sym}")

    # ── 3. Calibration analysis: does margin predict correctness? ─────────────
    print("\n" + "="*60)
    print("PART 2: Calibration — does margin predict correctness?")
    print("="*60)

    # Collect all per-example (margin, correct) pairs from baseline
    all_pairs = []
    for pattern, examples in BIAS_DATA.items():
        pd = per_pattern[pattern]
        for i in range(pd["n"]):
            correct = pd["base_margins"][i] > 0
            all_pairs.append((pd["base_margins"][i], correct))

    # Sort by margin, check accuracy in each quintile
    all_pairs.sort(key=lambda x: x[0])
    n = len(all_pairs)
    q = n // 5
    print(f"\n  Accuracy by margin quintile (baseline, n={n}):")
    print(f"  {'Quintile':12s} {'Margin range':20s} {'Accuracy':10s}")
    print(f"  {'─'*12} {'─'*20} {'─'*10}")
    for qi in range(5):
        chunk = all_pairs[qi*q : (qi+1)*q if qi < 4 else n]
        margins = [p[0] for p in chunk]
        acc = sum(p[1] for p in chunk) / len(chunk)
        print(f"  Q{qi+1} (bottom-top) {min(margins):+.2f}..{max(margins):+.2f}       {acc:.0%}")

    # Check: negative margins → always wrong?
    neg_correct   = sum(1 for m, c in all_pairs if m <= 0 and c)
    neg_total     = sum(1 for m, c in all_pairs if m <= 0)
    pos_correct   = sum(1 for m, c in all_pairs if m > 0 and c)
    pos_total     = sum(1 for m, c in all_pairs if m > 0)
    print(f"\n  Negative margin → correct: {neg_correct}/{neg_total} ({neg_correct/max(1,neg_total):.0%})")
    print(f"  Positive margin → correct: {pos_correct}/{pos_total} ({pos_correct/max(1,pos_total):.0%})")

    # ── 4. Mixed adapter margin improvement distribution ──────────────────────
    print("\n" + "="*60)
    print("PART 3: Mixed adapter — uniform vs concentrated improvement?")
    print("="*60)

    deltas_by_outcome = {"was_wrong_now_right": [], "was_right_stayed_right": [],
                         "was_right_now_wrong": [], "was_wrong_stayed_wrong": []}
    for pattern in BIAS_DATA:
        pd = per_pattern[pattern]
        for i in range(pd["n"]):
            b_win = pd["base_margins"][i] > 0
            m_win = pd["mixed_margins"][i] > 0
            d     = pd["deltas"][i]
            if not b_win and m_win:
                deltas_by_outcome["was_wrong_now_right"].append(d)
            elif b_win and m_win:
                deltas_by_outcome["was_right_stayed_right"].append(d)
            elif b_win and not m_win:
                deltas_by_outcome["was_right_now_wrong"].append(d)
            else:
                deltas_by_outcome["was_wrong_stayed_wrong"].append(d)

    total_corrections = len(deltas_by_outcome["was_wrong_now_right"])
    total_regressions = len(deltas_by_outcome["was_right_now_wrong"])
    total_still_wrong = len(deltas_by_outcome["was_wrong_stayed_wrong"])
    total_stayed_right = len(deltas_by_outcome["was_right_stayed_right"])

    print(f"\n  Outcome breakdown (mixed adapter vs baseline):")
    print(f"    Wrong → Correct:  {total_corrections:2d}  (margin delta: {sum(deltas_by_outcome['was_wrong_now_right'])/max(1,total_corrections):+.2f} avg)")
    print(f"    Right → Right:    {total_stayed_right:2d}  (margin delta: {sum(deltas_by_outcome['was_right_stayed_right'])/max(1,total_stayed_right):+.2f} avg)")
    print(f"    Right → Wrong:    {total_regressions:2d}  (margin delta: {sum(deltas_by_outcome['was_right_now_wrong'])/max(1,total_regressions):+.2f} avg)")
    print(f"    Wrong → Wrong:    {total_still_wrong:2d}  (margin delta: {sum(deltas_by_outcome['was_wrong_stayed_wrong'])/max(1,total_still_wrong):+.2f} avg)")

    # ── 5. Full STEM benchmark margin distribution ────────────────────────────
    print("\n" + "="*60)
    print("PART 4: Full 97-fact STEM benchmark baseline margin distribution")
    print("="*60)

    if os.path.exists(STEM_PATH):
        with open(STEM_PATH) as f:
            stem_data = json.load(f)

        # models is a list of dicts with model_id key
        models_list  = stem_data.get("models", [])
        stem_results = next(
            (m for m in models_list if "Qwen3-4B" in m.get("model_id", "")),
            {}
        )
        if "per_fact" in stem_results:
            margins = [f["margin"] for f in stem_results["per_fact"]]
            wins    = [f["win"] for f in stem_results["per_fact"]]
            domains = [f.get("domain", "unknown") for f in stem_results["per_fact"]]

            by_domain = {}
            for m, w, d in zip(margins, wins, domains):
                by_domain.setdefault(d, {"margins": [], "wins": []})
                by_domain[d]["margins"].append(m)
                by_domain[d]["wins"].append(w)

            print(f"\n  By domain (Qwen3-4B-Base, n=97):")
            print(f"  {'Domain':20s} {'n':4s} {'Acc':6s} {'Mean margin':12s} {'Min margin':10s}")
            print(f"  {'─'*20} {'─'*4} {'─'*6} {'─'*12} {'─'*10}")
            for domain, dd in sorted(by_domain.items()):
                m_mean = sum(dd["margins"]) / len(dd["margins"])
                acc    = sum(dd["wins"]) / len(dd["wins"])
                print(f"  {domain:20s} {len(dd['margins']):4d} {acc:6.0%} {m_mean:+12.3f} {min(dd['margins']):+10.3f}")

            # Margin distribution across all
            n_neg = sum(1 for m in margins if m <= 0)
            print(f"\n  Overall: {sum(wins)}/{len(wins)} correct, {n_neg} negative margins")
            print(f"  Mean={sum(margins)/len(margins):+.3f}  Min={min(margins):+.3f}  Max={max(margins):+.3f}")
            ascii_hist(margins, bins=12, label="Full STEM margin distribution (Qwen3-4B-Base)")
        else:
            print("  No per_fact data in stem_crossmodel.json — need to re-score")
            print("  Skipping domain breakdown (run eval with per_fact=True)")
    else:
        print(f"  STEM results not found at {STEM_PATH}")

    # ── 6. Save results ───────────────────────────────────────────────────────
    out = {
        "model": MODEL_ID,
        "n_bias_examples": 40,
        "per_pattern": {
            pat: {k: v for k, v in pd.items() if not k.endswith("_margins") and k != "deltas"}
            for pat, pd in per_pattern.items()
        },
        "per_pattern_full": {
            pat: {"base_margins": pd["base_margins"], "mixed_margins": pd["mixed_margins"],
                  "deltas": pd["deltas"]}
            for pat, pd in per_pattern.items()
        },
        "outcome_counts": {k: len(v) for k, v in deltas_by_outcome.items()},
        "outcome_delta_means": {k: sum(v)/len(v) if v else 0.0 for k, v in deltas_by_outcome.items()},
        "calibration": {
            "neg_margin_correct_rate": neg_correct / max(1, neg_total),
            "pos_margin_correct_rate": pos_correct / max(1, pos_total),
            "neg_total": neg_total,
            "pos_total": pos_total,
        },
        "all_baseline_margins": all_baseline,
        "all_mixed_margins":    all_mixed,
    }
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
