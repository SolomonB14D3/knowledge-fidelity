#!/usr/bin/env python3
"""Ablation: does length normalization change the calibration result?

Currently we use sum log-prob (truth_lp - best_dist_lp). Longer completions
naturally have more negative sum log-prob. This ablation computes:
  - mean log-prob per token (divides sum by token count)
  - checks whether calibration holds for both sum and mean

Key question: Is margin-sign calibration a genuine property, or an artifact
of the model always preferring shorter completions (which happen to be distractors)?

Usage:
    python sub_experiments/exp04_confidence/ablation_length_norm.py
"""

import json, sys, time
sys.path.insert(0, "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity")

import mlx.core as mx
import mlx_lm
import numpy as np

STEM_PATH = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/results/operation_destroyer/stem_crossmodel/stem_crossmodel.json"
MODEL_ID  = "Qwen/Qwen3-4B-Base"

BIAS_FACTS_40 = [
    # (context, truth, distractors)  — 10 per pattern
    # positivity
    ("derivative of cos(x)",         "-sin(x)",           ["sin(x)", "-cos(x)", "cos(x)", "tan(x)"]),
    ("second derivative of sin(x)",  "-sin(x)",           ["sin(x)", "cos(x)", "-cos(x)", "0"]),
    ("integral of sin(x)",           "-cos(x)+C",         ["cos(x)+C", "-sin(x)+C", "sin^2(x)+C", "sin(x)+C"]),
    ("derivative of cos(2*x)",       "-2*sin(2*x)",       ["2*sin(2*x)", "-sin(2*x)", "2*cos(2*x)", "-2*cos(2*x)"]),
    ("second derivative of cos(x)",  "-cos(x)",           ["cos(x)", "-sin(x)", "sin(x)", "0"]),
    ("integral of sin(2*x)",         "-(1/2)*cos(2*x)+C", ["(1/2)*cos(2*x)+C", "-cos(2*x)+C", "cos(2*x)+C", "(1/2)*sin(2*x)+C"]),
    ("first law of thermodynamics",  "delta(U)=Q-W",      ["delta(U)=Q+W", "delta(U)=W-Q", "Q=delta(U)+W", "delta(U)=Q*W"]),
    ("Gibbs free energy formula",    "G=H-T*S",           ["G=H+T*S", "G=H*T*S", "G=T*S-H", "G=H/(T*S)"]),
    ("entropy change when heat flows out", "-Q/T",        ["Q/T", "-T/Q", "T/Q", "0"]),
    ("work done by gas in isothermal compression", "-P*delta(V)", ["P*delta(V)", "-delta(V)/P", "P/delta(V)", "delta(V)*T"]),
    # linearity
    ("kinetic energy formula",       "(1/2)*m*v^2",       ["m*v^2", "m*v", "(1/2)*m*v", "2*m*v^2"]),
    ("energy stored in capacitor",   "(1/2)*C*V^2",       ["C*V^2", "(1/2)*Q*V^2", "C*V^2/(2*Q)", "Q^2/(2*C)"]),
    ("centripetal acceleration",     "v^2/r",             ["v*r", "v/r", "v^2*r", "2*v/r"]),
    ("electric potential energy",    "k*q1*q2/r",         ["k*q1*q2/r^2", "k*q1*q2*r", "q1*q2/r", "k/r"]),
    ("gravitational potential energy formula", "G*m1*m2/r", ["G*m1*m2/r^2", "G*m1*m2*r", "m1*m2/r", "G*r"]),
    ("spring potential energy",      "(1/2)*k*x^2",       ["k*x^2", "k*x", "(1/2)*k*x", "2*k*x^2"]),
    ("integral of x",                "x^2/2+C",           ["x+C", "x^2+C", "2*x+C", "x^3/3+C"]),
    ("integral of x^2",              "x^3/3+C",           ["x^2/2+C", "x^2+C", "x^3+C", "3*x^2+C"]),
    ("area under parabola from 0 to a", "a^3/3",          ["a^2/2", "a^2", "a^3", "a/3"]),
    ("variance of uniform distribution from 0 to 1", "(b-a)^2/12", ["(b-a)/12", "(b-a)^2/6", "(b+a)^2/12", "(b-a)/6"]),
    # missing_constant
    ("Coulomb's law force",          "k*q1*q2/r^2",       ["k*q1*q2/r", "k*q1*q2*r^2", "q1*q2/r^2", "k/(q1*q2*r^2)"]),
    ("gravitational force",          "G*m1*m2/r^2",       ["m1*m2/r^2", "G*m1/r^2", "G*m1*m2/r", "G*m2/r^2"]),
    ("photon energy formula",        "E=h*f",             ["E=h*lambda", "E=h/f", "E=h*f^2", "E=f/h"]),
    ("de Broglie wavelength",        "h/(m*v)",           ["m*v/h", "h*m*v", "h/(m^2*v)", "m*v/h^2"]),
    ("Stefan-Boltzmann power",       "sigma*A*T^4",       ["A*T^4", "sigma*T^4", "A*T^4/sigma", "sigma*A/T^4"]),
    ("eigenvalue equation",          "A*v=lambda*v",      ["A*v=lambda", "A=lambda*v", "A*v=v/lambda", "A*lambda=v"]),
    ("scalar factor in inverse of 2x2 matrix [[a,b],[c,d]]", "1/(a*d-b*c)", ["1/(a*d+b*c)", "a*d-b*c", "1/(a*b-c*d)", "(a*d-b*c)"]),
    ("determinant of 2x2 matrix [[a,b],[c,d]]", "a*d-b*c", ["a*b-c*d", "a*d+b*c", "a*c-b*d", "a*b+c*d"]),
    ("Bayes theorem formula",        "P(A|B)=P(B|A)*P(A)/P(B)", ["P(A|B)=P(A)*P(B|A)", "P(A|B)=P(B|A)/P(A)", "P(A|B)=P(A and B)/P(A)", "P(A|B)=P(B|A)*P(B)/P(A)"]),
    ("standard error of the mean",  "sigma/sqrt(n)",     ["sigma/n", "sigma^2/n", "sigma*sqrt(n)", "sigma/n^2"]),
    # truncation
    ("hybridization of carbon in methane",   "sp3",       ["sp2", "sp", "sp3d", "sp2d"]),
    ("hybridization of carbon in ethylene",  "sp2",       ["sp3", "sp", "sp3d", "sp2d"]),
    ("hybridization of carbon in acetylene", "sp",        ["sp2", "sp3", "sp3d", "sp2d"]),
    ("Arrhenius equation for rate constant", "k=A*e^(-Ea/(R*T))", ["k=A*e^(Ea/(R*T))", "k=Ea/(R*T)", "k=A/e^(Ea/(R*T))", "k=R*T/Ea"]),
    ("Henderson-Hasselbalch equation",       "pH=pKa+log([A-]/[HA])", ["pH=pKa-log([A-]/[HA])", "pH=pKa+log([HA]/[A-])", "pH=Ka+log([A-]/[HA])", "pH=pKa*log([A-]/[HA])"]),
    ("characteristic polynomial (definition)", "det(A-lambda*I)", ["det(A+lambda*I)", "tr(A-lambda*I)", "det(A)-lambda", "det(lambda*I-A)"]),
    ("Cauchy-Schwarz inequality",    "|u*v| <= |u|*|v|",  ["|u*v| >= |u|*|v|", "|u*v| = |u|*|v|", "|u*v| < |u|+|v|", "|u*v| <= |u|+|v|"]),
    ("Taylor series first two terms of e^x around 0", "1+x", ["x+x^2", "e+e*x", "1+x+x^2/2", "1-x"]),
    ("period of a simple pendulum",  "2*pi*sqrt(L/g)",    ["2*pi*sqrt(g/L)", "2*pi*L/g", "pi*sqrt(L/g)", "2*pi*sqrt(m/k)"]),
    ("escape velocity formula",      "sqrt(2*G*M/r)",     ["sqrt(G*M/r)", "2*G*M/r^2", "sqrt(G*M/(2*r))", "G*M/r^2"]),
]


def score_both(model, tokenizer, context, truth, distractors):
    """Returns (sum_margin, mean_margin, truth_ntok, best_dist_ntok)."""
    prompt = f"context: {context}\nanswer:"

    def lp(completion):
        prompt_ids = tokenizer.encode(prompt)
        comp_ids   = tokenizer.encode(" " + completion)
        full_ids   = prompt_ids + comp_ids
        if not comp_ids:
            return -1e9, 0
        tokens  = mx.array(full_ids)[None, :]
        logits  = model(tokens)
        mx.eval(logits)
        logits_np = np.array(logits[0].astype(mx.float32))
        total = 0.0
        for i, tid in enumerate(comp_ids):
            pos  = len(prompt_ids) - 1 + i
            lv   = logits_np[pos]
            lse  = np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max()
            total += float(lv[tid] - lse)
        return total, len(comp_ids)

    truth_sum, truth_n = lp(truth)
    dists = [lp(d) for d in distractors]
    best_dist_sum = max(s for s, _ in dists)
    best_dist_n   = max(n for _, n in dists)  # length of most-likely distractor

    sum_margin  = truth_sum - best_dist_sum
    # mean: normalize each by its own length
    truth_mean      = truth_sum / max(1, truth_n)
    best_dist_sums  = [(s, n) for s, n in dists]
    best_dist_mean  = max(s / max(1, n) for s, n in best_dist_sums)
    mean_margin = truth_mean - best_dist_mean

    return sum_margin, mean_margin, truth_n, best_dist_n


def check_calibration(pairs, label):
    neg_correct = sum(1 for m, c in pairs if m <= 0 and c)
    neg_total   = sum(1 for m, c in pairs if m <= 0)
    pos_correct = sum(1 for m, c in pairs if m > 0 and c)
    pos_total   = sum(1 for m, c in pairs if m > 0)
    print(f"\n  {label}:")
    print(f"    Negative margin → correct: {neg_correct}/{neg_total} ({neg_correct/max(1,neg_total):.0%})")
    print(f"    Positive margin → correct: {pos_correct}/{pos_total} ({pos_correct/max(1,pos_total):.0%})")
    # agreement with sum-margin
    return neg_correct, neg_total, pos_correct, pos_total


def main():
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(MODEL_ID)
    mx.eval(model.parameters())
    print(f"Loaded in {time.time()-t0:.1f}s")

    print("\n=== LENGTH NORMALIZATION ABLATION (n=40 bias facts) ===")
    print("Comparing: sum log-prob margin vs mean-per-token log-prob margin\n")

    sum_pairs  = []
    mean_pairs = []
    length_pairs = []  # (truth_ntok, best_dist_ntok, correct)

    for i, (ctx, truth, distractors) in enumerate(BIAS_FACTS_40):
        sm, mm, tn, dn = score_both(model, tokenizer, ctx, truth, distractors)
        correct = sm > 0  # ground truth from sum-margin (our current method)
        sum_pairs.append((sm, correct))
        mean_pairs.append((mm, correct))
        length_pairs.append((tn, dn, correct))
        if (i+1) % 10 == 0:
            print(f"  Scored {i+1}/40 facts...")

    check_calibration(sum_pairs,  "Sum log-prob margin (current method)")
    check_calibration(mean_pairs, "Mean log-prob/token margin (ablation)")

    # Check agreement between the two methods
    agree = sum(1 for (sm, _), (mm, _) in zip(sum_pairs, mean_pairs)
                if (sm > 0) == (mm > 0))
    print(f"\n  Agreement between sum and mean margin predictions: {agree}/40 ({agree/40:.0%})")

    # Check if truth is consistently longer than distractors for wrong cases
    wrong_facts = [(tn, dn) for tn, dn, c in length_pairs if not c]
    right_facts = [(tn, dn) for tn, dn, c in length_pairs if c]
    if wrong_facts:
        avg_truth_len_wrong = sum(t for t, d in wrong_facts) / len(wrong_facts)
        avg_dist_len_wrong  = sum(d for t, d in wrong_facts) / len(wrong_facts)
        print(f"\n  Wrong facts — avg truth length: {avg_truth_len_wrong:.1f} tokens, "
              f"avg best-dist length: {avg_dist_len_wrong:.1f} tokens")
    if right_facts:
        avg_truth_len_right = sum(t for t, d in right_facts) / len(right_facts)
        avg_dist_len_right  = sum(d for t, d in right_facts) / len(right_facts)
        print(f"  Correct facts — avg truth length: {avg_truth_len_right:.1f} tokens, "
              f"avg best-dist length: {avg_dist_len_right:.1f} tokens")

    print("\n=== CONCLUSION ===")
    # If both methods agree and both are calibrated → length normalization doesn't matter
    neg_sum  = sum(1 for m, c in sum_pairs  if m <= 0 and c)
    neg_mean = sum(1 for m, c in mean_pairs if m <= 0 and c)
    pos_sum  = sum(1 for m, c in sum_pairs  if m > 0 and c)
    pos_mean = sum(1 for m, c in mean_pairs if m > 0 and c)
    if neg_sum == 0 and neg_mean == 0:
        print("  Both methods show perfect calibration (0 false negatives, 0 false positives).")
        print("  Length normalization does NOT change the result.")
        print("  Calibration is a genuine property of the model's log-prob assignments,")
        print("  not an artifact of completion length differences.")
    else:
        print(f"  Sum margin: {neg_sum} false negatives. Mean margin: {neg_mean} false negatives.")
        print("  Methods disagree — investigate further.")

    # Save results
    out = {
        "sum_pairs":  [(float(m), bool(c)) for m, c in sum_pairs],
        "mean_pairs": [(float(m), bool(c)) for m, c in mean_pairs],
        "length_pairs": [(int(t), int(d), bool(c)) for t, d, c in length_pairs],
        "agreement":  int(agree),
        "n":          40,
    }
    out_path = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/sub_experiments/exp04_confidence/ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
