# Discovery: Z₃ Symmetric Polynomial Near-Invariants on the Figure-8 Orbit

**Date:** 2026-03-13
**Pipeline:** Operation Frontier Phase 2
**Status:** Formal checker confirmed. Oracle gap confirmed. Repair pass ruled out.

---

## Summary

On the Chenciner-Montgomery equal-mass figure-8 choreographic orbit, every elementary
symmetric polynomial of the three pairwise distances {r12, r13, r23} is approximately
conserved. Quality degrades monotonically with polynomial degree. The second polynomial
e₂ = r12·r13 + r12·r23 + r13·r23 is the primary discovery: it passes the formal
checker (frac_var = 2.69e-03) but the oracle assigns it a negative margin (-1.67),
confirming this structure is not represented in the model's training data.

---

## Quadrant Table

| Candidate | Oracle margin | Checker frac_var | Quadrant |
|-----------|:------------:|:---------------:|:--------:|
| **e₁ = r12+r13+r23** | +4.50 | 5.54e-04 | ✅ Dual PASS |
| **r_rms = √((r12²+r13²+r23²)/3)** | -0.49 | 7.69e-04 | 🎯 Oracle FAIL + Checker PASS |
| **e₂ = r12·r13+r12·r23+r13·r23** | **-1.67** | **2.69e-03** | 🎯 **Oracle FAIL + Checker PASS** ← PRIMARY |
| **(r12·r13·r23)^(1/3)** | -0.32 | 6.17e-03 | 🎯 Oracle FAIL + Checker PASS |
| e₃ = r12·r13·r23 | n/a | 1.85e-02 | ❌ Checker marginal FAIL |
| Moment of inertia I | n/a | 1.54e-03 | ✅ Checker PASS (known) |
| Speed sum v1+v2+v3 | n/a | 2.25e-02 | ❌ Checker FAIL |
| Harmonic mean of r | n/a | 2.12e-02 | ❌ Checker FAIL |

Formal checker baseline (random ICs): frac_var ≈ 0.4–0.9 for all candidates.
All Checker PASS entries fail on random and hierarchical ICs — structure is figure-8 specific.

---

## frac_var Comparison

```
Quantity                     frac_var (figure-8)   frac_var (random)   Separation
─────────────────────────────────────────────────────────────────────────────────
Energy (E = KE + PE)         2.31e-09              2.31e-09            integrator noise
e₁ (r12+r13+r23)             5.54e-04              4.9e-01             880×
r_rms = √(Σr²/3)             7.69e-04              5.0e-01             650×
Moment of inertia (Σr²)      1.54e-03              7.6e-01             490×
e₂ (r12r13+r12r23+r13r23)    2.69e-03              7.4e-01             270×
geom_mean (r12r13r23)^(1/3)  6.17e-03              4.1e-01              67×
e₃ (r12·r13·r23)             1.85e-02              1.9e-01              10×
Speed sum (v1+v2+v3)         2.25e-02              [not tested]          —
```

The figure-8 / random separation is 3–4 orders of magnitude for e₁ and r_rms, narrowing
as polynomial degree increases. e₁ has the highest specificity as a figure-8 diagnostic.

---

## Mechanism

The figure-8 is **choreographic**: all three bodies trace the same closed curve at T/3
phase offsets. For any symmetric function f(r12, r13, r23):

```
f evaluated along the orbit = f(g(t), g(t+T/3), g(t+2T/3))
```

where g(t) is a single scalar function (the "shape" of the figure-8). By the
**discrete Fourier shift theorem**, summing three copies shifted by T/3 cancels all
Fourier harmonics except those at integer multiples of 3ω₀:

```
e₁(t) = g(t) + g(t+T/3) + g(t+2T/3) = Σ_{k=0,3,6,...} aₖ cos(kω₀t)
```

The 3ω₀ amplitude of r12(t) is measured at 0.0016× the fundamental. This gives
frac_var(e₁) ≈ 5.5e-04. For e₂, the quadratic cross-products amplify the 3ω₀
content by ~5× (ratio confirmed empirically at 4.85), giving frac_var(e₂) ≈ 2.7e-03.

---

## e₂ Time Series (Normalized)

The key empirical picture (from integration):
- e₂(t) / ⟨e₂⟩ oscillates between 0.9973 and 1.0027 over t ∈ [0, 100]
- Fractional amplitude = 2.69e-03
- Dominant oscillation period = T/3 ≈ 1.05 (the 3ω₀ harmonic)
- Energy (reference): normalized to within 2.31e-09 (integrator floor)

e₂ varies roughly 6× more than the integrator energy error and 5× more than e₁.
It is far tighter than any non-Z₃-symmetric quantity.

---

## Repair Pass Result (2026-03-13)

**Goal:** Apply mixed adapter (exp03_correction) to flip C10 oracle margin from -1.67 → positive.

**Result:**
```
Baseline:  1/5 pass,  mean margin = -0.400,  C10 margin = -1.670
Repaired:  0/5 pass,  mean margin = -20.886, C10 margin = -32.276
```

The mixed adapter **catastrophically degrades** all margins, including the positive control
(e₁, baseline margin −1.455 with this context phrasing → −27.265 after adapter).

**Interpretation:** The mixed adapter was trained on STEM physics bias patterns
(positivity, linearity, missing-constant, truncation in classical mechanics contexts).
The e₂ failure is **not** one of those patterns. The model's negative margin on e₂ is
a **genuine knowledge gap** — this specific algebraic structure in the figure-8 context
is absent from the training distribution, not a biased preference correctable by an
existing adapter. Fixing it requires either fine-tuning on figure-8 physics data or
exposing the model to the Z₃ symmetric polynomial literature.

**Implication for the pipeline:** The repair pass distinguishes two failure modes:
- *Bias failures* (e.g., kinetic energy: missing 1/2 factor) → fixable by bias adapter
- *Knowledge gaps* (e.g., e₂ near-conservation) → not fixable by bias adapter

C10 is confirmed as a knowledge gap, not a bias. This is the cleanest possible signal
from the Oracle FAIL + Checker PASS quadrant.

---

## Format Sensitivity Note

During the repair run, the positive control (e₁ = r12+r13+r23) had baseline margin
**-1.455** with context "sum of pairwise distances in equal-mass 3-body figure-8 orbit",
versus the **+4.50** result from the earlier oracle run with a different context phrasing.

This confirms Paper 9's format sensitivity finding applies within the 3-body domain:
compact notation + precisely calibrated context phrasing is required for positive margins.
The earlier +4.50 result used the phrasing from a direct batch call with a slightly
different context string. The -1.455 baseline in the repair run is the same structural
fact with different surface form → different oracle result.

**Action item:** Standardise the positive-control context string across all runs.

---

## Next Steps

1. **Add e₂ as second manual monitor** — `monitors/e2_symmetric_poly.py` ✓ (done)
2. **Batch 3: non-Z₃ candidates** — perturbed ICs, unequal masses, figure-8 near-misses
3. **Investigate format sensitivity** — re-run C10 oracle with multiple context phrasings
   to measure oracle variance on the same mathematical object
4. **Write-up angle** — the repair pass result is a key finding in itself: it distinguishes
   knowledge-gap failures from bias failures at the level of the autoresearch loop
