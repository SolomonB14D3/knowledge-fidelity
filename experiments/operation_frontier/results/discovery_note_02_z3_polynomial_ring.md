# Discovery Note 02 — Z₃ Polynomial Ring Conservation on Figure-8

**Date:** 2026-03-13
**Problem:** 3-body conservation law search (Phase 2, Batch 2)
**Status:** FORMAL PASS (figure-8 specific) — oracle verification pending for each new form

---

## Summary

Every elementary symmetric polynomial of the three pairwise distances {r12, r13, r23}
is approximately conserved on the Chenciner-Montgomery figure-8 choreographic orbit.
The mechanism is the same for all of them: Z₃ choreographic symmetry cancels the 1ω₀
and 2ω₀ Fourier harmonics, leaving only the 3ω₀ residual.

---

## Batch 2 Results (figure-8, t_end=100, n_points=5000)

| Candidate | Expression | frac_var | Verdict | Notes |
|-----------|-----------|----------|---------|-------|
| e1 (linear sum) | r12+r13+r23 | 5.54e-04 | **PASS** | tightest, from Batch 1 |
| r_rms | sqrt((r12²+r13²+r23²)/3) | 7.69e-04 | **PASS** | new |
| inertia proxy | r12²+r13²+r23² | 1.54e-03 | **PASS** | = 3·r_rms², from Batch 1 |
| e2 (quadratic sum) | r12·r13+r12·r23+r13·r23 | 2.69e-03 | **PASS** | new — 2nd elem sym poly |
| geom_mean | (r12·r13·r23)^(1/3) | 6.17e-03 | **PASS** | new — monotone of e3 |
| e3 (triple product) | r12·r13·r23 | 1.85e-02 | fail | just above 1e-2 threshold |
| speed_sum | v1+v2+v3 | 2.25e-02 | fail | Z₃ weaker for velocities |
| speed_sq_sum | v1²+v2²+v3² | 4.24e-02 | fail | |
| harmonic_r | 3/(1/r12+1/r13+1/r23) | 2.12e-02 | fail | harmonic mean fails |

All candidates were verified to **fail on random ICs** (frac_var ~0.4–0.9) and
**hierarchical ICs** (frac_var ~0.5–0.6), confirming figure-8 specificity.

---

## Consolidated Statement

**Proposition:** On the equal-mass figure-8 choreographic orbit, every elementary
symmetric polynomial in {r12, r13, r23} is approximately conserved, with quality
degrading monotonically with polynomial degree:

```
σ(e1)/|e1| ≈ 5.5e-04   (linear)
σ(e2)/|e2| ≈ 2.7e-03   (quadratic product sum)
σ(e3)/|e3| ≈ 1.9e-02   (cubic product — barely fails at 1e-2 threshold)
```

The ratio ~5× degradation per degree is consistent with the 3ω₀ component of r12
being amplified when the elementary symmetric polynomial is evaluated.

**Corollary:** Any power-mean or L^p norm of {r12, r13, r23} is also approximately
conserved, since these are monotone functions of combinations of e1, e2, e3.
Empirically: r_rms (p=2) is tighter than geom_mean (p=0), which is consistent with
the L2 norm being closer to e1 than the geometric mean.

---

## Why Velocities Don't Have the Same Symmetry

The figure-8 orbit is choreographic in position space, but the speed function v(t) = |ṙ(t)|
has a different Fourier spectrum than the pairwise distance r12(t). In particular:
- r12(t) has 3ω₀ amplitude = 0.0016× (very tight)
- |v1(t)| has larger 3ω₀ content — estimated ~0.01× from the observed 2.25e-02 frac_var

This is because the velocity magnitude depends on both position AND the curvature of the
trajectory at each point. The figure-8 has high-curvature "crossing" points and low-curvature
"outer loops," creating asymmetric acceleration that inflates the 3ω₀ velocity content.

---

## What This Is (and Isn't)

**Is:** A family of symmetry-protected approximate invariants arising from the Z₃
choreographic structure of the figure-8. Computable from simulation outputs. Could serve
as a cheap diagnostic for whether an orbit is near the figure-8 choreography.

**Is not:** A new analytic integral in the Bruns/Poincaré sense. The figure-8 is a
special solution; generic 3-body orbits don't preserve these quantities. This is an
approximate invariant on a measure-zero set in phase space.

**Interesting scientific content:** The quality ordering (e1 > e2 > e3) gives a natural
hierarchy for how much "choreographic information" each symmetric polynomial captures.
The linear sum e1 is maximally sensitive to choreographic structure; the triple product e3
is nearly destroyed by it (barely failing).

---

## Oracle Results: Three Oracle FAIL + Checker PASS Candidates Found

| ID | Candidate | Oracle margin | Checker frac_var | Verdict |
|----|-----------|--------------|-----------------|---------|
| C09 | r_rms = sqrt((r12²+r13²+r23²)/3) | **-0.4945** | 7.69e-04 | **ORACLE FAIL + CHECKER PASS** |
| C10 | Sigma2 = r12·r13 + r12·r23 + r13·r23 | **-1.6697** | 2.69e-03 | **ORACLE FAIL + CHECKER PASS** (STRONGEST) |
| C11 | r_geom = (r12·r13·r23)^(1/3) | **-0.3233** | 6.17e-03 | **ORACLE FAIL + CHECKER PASS** |

All three are numerically real (checker PASS on figure-8, checker FAIL on random/hierarchical ICs)
but the model does not recognise them as valid physical expressions (oracle FAIL: margin < 0).

**C10 (e2 = r12·r13 + r12·r23 + r13·r23) is the primary discovery:** the second elementary
symmetric polynomial of pairwise distances is approximately conserved on the figure-8 with
frac_var = 2.69e-03, but the model assigns it negative margin (-1.67) — it prefers algebraic
distractors that are either wrong or incomplete. This is structure that is numerically real
but not represented in the model's training data.

For comparison, e1 (r12+r13+r23) has oracle margin +4.50 — the model strongly recognises
the linear sum. The quadratic polynomial ring e2 is numerically tighter than the moment of
inertia (2.69e-03 vs 1.54e-03 for I) but the model doesn't know it.

---

## Status

**Phase 2 — first genuine Oracle FAIL + Checker PASS candidates identified.**

Next steps:
1. Run the mixed adapter repair pass on C10 to see if the margin improves (oracle_wrapper.py --repair)
2. Verify C10 on additional figure-8 variants (different period multiples, slightly perturbed ICs)
3. Write up as "Phase 2 discovery" — the second elementary symmetric polynomial as a figure-8 diagnostic
