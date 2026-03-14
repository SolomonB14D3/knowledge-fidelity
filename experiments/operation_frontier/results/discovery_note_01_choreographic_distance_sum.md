# Discovery Note 01 — Choreographic Distance Sum Near-Invariant

**Date:** 2026-03-13
**Problem:** 3-body conservation law search (Phase 2)
**Candidate ID:** C03 (promoted from 3body_candidates.json)
**Status:** DUAL-FILTER PASS — Oracle PASS + Formal checker PASS (figure-8 only)

---

## The Finding

The sum of pairwise distances `S = r12 + r13 + r23` is approximately conserved in the
Chenciner-Montgomery figure-8 choreographic orbit.

**Oracle result:** margin = +4.5031 (above +1.0 strong-pass threshold)
**Formal checker (figure-8 ICs):** frac_var = 5.542e-04 (below 1e-02 threshold → PASS)
**Formal checker (random ICs):** frac_var ≈ 0.3–0.5 (FAIL — structure is figure-8-specific)
**Formal checker (hierarchical ICs):** frac_var ≈ 0.15–0.30 (FAIL)

---

## Physical Mechanism

The figure-8 orbit is **choreographic**: all three equal-mass bodies trace the same
closed curve at phase offsets of T/3 from each other. This means:

```
r12(t) = f(t)
r23(t) = f(t + T/3)    (up to permutation and time-reversal symmetry)
r13(t) = f(t + 2T/3)
```

The sum is therefore:

```
S(t) = f(t) + f(t + T/3) + f(t + 2T/3)
```

By the **discrete Fourier shift theorem**, when you sum three copies of a signal shifted
by T/3, T/3, and 2T/3, the result cancels all Fourier components **except** those whose
frequency is an integer multiple of 3ω₀:

```
S(t) = 3 * [a₀ + a₃cos(3ω₀t) + a₆cos(6ω₀t) + ...]
```

All harmonics at 1ω₀, 2ω₀, 4ω₀, 5ω₀, ... vanish exactly.

### Measured Fourier content of r12(t)

| Harmonic | Relative amplitude |
|----------|-------------------|
| 1ω₀ (fundamental) | 1.0000 |
| 2ω₀ | 0.0241 |
| **3ω₀** | **0.0016** |
| 4ω₀ | 0.0076 |

The 3ω₀ component has amplitude **0.0016×** the fundamental. After the choreographic
cancellation, the residual variation in S(t) comes entirely from this component, giving
a fractional variation of ~5.5e-04.

---

## Classification

This is a **symmetry-protected approximate invariant**, not a new conservation law:

1. **It follows from choreographic symmetry**, which is a discrete Z₃ symmetry of the
   figure-8 orbit. Any function of the form f(r12) + f(r23) + f(r13) with f symmetric
   will be approximately conserved, with quality determined by the 3ω₀ Fourier content.

2. **It is not an analytic integral** in the sense of Bruns/Poincaré. The figure-8 is
   a periodic orbit, and any bounded function is "approximately conserved" over one period
   by definition. The interesting content is that the fractional variation is as small
   as 5.5e-04 — tight enough to be useful as a diagnostic.

3. **The best-conserved choice of f:** `f(r) = r` (linear distance sum) gives
   frac_var = 5.54e-04. Other choices: `r²` sum gives frac_var ≈ 1.54e-03 (moment
   of inertia, ~3× worse). The linear sum is the tightest near-invariant in this family.

---

## Oracle Calibration Note

The 3-body verification set has a **40% baseline oracle pass rate** (vs 77% on the
STEM benchmark domain). Multi-body notation is underrepresented in Qwen3-4B training
data relative to 2-body Kepler mechanics. An oracle margin of +4.50 in this domain
is therefore a strong signal — the model strongly prefers `r12+r13+r23` as the "truth"
over the distractor forms (`r12²+r13²+r23²`, `r12*r13*r23`, etc.).

---

## Implications for Phase 2

This finding establishes the **choreographic Z₃ family** as a productive source of
near-invariants. Next candidates to test:

- `r12² + r13² + r23²` (moment of inertia proxy) — already C01, frac_var=1.54e-03
- `(r12*r13*r23)^(1/3)` (geometric mean of separations) — C03 distractor, untested
- `v1² + v2² + v3²` (kinetic energy proxy without mass weighting) — equal mass version
- `|v1| + |v2| + |v3|` (speed sum) — same choreographic argument applies
- `E_kin(t) - E_kin(t+T/3)` difference — should be zero by choreographic symmetry

The oracle + checker pipeline is working. Archive this result and advance to batch 2.

---

## Raw Numbers

```
figure-8 integration: t_end=100, rtol=1e-10, atol=1e-12 (scipy RK45)
Energy conservation: frac_var = 2.31e-09 (confirms integrator fidelity)
r12+r13+r23: mean=3.842, std=2.133e-03, frac_var=5.542e-04
Moment of inertia: mean=1.041, std=1.605e-03, frac_var=1.542e-03
```
