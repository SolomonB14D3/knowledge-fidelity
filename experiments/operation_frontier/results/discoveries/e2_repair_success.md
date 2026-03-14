# Discovery Note: e₂ Repair Success (2026-03-13)

**Candidate:** C10 — e₂ = r12·r13 + r12·r23 + r13·r23
**Baseline margin:** -1.670 (KNOWLEDGE_GAP quadrant)
**After choreography mini-adapter:** +1.296 (FIXABLE_BIAS quadrant)
**Mean margin delta across verification set:** +0.22
**frac_var stability:** unchanged (2.69e-03 — checker PASS preserved)

---

## What Happened

The second elementary symmetric polynomial of pairwise distances is a genuine knowledge
gap in Qwen3-4B-Base. The mixed STEM adapter (exp03_correction) confirmed this by making
the margin 20× worse (Δ = -20.49). A choreography-specific mini-adapter (25 examples,
500 steps, d_inner=64, ~2 min) flipped the margin from -1.670 → +1.296.

The detect-and-fill cycle works:

```
baseline oracle → FAIL (margin -1.67)
       ↓
mixed adapter   → FAIL harder (margin -32.3) → KNOWLEDGE_GAP triggered
       ↓
domain adapter  → PASS (margin +1.30) → quadrant FIXABLE_BIAS
```

---

## New Capability Unlocked

The pipeline now has three operating modes:

| Mode | Trigger | Action |
|------|---------|--------|
| Known good | Oracle PASS + Checker PASS | Archive |
| Bias repair | Oracle FAIL + adapter ↑ | Use targeted adapter |
| **Knowledge gap fill** | Oracle FAIL + adapter ↓ → train domain adapter | **NEW** |

---

## What Still Needs Work

1. **Holdout vis-viva regressed** (-8.48): the adapter doesn't have a Kepler domain guard.
   Fix: add 3–5 vis-viva/orbital-energy examples to the holdout-preservation set in training.
2. **e₁ anchor (figure-8 context) still failing** (-2.876): this is a context-phrasing issue —
   the +4.50 result used a slightly different context string. Add the exact phrasing to training.
3. **Newton's identity improved but not flipped** (-0.627): close. 2–3 more Newton examples
   would likely push it over.

These are all addressable with a v2 training run adding ~8 examples and 200 more steps.

---

## Files

- `train_choreography_adapter.py` — training script (committed)
- `adapters/adapter_choreography.npz` — trained adapter (local, gitignored)
- `monitors/e2_symmetric_poly.py` — live checker monitor with e₂/e₁ ratio discriminator
- `problems/c10_repair_facts.json` — verification set used for eval
