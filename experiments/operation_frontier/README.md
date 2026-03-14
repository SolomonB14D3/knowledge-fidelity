# NoetherSolve

**Find what LLMs don't know about what the universe conserves. Then fix it.**

Emmy Noether proved that every continuous symmetry of a physical system
corresponds to a conserved quantity. NoetherSolve finds where LLMs fail to
recognize those quantities — and closes the gap with targeted adapters.

---

## What It Does

NoetherSolve runs a **dual-filter pipeline**:

```
Hypothesis (expression)
       │
       ▼
 Numerical checker          ← Is this quantity actually conserved?
 (RK45 integration,           frac_var = σ/|mean| < threshold
  frac_var test)
       │ PASS
       ▼
 Oracle filter              ← Does the model know it?
 (log-prob margin,            margin = log P(truth) − log P(best distractor)
  base LLM)
       │
       ├─ PASS  → DUAL-PASS: known conserved quantity, archive it
       │
       └─ FAIL  → Run repair pass (adapter):
                    ├─ margin improves  → FIXABLE BIAS: apply domain adapter
                    └─ margin worsens   → KNOWLEDGE GAP: train new adapter
```

Every discovery lands in one of four diagnostic quadrants. The pipeline
tells you exactly which one and what to do next.

---

## Quick Start

```bash
# 1. Run the checker on a hypothesis
python vortex_checker.py --ic restricted --expr "s['r12'] + 0.01*(s['r13']+s['r23'])"

# 2. If checker passes, run the oracle
python oracle_wrapper.py --problem problems/vortex_pair_conservation.yaml

# 3. If oracle fails, diagnose and repair
python oracle_wrapper.py --problem problems/vortex_pair_conservation.yaml \
    --repair --diagnose

# 4. Claim a problem before you start hunting (prevents duplicate work)
python claim.py claim \
    --problem vortex_pair_conservation \
    --expr "r12 + eps*(r13+r23)" \
    --handle your_handle
```

---

## Adding a New Domain (Fork This)

Every domain is three files in `problems/`:

| File | Purpose |
|------|---------|
| `my_domain.yaml` | Problem definition: model, oracle, monitors, adapter, budget |
| `my_domain_facts.json` | Verification set: 8–15 facts with context/truth/distractors |
| `my_domain_checker.py` | Numerical integrator: `integrate()` + `parse_state()` + `frac_var()` |

Copy `problem_template.yaml` and follow `CONTRIBUTING.md` for the full protocol.

**Format rule:** Use compact symbolic notation in facts.
`"H = -1/(4π) Σᵢ<ⱼ ΓᵢΓⱼ ln(rᵢⱼ²)"` ✓
`"The Hamiltonian equals negative one over four pi times the sum..."` ✗

---

## Discoveries So Far

| Date | Domain | Expression | frac_var | Oracle | Status |
|------|--------|------------|----------|--------|--------|
| 2026-03-13 | Figure-8 3-body | e₁ = r₁₂+r₁₃+r₂₃ | 5.54e-04 | +4.50 | **DUAL-PASS** |
| 2026-03-13 | Figure-8 3-body | e₂ = r₁₂r₁₃+r₁₂r₂₃+r₁₃r₂₃ | 2.69e-03 | -1.67→**+1.30** | **FLIPPED** |
| 2026-03-13 | Figure-8 3-body | r_rms = √((r₁₂²+r₁₃²+r₂₃²)/3) | 7.69e-04 | -0.49 | GAP (open) |
| 2026-03-13 | Point-vortex | Q = r₁₂+Γ₃(r₁₃+r₂₃) | 5.36e-06 | -29.96 | GAP (adapter training) |

Full history: `results/candidates.tsv`

---

## Coordination

NoetherSolve uses the **THINK → CLAIM → RUN → PUBLISH** protocol
to prevent duplicate work across contributors.

> Coordination design adapted from
> [autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home)
> (mutable-state-inc), which pioneered asynchronous multi-agent research
> coordination with semantic duplicate detection and claim expiry.
> We adapt it here for human-in-the-loop physics hunting.

```bash
python claim.py list     # see what's in flight
python claim.py claim    # reserve your problem before running
python claim.py release  # publish your results, free the claim
```

Claims expire after 4 hours. See `CONTRIBUTING.md` for the full protocol.

---

## Architecture

```
NoetherSolve
├── oracle_wrapper.py          ← Oracle + repair + quadrant diagnosis
├── claim.py                   ← Coordination: THINK/CLAIM/RELEASE
├── claims.json                ← Live claims registry
│
├── problems/                  ← Domain plugins (fork here)
│   ├── problem_template.yaml  ← Starting point for new domains
│   ├── *_facts.json           ← Verification sets
│   └── *_checker.py           ← Numerical integrators
│
├── monitors/                  ← Reusable checker monitors
│   ├── sum_pairwise_distances.py
│   ├── e2_symmetric_poly.py
│   └── __init__.py
│
├── adapters/                  ← Trained domain adapters (gitignored)
│   ├── adapter_choreography.npz
│   └── adapter_vortex.npz
│
└── results/
    ├── candidates.tsv         ← All tested hypotheses
    └── discoveries/           ← Discovery notes for DUAL-PASS / FLIPPED
```

---

## Built On

- **STEM Truth Oracle** (Paper 9) — log-prob margin as a zero-FP/FN binary
  classifier for factual correctness.
  DOI: [10.5281/zenodo.19005729](https://doi.org/10.5281/zenodo.19005729)

- **Snap-On Communication Modules** (Paper 8) — frozen logit-space adapters
  that close knowledge gaps without touching base model weights.
  DOI: [10.5281/zenodo.18902616](https://doi.org/10.5281/zenodo.18902616)

- **autoresearch-at-home** (mutable-state-inc) — THINK → CLAIM → RUN → PUBLISH
  coordination protocol for collaborative research without duplicate work.
  [github.com/mutable-state-inc/autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home)

- **Noether's theorem** (Emmy Noether, 1915) — the reason any of this works.
