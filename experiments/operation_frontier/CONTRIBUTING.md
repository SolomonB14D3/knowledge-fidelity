# Contributing to NoetherSolve

**NoetherSolve** is a collaborative knowledge-gap hunting system for LLMs.
The goal: find physical/mathematical structures that are numerically real but
not recognized by the base model, then close those gaps with targeted adapters.

> *Emmy Noether proved that every continuous symmetry of a physical system
> corresponds to a conserved quantity. NoetherSolve finds where LLMs don't
> know what she proved — and fixes it.*

Each contribution is a **problem plugin** — three files that any researcher can
fork, run, and publish results from.

---

## Coordination Protocol

> **Inspired by [autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home)**
> (mutable-state-inc), which pioneered the THINK → CLAIM → RUN → PUBLISH
> protocol for asynchronous multi-agent research coordination with automatic
> claim expiry and semantic duplicate detection. We adapt it here for
> human-in-the-loop physics hunting with longer claim windows.

Before starting a hunt, check `claims.json` and `results/candidates.tsv`.
Claims expire after **4 hours**; if you see an expired claim, it's free to take.

### The Four Steps

```
1. THINK   — Read candidates.tsv (tried) + claims.json (in flight).
             Does your hypothesis already exist? Check expression column for
             semantic near-duplicates (e.g. r12+r13+r23 ≡ r13+r12+r23).

2. CLAIM   — Add an entry to claims.json before you start running.
             Use the claim.py helper or edit manually. Include:
               - problem name (matches problems/*.yaml)
               - expression you're testing
               - your handle
               - claimed_at (ISO 8601), expires_at (+4 hours)

3. RUN     — Execute the dual-filter pipeline:
               python vortex_checker.py --ic <IC> --expr "<expr>"   # checker first
               python oracle_wrapper.py --problem problems/<yaml>   # then oracle
             If checker PASS: record frac_var. If oracle FAIL: run --repair --diagnose.

4. PUBLISH — Open a PR that:
               - Adds a row to results/candidates.tsv
               - Adds a discovery note to results/discoveries/ (if DUAL-PASS or FLIPPED)
               - Updates the problem YAML notes section with checker results
               - Removes your entry from claims.json
```

---

## Plugin Contract: Three Files

To contribute a new domain, add these three files to `problems/`:

### 1. `problems/my_domain.yaml`
Use `problem_template.yaml` as your starting point. Required fields:
```yaml
name: "my_domain_conservation"
description: |
  One paragraph: what structure you're hunting.
model: "Qwen/Qwen3-4B-Base"
oracle: "stem_margin"
verification_set: "my_domain_facts.json"
pass_threshold: 0.8
monitors:
  - margin_sign        # required
monitors:              # add domain-specific ones as needed
adapter: null          # or path to domain adapter if you've trained one
```

### 2. `problems/my_domain_facts.json`
Verification set for the oracle. Format:
```json
{
  "facts": [
    {
      "id": "md01_some_law",
      "context": "Brief symbolic setup. Which quantity is conserved?",
      "truth": "Q = <compact symbolic expression>",
      "distractors": ["wrong1", "wrong2", "wrong3"]
    }
  ]
}
```
**Critical:** Use compact symbolic notation. Verbose prose fails the oracle
(confirmed across pilot and repair runs — see `results/discoveries/`).
Best context: `"[domain setup]. Which quantity is [conserved/constant]?"`
Best truth: `"Q = <formula>"` or `"Q = <formula> = const"`

### 3. `problems/my_domain_checker.py`
Numerical checker for your domain. Must implement:
```python
def integrate(params, pos0, t_end, n_points) -> (t, state):
    """Integrate equations of motion. Return (t_array, state_array)."""

def parse_state(t, state, params) -> dict:
    """Extract named quantities: distances, conserved quantities, etc."""

def frac_var(arr) -> float:
    """Fractional variation σ/|mean|. Values < 1e-3 → PASS."""
```
CLI convention: `python my_checker.py --ic <name> --expr "<python_expr>" --threshold 1e-3`

---

## Diagnostic Quadrants

Every candidate lands in one of four quadrants after the dual-filter pass:

| # | Oracle | Checker | Adapter Δ | Diagnosis | Action |
|---|--------|---------|-----------|-----------|--------|
| 1 | PASS | PASS | — | Known/confirmed structure | Archive to candidates.tsv, add to verification set |
| 2 | FAIL | PASS | improves | Fixable bias | Apply domain adapter, re-verify |
| 3 | FAIL | PASS | worsens | **Knowledge gap** | Document, generate training data, train domain adapter |
| 4 | — | FAIL | — | Numerical artifact | Discard (record frac_var in notes) |

The `--diagnose` flag in `oracle_wrapper.py` prints the quadrant automatically.
Threshold for knowledge gap: `margin_delta < -5.0` after repair pass.

---

## Candidates Registry Format

`results/candidates.tsv` — one row per tested hypothesis:

```
timestamp  hypothesis                          margin_mean  n_pass  verdict                   classification
```

Verdict options: `DUAL-PASS`, `ORACLE-FAIL+CHECKER-PASS`, `CHECKER-FAIL`,
`QUADRANT3→FLIPPED`, `ORACLE-FAIL`, `pending`

---

## What Makes a Good Discovery Note

Add a file to `results/discoveries/` when you have a DUAL-PASS or a FLIPPED quadrant.
Include:
- **Expression** — exact symbolic form
- **frac_var** — numerical evidence (which ICs, what time span)
- **Oracle margin** — baseline, after repair (if applicable)
- **Mechanism** (if known) — why is this conserved? Symmetry? Perturbative limit?
- **What still needs work** — open questions, regressions, related targets

See `results/discoveries/e2_repair_success.md` for an example.

---

## Current Open Problems

| Domain | File | Status | Next target |
|--------|------|--------|-------------|
| Figure-8 3-body | `3body_conservation.yaml` | Active | r_rms oracle flip (C09) |
| 2D point-vortex | `vortex_pair_conservation.yaml` | Active | vp11 Q=r₁₂+ε(r₁₃+r₂₃) adapter flip |
| *(your domain)* | `problem_template.yaml` | Open | — |

---

## Claiming a Problem (claim.py)

```bash
# See what's in flight
python claim.py list

# Claim before you start
python claim.py claim \
  --problem vortex_pair_conservation \
  --expr "r12 + eps*(r13+r23)" \
  --handle your_github_handle

# Release when done (or it auto-expires in 4h)
python claim.py release --id <claim_id>
```

Claim format in `claims.json`:
```json
{
  "id": "abc123",
  "problem": "vortex_pair_conservation",
  "expression": "r12 + eps*(r13+r23)",
  "claimer": "your_handle",
  "claimed_at": "2026-03-13T19:00:00Z",
  "expires_at": "2026-03-13T23:00:00Z",
  "status": "active"
}
```

---

## Acknowledgements

- **Coordination protocol** (THINK → CLAIM → RUN → PUBLISH, semantic duplicate
  detection, claim expiry) adapted from
  [autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home)
  by mutable-state-inc. Their design for asynchronous multi-agent research
  without centralized control directly inspired this human-in-the-loop variant.

- **Oracle infrastructure** built on top of the STEM Truth Oracle
  (Paper 9, DOI: [10.5281/zenodo.19005729](https://doi.org/10.5281/zenodo.19005729))
  and the Snap-On Communication Module architecture
  (Paper 8, DOI: [10.5281/zenodo.18902616](https://doi.org/10.5281/zenodo.18902616)).

- **Figure-8 choreography** problem domain inspired by Chenciner & Montgomery (2000)
  and the broader N-body choreography literature (Simó 2002).

- **Point-vortex domain** based on Kirchhoff (1876) and Aref (1979, 1983).
