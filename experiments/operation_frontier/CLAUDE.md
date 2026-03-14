# NoetherSolve — AI Agent Instructions

**What this project does:** Find physical/mathematical structures that are numerically conserved but not recognized by LLMs — then close those gaps with targeted adapters.

Emmy Noether proved every continuous symmetry corresponds to a conserved quantity. NoetherSolve finds where LLMs fail to recognize those quantities and fixes it.

---

## Your First Move — Always

Before doing anything else:

```bash
# 1. See what's already been tried (avoid duplicates)
cat results/candidates.tsv

# 2. See what's currently being hunted
python claim.py list

# 3. See the current state of all discoveries
python dashboard.py --open
```

Then read `README.md` for the architecture and `CONTRIBUTING.md` for the full protocol.

---

## The Dual-Filter Pipeline

Every hypothesis goes through two filters:

```
Hypothesis (expression)
       │
       ▼
 Numerical checker          ← Is this actually conserved?
 (RK45, frac_var test)        frac_var = σ/|mean| < 5e-3 → PASS
       │ PASS
       ▼
 Oracle filter              ← Does the model know it?
 (log-prob margin)            margin = log P(truth) − log P(best distractor)
       │
       ├─ PASS  → DUAL-PASS (archive it)
       └─ FAIL  → Run repair (adapter):
                    ├─ margin improves → FIXABLE BIAS (apply adapter)
                    └─ margin worsens  → KNOWLEDGE GAP (train new adapter)
```

**Diagnostic quadrants:**
| # | Oracle | Checker | Adapter Δ | Action |
|---|--------|---------|-----------|--------|
| 1 | PASS | PASS | — | Archive, add to verification set |
| 2 | FAIL | PASS | improves | Apply adapter, re-verify |
| 3 | FAIL | PASS | worsens | **Knowledge gap** — train domain adapter |
| 4 | — | FAIL | — | Discard |

---

## Running an Experiment — Step by Step

### Step 1: Claim your hypothesis (prevents duplicate work)
```bash
python claim.py claim \
  --problem vortex_pair_conservation \
  --expr "your expression here" \
  --handle your-name
```
Claims expire after 4 hours. Check `claims.json` to see active claims.

### Step 2: Run the numerical checker
```bash
# Figure-8 3-body
python conservation_checker.py --ic figure8 --expr "s['r12']+s['r13']+s['r23']"
python conservation_checker.py --all   # run all ICs and known candidates

# 2D point-vortex
python vortex_checker.py --ic restricted --expr "s['r12'] + 0.3*(s['r13']+s['r23'])"
python vortex_checker.py --all

# frac_var < 5e-3 → PASS (proceed to oracle)
# frac_var > 5e-3 → FAIL (discard, record in candidates.tsv)
```

### Step 3: Run the oracle (if checker passes)
```bash
# Apple Silicon (MLX)
python oracle_wrapper.py --problem problems/vortex_pair_conservation.yaml

# Linux/CUDA (PyTorch — no MLX needed)
python noethersolve_torch.py eval-oracle \
  --problem problems/vortex_pair_conservation.yaml --diagnose
```

### Step 4: If oracle fails, diagnose and repair
```bash
python oracle_wrapper.py --problem problems/vortex_pair_conservation.yaml \
    --repair --diagnose
# Prints quadrant: FIXABLE_BIAS (adapter helps) or KNOWLEDGE_GAP (need training data)
```

### Step 5: If knowledge gap, train a domain adapter
```bash
# Apple Silicon (MLX)
python train_vortex_adapter.py --data my_training_data.json --steps 1500

# Linux/CUDA (PyTorch)
python noethersolve_torch.py train-adapter \
  --data my_training_data.json \
  --model Qwen/Qwen3-4B-Base \
  --out adapters/my_adapter.npz
```

### Step 6: Publish results
Add a row to `results/candidates.tsv` and open a PR. If DUAL-PASS or FLIPPED, add a discovery note to `results/discoveries/`. Remove your entry from `claims.json`.

---

## Finding Open Problems — Read the Live Sources

**Never rely on static lists in this file — they go stale.** Always query the live sources:

```bash
# What's already been tried? (closed holes — don't duplicate)
cat results/candidates.tsv

# What's actively being hunted right now? (in-flight claims)
python claim.py list

# What's still open in each domain? (suggested next targets from domain experts)
grep -A 20 "Next interesting targets" problems/vortex_pair_conservation.yaml
grep -A 20 "Next interesting targets" problems/3body_conservation.yaml

# Full picture with charts
python dashboard.py --open
```

**Interpreting candidates.tsv to find open work:**
- `ORACLE-FAIL+CHECKER-PASS` with no claim → open gap, good target for adapter repair
- `QUADRANT3→FLIPPED` → closed, but suggests related expressions worth trying
- `CHECKER-FAIL` → dead end, skip entirely
- Rows in `claims.json` with future `expires_at` → someone is working on it, pick something else

**To propose a new domain entirely:**
Copy `problems/problem_template.yaml` and add three files: `my_domain.yaml` + `my_domain_facts.json` + `my_domain_checker.py`. See `CONTRIBUTING.md` for the plugin contract.

---

## What NOT to Do

- **Do not re-test already-closed hypotheses.** Check `candidates.tsv` first. Semantic near-duplicates count (r12+r13+r23 ≡ r13+r12+r23).
- **Do not use the mixed STEM adapter on vortex facts.** It makes vortex margins catastrophically worse (confirmed: -10.6 → -30.5). Use the domain-specific vortex adapter.
- **Do not use the choreography adapter on vortex problems** (wrong domain, cross-domain interference confirmed).
- **Do not naively merge/average adapters across domains.** `multi_domain_v2` (averaged weights of vortex + H-Lz adapters) underperforms both specialists on every benchmark. Adapter averaging degrades specialist performance. If you need multi-domain coverage, use task-vector merging or keep adapters separate and swap them per domain.
- **Do not test equilateral triangle ICs as interesting.** Equilateral = relative equilibrium for ANY circulation values — all rᵢⱼ=const exactly. Trivially conserved, not interesting.
- **Do not use verbose prose in oracle facts.** Compact symbolic notation only: `"Q = r₁₂ + ε(r₁₃+r₂₃) = const"`. Verbose prose fails the oracle (confirmed in pilot runs).
- **Do not hardcode absolute paths** in any script. Use `os.path.dirname(__file__)` for relative resolution.

---

## Key Files

| File | What it does |
|------|-------------|
| `conservation_checker.py` | Figure-8 3-body RK45 integrator + frac_var checker |
| `vortex_checker.py` | 2D point-vortex Kirchhoff integrator + frac_var checker |
| `oracle_wrapper.py` | Log-prob margin oracle + repair pass + quadrant diagnosis (MLX) |
| `noethersolve_torch.py` | Same as oracle_wrapper but PyTorch/CUDA — no MLX needed |
| `claim.py` | THINK→CLAIM→RUN→PUBLISH coordination (4h claim expiry) |
| `dashboard.py` | Regenerate results dashboard from candidates.tsv |
| `train_vortex_adapter.py` | Train vortex-specific logit adapter (MLX) |
| `train_choreography_adapter.py` | Train figure-8 choreography adapter (MLX) |
| `results/candidates.tsv` | **The shared ledger** — all tested hypotheses and verdicts |
| `claims.json` | Active claims registry — check before starting |
| `problems/*.yaml` | Domain plugin definitions |
| `problems/*_facts.json` | Oracle verification sets (8–15 facts per domain) |
| `adapters/` | Trained adapter weights (gitignored — local only) |

---

## Checker Output Interpretation

```
frac_var < 1e-6   → Near-exact conservation (e.g. H, Lz — fundamental laws)
frac_var < 5e-3   → PASS threshold — approximate invariant worth checking oracle
frac_var < 1e-2   → Borderline (record, may be IC-dependent)
frac_var > 1e-2   → FAIL — not conserved on this IC
```

## Oracle Output Interpretation

```
margin > +1.5     → Strong PASS — model confidently knows this
margin  0 to +1.5 → Weak PASS — model leans correct
margin -1 to 0    → Borderline — try adapter repair
margin < -5       → Strong FAIL — likely knowledge gap
margin < -20      → Extreme gap — domain-specific adapter required
```

---

## Hardware Notes

**Apple Silicon (M-series):**
- Use `oracle_wrapper.py` and `train_*_adapter.py` (MLX backend)
- MLX loads Qwen3-4B-Base in ~1.5s
- Do NOT use PyTorch MPS for training — deadlocks on backward passes

**Linux / NVIDIA GPU:**
- Use `noethersolve_torch.py` (PyTorch backend, no MLX dependency)
- `pip install torch transformers accelerate` then run normally
- CUDA auto-detected; falls back to CPU if no GPU

**Both backends** produce `.npz` adapter files with identical key names (`gate_proj.weight`, `up_proj.weight`, `down_proj.weight`) — adapters are cross-platform.

---

## Credits

- **Coordination protocol** (THINK→CLAIM→RUN→PUBLISH) adapted from [autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home) by mutable-state-inc.
- **Oracle infrastructure** built on STEM Truth Oracle (Paper 9, DOI: 10.5281/zenodo.19005729) and Snap-On Communication Modules (Paper 8, DOI: 10.5281/zenodo.18902616).
- **Noether's theorem** (Emmy Noether, 1915) — the reason any of this works.
