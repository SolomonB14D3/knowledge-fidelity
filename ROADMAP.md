# Operation Frontier: Roadmap to an Auto-Solve Researching Machine

**Author:** SolomonB14D3
**Goal:** Turn the STEM Truth Oracle (perfect log-prob margin oracle + mixed adapter) into a fully autonomous "solver" for unsolved math and physics problems.

The machine will:
- Take any unsolved problem (or toy version of one)
- Let Claude propose aggressive, destructive hypotheses (new terms, rewrites, architectures)
- Run cheap experiments on a small model / PINN / symbolic regressor
- Use the **margin oracle** (from the paper) + rho-eval + formal checkers as the verifier
- Repair failures with the mixed adapter
- Keep only positive-margin, verifiable discoveries
- Iterate overnight until a novel, correct insight emerges

This runs on a laptop (MLX + Apple Silicon) and scales with more GPUs.

---

## Current State

- STEM Truth Oracle paper written and published — perfect binary margin oracle + mixed adapter. DOI: [10.5281/zenodo.19005729](https://doi.org/10.5281/zenodo.19005729)
- Benchmark, adapters, eval code, and autoresearch wrapper (yaml + program.md) all in place.
- Operation Destroyer proved the destructive-discovery loop works.

---

## Phase 0: Ship the Foundation

1. ~~Polish the paper (title/abstract tweaks, fix self-references, add GitHub link).~~ ✓ Done (v1.1)
2. ~~Publish on Zenodo.~~ ✓ Done (v1.1 — DOI: 10.5281/zenodo.19010629)
3. Create a new repo: `SolomonB14D3/stem-truth-oracle` (or add to existing).
   - Include: benchmark.json, trained adapters, eval scripts, margin-oracle.py.
4. Tweet thread (hook: the margin oracle result from Fig 4 in the paper).
5. Add a "Code & Data" section at the end of the paper linking to the repo.

**Milestone:** Paper live + repo public. This alone opens doors — people start using the oracle.

---

## Phase 1: Wire the Oracle into the Autoresearch Loop

Create a new folder: `experiments/operation_frontier/`

1. **Copy the autoresearch wrapper** from Operation Destroyer setup.

2. Add a new `problem.yaml` template:
   ```yaml
   problem:
     name: "discover_new_3body_conservation"
     model: "qwen3-4b-base"
     oracle: "stem_margin"
     monitor:
       - margin_sign
       - rho_expression_unlock
       - energy_violation
       - sympy_simplify
     adapter: "mixed_stem_oracle"
     destructive_mode: true
     budget: "3000 steps or 5M tokens"
   ```

3. Update `program.md` for Claude:
   ```
   You are a destructive math/physics archaeologist. Propose aggressive changes
   that might temporarily break known cases if they could unlock new truths.
   After every run, apply the STEM margin oracle. Only keep configs with positive
   margin on verification set + passing formal checks. Explain what hidden
   structure you suspect was revealed.
   ```

4. Write `oracle_wrapper.py` (~30 lines):
   - Compute log-prob margin on a fixed verification set of analogous known facts.
   - If margin > 0 → success signal; else → apply mixed adapter and retry.

5. Test on a trivial "solved" problem first (e.g., rediscover kinetic energy coefficient) to confirm the loop closes.

**Milestone:** One full overnight run that recovers a known fact with positive margin. Archive as `pilot_known_fact.md`.

---

## Phase 2: Toy Unsolved Problems

Pick one of these (all toy-scale, high-signal, laptop-friendly):

**Top recommendation:** Simplified 3-body gravitational conservation law search
**Alternatives:**
- Missing symmetry term in 2D Navier-Stokes
- New identity in elementary number theory (Erdős-style toy version)
- Conservation-law discovery in a reduced PINN

For each:
1. Define a small verification set of known analogs (e.g., 2-body Kepler cases).
2. Run 50–100 experiments overnight.
3. Require **positive margin** + rho-unlock improvement + formal check (SymPy or energy error < 1e-6).
4. Log every candidate in `results/<problem>/candidates.tsv`.
5. When you get a positive-margin winner that passes formal check on unseen trajectories → you have a discovery.

**Milestone:** At least one verifiable new term/identity that generalizes. Write a short "Discovery Note" markdown with plots.

---

## Monitor Evolution Rule (2026-03-13)

New monitors start manual — one Python file in `monitors/`, one line in your problem yaml.
After 3–5 discoveries we turn the common patterns into an auto-registry (`monitors/registry.py`).
Until then: manual addition = maximum flexibility and speed. Takes <10 minutes per new monitor.
Forces the loop to stay grounded in real physics/math signals instead of guessing in advance.

This is exactly how every good research tool in this codebase evolved: rho-eval started with
one dimension and grew; Snap-On started with one mode. Monitors will follow the same arc.

**Current manual monitors** (`monitors/`):

| File | Monitor name (yaml) | Discovery | Status |
|------|--------------------|-----------|-|
| `monitors/sum_pairwise_distances.py` | `sum_pairwise_distances_variance` | r12+r13+r23 on figure-8 (C01) | ✓ Live |
| `monitors/e2_symmetric_poly.py` | `e2_symmetric_poly_variance` | r12·r13+r12·r23+r13·r23 (C10, knowledge gap) | ✓ Live |
| *(next)* | `rms_pairwise_variance` | sqrt((r12²+r13²+r23²)/3) (C09) | pending file |

**When to promote to registry.py:** when ≥3 monitors share a common structure (e.g., all are
"fractional variance of a symmetric function of pairwise distances"), factor out the pattern.

---

## Phase 3: Scale to Real Unsolved Problems

Once the loop works on toys:

1. Move to actual open problems (check erdosproblems.com Tier 1–3 or recent physics open questions).
2. Add more monitors:
   - Lean proof attempt (via API)
   - PINN energy violation on high-fidelity sim
   - Novelty score (semantic distance from known literature)
3. Parallelize: run 2–3 problems at once on different machines.
4. When a candidate survives 3 independent formal checks + positive margin → write it up.

**Target problems:**
- A constrained version of Andrews-Curtis conjecture
- Missing term in reduced fluid equations
- New conserved quantity in N-body with variable masses

---

## Phase 4: Turn It Into a Public Researching Machine

1. Package everything as `autoresearch-solver` PyPI tool.
2. Add web dashboard (simple Streamlit) showing live candidates.
3. Open to the community: others plug in their unsolved problems and let the swarm run.
4. Continue the rho/Snap-On line in parallel — use the solver to test new geometric interventions.

---

## Success Criteria

- **Phase 0:** Paper + repo live, published on Zenodo. ✓ Done
- **Phase 1:** Oracle wrapper closes the loop on a known fact. ✓ Done (pilot confirmed, oracle working)
- **Phase 2:** First toy discovery (positive margin + generalizes to held-out set). ✓ First candidate: choreographic distance sum (dual-filter pass, see `results/discovery_note_01_choreographic_distance_sum.md`)
- **Phase 3:** First real unsolved-problem candidate that survives formal checks.
- **Phase 4:** Public repo + first co-discovery with another researcher.

---

## Resources Already in Place

| Asset | Location |
|-------|----------|
| Margin oracle code | `experiments/operation_destroyer/` |
| Mixed adapter (trained) | `experiments/operation_destroyer/adapters/` |
| Autoresearch loop + yaml | `autoresearch-sample-efficiency-macos/` |
| rho-eval suite | `src/rho_eval/` |
| MLX on Apple Silicon | local |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| False positives | Require two independent formal checks |
| Data contradictions (stubborn 9) | Flag explicitly in oracle; don't retry endlessly |
| Compute creep | Start with 1.5B–4B models; scale only after pilot success |
| Scope creep | One problem per phase; archive everything else |

---

The hard parts are done: mechanism works, oracle is perfect, adapter repairs biases. The rest is wiring and running.

**Next action:** Phase 2 batch 2 (speed sum, geometric mean, kinetic energy sum on figure-8) + stem-truth-oracle repo spinout (Phase 0).
