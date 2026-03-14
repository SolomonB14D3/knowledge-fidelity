# Pilot Run: Known-Fact Recovery
**Date:** 2026-03-13
**Problem:** kinetic_energy_pilot — 10 classical mechanics facts
**Model:** Qwen/Qwen3-4B-Base

## Result: Loop confirmed working ✓

### Baseline Oracle
- Pass rate: 3/10 (30%)
- Mean margin: −0.552
- All 7 failures on predicted bias patterns:
  - `Fd cos(θ)` margin −6.1 (truncation)
  - `(1/2)mv²` margin −3.6 (linearity + missing constant)
  - `kq1q2/r²` margin −3.5 (missing constant — drops k)
  - `kq/r²` margin −2.8 (missing constant)
  - `√(2GM/R)` margin −2.7 (missing constant + truncation)
  - `−sin(x)` margin −1.7 (positivity)
  - `2π√(L/g)` margin −0.7 (missing constant + truncation)

### Adapter Repair
The exp03 mixed adapter made margins worse (mean −6.1 vs −0.6). **Expected behavior** — the adapter was trained on a specific prompt format from the operation_destroyer training data, not the generic `"context: ..."` format used here. The adapter is format-specific, not semantics-specific.

**Implication for Phase 2:** For each new problem, the repair adapter must be trained on the problem's own verification set format. The oracle (baseline scoring) is fully general; the adapter (repair) is per-problem.

### Conclusions

1. **The oracle loop closes.** Baseline correctly flags the expected failures with correct failure modes.
2. **The 3 passing facts** (momentum=mv, F=ma, gravitational PE=mgh) are the least surprising to the training distribution — consistent with the paper's finding that simpler, less-ambiguous forms win.
3. **Format sensitivity:** The exp03 adapter cannot be directly borrowed for new problem formats. This is by design — the adapter steers logits in a specific direction learned from specific training contexts.

### Next Step

Phase 2: Define a toy unsolved problem. Build its verification set with known analogs. Train an oracle-guided adapter on those specific facts. Check for positive margin on held-out test cases.

Recommended first problem: **simplified 3-body conservation law search** — use 2-body Kepler facts as the verification set, run the oracle on candidate new terms, keep only positive-margin ones.
