#!/bin/bash
# Overnight experiment queue — March 5, 2026
#
# Experiments in priority order:
#   1. 7M bias-only injection (5%) → measures cross-transfer to sycophancy  ~25 min
#   2. 7M sycophancy-only injection (5%) → complement                       ~25 min
#   3. 34M contrastive (bias+syco, 5%) → critical scaling test             ~90 min
#   4. 64M contrastive (bias+syco, 5%) → perplexity comparison            ~3-5 hrs
#
# Total estimated: ~5-7 hours
#
# These experiments directly test:
#   - Topological transfer (Paper 4, Section 5.3): does one behavior lift another?
#   - Scaling beyond 12M: does the effect hold, grow, or diminish?
#
# Usage:
#   nohup bash experiments/run_overnight_mar5.sh > experiments/overnight_mar5.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="mps"
TIMESTAMP=$(date +%Y%m%d_%H%M)

echo "============================================================"
echo "  Overnight Queue — Started at $(date)"
echo "  Device: $DEVICE"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────
# Experiment 1: 7M bias-only injection (5% = rate 20)
# Tests: Does bias-only injection improve sycophancy? (topological transfer)
# ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/4] 7M bias-only injection (5%, rate=20)"
echo "    Start: $(date)"
python experiments/scale_ladder/train_contrastive.py \
    --size 7M --seed 42 --device "$DEVICE" \
    --inject-rate 20 --inject-behaviors bias \
    --probe-seed 999 --audit-interval 500

# Audit: full 8-dim rho + subspace extraction
BIAS_ONLY_DIR="results/scale_ladder/7M_seed42_contr_bia_r20"
echo "    Auditing bias-only model..."
python experiments/scale_ladder/scale_audit.py \
    --checkpoint "$BIAS_ONLY_DIR" --device "$DEVICE"

echo "    Done: $(date)"

# ─────────────────────────────────────────────────────────────────
# Experiment 2: 7M sycophancy-only injection (5% = rate 20)
# Tests: Does sycophancy-only injection improve bias? (complement to Exp 1)
# ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/4] 7M sycophancy-only injection (5%, rate=20)"
echo "    Start: $(date)"
python experiments/scale_ladder/train_contrastive.py \
    --size 7M --seed 42 --device "$DEVICE" \
    --inject-rate 20 --inject-behaviors sycophancy \
    --probe-seed 999 --audit-interval 500

SYCO_ONLY_DIR="results/scale_ladder/7M_seed42_contr_syc_r20"
echo "    Auditing sycophancy-only model..."
python experiments/scale_ladder/scale_audit.py \
    --checkpoint "$SYCO_ONLY_DIR" --device "$DEVICE"

echo "    Done: $(date)"

# ─────────────────────────────────────────────────────────────────
# Experiment 3: 34M contrastive (bias+sycophancy, 5%)
# Tests: Does the effect scale beyond 12M? Critical for Paper 4 revision.
# 34M vanilla already has bias=0.238, sycophancy=0.300 — does contrastive
# injection push these higher, or are diminishing returns already visible?
# ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> [3/4] 34M contrastive injection (5%, bias+sycophancy)"
echo "    Start: $(date)"
python experiments/scale_ladder/train_contrastive.py \
    --size 34M --seed 42 --device "$DEVICE" \
    --inject-rate 20 --inject-behaviors bias sycophancy \
    --probe-seed 999 --audit-interval 1000

CONTR_34M_DIR="results/scale_ladder/34M_seed42_contr_bia_syc_r20"
echo "    Auditing 34M contrastive model..."
python experiments/scale_ladder/scale_audit.py \
    --checkpoint "$CONTR_34M_DIR" --device "$DEVICE"

echo "    Done: $(date)"

# ─────────────────────────────────────────────────────────────────
# Experiment 4: 64M contrastive (bias+sycophancy, 5%)
# Tests: Does injection fix the 64M bias regression (0.238→0.087)?
# Also the largest scale we've tested — measures diminishing returns.
# ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> [4/4] 64M contrastive injection (5%, bias+sycophancy)"
echo "    Start: $(date)"
python experiments/scale_ladder/train_contrastive.py \
    --size 64M --seed 42 --device "$DEVICE" \
    --inject-rate 20 --inject-behaviors bias sycophancy \
    --probe-seed 999 --audit-interval 1000

CONTR_64M_DIR="results/scale_ladder/64M_seed42_contr_bia_syc_r20"
echo "    Auditing 64M contrastive model..."
python experiments/scale_ladder/scale_audit.py \
    --checkpoint "$CONTR_64M_DIR" --device "$DEVICE"

echo "    Done: $(date)"

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  All experiments complete — $(date)"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  1. $BIAS_ONLY_DIR"
echo "  2. $SYCO_ONLY_DIR"
echo "  3. $CONTR_34M_DIR"
echo "  4. $CONTR_64M_DIR"
echo ""
echo "Next steps:"
echo "  - Check bias-only/syco-only for cross-transfer (topological transfer test)"
echo "  - Compare 34M/64M contrastive vs vanilla for scaling curve"
echo "  - Run: python scripts/build_master_db.py"
