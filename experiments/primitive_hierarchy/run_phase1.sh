#!/bin/bash
# Phase 1: Single-injection hierarchy ladder at 7M
# Tests whether primitive evaluative contrast cascades to downstream behaviors.
#
# Run 1: Primitive only (NEW)
# Run 2: Sycophancy only (DONE — exists at 7M_seed42_contr_syc_r20)
# Run 3: Bias only (DONE — exists at 7M_seed42_contr_bia_r20)
# Run 4: Primitive + sycophancy combined (NEW)
# Run 5: Primitive + sycophancy + bias combined (NEW)
#
# All runs: 7M, 5% injection (rate=20), seed=42, probe_seed=999
# After training, each model is audited for all 8 rho dimensions.

set -e
cd "$(dirname "$0")/../.."

PAIRS_JSON="data/contrastive/primitive_pairs.json"
DEVICE="mps"
SIZE="7M"
SEED=42
RATE=20
PROBE_SEED=999
AUDIT_INTERVAL=500

echo "============================================"
echo "Phase 1: Primitive Evaluative Hierarchy"
echo "============================================"
echo ""

# Run 1: Primitive only
echo "--- Run 1: Primitive only ---"
python experiments/scale_ladder/train_contrastive.py \
    --size $SIZE --seed $SEED --device $DEVICE \
    --inject-rate $RATE --inject-behaviors primitive \
    --probe-seed $PROBE_SEED --audit-interval $AUDIT_INTERVAL \
    --pairs-json "$PAIRS_JSON"

echo "  Auditing Run 1..."
python experiments/scale_ladder/scale_audit.py \
    --checkpoint "results/scale_ladder/${SIZE}_seed${SEED}_contr_pri_r${RATE}" \
    --device $DEVICE
echo ""

# Run 4: Primitive + sycophancy
echo "--- Run 4: Primitive + sycophancy ---"
python experiments/scale_ladder/train_contrastive.py \
    --size $SIZE --seed $SEED --device $DEVICE \
    --inject-rate $RATE --inject-behaviors primitive sycophancy \
    --probe-seed $PROBE_SEED --audit-interval $AUDIT_INTERVAL \
    --pairs-json "$PAIRS_JSON"

echo "  Auditing Run 4..."
python experiments/scale_ladder/scale_audit.py \
    --checkpoint "results/scale_ladder/${SIZE}_seed${SEED}_contr_pri_syc_r${RATE}" \
    --device $DEVICE
echo ""

# Run 5: Primitive + sycophancy + bias
echo "--- Run 5: Primitive + sycophancy + bias ---"
python experiments/scale_ladder/train_contrastive.py \
    --size $SIZE --seed $SEED --device $DEVICE \
    --inject-rate $RATE --inject-behaviors primitive sycophancy bias \
    --probe-seed $PROBE_SEED --audit-interval $AUDIT_INTERVAL \
    --pairs-json "$PAIRS_JSON"

echo "  Auditing Run 5..."
python experiments/scale_ladder/scale_audit.py \
    --checkpoint "results/scale_ladder/${SIZE}_seed${SEED}_contr_pri_syc_bia_r${RATE}" \
    --device $DEVICE
echo ""

echo "============================================"
echo "Phase 1 complete. Compare results:"
echo "  Run 1 (primitive):     results/scale_ladder/${SIZE}_seed${SEED}_contr_pri_r${RATE}/audit_report.json"
echo "  Run 2 (sycophancy):    results/scale_ladder/${SIZE}_seed${SEED}_contr_syc_r${RATE}/audit_report.json"
echo "  Run 3 (bias):          results/scale_ladder/${SIZE}_seed${SEED}_contr_bia_r${RATE}/audit_report.json"
echo "  Run 4 (prim+syco):     results/scale_ladder/${SIZE}_seed${SEED}_contr_pri_syc_r${RATE}/audit_report.json"
echo "  Run 5 (prim+syco+bia): results/scale_ladder/${SIZE}_seed${SEED}_contr_pri_syc_bia_r${RATE}/audit_report.json"
echo "============================================"
