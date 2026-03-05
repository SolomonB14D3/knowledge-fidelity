#!/bin/bash
# Phase 2: Curriculum ordering at 7M
# Tests whether injection ORDER matters (Kohlberg: you can't skip stages).
#
# Uses the best combination from Phase 1. Update SCHEDULE_BEHAVIORS if needed.
#
# Run 6: Primitive first (0-50%), sycophancy second (50-100%)
# Run 7: Sycophancy first (0-50%), primitive second (50-100%)
# Run 8: Both simultaneously throughout (standard interleaved)
#
# All runs: 7M, 5% injection (rate=20), seed=42

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
echo "Phase 2: Curriculum Ordering"
echo "============================================"
echo ""

# Run 6: Primitive first, sycophancy second
echo "--- Run 6: Primitive→Sycophancy ---"
python experiments/scale_ladder/train_contrastive.py \
    --size $SIZE --seed $SEED --device $DEVICE \
    --inject-rate $RATE --inject-behaviors primitive sycophancy \
    --probe-seed $PROBE_SEED --audit-interval $AUDIT_INTERVAL \
    --pairs-json "$PAIRS_JSON" \
    --inject-schedule "primitive:0-50,sycophancy:50-100" \
    -o "results/scale_ladder/${SIZE}_seed${SEED}_sched_pri_then_syc_r${RATE}"

echo "  Auditing Run 6..."
python experiments/scale_ladder/scale_audit.py \
    --checkpoint "results/scale_ladder/${SIZE}_seed${SEED}_sched_pri_then_syc_r${RATE}" \
    --device $DEVICE
echo ""

# Run 7: Sycophancy first, primitive second
echo "--- Run 7: Sycophancy→Primitive ---"
python experiments/scale_ladder/train_contrastive.py \
    --size $SIZE --seed $SEED --device $DEVICE \
    --inject-rate $RATE --inject-behaviors sycophancy primitive \
    --probe-seed $PROBE_SEED --audit-interval $AUDIT_INTERVAL \
    --pairs-json "$PAIRS_JSON" \
    --inject-schedule "sycophancy:0-50,primitive:50-100" \
    -o "results/scale_ladder/${SIZE}_seed${SEED}_sched_syc_then_pri_r${RATE}"

echo "  Auditing Run 7..."
python experiments/scale_ladder/scale_audit.py \
    --checkpoint "results/scale_ladder/${SIZE}_seed${SEED}_sched_syc_then_pri_r${RATE}" \
    --device $DEVICE
echo ""

# Run 8: Both simultaneously (control — same as Phase 1 Run 4)
echo "--- Run 8: Simultaneous (control) ---"
echo "  This is identical to Phase 1 Run 4 (primitive + sycophancy)."
echo "  Check: results/scale_ladder/${SIZE}_seed${SEED}_contr_pri_syc_r${RATE}/audit_report.json"
echo ""

echo "============================================"
echo "Phase 2 complete. Compare results:"
echo "  Run 6 (pri→syco): results/scale_ladder/${SIZE}_seed${SEED}_sched_pri_then_syc_r${RATE}/audit_report.json"
echo "  Run 7 (syco→pri): results/scale_ladder/${SIZE}_seed${SEED}_sched_syc_then_pri_r${RATE}/audit_report.json"
echo "  Run 8 (simultaneous): results/scale_ladder/${SIZE}_seed${SEED}_contr_pri_syc_r${RATE}/audit_report.json"
echo "============================================"
