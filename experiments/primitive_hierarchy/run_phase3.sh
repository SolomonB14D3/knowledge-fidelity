#!/bin/bash
# Phase 3: Cascade test at larger scale
# Tests whether cleaning up lower hierarchy levels creates conditions
# for reasoning to emerge spontaneously.
#
# Inject winning protocol from Phases 1+2 for first 20% of training only,
# then train to convergence on pure LM data. Monitor reasoning rho
# throughout post-injection training with frequent audits.
#
# Run at 12M (or change SIZE below for 34M).

set -e
cd "$(dirname "$0")/../.."

PAIRS_JSON="data/contrastive/primitive_pairs.json"
DEVICE="mps"
SIZE="12M"
SEED=42
RATE=20
PROBE_SEED=999
AUDIT_INTERVAL=200  # More frequent audits to catch emergence

# UPDATE these based on Phase 1+2 results:
# If primitive alone works best, use just "primitive"
# If primitive+sycophancy with ordering works best, use schedule
INJECT_BEHAVIORS="primitive sycophancy"
INJECT_UNTIL=0.2  # Stop injection at 20% of training

echo "============================================"
echo "Phase 3: Cascade Test — ${SIZE}"
echo "  Injection: ${INJECT_BEHAVIORS} for first ${INJECT_UNTIL}x of training"
echo "  Then: pure LM training to convergence"
echo "  Monitoring: reasoning rho every ${AUDIT_INTERVAL} steps"
echo "============================================"
echo ""

python experiments/scale_ladder/train_contrastive.py \
    --size $SIZE --seed $SEED --device $DEVICE \
    --inject-rate $RATE --inject-behaviors $INJECT_BEHAVIORS \
    --probe-seed $PROBE_SEED --audit-interval $AUDIT_INTERVAL \
    --pairs-json "$PAIRS_JSON" \
    --inject-until $INJECT_UNTIL \
    -o "results/scale_ladder/${SIZE}_seed${SEED}_cascade_pri_syc_until20_r${RATE}"

echo "  Running final audit..."
python experiments/scale_ladder/scale_audit.py \
    --checkpoint "results/scale_ladder/${SIZE}_seed${SEED}_cascade_pri_syc_until20_r${RATE}" \
    --device $DEVICE

echo ""
echo "============================================"
echo "Phase 3 complete."
echo "  Results: results/scale_ladder/${SIZE}_seed${SEED}_cascade_pri_syc_until20_r${RATE}/"
echo "  Check audit_step_*/audit_data.json for reasoning emergence trajectory"
echo "============================================"
