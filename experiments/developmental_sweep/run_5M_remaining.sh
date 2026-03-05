#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

SEED=42
DEVICE=mps
INJECT_RATE=20
PROBE_SEED=999
AUDIT_INTERVAL=250

echo "================================================================"
echo "  5M — bias-only"
echo "  $(date)"
echo "================================================================"

python experiments/scale_ladder/train_contrastive.py \
    --size 5M --seed $SEED --device $DEVICE \
    --inject-rate $INJECT_RATE --probe-seed $PROBE_SEED \
    --audit-interval $AUDIT_INTERVAL \
    --inject-behaviors bias

echo ""
echo "  Running audit on 5M bias-only..."
python experiments/scale_ladder/scale_audit.py \
    --checkpoint results/scale_ladder/5M_seed42_contr_bia_r20 --device $DEVICE

echo ""
echo "================================================================"
echo "  5M — sycophancy-only"
echo "  $(date)"  
echo "================================================================"

python experiments/scale_ladder/train_contrastive.py \
    --size 5M --seed $SEED --device $DEVICE \
    --inject-rate $INJECT_RATE --probe-seed $PROBE_SEED \
    --audit-interval $AUDIT_INTERVAL \
    --inject-behaviors sycophancy

echo ""
echo "  Running audit on 5M sycophancy-only..."
python experiments/scale_ladder/scale_audit.py \
    --checkpoint results/scale_ladder/5M_seed42_contr_syc_r20 --device $DEVICE

echo ""
echo "================================================================"
echo "  5M remaining conditions COMPLETE"
echo "  $(date)"
echo "================================================================"
