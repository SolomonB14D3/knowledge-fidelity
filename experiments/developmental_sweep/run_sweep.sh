#!/bin/bash
# Developmental Scale Sweep — mapping signal utility across scales
#
# 3 scales x 4 conditions = 12 runs. 7M already has all 4 conditions.
# New runs: 8 (3M and 5M, 4 conditions each).
#
# Scales:
#   3M  = 2L/64d/2H/ctx512, ~3.3M actual params, 50M tokens, ~3K steps
#   5M  = 2L/96d/2H/ctx512, ~5.1M actual params, 75M tokens, ~4.6K steps
#   7M  = 4L/128d/4H/ctx512, 7.3M params, 100M tokens (existing)
#
# Conditions:
#   1. Vanilla (no injection)  — via train_model.py
#   2. Primitive-only 5%       — via train_contrastive.py
#   3. Bias-only 5%            — via train_contrastive.py
#   4. Sycophancy-only 5%      — via train_contrastive.py
#
# Usage:
#   bash experiments/developmental_sweep/run_sweep.sh [--scale 3M|5M|all] [--skip-audit]

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_ROOT"

SEED=42
DEVICE=mps
INJECT_RATE=20
PROBE_SEED=999
AUDIT_INTERVAL=250
PAIRS_JSON="data/contrastive/primitive_pairs.json"

# Parse args
SCALE_FILTER="all"
SKIP_AUDIT=false
for arg in "$@"; do
    case $arg in
        --scale=*) SCALE_FILTER="${arg#*=}" ;;
        --skip-audit) SKIP_AUDIT=true ;;
    esac
done

run_vanilla() {
    local SIZE=$1

    echo ""
    echo "================================================================"
    echo "  TRAINING: ${SIZE} — vanilla (no injection)"
    echo "  $(date)"
    echo "================================================================"

    python experiments/scale_ladder/train_model.py \
        --size "$SIZE" \
        --seed "$SEED" \
        --device "$DEVICE"

    echo "  Training complete: ${SIZE} — vanilla"
}

run_contrastive() {
    local SIZE=$1
    local LABEL=$2
    shift 2

    echo ""
    echo "================================================================"
    echo "  TRAINING: ${SIZE} — ${LABEL}"
    echo "  $(date)"
    echo "================================================================"

    python experiments/scale_ladder/train_contrastive.py \
        --size "$SIZE" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --inject-rate "$INJECT_RATE" \
        --probe-seed "$PROBE_SEED" \
        --audit-interval "$AUDIT_INTERVAL" \
        "$@"

    echo "  Training complete: ${SIZE} — ${LABEL}"
}

run_audit() {
    local CHECKPOINT=$1

    if [ "$SKIP_AUDIT" = true ]; then
        echo "  Skipping audit for $CHECKPOINT (--skip-audit)"
        return
    fi

    if [ ! -d "$CHECKPOINT" ]; then
        echo "  WARNING: Checkpoint dir not found: $CHECKPOINT"
        return
    fi

    echo ""
    echo "  Running scale_audit.py on ${CHECKPOINT}..."
    python experiments/scale_ladder/scale_audit.py \
        --checkpoint "$CHECKPOINT" \
        --device "$DEVICE"
}

###############################################################################
# 3M Scale
###############################################################################
if [ "$SCALE_FILTER" = "all" ] || [ "$SCALE_FILTER" = "3M" ]; then
    echo ""
    echo "========================================"
    echo "  SCALE: 3M (2L/64d/2H, 50M tokens)"
    echo "  $(date)"
    echo "========================================"

    # Condition 1: Vanilla
    run_vanilla "3M"
    run_audit "results/scale_ladder/3M_seed42"

    # Condition 2: Primitive-only
    run_contrastive "3M" "primitive-only" \
        --inject-behaviors primitive --pairs-json "$PAIRS_JSON"
    run_audit "results/scale_ladder/3M_seed42_contr_pri_r20"

    # Condition 3: Bias-only
    run_contrastive "3M" "bias-only" \
        --inject-behaviors bias
    run_audit "results/scale_ladder/3M_seed42_contr_bia_r20"

    # Condition 4: Sycophancy-only
    run_contrastive "3M" "sycophancy-only" \
        --inject-behaviors sycophancy
    run_audit "results/scale_ladder/3M_seed42_contr_syc_r20"

    echo ""
    echo "  3M scale complete! $(date)"
fi

###############################################################################
# 5M Scale
###############################################################################
if [ "$SCALE_FILTER" = "all" ] || [ "$SCALE_FILTER" = "5M" ]; then
    echo ""
    echo "========================================"
    echo "  SCALE: 5M (2L/96d/2H, 75M tokens)"
    echo "  $(date)"
    echo "========================================"

    # Condition 1: Vanilla
    run_vanilla "5M"
    run_audit "results/scale_ladder/5M_seed42"

    # Condition 2: Primitive-only
    run_contrastive "5M" "primitive-only" \
        --inject-behaviors primitive --pairs-json "$PAIRS_JSON"
    run_audit "results/scale_ladder/5M_seed42_contr_pri_r20"

    # Condition 3: Bias-only
    run_contrastive "5M" "bias-only" \
        --inject-behaviors bias
    run_audit "results/scale_ladder/5M_seed42_contr_bia_r20"

    # Condition 4: Sycophancy-only
    run_contrastive "5M" "sycophancy-only" \
        --inject-behaviors sycophancy
    run_audit "results/scale_ladder/5M_seed42_contr_syc_r20"

    echo ""
    echo "  5M scale complete! $(date)"
fi

echo ""
echo "================================================================"
echo "  DEVELOPMENTAL SWEEP COMPLETE"
echo "  $(date)"
echo "================================================================"
