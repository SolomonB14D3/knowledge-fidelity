#!/usr/bin/env bash
# Scale Ladder — Train all models, audit, and analyze.
#
# Usage:
#   bash experiments/scale_ladder/run_all.sh              # CPU, all sizes
#   bash experiments/scale_ladder/run_all.sh --device mps  # MPS for larger models
#   bash experiments/scale_ladder/run_all.sh --sizes "7M 12M 18M"  # Subset

set -e
cd "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"

DEVICE="${1:---device}"
DEVICE_VAL="${2:-cpu}"
SIZES="${3:-7M 12M 18M 34M 64M 153M 210M}"
SEED=42

# Parse --device and --sizes flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --device) DEVICE_VAL="$2"; shift 2 ;;
        --sizes) SIZES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "  SCALE LADDER — $(date)"
echo "  Device: $DEVICE_VAL"
echo "  Sizes: $SIZES"
echo "  Seed: $SEED"
echo "============================================================"
echo ""

# Phase 1: Train all models
for SIZE in $SIZES; do
    OUTDIR="results/scale_ladder/${SIZE}_seed${SEED}"
    if [ -f "$OUTDIR/model/config.json" ]; then
        echo "  [SKIP] $SIZE already trained at $OUTDIR"
    else
        echo "  [TRAIN] $SIZE..."
        python experiments/scale_ladder/train_model.py \
            --size "$SIZE" --seed "$SEED" --device "$DEVICE_VAL" \
            -o "$OUTDIR"
    fi
done

echo ""
echo "============================================================"
echo "  AUDITING ALL MODELS"
echo "============================================================"
echo ""

# Phase 2: Audit all models
for SIZE in $SIZES; do
    OUTDIR="results/scale_ladder/${SIZE}_seed${SEED}"
    if [ -f "$OUTDIR/audit_report.json" ]; then
        echo "  [SKIP] $SIZE already audited"
    else
        echo "  [AUDIT] $SIZE..."
        python experiments/scale_ladder/scale_audit.py \
            --checkpoint "$OUTDIR" --device "$DEVICE_VAL"
    fi
done

echo ""
echo "============================================================"
echo "  AUDITING TRAINING CHECKPOINTS (subspace emergence within-scale)"
echo "============================================================"
echo ""

# Phase 2b: Audit intermediate training checkpoints
# Shows when behavioral subspaces emerge DURING pretraining at each scale
for SIZE in $SIZES; do
    OUTDIR="results/scale_ladder/${SIZE}_seed${SEED}"
    for CKPT in "$OUTDIR"/checkpoint_*/; do
        [ -d "$CKPT" ] || continue
        if [ -f "$CKPT/audit_report.json" ]; then
            echo "  [SKIP] $SIZE checkpoint $(basename $CKPT) already audited"
        else
            echo "  [AUDIT] $SIZE checkpoint $(basename $CKPT)..."
            python experiments/scale_ladder/scale_audit.py \
                --checkpoint "$CKPT" --device "$DEVICE_VAL" --skip-dprime
        fi
    done
done

echo ""
echo "============================================================"
echo "  ANALYZING SCALING CURVES"
echo "============================================================"
echo ""

# Phase 3: Analyze
python experiments/scale_ladder/analyze_scaling.py

echo ""
echo "============================================================"
echo "  SCALE LADDER COMPLETE — $(date)"
echo "============================================================"
