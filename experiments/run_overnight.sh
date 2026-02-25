#!/bin/bash
# Overnight runner: waits for TruthfulQA v2 to finish, then runs Phases 2-4.
# Launched with caffeinate to prevent sleep.
#
# Usage: caffeinate -dims bash experiments/run_overnight.sh
#
# Estimated wall time: ~7-8 hours total

set -e

PYTHON="/Users/bryan/miniconda3/envs/confidence/bin/python"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

LOG="results/alignment/overnight_log.txt"
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo "  OVERNIGHT RUN: Phases 2-4"
echo "  Working dir: $DIR"
echo "  Started: $(date)"
echo "============================================"

# ── Wait for TruthfulQA v2 to finish ────────────────────────────────
TQA_PID=72544
if kill -0 $TQA_PID 2>/dev/null; then
    echo ""
    echo "Waiting for TruthfulQA v2 (PID $TQA_PID) to finish..."
    while kill -0 $TQA_PID 2>/dev/null; do
        sleep 60
    done
    echo "TruthfulQA v2 finished at $(date)"
else
    echo "TruthfulQA v2 already finished."
fi

# ── Phase 2: Ablation Study ─────────────────────────────────────────
echo ""
echo "============================================"
echo "  Phase 2: Ablation study (4 conditions × 2 seeds)"
echo "  Start: $(date)"
echo "============================================"
$PYTHON -u experiments/ablation_sft_mlx.py \
    --model qwen2.5-7b \
    --conditions sft-only,rho-guided,contrastive-only,shuffled-pairs \
    --rho-weight 0.2 \
    --seeds 42,123 \
    --sft-size 1000 --epochs 1

echo "  Phase 2 done: $(date)"

# ── Phase 3: Calibration Evaluation ─────────────────────────────────
echo ""
echo "============================================"
echo "  Phase 3: Calibration evaluation (ECE + Brier)"
echo "  Start: $(date)"
echo "============================================"
$PYTHON -u experiments/calibration_eval_mlx.py \
    --model qwen2.5-7b \
    --rho-weights 0.0,0.2,0.5 \
    --seeds 42,123 \
    --sft-size 1000 --epochs 1

echo "  Phase 3 done: $(date)"

# ── Phase 4: OOD Validation ─────────────────────────────────────────
echo ""
echo "============================================"
echo "  Phase 4: OOD validation (Fidelity-Bench 2.0)"
echo "  Start: $(date)"
echo "============================================"
$PYTHON -u experiments/ood_validation_mlx.py \
    --model qwen2.5-7b \
    --rho-weights 0.0,0.2 \
    --seeds 42,123 \
    --sft-size 1000 --epochs 1

echo "  Phase 4 done: $(date)"

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  ALL PHASES COMPLETE"
echo "  Finished: $(date)"
echo "============================================"
echo ""
echo "Results:"
echo "  TruthfulQA v2: results/alignment/truthfulqa_*.json"
echo "  Phase 2:       results/alignment/ablation_*.json"
echo "  Phase 3:       results/alignment/calibration_*.json"
echo "  Phase 4:       results/alignment/ood_*.json"
echo "  This log:      $LOG"
