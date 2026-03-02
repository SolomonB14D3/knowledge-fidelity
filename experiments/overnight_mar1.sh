#!/usr/bin/env bash
# Overnight GPU run — March 1, 2026
#
# Job 1: CatSAE hybrid floor weighting (~3-4h)
#   - 5 protected categories with per-category floor overrides
#   - Should preserve Religion breakthrough while recovering Age
#
# Job 2: γ critical margin sweep at γ=0.02, 0.03 (~4-6h)
#   - Tests whether bias protection collapses below γ=0.05
#   - Already have γ=0.05 from coarse sweep (Δρ=+0.090, 0 FAILs)
#
# Total estimated: ~7-10h
#
# Usage:
#   nohup bash experiments/overnight_mar1.sh > results/overnight_mar1.log 2>&1 &

set -e
cd "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"

MODEL="Qwen/Qwen2.5-7B-Instruct"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "============================================================"
echo "  OVERNIGHT RUN — $TIMESTAMP"
echo "  Model: $MODEL"
echo "  Jobs:  (1) CatSAE floor  (2) γ critical margin"
echo "============================================================"
echo ""

# ── Job 1: CatSAE hybrid floor weighting ─────────────────────────
echo ">>> JOB 1: CatSAE hybrid floor weighting"
echo ">>> Started: $(date)"
echo ""

python experiments/cat_sae_mlx.py "$MODEL" \
    --use-floor-overrides \
    --gamma 0.10 \
    --categories Age Gender_biology Race_ethnicity Sexual_orientation_biology Religion \
    -o results/overnight_sweep/Qwen2.5-7B-Instruct/cat_sae

echo ""
echo ">>> JOB 1 COMPLETE: $(date)"
echo ""

# ── Job 2: γ critical margin sweep ───────────────────────────────
echo ">>> JOB 2: γ critical margin sweep (γ=0.02, 0.03)"
echo ">>> Started: $(date)"
echo ""

python experiments/overnight_sweep.py "$MODEL" \
    --gammas 0.02 0.03 \
    --gamma-only \
    -o results/overnight_sweep

echo ""
echo ">>> JOB 2 COMPLETE: $(date)"
echo ""

echo "============================================================"
echo "  ALL JOBS COMPLETE — $(date)"
echo "============================================================"
