#!/bin/bash
# Run all experimental phases sequentially after seed 456 sweep completes.
#
# Prerequisites:
#   - Qwen 3-seed sweep (42, 123, 456) must be complete
#   - conda env "confidence" active
#   - Run from knowledge-fidelity root directory
#
# Estimated wall time: ~6-8 hours total
#   Phase 1a (Llama cross-check):  ~2h (8 runs)
#   Phase 1b (3-seed stats):       <1min
#   Phase 2  (Ablation):           ~3h (8 runs)
#   Phase 3  (Calibration):        ~3h (8 runs) — CAN COMBINE with ablation
#   Phase 4  (OOD validation):     ~1h (4 runs, only λ=0.0,0.2)

set -e

PYTHON="/Users/bryan/miniconda3/envs/confidence/bin/python"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

echo "============================================"
echo "  Rho-Guided SFT: Full Experimental Pipeline"
echo "  Working dir: $DIR"
echo "  Python: $PYTHON"
echo "  Time: $(date)"
echo "============================================"

# ── Phase 1b: 3-Seed Statistics ──────────────────────────────────────
echo ""
echo "Phase 1b: Compiling 3-seed Qwen statistics..."
$PYTHON experiments/analyze_sweep_stats.py \
    results/alignment/mlx_rho_sft_sweep_7B_seeds42_123.json \
    results/alignment/mlx_rho_sft_sweep_Qwen_Qwen2.5-7B-Instruct.json \
    --json-out results/alignment/qwen7b_3seed_analysis.json

# ── Phase 1a: Llama Cross-Check ─────────────────────────────────────
echo ""
echo "Phase 1a: Llama-3.1-8B cross-check sweep..."
echo "  Start: $(date)"
$PYTHON -u experiments/rho_guided_sft_mlx.py \
    --model llama3.1-8b \
    --rho-weights 0.0,0.1,0.2,0.5 \
    --seeds 42,123 \
    --sft-size 1000 --epochs 1

echo "  End: $(date)"

# ── Phase 2: Ablation Study ─────────────────────────────────────────
echo ""
echo "Phase 2: Ablation study (4 conditions × 2 seeds)..."
echo "  Start: $(date)"
$PYTHON -u experiments/ablation_sft_mlx.py \
    --model qwen2.5-7b \
    --conditions sft-only,rho-guided,contrastive-only,shuffled-pairs \
    --rho-weight 0.2 \
    --seeds 42,123 \
    --sft-size 1000 --epochs 1

echo "  End: $(date)"

# ── Phase 3: Calibration Evaluation ─────────────────────────────────
echo ""
echo "Phase 3: Calibration evaluation (ECE + Brier)..."
echo "  Start: $(date)"
$PYTHON -u experiments/calibration_eval_mlx.py \
    --model qwen2.5-7b \
    --rho-weights 0.0,0.2,0.5 \
    --seeds 42,123 \
    --sft-size 1000 --epochs 1

echo "  End: $(date)"

# ── Phase 4: OOD Validation ─────────────────────────────────────────
echo ""
echo "Phase 4: OOD validation (Fidelity-Bench 2.0)..."
echo "  Start: $(date)"
$PYTHON -u experiments/ood_validation_mlx.py \
    --model qwen2.5-7b \
    --rho-weights 0.0,0.2 \
    --seeds 42,123 \
    --sft-size 1000 --epochs 1

echo "  End: $(date)"

# ── Final Analysis ──────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  ALL PHASES COMPLETE"
echo "  Time: $(date)"
echo "============================================"
echo ""
echo "Results:"
echo "  Phase 1b: results/alignment/qwen7b_3seed_analysis.json"
echo "  Phase 1a: results/alignment/mlx_rho_sft_sweep_*llama*.json"
echo "  Phase 2:  results/alignment/ablation_*.json"
echo "  Phase 3:  results/alignment/calibration_*.json"
echo "  Phase 4:  results/alignment/ood_*.json"
