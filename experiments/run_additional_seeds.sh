#!/usr/bin/env bash
# run_additional_seeds.sh — Expand statistical power for rho-guided SFT paper
#
# Phase A: Additional ablation seeds (n=2 → n=5)
#   - 3 new seeds (456, 789, 1337) × 4 conditions = 12 runs
#   - ~30 min each on M3 Ultra = ~6 hours
#
# Phase B: Additional dose-response seeds (n=3 → n=5)
#   - 2 new seeds (789, 1337) × 4 rho_weights = 8 runs
#   - ~25 min each = ~3.3 hours
#
# Phase C: No-margin ablation control (γ=0)
#   - rho-guided with margin=0.0 for 5 seeds = 5 runs
#   - ~25 min each = ~2 hours
#
# Total: ~25 runs, ~11 hours on M3 Ultra
#
# Usage:
#   cd /Volumes/4TB\ SD/ClaudeCode/knowledge-fidelity
#   bash experiments/run_additional_seeds.sh [--phase A|B|C|all]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results/alignment"

# Use the confidence conda env with torch + mlx + mlx_lm
PYTHON="/Users/bryan/miniconda3/envs/confidence/bin/python"
export PATH="/Users/bryan/miniconda3/envs/confidence/bin:$PATH"
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"

# Parse phase argument
PHASE="${1:-all}"
PHASE="${PHASE#--phase=}"
PHASE="${PHASE#--phase }"

echo "============================================================"
echo "  Additional Seeds Runner — rho-guided SFT"
echo "  Phase: $PHASE"
echo "  Started: $(date)"
echo "============================================================"


# ── Phase A: Additional Ablation Seeds ──────────────────────────────

run_phase_a() {
    echo ""
    echo "============================================================"
    echo "  PHASE A: Additional Ablation Seeds (456, 789, 1337)"
    echo "  4 conditions × 3 seeds = 12 runs"
    echo "  Started: $(date)"
    echo "============================================================"

    $PYTHON experiments/ablation_sft_mlx.py \
        --model qwen2.5-7b \
        --conditions sft-only,rho-guided,contrastive-only,shuffled-pairs \
        --rho-weight 0.2 \
        --seeds 456,789,1337 \
        --sft-size 1000 \
        --epochs 1 \
        --lr 2e-4 \
        --lora-rank 8 \
        --margin 0.1

    # The script writes to ablation_Qwen_Qwen2.5-7B-Instruct.json
    # Rename to preserve (the original is already committed in git)
    ABLATION_NEW="$RESULTS_DIR/ablation_Qwen_Qwen2.5-7B-Instruct_s456_789_1337.json"
    if [ -f "$RESULTS_DIR/ablation_Qwen_Qwen2.5-7B-Instruct.json" ]; then
        cp "$RESULTS_DIR/ablation_Qwen_Qwen2.5-7B-Instruct.json" "$ABLATION_NEW"
        echo "  Saved new ablation results to: $(basename $ABLATION_NEW)"
    fi

    echo ""
    echo "  Phase A complete: $(date)"
}


# ── Phase B: Additional Dose-Response Seeds ─────────────────────────

run_phase_b() {
    echo ""
    echo "============================================================"
    echo "  PHASE B: Additional Dose-Response Seeds (789, 1337)"
    echo "  4 rho_weights × 2 seeds = 8 runs"
    echo "  Started: $(date)"
    echo "============================================================"

    # Use the same 4 rho_weights as the existing 3-seed sweep
    $PYTHON experiments/rho_guided_sft_mlx.py \
        --model qwen2.5-7b \
        --rho-weights 0.0,0.1,0.2,0.5 \
        --seeds 789,1337 \
        --sft-size 1000 \
        --epochs 1 \
        --lr 2e-4 \
        --lora-rank 8 \
        --margin 0.1

    echo ""
    echo "  Phase B complete: $(date)"
}


# ── Phase C: No-Margin Control (γ=0) ───────────────────────────────

run_phase_c() {
    echo ""
    echo "============================================================"
    echo "  PHASE C: No-Margin Control (margin=0.0)"
    echo "  rho-guided condition only, 5 seeds"
    echo "  Started: $(date)"
    echo "============================================================"

    # Run rho-guided (rho_weight=0.2) with margin=0.0
    # Compare against same condition with margin=0.1 from ablation
    $PYTHON experiments/rho_guided_sft_mlx.py \
        --model qwen2.5-7b \
        --rho-weights 0.2 \
        --seeds 42,123,456,789,1337 \
        --sft-size 1000 \
        --epochs 1 \
        --lr 2e-4 \
        --lora-rank 8 \
        --margin 0.0 \
        --output "$RESULTS_DIR/mlx_no_margin_Qwen_Qwen2.5-7B-Instruct.json"

    echo ""
    echo "  Phase C complete: $(date)"
}


# ── Run Phases ──────────────────────────────────────────────────────

case "$PHASE" in
    A|a|phase-a)
        run_phase_a
        ;;
    B|b|phase-b)
        run_phase_b
        ;;
    C|c|phase-c)
        run_phase_c
        ;;
    all|ALL)
        run_phase_a
        run_phase_b
        run_phase_c
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: $0 [A|B|C|all]"
        exit 1
        ;;
esac


# ── Summary ─────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  ALL PHASES COMPLETE: $(date)"
echo ""
echo "  Next steps:"
echo "    1. Merge ablation results:"
echo "       $PYTHON experiments/merge_ablation_results.py"
echo "    2. Rerun analysis:"
echo "       $PYTHON experiments/analyze_ablation_stats.py \\"
echo "           results/alignment/ablation_Qwen_Qwen2.5-7B-Instruct_merged.json \\"
echo "           --json-out results/alignment/ablation_analysis_5seed.json"
echo "    3. Merge dose-response results:"
echo "       $PYTHON experiments/analyze_sweep_stats.py \\"
echo "           results/alignment/mlx_rho_sft_sweep_7B_seeds42_123.json \\"
echo "           results/alignment/mlx_rho_sft_sweep_Qwen_Qwen2.5-7B-Instruct.json \\"
echo "           results/alignment/mlx_rho_sft_sweep_Qwen_Qwen2.5-7B-Instruct_s789_1337.json \\"
echo "           --json-out results/alignment/qwen7b_5seed_analysis.json"
echo "============================================================"
