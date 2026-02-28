#!/bin/bash
# Margin sweep: test critical γ* ≈ 0.024 prediction
# Tests 5 margin values around the predicted threshold
# Usage: caffeinate -dims bash experiments/margin_sweep.sh
#
# Theory predicts γ* ≈ 0.024 for bias preservation at λ_ρ=0.2
# Below γ*: bias inverts (Δρ_bias < 0)
# Above γ*: bias preserved (Δρ_bias > 0)
#
# Expected results:
#   γ=0.00  → Δρ_bias ≈ -0.011 (known, already measured)
#   γ=0.02  → Δρ_bias < 0 (below threshold)
#   γ=0.03  → Δρ_bias ≈ 0 (near threshold)
#   γ=0.05  → Δρ_bias > 0 (above threshold)
#   γ=0.10  → Δρ_bias ≈ +0.034 (known, already measured)
#
# Estimated time: ~1.5-2 hours on M3 Ultra (3 margins × 5 seeds × ~10 min/run)
set -e
cd "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"
export PYTHONUNBUFFERED=1

LOG="results/alignment/margin_sweep_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results/alignment

echo "=== Margin sweep started: $(date) ===" | tee "$LOG"
echo "  Testing γ ∈ {0.02, 0.03, 0.05} at λ_ρ=0.2, 5 seeds" | tee -a "$LOG"
echo "  Theory predicts γ* ≈ 0.024" | tee -a "$LOG"

for GAMMA in 0.02 0.03 0.05; do
    echo "" | tee -a "$LOG"
    echo ">>> Margin γ=${GAMMA} — started $(date)" | tee -a "$LOG"
    python experiments/rho_guided_sft_mlx.py \
        qwen2.5-7b \
        --rho-weights 0.2 \
        --seeds 42,123,456,789,1337 \
        --margin ${GAMMA} \
        --output "results/alignment/margin_sweep_gamma${GAMMA}.json" \
        2>&1 | tee -a "$LOG"
    echo ">>> Margin γ=${GAMMA} — finished $(date)" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== Margin sweep finished: $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Next: run 'python experiments/analyze_margin_sweep.py' to test predictions" | tee -a "$LOG"
