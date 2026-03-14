#!/bin/bash
# Monitor training and run comprehensive eval when done.
# Usage: bash experiments/operation_destroyer/monitor_and_eval.sh [version]

set -e
cd "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"

VERSION=${1:-v13}
echo "Monitoring ${VERSION} training process..."

# Wait for training to finish
while ps aux | grep "train_v2_run.*${VERSION}" | grep -v grep > /dev/null 2>&1; do
    STEP=$(cat results/operation_destroyer/${VERSION}/checkpoint_meta.json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['global_step'])" 2>/dev/null || echo "?")
    LOSS=$(cat results/operation_destroyer/${VERSION}/checkpoint_meta.json 2>/dev/null | python3 -c "import sys,json; print(f\"{json.load(sys.stdin)['best_val_loss']:.4f}\")" 2>/dev/null || echo "?")
    echo "[$(date '+%H:%M')] Training running... step=$STEP best_val_loss=$LOSS"
    sleep 300
done

echo ""
echo "Training completed at $(date)"
echo ""

# Run comprehensive evaluation
echo "Starting comprehensive evaluation..."
caffeinate -i python experiments/operation_destroyer/eval_comprehensive.py \
    --checkpoint results/operation_destroyer/${VERSION}/best.npz \
    --save_dir results/operation_destroyer/${VERSION} \
    --softcap 30.0 \
    --mmlu_n 500 \
    --truthfulqa_n 200 \
    --arc_n 200 \
    --safety_n 50 \
    2>&1 | tee results/operation_destroyer/${VERSION}/eval_output.log

echo ""
echo "All evaluations complete at $(date)"
echo "Results: results/operation_destroyer/${VERSION}/eval_comprehensive_v11.json"
