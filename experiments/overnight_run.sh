#!/bin/bash
cd "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"

echo "=========================================="
echo "OVERNIGHT RUN — $(date)"
echo "=========================================="

# 1. Cross-behavioral sweep: Mistral (Qwen-7B already done)
echo ""
echo "[1/4] Cross-behavioral sweep: Mistral-7B"
echo "=========================================="
/Users/bryan/miniconda3/bin/python experiments/cross_behavioral_sweep.py --models mistral-7b 2>&1 || echo "Mistral sweep failed, continuing..."

# 2. Cross-behavioral sweep: Llama-3.1-8B
echo ""
echo "[2/4] Cross-behavioral sweep: Llama-3.1-8B"
echo "=========================================="
/Users/bryan/miniconda3/bin/python experiments/cross_behavioral_sweep.py --models llama3.1-8b 2>&1 || echo "Llama sweep failed, continuing..."

# 3. Freeze-ratio sweep: Qwen-7B
echo ""
echo "[3/4] Freeze-ratio sweep: Qwen-7B"
echo "=========================================="
/Users/bryan/miniconda3/bin/python experiments/freeze_ratio_sweep.py --models qwen2.5-7b 2>&1 || echo "Freeze sweep failed, continuing..."

# 4. BO prototype: refit with all new data + generate plots
echo ""
echo "[4/4] BO prototype: refit with new data"
echo "=========================================="
/Users/bryan/miniconda3/bin/python experiments/bo_prototype.py --target bias --plot 2>&1 || echo "BO prototype failed"

echo ""
echo "=========================================="
echo "OVERNIGHT RUN COMPLETE — $(date)"
echo "=========================================="
