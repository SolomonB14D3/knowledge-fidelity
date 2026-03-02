#!/usr/bin/env bash
# Run pending experiments: γ=0.03 → entropy map → 7M scale ladder
# Launched: 2026-03-02
set -e
cd "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"

echo "============================================================"
echo "  PENDING RUNS — $(date)"
echo "  Jobs: (1) γ=0.03  (2) 7B entropy map  (3) 7M training"
echo "============================================================"

# Job 1: γ=0.03 re-run (~2h)
echo ""
echo ">>> JOB 1: γ=0.03 critical margin sweep"
echo ">>> Started: $(date)"
/Users/bryan/miniconda3/bin/python experiments/overnight_sweep.py \
    Qwen/Qwen2.5-7B-Instruct \
    --gammas 0.03 --gamma-only \
    -o results/overnight_sweep
echo ">>> JOB 1 COMPLETE: $(date)"

# Job 2: 7B entropy map (~1-2h)
echo ""
echo ">>> JOB 2: 7B entropy map"
echo ">>> Started: $(date)"
/Users/bryan/miniconda3/bin/python scripts/invariant_gate.py \
    Qwen/Qwen2.5-7B-Instruct \
    --entropy-map \
    --output-dir results/7b_entropy_map
echo ">>> JOB 2 COMPLETE: $(date)"

# Job 3: 7M scale ladder on MPS (~1.5h)
echo ""
echo ">>> JOB 3: 7M scale ladder training (MPS)"
echo ">>> Started: $(date)"
/Users/bryan/miniconda3/bin/python experiments/scale_ladder/train_model.py \
    --size 7M --seed 42 --device mps \
    -o results/scale_ladder/7M_seed42
echo ">>> JOB 3 COMPLETE: $(date)"

# Job 4: Audit 7M model
echo ""
echo ">>> JOB 4: 7M scale ladder audit"
echo ">>> Started: $(date)"
/Users/bryan/miniconda3/bin/python experiments/scale_ladder/scale_audit.py \
    --checkpoint results/scale_ladder/7M_seed42 \
    --device cpu
echo ">>> JOB 4 COMPLETE: $(date)"

echo ""
echo "============================================================"
echo "  ALL PENDING RUNS COMPLETE — $(date)"
echo "============================================================"
