#!/bin/bash
# Overnight GPU tasks — March 5 night
# Task 1: Toxicity-only injection at 7M (~30 min)
# Task 2: Factual-only injection at 7M (~30 min)
# Task 3: SVD spectrum extraction on all checkpoints (~2-4h)
set -euo pipefail
cd "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"

DEVICE="mps"
LOG="experiments/overnight_mar5b.log"

echo "=== Overnight GPU Tasks ($(date)) ===" | tee "$LOG"

# ── Task 1: Toxicity-only injection at 7M ────────────────────────────
echo "" | tee -a "$LOG"
echo ">>> [1/3] Toxicity-only injection (7M, rate=20)..." | tee -a "$LOG"
echo ">>> Start: $(date)" | tee -a "$LOG"

python experiments/scale_ladder/train_contrastive.py \
    --size 7M --seed 42 --device "$DEVICE" \
    --inject-rate 20 --inject-behaviors toxicity \
    --probe-seed 999 --audit-interval 500 2>&1 | tee -a "$LOG"

echo ">>> Auditing 7M toxicity-only..." | tee -a "$LOG"
python experiments/scale_ladder/scale_audit.py \
    --checkpoint results/scale_ladder/7M_seed42_contr_tox_r20 --device "$DEVICE" 2>&1 | tee -a "$LOG"

echo ">>> [1/3] Done: $(date)" | tee -a "$LOG"

# ── Task 2: Factual-only injection at 7M ─────────────────────────────
echo "" | tee -a "$LOG"
echo ">>> [2/3] Factual-only injection (7M, rate=20)..." | tee -a "$LOG"
echo ">>> Start: $(date)" | tee -a "$LOG"

python experiments/scale_ladder/train_contrastive.py \
    --size 7M --seed 42 --device "$DEVICE" \
    --inject-rate 20 --inject-behaviors factual \
    --probe-seed 999 --audit-interval 500 2>&1 | tee -a "$LOG"

echo ">>> Auditing 7M factual-only..." | tee -a "$LOG"
python experiments/scale_ladder/scale_audit.py \
    --checkpoint results/scale_ladder/7M_seed42_contr_fac_r20 --device "$DEVICE" 2>&1 | tee -a "$LOG"

echo ">>> [2/3] Done: $(date)" | tee -a "$LOG"

# ── Task 3: SVD spectrum extraction ──────────────────────────────────
echo "" | tee -a "$LOG"
echo ">>> [3/3] SVD spectrum extraction (all checkpoints)..." | tee -a "$LOG"
echo ">>> Start: $(date)" | tee -a "$LOG"

python experiments/developmental_sweep/extract_svd_spectrum.py \
    --sweep results/scale_ladder \
    --pattern "*_seed42*" \
    --device "$DEVICE" \
    --skip-existing 2>&1 | tee -a "$LOG"

echo ">>> [3/3] Done: $(date)" | tee -a "$LOG"

# ── Rebuild master DB ────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo ">>> Rebuilding master.db..." | tee -a "$LOG"
python scripts/build_master_db.py 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== All overnight tasks complete: $(date) ===" | tee -a "$LOG"
