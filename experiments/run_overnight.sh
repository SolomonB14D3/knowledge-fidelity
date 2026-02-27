#!/bin/bash
# Overnight experiment runner — chains attack/defense + hybrid sweeps
# Usage: caffeinate -dims bash experiments/run_overnight.sh
# Estimated: ~3-5 hours total
set -e
cd "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"
export PYTHONUNBUFFERED=1

LOG="results/overnight_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results

echo "=== Overnight run started: $(date) ===" | tee "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"

# ── Step 1: Attack/defense asymmetry (all 3 phases) ──────────────────────
echo "" | tee -a "$LOG"
echo ">>> Step 1: Attack/defense asymmetry on Qwen2.5-0.5B-Instruct" | tee -a "$LOG"
echo "    Started: $(date)" | tee -a "$LOG"
python experiments/attack_defense_asymmetry.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer 17 --device mps 2>&1 | tee -a "$LOG"
echo "    Finished: $(date)" | tee -a "$LOG"

# ── Step 2: Hybrid control sweep --quick on 0.5B ─────────────────────────
echo "" | tee -a "$LOG"
echo ">>> Step 2: Hybrid control sweep (quick) on Qwen2.5-0.5B-Instruct" | tee -a "$LOG"
echo "    Started: $(date)" | tee -a "$LOG"
python experiments/hybrid_control_sweep.py \
  Qwen/Qwen2.5-0.5B-Instruct --quick 2>&1 | tee -a "$LOG"
echo "    Finished: $(date)" | tee -a "$LOG"

# ── Step 3: Hybrid control sweep --quick on 7B ───────────────────────────
echo "" | tee -a "$LOG"
echo ">>> Step 3: Hybrid control sweep (quick) on Qwen2.5-7B-Instruct" | tee -a "$LOG"
echo "    Started: $(date)" | tee -a "$LOG"
python experiments/hybrid_control_sweep.py \
  Qwen/Qwen2.5-7B-Instruct --quick 2>&1 | tee -a "$LOG"
echo "    Finished: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Overnight run finished: $(date) ===" | tee -a "$LOG"
