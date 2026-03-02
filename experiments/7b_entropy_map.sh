#!/usr/bin/env bash
# 7B Layer 21 Entropy Map — Follow-up after overnight_mar1.sh
#
# Generates the comprehensive entropy map for Qwen2.5-7B-Instruct,
# scanning all 28 layers to identify optimal gate coordinates for the
# invariant gate (inference-time behavioral verification).
#
# Phase 1 target (from Arjtriv-inspired design):
#   - Layer 21: expected peak entropy delta (from 0.5B trajectory analysis)
#   - 1.0 bit threshold mark
#   - <5% fire rate while catching sycophantic deviations
#
# Estimated time: ~20-40 min (depends on n_probes)
#
# Usage:
#   # Run after overnight jobs complete:
#   bash experiments/7b_entropy_map.sh
#
#   # Or queue after overnight_mar1.sh:
#   bash experiments/overnight_mar1.sh && bash experiments/7b_entropy_map.sh

set -e
cd "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"

MODEL="Qwen/Qwen2.5-7B-Instruct"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "============================================================"
echo "  7B ENTROPY MAP — $TIMESTAMP"
echo "  Model: $MODEL"
echo "  Scan: all 28 layers, 50 probes"
echo "============================================================"
echo ""

# Full entropy map — scan every layer with 50 probes
python scripts/invariant_gate.py "$MODEL" \
    --entropy-map \
    --layer-step 1 \
    --n-cal-probes 50 \
    -o results/invariant_gate/Qwen_Qwen2.5-7B-Instruct

echo ""
echo "============================================================"
echo "  ENTROPY MAP COMPLETE — $(date)"
echo "  Results: results/invariant_gate/Qwen_Qwen2.5-7B-Instruct/"
echo "============================================================"
