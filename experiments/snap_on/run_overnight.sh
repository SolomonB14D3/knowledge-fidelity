#!/bin/bash
# Snap-On Module overnight queue
# Waits for Phase 1 (PID 78774), then runs Phase 2 + qualitative eval
#
# Phase 1: 10K Alpaca, 3 epochs, MLP adapter (ALREADY RUNNING)
# Phase 2a: Same config, different seed → test consistency
# Phase 2b: Same config, different data split → test data sensitivity
# Phase 4: Cross-scale transfer → test if 7B adapter works on 3B/1.5B
#           (needs dimension projection — runs eval-only with projection script)
# Qualitative: Generate side-by-side base vs adapter samples

set -e
cd "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"

RESULTS="results/snap_on"
LOG="$RESULTS/overnight_log.txt"
mkdir -p "$RESULTS"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

log "=== Overnight queue started ==="

# ── Wait for Phase 1 (PID 78774) ─────────────────────────────────
if ps -p 78774 > /dev/null 2>&1; then
    log "Phase 1 still running (PID 78774), waiting..."
    while ps -p 78774 > /dev/null 2>&1; do
        sleep 60
    done
    log "Phase 1 finished."
else
    log "Phase 1 already completed."
fi

# Check Phase 1 produced output
if [ ! -f "$RESULTS/phase1_mlp/best.npz" ]; then
    log "ERROR: Phase 1 best.npz not found. Aborting."
    exit 1
fi
log "Phase 1 checkpoint found: $RESULTS/phase1_mlp/best.npz"

# ── Phase 2a: Different seed ─────────────────────────────────────
log "=== Phase 2a: Training with seed=123 ==="
python experiments/snap_on/train.py \
    --n_train 10000 \
    --n_val 500 \
    --epochs 3 \
    --lr 1e-4 \
    --warmup_steps 200 \
    --log_every 200 \
    --eval_every 2000 \
    --mmlu_n 200 \
    --save_dir "$RESULTS/phase2a_seed123" \
    2>&1 | tee -a "$LOG"

log "Phase 2a complete."

# ── Phase 2b: Fewer examples (3K) to test data efficiency ────────
log "=== Phase 2b: Training with 3K examples (data efficiency) ==="
python experiments/snap_on/train.py \
    --n_train 3000 \
    --n_val 500 \
    --epochs 3 \
    --lr 1e-4 \
    --warmup_steps 60 \
    --log_every 100 \
    --eval_every 1000 \
    --mmlu_n 200 \
    --save_dir "$RESULTS/phase2b_3k" \
    2>&1 | tee -a "$LOG"

log "Phase 2b complete."

# ── Qualitative: Generate comparison samples ─────────────────────
log "=== Generating qualitative comparison samples ==="
python -c "
import sys
sys.path.insert(0, '.')
import json
from mlx_lm import load as mlx_load
from experiments.snap_on.train import load_adapter, generate_with_adapter, generate_base_only

# Load model + best adapter
model, tokenizer = mlx_load('Qwen/Qwen2.5-7B')
adapter = load_adapter('$RESULTS/phase1_mlp', 'best')

prompts = [
    'Below is an instruction that describes a task.\\n\\n### Instruction:\\nExplain what a neural network is in simple terms.\\n\\n### Response:\\n',
    'Below is an instruction that describes a task.\\n\\n### Instruction:\\nWrite a Python function to check if a number is prime.\\n\\n### Response:\\n',
    'Below is an instruction that describes a task.\\n\\n### Instruction:\\nWhat are the main causes of climate change?\\n\\n### Response:\\n',
    'Below is an instruction that describes a task.\\n\\n### Instruction:\\nSummarize the plot of Romeo and Juliet in 3 sentences.\\n\\n### Response:\\n',
    'Below is an instruction that describes a task.\\n\\n### Instruction:\\nWhat is the difference between a list and a tuple in Python?\\n\\n### Response:\\n',
]

results = []
for i, prompt in enumerate(prompts):
    print(f'  Sample {i+1}/{len(prompts)}...', flush=True)
    base_out = generate_base_only(model, tokenizer, prompt, max_tokens=200)
    adapter_out = generate_with_adapter(model, adapter, tokenizer, prompt, max_tokens=200)
    results.append({
        'prompt': prompt.split('### Instruction:\\n')[1].split('\\n\\n### Response')[0],
        'base': base_out,
        'adapter': adapter_out,
    })

with open('$RESULTS/qualitative_samples.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved qualitative_samples.json')
" 2>&1 | tee -a "$LOG"

log "Qualitative samples complete."

# ── Summary ──────────────────────────────────────────────────────
log "=== Overnight queue complete ==="
log "Results:"
log "  Phase 1:  $RESULTS/phase1_mlp/"
log "  Phase 2a: $RESULTS/phase2a_seed123/"
log "  Phase 2b: $RESULTS/phase2b_3k/"
log "  Samples:  $RESULTS/qualitative_samples.json"

# Print MMLU comparison if all phases have results
python -c "
import json, os
results_dir = '$RESULTS'
phases = [
    ('Phase 1 (10K, seed=42)', 'phase1_mlp'),
    ('Phase 2a (10K, seed=123)', 'phase2a_seed123'),
    ('Phase 2b (3K, seed=42)', 'phase2b_3k'),
]
print()
print('=== MMLU Comparison ===')
print(f'{\"Config\":<30s} {\"Base\":>8s} {\"Adapter\":>8s} {\"Delta\":>8s}')
print('-' * 56)
for label, dirname in phases:
    # Check for final results in the log or saved files
    path = os.path.join(results_dir, dirname)
    if os.path.exists(path):
        print(f'{label:<30s} (see log for numbers)')
    else:
        print(f'{label:<30s} NOT RUN')
print()
print('Full log:', '$LOG')
" 2>&1 | tee -a "$LOG"
