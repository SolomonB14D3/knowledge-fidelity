#!/bin/bash
# Wait for 1.5B snap-on training to complete, then run full pipeline.
# Usage: nohup bash experiments/run_after_training.sh > experiments/pipeline.log 2>&1 &

cd "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"

echo "[$(date)] Waiting for snap-on training (PID 96892) to complete..."

# Wait for training process to finish
while kill -0 96892 2>/dev/null; do
    sleep 60
    LAST=$(tail -1 /private/tmp/claude-501/-Volumes-4TB-SD-ClaudeCode/tasks/b735c76.output 2>/dev/null || echo "unknown")
    echo "[$(date)] Training still running: $LAST"
done

echo "[$(date)] Training process complete. Waiting 10s for file writes..."
sleep 10

# Verify training output exists
if [ -f "results/pipeline_demo_1.5b/adapter/results.json" ]; then
    echo "[$(date)] Training results found. Starting full pipeline..."
else
    echo "[$(date)] WARNING: results.json not found, training may have crashed."
    echo "[$(date)] Starting pipeline anyway (will attempt to retrain)..."
fi

# Run the full pipeline
echo "[$(date)] Launching full_pipeline_runner.py..."
/Users/bryan/miniconda3/bin/python experiments/full_pipeline_runner.py 2>&1

echo "[$(date)] Pipeline complete."
