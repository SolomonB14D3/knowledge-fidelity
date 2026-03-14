#!/usr/bin/env bash
# Wait for scale ladder (run_all.sh, PID 53138) to finish,
# then launch the Grassmann trajectory experiment.
#
# Usage:
#   nohup bash experiments/queue_grassmann_after_ladder.sh > /tmp/grassmann_queue.log 2>&1 &

set -e
cd "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"

LADDER_PID=53138
POLL_SEC=120  # check every 2 minutes

echo "============================================================"
echo "  GRASSMANN QUEUE — $(date)"
echo "  Waiting for scale ladder PID $LADDER_PID to finish..."
echo "============================================================"

# Wait for run_all.sh to exit
while kill -0 $LADDER_PID 2>/dev/null; do
    echo "  $(date +%H:%M) — scale ladder still running (PID $LADDER_PID)"
    sleep $POLL_SEC
done

echo ""
echo "  $(date) — Scale ladder finished!"
echo ""

# Brief cooldown to let GPU memory settle
sleep 10

echo "============================================================"
echo "  LAUNCHING GRASSMANN TRAJECTORY EXPERIMENT — $(date)"
echo "============================================================"
echo ""

python experiments/grassmann_trajectory_runner.py \
    Qwen/Qwen2.5-7B-Instruct \
    -o results/grassmann_trajectory

echo ""
echo "============================================================"
echo "  GRASSMANN TRAJECTORY COMPLETE — $(date)"
echo "============================================================"
