#!/usr/bin/env bash
# Wait for 64M training to finish → audit → analyze → launch Grassmann trajectory.
#
# Usage:
#   nohup bash experiments/queue_64m_then_grassmann.sh > /tmp/queue_64m_grassmann.log 2>&1 &

set -e
cd "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity"

TRAIN_PID=59168
POLL_SEC=60
DEVICE=mps

echo "============================================================"
echo "  QUEUE: 64M → Audit → Grassmann — $(date)"
echo "  Waiting for 64M training PID $TRAIN_PID..."
echo "============================================================"

# ── Phase 1: Wait for 64M training ──
while kill -0 $TRAIN_PID 2>/dev/null; do
    echo "  $(date +%H:%M) — 64M training still running"
    sleep $POLL_SEC
done
echo ""
echo "  $(date) — 64M training finished!"
sleep 5

# ── Phase 2: Audit 64M final model ──
OUTDIR="results/scale_ladder/64M_seed42"
if [ ! -f "$OUTDIR/audit_report.json" ]; then
    echo ""
    echo "============================================================"
    echo "  AUDITING 64M final model — $(date)"
    echo "============================================================"
    python experiments/scale_ladder/scale_audit.py \
        --checkpoint "$OUTDIR" --device "$DEVICE"
else
    echo "  [SKIP] 64M already audited"
fi

# ── Phase 3: Audit 64M training checkpoints ──
echo ""
echo "============================================================"
echo "  AUDITING 64M checkpoints — $(date)"
echo "============================================================"
for CKPT in "$OUTDIR"/checkpoint_*/; do
    [ -d "$CKPT" ] || continue
    if [ -f "$CKPT/audit_report.json" ]; then
        echo "  [SKIP] $(basename $CKPT) already audited"
    else
        echo "  [AUDIT] $(basename $CKPT)..."
        python experiments/scale_ladder/scale_audit.py \
            --checkpoint "$CKPT" --device "$DEVICE" --skip-dprime
    fi
done

# ── Phase 4: Analyze scaling curves (7M→64M) ──
echo ""
echo "============================================================"
echo "  ANALYZING SCALING CURVES — $(date)"
echo "============================================================"
python experiments/scale_ladder/analyze_scaling.py

# ── Phase 5: Grassmann trajectory experiment (7B, 3 conditions) ──
echo ""
echo "============================================================"
echo "  LAUNCHING GRASSMANN TRAJECTORY — $(date)"
echo "============================================================"
python experiments/grassmann_trajectory_runner.py \
    Qwen/Qwen2.5-7B-Instruct \
    -o results/grassmann_trajectory

echo ""
echo "============================================================"
echo "  ALL COMPLETE — $(date)"
echo "============================================================"
