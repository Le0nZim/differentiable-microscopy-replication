#!/usr/bin/env bash
# Auto-finalize the Fig-3 SwinIR l1_ssim sweep: wait for both training PIDs to
# exit, then build metrics_summary.csv + diagnostic/random-field panels + full
# Fig-3 panel and refresh REPORT.md's LIVE RESULTS block.
#
# Usage: nohup bash finalize_watch.sh <pid_cuda0> <pid_cuda1> > finalize_watch.log 2>&1 &
set -u
PID0="${1:?need cuda0 train pid}"
PID1="${2:?need cuda1 train pid}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

echo "[finalize] waiting for train PIDs $PID0 (cuda0) and $PID1 (cuda1) ..."
while kill -0 "$PID0" 2>/dev/null || kill -0 "$PID1" 2>/dev/null; do
    sleep 120
done
echo "[finalize] both training processes exited at $(date -Is). Generating report ..."

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONWARNINGS=ignore
python scripts/figure03_content_aware/report.py \
    --name paper_faithful_l1_ssim --device cuda:0 \
    --fixed-index 7 --random-comp x256 --random-indices 3,11,19,27,41 \
    --full-panel
echo "[finalize] DONE at $(date -Is)"
