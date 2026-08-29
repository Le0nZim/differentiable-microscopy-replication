#!/usr/bin/env bash
# Waits for all 6 task-aware cells to finish, then generates the metrics CSV,
# paper-layout Figure 4, bar chart, and fills REPORT.md. Detached + idempotent.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=python
RUNS=experiments/figure4_bbbc022_segmentation_fix_v1/runs
LOG=experiments/figure4_bbbc022_segmentation_fix_v1/logs/finalize.log
CELLS=(
  taskaware_x64_random_fixed_seed42
  taskaware_x64_learnable_frequency_seed42
  taskaware_x256_random_fixed_seed42
  taskaware_x256_learnable_frequency_seed42
  taskaware_x1024_random_fixed_seed42
  taskaware_x1024_learnable_frequency_seed42
)
echo "[finalize] watching for 6 cells at $(date)" > "$LOG"
for _ in $(seq 1 720); do   # up to ~2h (720 * 10s)
  done_count=0
  for c in "${CELLS[@]}"; do
    [ -f "$RUNS/$c/metrics/run_summary.json" ] && done_count=$((done_count+1))
  done
  echo "[finalize] $(date +%H:%M:%S) cells done: $done_count/6" >> "$LOG"
  if [ "$done_count" -eq 6 ]; then
    echo "[finalize] all cells done; generating report + figures" >> "$LOG"
    CUDA_VISIBLE_DEVICES=0 $PY scripts/fig4_seg_fix_report.py --device cuda:0 --seed 42 >> "$LOG" 2>&1
    echo "[finalize] DONE at $(date)" >> "$LOG"
    exit 0
  fi
  sleep 10
done
echo "[finalize] TIMEOUT waiting for cells" >> "$LOG"
exit 1
