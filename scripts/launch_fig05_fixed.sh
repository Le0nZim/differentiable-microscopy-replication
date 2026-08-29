#!/usr/bin/env bash
# Fig. 5 clean-split rerun on the paper grid, sharded across both GPUs.
# Shards are balanced by cost: the 512 runs dominate (~33 min each).
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=experiments/upsampling_ablation/patchmnist_upsampling_analysis_fixed
LOGS=logs/fig05_fixed
mkdir -p "$LOGS" "$OUT"

# GPU 0: the two heaviest 512 columns.
$PY scripts/run_fig05_upsampling_fixed.py --device cuda:0 \
    --sizes 512 --counts 600,6000 --output-root "$OUT" \
    > "$LOGS/gpu0.log" 2>&1 &
PID0=$!

# GPU 1: all small sizes, then the remaining 512 column.
(
  $PY scripts/run_fig05_upsampling_fixed.py --device cuda:1 \
      --sizes 64,128,256 --counts 600,3000,6000 --output-root "$OUT"
  $PY scripts/run_fig05_upsampling_fixed.py --device cuda:1 \
      --sizes 512 --counts 3000 --output-root "$OUT"
) > "$LOGS/gpu1.log" 2>&1 &
PID1=$!

wait $PID0 $PID1

$PY scripts/run_fig05_upsampling_fixed.py --collect-only --output-root "$OUT"
echo "FIG05_FIXED_GRID_DONE"
