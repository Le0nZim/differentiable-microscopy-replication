#!/usr/bin/env bash
# Launch the no-frequency-domain-optimization noise sweep on both GPUs, then aggregate
# against the frozen Table-1 baselines.
#
#   PY=path/to/python bash scripts/figure06_noise_robustness_no_freq/launch.sh [corners|full]
set -u

PY=${PY:-./.venv/bin/python}
CELLS=${1:-corners}
EXP=experiments/figure06_noise_robustness_no_freq
mkdir -p "$EXP"
STATUS=$EXP/runs_STATUS.txt
: > "$STATUS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

log "=== Fig6/Table1 NO-FREQ sweep start: learnable_spatial, cells=$CELLS ==="

$PY scripts/figure06_noise_robustness_no_freq/run.py \
    --cells "$CELLS" --device cuda:0 --shard 0 --num-shards 2 > "$EXP/shard0.log" 2>&1 &
P0=$!
log "GPU0 shard0 PID=$P0 (log: shard0.log)"

$PY scripts/figure06_noise_robustness_no_freq/run.py \
    --cells "$CELLS" --device cuda:1 --shard 1 --num-shards 2 > "$EXP/shard1.log" 2>&1 &
P1=$!
log "GPU1 shard1 PID=$P1 (log: shard1.log)"

wait $P0; R0=$?
wait $P1; R1=$?
log "training finished: shard0 exit=$R0  shard1 exit=$R1"

if [ "$R0" -ne 0 ] || [ "$R1" -ne 0 ]; then
  log "!! a shard exited non-zero; SKIPPING aggregate. Inspect shard*.log."
  exit 1
fi

log "aggregating against frozen Table-1 baselines ..."
$PY scripts/figure06_noise_robustness_no_freq/run.py \
    --cells "$CELLS" --aggregate-only --device cuda:0 >> "$STATUS" 2>&1
log "=== NO-FREQ sweep DONE: $EXP/results.md, results.json, plots/ ==="
