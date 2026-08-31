#!/usr/bin/env bash
# After independent-warmup C42/D42, do NOT start seeds 43/44.
# Frozen C vs D warmups diverged materially; the scientific protocol is
# shared-warmup branching (see train.py --protocol shared_warmup).
set -u
cd "$(dirname "$0")/../.."
EXP=experiments/figure10_ablation_patchmnist_udith_schedule
D42=$EXP/runs/D_seed42/metrics/run_summary.json
STATUS=$EXP/prevent_seed44_STATUS.txt
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

ORIG_PID=${1:-462543}
log "watching D42 summary; will SIGTERM PID=$ORIG_PID and any seed 43/44 launchers"

while true; do
  if [ -f "$D42" ]; then
    log "D42 summary present"
    break
  fi
  if ! kill -0 "$ORIG_PID" 2>/dev/null; then
    log "original driver PID=$ORIG_PID already exited"
    break
  fi
  sleep 1
done

for pid in $ORIG_PID $(pgrep -f "train.py --device cuda:1 --variants C D --seeds 42 43 44" || true); do
  if kill -0 "$pid" 2>/dev/null; then
    log "SIGTERM $pid (block 43/44)"
    kill -TERM "$pid" 2>/dev/null || true
  fi
done
sleep 2
for pid in $(pgrep -f "figure10_ablation_patchmnist_udith_schedule/train.py --seeds 4[34]" || true); do
  log "SIGTERM stray 43/44 PID $pid"
  kill -TERM "$pid" 2>/dev/null || true
done
# Do not launch seed 44. Shared-warmup C/D is the scientific continuation.
log "seeds 43/44 will not be started. independent D42 kept as a divergent-warmup diagnostic."
log "=== prevent_seed44 done ==="
