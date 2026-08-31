#!/usr/bin/env bash
# Split remaining Udith-schedule seeds across GPUs without interrupting seed 42:
#   GPU 0: seed 43 (started immediately)
#   GPU 1: seed 42 (already running) then seed 44 (started after 42 finishes)
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-../replication/.venv/bin/python}
EXP=experiments/figure10_ablation_patchmnist_udith_schedule
RUNS=$EXP/runs
STATUS=$EXP/gpu_split_STATUS.txt
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

ORIG_PID=${1:-}
if [ -z "$ORIG_PID" ]; then
  ORIG_PID=$(pgrep -f "train.py --device cuda:1 --variants C D --seeds 42 43 44" | head -n1 || true)
fi
if [ -z "$ORIG_PID" ]; then
  log "ERROR: could not find the original cuda:1 seed 42/43/44 process"
  exit 1
fi
log "original GPU1 driver PID=$ORIG_PID (leave it running through seed 42)"

# --- seed 43 on GPU 0 now ---
if nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qE '[0-9]'; then
  log "ERROR: GPU 0 is busy; not launching seed 43"
  exit 1
fi
log "launching seed 43 C then D on cuda:0"
nohup $PY scripts/figure10_ablation_patchmnist_udith_schedule/train.py \
    --device cuda:0 --allow-gpu0 --skip-gpu-check \
    --variants C D --seeds 43 \
    > "$EXP/train_seed43_gpu0.log" 2>&1 &
PID43=$!
log "seed43 GPU0 PID=$PID43"

# --- after seed 42 completes, stop the original driver before it starts 43, then run 44 on GPU 1 ---
GATES=$RUNS/seed42_gates.json
D42=$RUNS/D_seed42/metrics/run_summary.json
log "watching for seed-42 completion (D summary + gates) before starting seed 44 on GPU 1"
while true; do
  if [ -f "$D42" ] && [ -f "$GATES" ]; then
    log "seed 42 artifacts present"
    break
  fi
  if ! kill -0 "$ORIG_PID" 2>/dev/null; then
    log "original driver exited before gates appeared; checking artifacts"
    break
  fi
  sleep 5
done

if [ -f "$GATES" ]; then
  PASS=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('pass', False))" "$GATES")
  log "seed42 gates pass=$PASS"
  if [ "$PASS" != "True" ]; then
    log "seed 42 gates failed; not launching seed 44"
    exit 1
  fi
else
  log "ERROR: seed42_gates.json missing; not launching seed 44"
  exit 1
fi

# Prevent the original driver from starting seed 43 on GPU 1.
if kill -0 "$ORIG_PID" 2>/dev/null; then
  log "stopping original driver PID=$ORIG_PID so it cannot start seed 43 on GPU 1"
  kill -TERM "$ORIG_PID" 2>/dev/null || true
  for i in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$ORIG_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$ORIG_PID" 2>/dev/null; then
    log "original still alive; sending KILL"
    kill -KILL "$ORIG_PID" 2>/dev/null || true
  fi
else
  log "original driver already exited"
fi

# Wait until GPU 1 is free of the old process (seed 43 is on GPU 0 and must not be killed).
log "waiting for GPU 1 to be free of PID $ORIG_PID"
for i in $(seq 1 60); do
  if ! nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qE '[0-9]'; then
    break
  fi
  # If the only GPU1 process is gone, continue; if some other job appeared, still wait a bit.
  if nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q "^${ORIG_PID}$"; then
    sleep 2
    continue
  fi
  # GPU 1 has some pid but not ORIG_PID — if seed 42 python is gone, proceed.
  if ! kill -0 "$ORIG_PID" 2>/dev/null; then
    sleep 2
    break
  fi
  sleep 2
done

log "launching seed 44 C then D on cuda:1"
nohup $PY scripts/figure10_ablation_patchmnist_udith_schedule/train.py \
    --device cuda:1 --skip-gpu-check \
    --variants C D --seeds 44 \
    > "$EXP/train_seed44_gpu1.log" 2>&1 &
PID44=$!
log "seed44 GPU1 PID=$PID44"
log "split complete: GPU0=seed43 PID=$PID43  GPU1=seed44 PID=$PID44 (seed 42 finished)"
wait $PID43
log "seed43 GPU0 exit=$?"
wait $PID44
log "seed44 GPU1 exit=$?"
$PY scripts/figure10_ablation_patchmnist_udith_schedule/train.py --aggregate-only --device cuda:1 --skip-gpu-check >> "$STATUS" 2>&1 || true
log "=== GPU split done ==="
