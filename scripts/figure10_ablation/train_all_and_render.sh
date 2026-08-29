#!/usr/bin/env bash
# Train Fig-10 ablation A/B/C/D on the Fig-3 BBBC022 substitute data, then render
# the paper-style figure. GPU0 runs A then B; GPU1 runs C then D (in parallel).
set -u

PY=./.venv/bin/python
EXP=experiments/figure10_ablation
RUNS=$EXP/runs
mkdir -p "$RUNS"
STATUS=$EXP/runs_STATUS.txt
: > "$STATUS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

log "=== Fig10 ablation start: A/B/C/D on Fig-3 substitute (1980/40/60), x16 T=4, uniform Alg1 8500 steps ==="

# GPU0 chain: A then B
(
  $PY scripts/figure10_ablation/train.py --variants A --device cuda:0 --out "$RUNS" > "$EXP/train_A.log" 2>&1
  echo "A_exit=$?" >> "$STATUS"
  $PY scripts/figure10_ablation/train.py --variants B --device cuda:0 --out "$RUNS" > "$EXP/train_B.log" 2>&1
  echo "B_exit=$?" >> "$STATUS"
) &
P0=$!
log "GPU0 chain PID=$P0 -> A then B  (logs: train_A.log, train_B.log)"

# GPU1 chain: C then D
(
  $PY scripts/figure10_ablation/train.py --variants C --device cuda:1 --out "$RUNS" > "$EXP/train_C.log" 2>&1
  echo "C_exit=$?" >> "$STATUS"
  $PY scripts/figure10_ablation/train.py --variants D --device cuda:1 --out "$RUNS" > "$EXP/train_D.log" 2>&1
  echo "D_exit=$?" >> "$STATUS"
) &
P1=$!
log "GPU1 chain PID=$P1 -> C then D  (logs: train_C.log, train_D.log)"

wait $P0; R0=$?
wait $P1; R1=$?
log "training finished: gpu0-chain exit=$R0  gpu1-chain exit=$R1"

if [ "$R0" -ne 0 ] || [ "$R1" -ne 0 ]; then
  log "!! a training chain exited non-zero; SKIPPING render. Inspect train_*.log."
  exit 1
fi

log "rebuilding aggregate + rendering paper-style Figure 10 ..."
$PY scripts/figure10_ablation/train.py --aggregate-only --out "$RUNS" >> "$STATUS" 2>&1
$PY scripts/figure10_ablation/reproduce.py --runs "$RUNS" >> "$STATUS" 2>&1
log "=== Fig10 DONE: figures in $EXP/figures/ and results/reproduced_figures/fig10/ ==="
