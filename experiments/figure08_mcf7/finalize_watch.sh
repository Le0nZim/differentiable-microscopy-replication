#!/usr/bin/env bash
# Wait for all three MCF7 Fig 8/9 conditions to finish, then render figures + fill REPORT.md.
# Usage: setsid nohup bash experiments/figure08_mcf7/finalize_watch.sh > .../finalize.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."          # -> replication/
PY=python
EXP=experiments/figure08_mcf7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[finalize] waiting for result.json from wswinir, transpose256, wcnn64 ..."
while true; do
  done=0
  for c in wswinir transpose256 wcnn64; do
    [ -f "$EXP/runs/$c/result.json" ] && done=$((done+1))
  done
  echo "[finalize] $(date -u +%H:%M:%S) ready=$done/3"
  [ "$done" -ge 3 ] && break
  sleep 120
done

echo "[finalize] all three done -> rendering figures + report"
$PY scripts/figure08_mcf7/report.py --device cuda:1
echo "[finalize] DONE"
