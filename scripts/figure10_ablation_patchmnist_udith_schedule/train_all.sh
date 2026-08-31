#!/usr/bin/env bash
# C vs D on PatchMNIST with Udith's 121,500-step schedule. GPU 1 only.
set -u
PY=${PY:-../replication/.venv/bin/python}
EXP=experiments/figure10_ablation_patchmnist_udith_schedule
mkdir -p "$EXP"
STATUS=$EXP/runs_STATUS.txt
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

if nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qE '[0-9]'; then
  log "GPU 1 is busy; not launching. Implementation and tests still stand."
  exit 2
fi

log "=== Udith-schedule C/D start on cuda:1 (seed 42 gates, then 43/44) ==="
$PY scripts/figure10_ablation_patchmnist_udith_schedule/train.py \
    --device cuda:1 --variants C D --seeds 42 43 44 \
    > "$EXP/orchestrator.log" 2>&1
echo "exit=$?" >> "$STATUS"
log "=== driver finished; see $EXP/UDITH_SCHEDULE_REPORT.md ==="
