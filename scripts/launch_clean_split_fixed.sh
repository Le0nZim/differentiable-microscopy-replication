#!/usr/bin/env bash
# Launch Fig.6 / Table 1 and Fig.7 / Table 2 clean-split reruns on both GPUs.
set -euo pipefail

REPL="$(cd "$(dirname "$0")/.." && pwd)"
PY="${REPL}/.venv/bin/python"
LOG="${REPL}/experiments/clean_split_fixed_logs"
mkdir -p "${LOG}"

NOISE_CFG="${REPL}/configs/paper_aligned_patchmnist_noise_table_v3_fixed.yaml"
NOISE_OUT="${REPL}/experiments/noise_robustness/rr1_v3_normalized_full_fixed"
SWIN_CFG="${REPL}/configs/swinir/am4_table2_full_fixed.yaml"
SWIN_BASE="${REPL}/experiments/swinir_or_highres/am4_swinir_table2_resolution_fixed"

echo "[$(date -Is)] clean-split launch  logs=${LOG}" | tee "${LOG}/launch.log"

# --- Phase 1: noise table, sharded across both GPUs ---
echo "[$(date -Is)] phase 1: noise table shards" | tee -a "${LOG}/launch.log"
"${PY}" "${REPL}/scripts/run_rr1_v3_noise_table.py" \
  --config "${NOISE_CFG}" --output-root "${NOISE_OUT}" \
  --device cuda:0 --shard 0 --num-shards 2 \
  > "${LOG}/noise_shard0.log" 2>&1 &
PID0=$!
"${PY}" "${REPL}/scripts/run_rr1_v3_noise_table.py" \
  --config "${NOISE_CFG}" --output-root "${NOISE_OUT}" \
  --device cuda:1 --shard 1 --num-shards 2 \
  > "${LOG}/noise_shard1.log" 2>&1 &
PID1=$!
echo "noise PIDs shard0=${PID0} shard1=${PID1}" | tee -a "${LOG}/launch.log"
wait ${PID0}
EC0=$?
wait ${PID1}
EC1=$?
echo "[$(date -Is)] noise shards done exit ${EC0} ${EC1}" | tee -a "${LOG}/launch.log"
if [[ ${EC0} -ne 0 || ${EC1} -ne 0 ]]; then
  echo "noise shards failed" | tee -a "${LOG}/launch.log"
  exit 1
fi

"${PY}" "${REPL}/scripts/run_rr1_v3_noise_table.py" \
  --config "${NOISE_CFG}" --output-root "${NOISE_OUT}" \
  --aggregate-only --device cuda:0 \
  > "${LOG}/noise_aggregate.log" 2>&1
echo "[$(date -Is)] noise aggregate done" | tee -a "${LOG}/launch.log"

# --- Phase 2: SwinIR conditions in parallel ---
echo "[$(date -Is)] phase 2: SwinIR both conditions" | tee -a "${LOG}/launch.log"
"${PY}" "${REPL}/scripts/run_am4_swinir_table2.py" \
  --config "${SWIN_CFG}" --output-base "${SWIN_BASE}" \
  --only-condition swinir_wo_li --device cuda:0 \
  > "${LOG}/swinir_wo_li.log" 2>&1 &
PID0=$!
"${PY}" "${REPL}/scripts/run_am4_swinir_table2.py" \
  --config "${SWIN_CFG}" --output-base "${SWIN_BASE}" \
  --only-condition swinir_with_li --device cuda:1 \
  > "${LOG}/swinir_with_li.log" 2>&1 &
PID1=$!
echo "swinir PIDs wo=${PID0} with=${PID1}" | tee -a "${LOG}/launch.log"
wait ${PID0}
EC0=$?
wait ${PID1}
EC1=$?
echo "[$(date -Is)] swinir conditions done exit ${EC0} ${EC1}" | tee -a "${LOG}/launch.log"
if [[ ${EC0} -ne 0 || ${EC1} -ne 0 ]]; then
  echo "swinir conditions failed" | tee -a "${LOG}/launch.log"
  exit 1
fi

"${PY}" "${REPL}/scripts/run_am4_swinir_table2.py" \
  --config "${SWIN_CFG}" --output-base "${SWIN_BASE}" \
  --aggregate-only \
  > "${LOG}/swinir_aggregate.log" 2>&1
echo "[$(date -Is)] swinir aggregate done" | tee -a "${LOG}/launch.log"

# --- Phase 3: write *_fixed figure and table folders ---
"${PY}" "${REPL}/scripts/finalize_clean_split_fixed.py" \
  > "${LOG}/finalize.log" 2>&1
echo "[$(date -Is)] finalize done" | tee -a "${LOG}/launch.log"
echo COMPLETE | tee -a "${LOG}/launch.log"
