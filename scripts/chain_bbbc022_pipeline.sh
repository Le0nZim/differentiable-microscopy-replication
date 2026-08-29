#!/bin/bash
# Chain BBBC022 substitute experiments after ablation completes.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=experiments/ablations/pipeline_chain.log

echo "$(date) waiting for ablation..." | tee -a "$LOG"
while pgrep -f "run_bbbc022_substitute.py --phase ablation" >/dev/null; do sleep 120; done
echo "$(date) ablation done; content_aware..." | tee -a "$LOG"
$PY scripts/run_bbbc022_substitute.py --phase content_aware --device cuda:1 --seeds 42 2>&1 | tee -a "$LOG"
echo "$(date) segmentation..." | tee -a "$LOG"
$PY scripts/run_bbbc022_segmentation.py --device cuda:1 --seed 42 2>&1 | tee -a "$LOG"
$PY scripts/run_bbbc022_substitute.py --phase swinir 2>&1 | tee -a "$LOG"
echo "$(date) pipeline complete" | tee -a "$LOG"
