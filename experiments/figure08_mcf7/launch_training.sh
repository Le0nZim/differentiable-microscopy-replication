#!/usr/bin/env bash
# Launch the MCF7 Figure 8/9 SwinIR fix training, fully detached (survives shell/session close).
#   GPU 0: wswinir  (heavy: SwinIR-M + full pixel+perceptual+GAN loss, ~7h)
#   GPU 1: transpose256 then wcnn64 (cheap conventional baselines, ~1.5h total)
# Ratio-preserving compute-scaled Algorithm-1 schedule (baseline/epochs~0.65, step/epochs~0.09).
#
# Usage:  bash experiments/figure08_mcf7/launch_training.sh
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
PY=python
EXP=experiments/figure08_mcf7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$EXP/runs/wswinir" "$EXP/runs/transpose256" "$EXP/runs/wcnn64"

# --- GPU 0: SwinIR (Q / wSwinIR) — the core fix ---
setsid nohup $PY scripts/figure08_mcf7/train.py \
    --condition wswinir --device cuda:0 \
    --epochs 54 --epoch-baseline 35 --epoch-step 5 --max-steps-per-epoch 120 \
    --val-subset 40 --n-examples 8 \
    > "$EXP/runs/wswinir/train.log" 2>&1 &
echo "launched wswinir (GPU0) PID $!"

# --- GPU 1: conventional baselines, sequential ---
setsid nohup bash -c "
  $PY scripts/figure08_mcf7/train.py --condition transpose256 --device cuda:1 \
      --epochs 54 --epoch-baseline 35 --epoch-step 5 --n-examples 8 \
      > $EXP/runs/transpose256/train.log 2>&1
  $PY scripts/figure08_mcf7/train.py --condition wcnn64 --device cuda:1 \
      --epochs 72 --epoch-baseline 46 --epoch-step 6 --n-examples 8 \
      > $EXP/runs/wcnn64/train.log 2>&1
" &
echo "launched transpose256->wcnn64 (GPU1) PID $!"
echo "logs: $EXP/runs/<condition>/train.log"
