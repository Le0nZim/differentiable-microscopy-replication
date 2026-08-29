#!/usr/bin/env bash
# Extend the paper-faithful (pixel+perceptual+GAN) SwinIR sweep from {x16,x256} to
# ALL FOUR compressions so it matches the paper's Table-S1 protocol (SwinIR loss is
# used for every SwinIR experiment; Table S1 reports all of x16/x64/x256/x1024).
#
# The two currently-running jobs (x16 on cuda:0, x256 on cuda:1) are left untouched.
# Each GPU only has ~25GB free while a cell trains, so we CHAIN (not co-locate):
#   cuda:0:  wait for x16 job  -> train x64   (random_fixed, learnable_frequency)
#   cuda:1:  wait for x256 job -> train x1024 (random_fixed, learnable_frequency)
# When all four compressions are done, build the full 8-cell GAN CSV, the full
# Fig-3 panel (GAN columns), the diagnostic/random panels, and the l1_ssim-vs-GAN
# comparison table.
#
# Usage: nohup bash extend_gan_and_finalize.sh <x16_pid> <x256_pid> > extend_gan_and_finalize.log 2>&1 &
set -u
X16_PID="${1:?need running x16 GAN pid (cuda0)}"
X256_PID="${2:?need running x256 GAN pid (cuda1)}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CFG="configs/figure03_content_aware/paper_faithful_pixel_perceptual_gan.yaml"
EXP="experiments/figure03_content_aware/swinir"
cd "$REPO" || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONWARNINGS=ignore

echo "[extend-gan] $(date -Is) waiting to chain x64 (after x16=$X16_PID) and x1024 (after x256=$X256_PID)"

# cuda:0 chain: x16 -> x64
(
  while kill -0 "$X16_PID" 2>/dev/null; do sleep 120; done
  echo "[extend-gan] $(date -Is) x16 done -> starting x64 on cuda:0"
  python scripts/figure03_content_aware/train_swinir.py --config "$CFG" \
      --comps x64 --patterns random_fixed,learnable_frequency --device cuda:0 \
      > "$EXP/train_gan_cuda0_x64.log" 2>&1
  echo "[extend-gan] $(date -Is) x64 done"
) &
C0=$!

# cuda:1 chain: x256 -> x1024
(
  while kill -0 "$X256_PID" 2>/dev/null; do sleep 120; done
  echo "[extend-gan] $(date -Is) x256 done -> starting x1024 on cuda:1"
  python scripts/figure03_content_aware/train_swinir.py --config "$CFG" \
      --comps x1024 --patterns random_fixed,learnable_frequency --device cuda:1 \
      > "$EXP/train_gan_cuda1_x1024.log" 2>&1
  echo "[extend-gan] $(date -Is) x1024 done"
) &
C1=$!

wait "$C0" "$C1"
echo "[extend-gan] $(date -Is) all 4 compressions trained. Generating full GAN report ..."

python scripts/figure03_content_aware/report.py \
    --name paper_faithful_pixel_perceptual_gan --device cuda:0 \
    --fixed-index 7 --random-comp x256 --random-indices 3,11,19,27,41 \
    --full-panel
echo "[extend-gan] DONE $(date -Is)"
