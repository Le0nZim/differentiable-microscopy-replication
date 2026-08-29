#!/usr/bin/env bash
# Auto-finalize the Fig-3 SwinIR pixel+perceptual+GAN run (key cells x16, x256):
# wait for both GAN training PIDs to exit, then build the name-suffixed metrics CSV,
# the l1_ssim-vs-GAN comparison table, and the GAN diagnostic + random-field panels.
# (No --full-panel: only x16/x256 were trained, so a 4-row Fig-3 would show base
#  fallbacks for x64/x1024.)
#
# Usage: nohup bash finalize_watch_gan.sh <pid_cuda0> <pid_cuda1> > finalize_watch_gan.log 2>&1 &
set -u
PID0="${1:?need cuda0 GAN pid}"
PID1="${2:?need cuda1 GAN pid}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

echo "[finalize-gan] waiting for GAN PIDs $PID0 (cuda0/x16) and $PID1 (cuda1/x256) ..."
while kill -0 "$PID0" 2>/dev/null || kill -0 "$PID1" 2>/dev/null; do
    sleep 120
done
echo "[finalize-gan] both GAN training processes exited at $(date -Is). Generating report ..."

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONWARNINGS=ignore
python scripts/fig3_swinir_fix_report.py \
    --name paper_faithful_pixel_perceptual_gan --device cuda:0 \
    --fixed-index 7 --random-comp x256 --random-indices 3,11,19,27,41
echo "[finalize-gan] DONE at $(date -Is)"
