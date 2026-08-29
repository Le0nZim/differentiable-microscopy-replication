#!/usr/bin/env bash
# Full-length corrected-recipe wSwinIR retrain, matched to the frozen run's compute
# budget (54ep x 120 steps, baseline 35, step 5) but WITH the fixes:
#   * GAN warmup (--gan-warmup-epochs 40)
#   * max-val-SSIM checkpoint selection (automatic for full-loss condition)
# On training success it regenerates Fig.8 (matched indices 12 6 4) + Fig.9 and a
# frozen-vs-full Q comparison strip. Designed to be launched under nohup.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
EXP=experiments/figure08_mcf7
RUNS=$EXP/runs_full
DEV=${DEVICE:-cuda:0}

echo "=== [1/4] TRAIN wswinir (full budget, corrected recipe) on $DEV ==="
$PY scripts/figure08_mcf7/train.py --condition wswinir --device "$DEV" \
    --epochs 54 --epoch-baseline 35 --epoch-step 5 --max-steps-per-epoch 120 \
    --gan-warmup-epochs 40 \
    --out "$RUNS/wswinir"

echo "=== [2/4] symlink frozen R/wCNN baselines into runs_full ==="
ln -sfn ../runs/transpose256 "$RUNS/transpose256"
ln -sfn ../runs/wcnn64       "$RUNS/wcnn64"

echo "=== [3/4] regenerate Fig.8 (matched idx 12 6 4) + Fig.9 from runs_full ==="
$PY scripts/figure08_mcf7/reproduce_fig8.py \
    --config configs/figure08_mcf7/reproduce_fig8_tubulin.yaml \
    --device "$DEV" --runs-dir "$RUNS" --indices 12 6 4 \
    --out results/reproduced_figures/fig8_v3_matched
$PY scripts/figure08_mcf7/reproduce_fig9.py \
    --config configs/figure08_mcf7/reproduce_fig9_tubulin.yaml \
    --device "$DEV" --runs-dir "$RUNS" \
    --out results/reproduced_figures/fig9_v3

echo "=== [4/4] build frozen-vs-full Q comparison strip ==="
$PY - <<'PYEOF'
import numpy as np
from PIL import Image, ImageDraw, ImageFont
froz = np.asarray(Image.open('experiments/figure08_mcf7/figures/figure8_paper_style.png').convert('RGB'))
v3   = np.asarray(Image.open('results/reproduced_figures/fig8_v3_matched/figure8_paper_style.png').convert('RGB'))
H,W,_=froz.shape; r=H//3
def lab(band,text):
    im=Image.fromarray(band.copy()); d=ImageDraw.Draw(im)
    try: f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',26)
    except Exception: f=ImageFont.load_default()
    d.rectangle([0,0,470,34],fill=(0,0,0)); d.text((6,4),text,fill=(255,255,255),font=f)
    return np.asarray(im)
out=np.concatenate([lab(froz[0:r],'P: Ground Truth'),
                    lab(froz[r:2*r],'Q frozen (54ep, no-warmup, min-MSE)'),
                    lab(v3[r:2*r],'Q full retrain (54ep, warmup40, SSIM)')],axis=0)
Image.fromarray(out).save('results/reproduced_figures/fig8_Q_frozen_vs_full.png')
print('saved results/reproduced_figures/fig8_Q_frozen_vs_full.png')
PYEOF

echo "=== ALL DONE ==="
