#!/usr/bin/env bash
# Full-resolution-pattern (superpixel_factor=1) retrain of ALL THREE Fig 8/9 models,
# matching each frozen run's compute recipe, then regenerate Fig 8 + Fig 9 and the
# pattern diagnosis. Fixes the superpixel_factor=8 bug (hard 8px illumination blocks).
#
#   GPU0 : wSwinIR  (proven recipe: 54ep, baseline35, step5, 120 steps/ep, warmup40, SSIM-select)
#   GPU1 : transpose256 (54ep, baseline35, step5)  ->  wCNN64 (72ep, baseline46, step6)
# All three write to runs_sp1/. Frozen runs (runs/, runs_full/) are untouched.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
EXP=experiments/figure08_mcf7
RUNS=$EXP/runs_sp1
mkdir -p "$RUNS"

echo "=== [launch] wSwinIR on cuda:0 (background) ==="
$PY scripts/figure08_mcf7/train.py --condition wswinir --device cuda:0 \
    --epochs 54 --epoch-baseline 35 --epoch-step 5 --max-steps-per-epoch 120 \
    --gan-warmup-epochs 40 --out "$RUNS/wswinir" > "$EXP/runs_sp1_wswinir.log" 2>&1 &
PID_SWIN=$!

echo "=== [launch] transpose256 -> wCNN64 on cuda:1 (background chain) ==="
(
  $PY scripts/figure08_mcf7/train.py --condition transpose256 --device cuda:1 \
      --epochs 54 --epoch-baseline 35 --epoch-step 5 --out "$RUNS/transpose256" \
      > "$EXP/runs_sp1_transpose256.log" 2>&1
  $PY scripts/figure08_mcf7/train.py --condition wcnn64 --device cuda:1 \
      --epochs 72 --epoch-baseline 46 --epoch-step 6 --out "$RUNS/wcnn64" \
      > "$EXP/runs_sp1_wcnn64.log" 2>&1
) &
PID_GPU1=$!

echo "=== [wait] wSwinIR PID=$PID_SWIN ; GPU1-chain PID=$PID_GPU1 ==="
FAIL=0
wait $PID_SWIN || { echo "wSwinIR FAILED"; FAIL=1; }
wait $PID_GPU1 || { echo "GPU1 chain FAILED"; FAIL=1; }
if [ "$FAIL" -ne 0 ]; then echo "=== ABORT: a training job failed, skipping regen ==="; exit 1; fi

echo "=== [verify] patterns are now full-resolution (std within 8x8 block should be > 0) ==="
$PY - <<'PYEOF'
import torch, numpy as np
for c,res in [('wswinir',256),('transpose256',256),('wcnn64',64)]:
    p=torch.load(f'experiments/figure08_mcf7/runs_sp1/{c}/illumination/patterns.pt',map_location='cpu')[0,0]
    H=p.shape[0]; pr=p[:H//8*8,:H//8*8].reshape(H//8,8,H//8,8)
    within=pr.std(dim=(1,3)).mean().item(); nu=len(np.unique(np.round(p.numpy(),4)))
    print(f'{c:12s} res={H} std_within_8x8={within:.5f} (>0 => full-res) #unique={nu}')
PYEOF

echo "=== [regen] Fig 8 (idx 12 6 4) + Fig 9 from runs_sp1 ==="
$PY scripts/figure08_mcf7/reproduce_fig8.py --config configs/figure08_mcf7/reproduce_fig8_tubulin.yaml \
    --device cuda:0 --runs-dir "$RUNS" --indices 12 6 4 \
    --out results/reproduced_figures/fig8_sp1
$PY scripts/figure08_mcf7/reproduce_fig9.py --config configs/figure08_mcf7/reproduce_fig9_tubulin.yaml \
    --device cuda:0 --runs-dir "$RUNS" \
    --out results/reproduced_figures/fig9_sp1

echo "=== [regen] pattern diagnosis: paper vs sp8(old) vs sp1(new) ==="
$PY - <<'PYEOF'
import torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
def font(s):
    try: return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',s)
    except Exception: return ImageFont.load_default()
def grid(path, disp=170, gap=6):
    p=torch.load(path,map_location='cpu').squeeze(1).clamp(0,1).numpy()
    T=p.shape[0]; cols=2; rows=(T+1)//2
    W=cols*disp+(cols-1)*gap; H=rows*disp+(rows-1)*gap
    c=np.full((H,W),255,np.uint8)
    for t in range(T):
        r,cc=divmod(t,cols); b=(p[t]>0.5).astype(np.uint8)*255
        c[r*(disp+gap):r*(disp+gap)+disp, cc*(disp+gap):cc*(disp+gap)+disp]=np.array(Image.fromarray(b).resize((disp,disp),Image.NEAREST))
    return Image.fromarray(c).convert('RGB')
R='experiments/figure08_mcf7'
cells=[('wSwinIR sp8 (OLD 32x32)',f'{R}/runs/wswinir/illumination/patterns.pt'),
       ('wSwinIR sp1 (NEW full)',f'{R}/runs_sp1/wswinir/illumination/patterns.pt'),
       ('wCNN sp8 (OLD 8x8)',f'{R}/runs/wcnn64/illumination/patterns.pt'),
       ('wCNN sp1 (NEW full)',f'{R}/runs_sp1/wcnn64/illumination/patterns.pt')]
imgs=[(t,grid(p)) for t,p in cells]
colw=max(i.width for _,i in imgs); th=30
def titled(t,im):
    c=Image.new('RGB',(colw,im.height+th),(255,255,255)); c.paste(im,((colw-im.width)//2,th))
    ImageDraw.Draw(c).text((4,4),t,fill=(0,0,0),font=font(18)); return c
cols=[titled(t,i) for t,i in imgs]
H=max(c.height for c in cols); pad=24
out=Image.new('RGB',(colw*len(cols)+pad*(len(cols)-1),H),(255,255,255))
for k,c in enumerate(cols): out.paste(c,(k*(colw+pad),0))
out.save('results/reproduced_figures/pattern_fix_sp8_vs_sp1.png')
print('saved results/reproduced_figures/pattern_fix_sp8_vs_sp1.png')
PYEOF

echo "=== ALL DONE (sp1 retrain + regen) ==="
