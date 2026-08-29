#!/usr/bin/env bash
# Paper-FAITHFUL full-budget retrain for MCF7 Fig 8 & 9 (paper §5.6, line 317).
#
#   Algorithm 1: epochs=230, epoch_baseline=150, epoch_cutoff=150, epoch_step=20  (--full-budget)
#   Batch size : 32   (SwinIR micro=4 x accum=8 ; conv baselines conv_batch_size=32)
#   Data loader: FULL (no max-steps-per-epoch cap) -> ~94 optimizer steps/epoch @ 3000 train
#   H_t lr=0.1, SwinIR lr=2e-4, pixel+perceptual+adversarial (SwinIR); L1 (conv baselines)
#
# Prior canonical run (runs_sp1) was compute-scaled: 54 epochs / batch 8 / 120-step cap.
# This run removes ALL of that scaling. Estimated wSwinIR wall time ~= 230*94*14.8s ~= 3.7 days.
# GPU1 baselines finish in a few hours. Output goes to a NEW dir (runs_paper) so the current
# canonical (runs_sp1) stays intact until we verify the new run is better.
set -uo pipefail
cd "$(dirname "$0")/.."                     # -> replication/
PY=./.venv/bin/python
EXP=experiments/figure89_mcf7_swinir_highres_fix_v1
RUNS=$EXP/runs_paper
mkdir -p "$RUNS"
SENTINEL="$EXP/runs_paper_STATUS.txt"
: > "$SENTINEL"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SENTINEL"; }

log "=== PAPER-FULL retrain start: Alg1 230/150/20, batch 32, full loader ==="

# --- GPU0: wSwinIR (Q) -- full pixel+perceptual+adversarial loss, ~3.7 days ---
$PY scripts/fig89_mcf7_swinir_fix_train.py --condition wswinir --device cuda:0 \
    --full-budget --gan-warmup-epochs 50 --out "$RUNS/wswinir" \
    > "$EXP/runs_paper_wswinir.log" 2>&1 &
PID_SWIN=$!
log "launched wswinir  PID=$PID_SWIN  -> $EXP/runs_paper_wswinir.log"

# --- GPU1: transpose256 (R) -> wCNN64 (Fig 9 baseline), L1, sequential chain ---
(
  $PY scripts/fig89_mcf7_swinir_fix_train.py --condition transpose256 --device cuda:1 \
      --full-budget --out "$RUNS/transpose256" > "$EXP/runs_paper_transpose256.log" 2>&1
  $PY scripts/fig89_mcf7_swinir_fix_train.py --condition wcnn64 --device cuda:1 \
      --full-budget --out "$RUNS/wcnn64" > "$EXP/runs_paper_wcnn64.log" 2>&1
) &
PID_GPU1=$!
log "launched gpu1 chain PID=$PID_GPU1 -> transpose256 then wcnn64"

wait $PID_SWIN;  SW=$?
wait $PID_GPU1;  G1=$?
log "training finished: wswinir exit=$SW  gpu1-chain exit=$G1"

if [ "$SW" -ne 0 ] || [ "$G1" -ne 0 ]; then
  log "!! a training job exited non-zero; SKIPPING figure regeneration. Inspect the .log files."
  log "=== PAPER-FULL retrain ABORTED ==="
  exit 1
fi

# --- pattern-resolution sanity (full-res speckle, NOT constant blocks) ---
$PY - <<'PYEOF' 2>&1 | tee -a "$SENTINEL"
import torch, os
RUNS="experiments/figure89_mcf7_swinir_highres_fix_v1/runs_paper"
for c in ["wswinir","transpose256","wcnn64"]:
    p=os.path.join(RUNS,c,"illumination","patterns.pt")
    if os.path.exists(p):
        t=torch.load(p,map_location="cpu").float()
        # count distinct 2x2-block variance as a crude "is it blocky" probe
        v=float(t.var().item())
        print(f"[pattern] {c:12s} shape={tuple(t.shape)} min={float(t.min()):.3f} "
              f"max={float(t.max()):.3f} var={v:.4f}")
    else:
        print(f"[pattern] {c:12s} MISSING {p}")
PYEOF

# --- regenerate canonical Fig 8 & Fig 9 (tubulin / channel-2) from the new run ---
log "regenerating Fig 8 ..."
$PY scripts/reproduce_fig8.py \
    --config configs/reproduce_fig8_bbbc021_tubulin_x16_swinir.yaml \
    --device cuda:0 --runs-dir "$RUNS" --out "$RUNS/figures" 2>&1 | tee -a "$SENTINEL" || log "fig8 regen FAILED"
log "regenerating Fig 9 ..."
$PY scripts/reproduce_fig9.py \
    --config configs/reproduce_fig9_bbbc021_tubulin_x16_swinir_vs_cnn.yaml \
    --device cuda:0 --runs-dir "$RUNS" --out "$RUNS/figures" 2>&1 | tee -a "$SENTINEL" || log "fig9 regen FAILED"

# --- dump final test metrics for quick comparison vs runs_sp1 ---
$PY - <<'PYEOF' 2>&1 | tee -a "$SENTINEL"
import json, os
RUNS="experiments/figure89_mcf7_swinir_highres_fix_v1/runs_paper"
for c in ["wswinir","transpose256","wcnn64"]:
    rp=os.path.join(RUNS,c,"result.json")
    if os.path.exists(rp):
        r=json.load(open(rp))
        print(f"[metric] {c:12s} PSNR={r.get('test_psnr'):.2f} SSIM={r.get('test_ssim'):.4f} "
              f"MSE={r.get('test_mse'):.6f} epochs={r.get('epochs')} eff_batch={r.get('effective_batch')} "
              f"wall_h={r.get('wall_seconds',0)/3600:.1f}")
PYEOF

log "=== PAPER-FULL retrain DONE (figures in $RUNS/figures) ==="
