# AM-2 audit note (before editing)

Goal: replace post-hoc reconstruction-thresholding "segmentation" with the
paper's actual task-aware procedure (supplement B.0.1 / §5.3): content-aware
pretrain → frozen segmentation-head training → end-to-end finetune of
excitation patterns + inverse model + segmentation head with the segmentation
task loss.

## Where post-hoc thresholding happens today

- `scripts/run_bbbc022_segmentation.py:134-167` — trains a **plain L1
  reconstruction** model (`train` / `train_staged_hardening`) and then computes
  Dice/IoU from `pred_mask = (out["x_recon"] > 0.3)`. The "segmentation" is just
  thresholding the reconstruction; the predicted mask never comes from a learned
  segmentation network.
- `src/training/train_reconstruction.py:74` — the only training loss is
  `reconstruction_loss_l1(outputs["x_recon"], specimen)`. No segmentation term.
- `scripts/run_bbbc022_segmentation.py:130` — sets
  `config["training"]["segmentation_bce_weight"] = 1.0`, but **no trainer reads
  it** (dead config; grep confirms it appears only in scripts/saved configs).

## What already exists (prior partial RR-3 work) and its gaps vs. the paper

- `src/models/segmentation_head.py` — exists, but the head ends in **`nn.Sigmoid`**
  and returns probabilities, not raw logits. The spec/paper want raw logits with
  `BCEWithLogitsLoss`.
- `src/models/task_aware_microscope.py` — forward returns only `seg_pred`
  (probabilities), not `seg_logits` / `seg_prob`.
- `src/training/train_task_aware_segmentation.py` — implements **only stages 2–3**
  (seg-head warmup + finetune) and **loads Stage 1 from an external
  content-aware checkpoint**; Stage 1 (content-aware pretraining) is not run by
  the trainer. Uses `F.binary_cross_entropy` on probabilities (not
  `BCEWithLogitsLoss`), has **no soft-Dice option**, no reconstruction
  regularization weight, **no gradient-norm / requires_grad evidence logging**,
  no per-stage checkpoints, no qualitative panels, and no validation-selected
  threshold.
- `scripts/run_bbbc022_segmentation_taskaware.py` — runner for the partial
  version; reused post-hoc reconstruction checkpoints as Stage 1.

## Pseudo-ground-truth masks (already paper-faithful — keep)

- `src/datasets/bbbc022_hoechst.py:179-188` `make_pseudo_mask`: normalize to
  [0,1] (done in `preprocess_image`), threshold `>= 0.3`, morphological
  **closing** with a (10,10) kernel (max-pool dilation then erosion). This
  matches supplement **A.2.3** exactly and is computed from the **ground-truth
  image** (not the reconstruction), so it is a valid training target. Keep it;
  surface its parameters via config and do NOT train against `x_recon > 0.3`.

## What needs to be replaced / added

1. Segmentation head must output **raw logits** `[B,1,H,W]` (Conv/ReLU blocks,
   final 1-channel conv, no activation).
2. Task-aware forward must return `x_recon`, `seg_logits`, `seg_prob`.
3. Real segmentation losses: `BCEWithLogitsLoss` (main) + optional soft Dice,
   weighted by `seg_bce_weight`, `seg_dice_weight`, `reconstruction_l1_weight`.
4. A single staged trainer that runs **all three** stages, freezes the
   microscope in Stage 2, finetunes everything in Stage 3, and **logs
   requires_grad + gradient norms** as evidence (illumination grads must be
   nonzero in Stage 3 for the learnable variant).
5. Explicit config, dedicated runner, per-stage checkpoints, qualitative panels,
   saved learned patterns, and a resolution report — without overwriting the old
   post-hoc diagnostics.

## Constraints honored

- Same BBBC022 substitute dataset/split/mask pipeline as the current experiment
  (split-by-well, 168/21/21, paper_strict preprocessing, patch 256).
- Results are a **BBBC022 substitute proxy**, not exact U2OS reproduction.
