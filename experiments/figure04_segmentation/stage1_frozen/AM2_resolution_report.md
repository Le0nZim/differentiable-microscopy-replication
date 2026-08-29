# AM-2 resolution report — Task-aware segmentation is now end-to-end

**Final status: `AM-2 — FULLY_RESOLVED_IMPLEMENTATION_PASS / RESULTS_PROXY_BBBC022`**

This resolves AM-2 ("Task-aware segmentation is post-hoc, not end-to-end") by
implementing the paper's actual task-aware training procedure (§5.3, supplement
B.0.1) and rerunning the ×64 / ×256 / ×1024 matrix for fixed pseudo-random vs.
learnable illumination. Results are a **BBBC022 Hoechst substitute proxy** — the
original U2OS data are unavailable, so this is **not** an exact paper
reproduction.

Authoritative artifacts (this directory):
- `aggregate_summary.json` — per-cell results + post-hoc comparison.
- `metrics_summary.json` — compact Dice/IoU matrix, resolution-criteria checklist, status.
- `report.md` — generated tables (Dice/IoU, learnable-vs-fixed, gradient evidence).
- `run.log` — full training log for all 6 cells (all three stages each).
- `AUDIT_NOTE.md` — pre-edit audit.
- `taskaware_{x64,x256,x1024}_{random_fixed,learnable_frequency}_seed42/` — per-run
  checkpoints, metrics, stage evidence, qualitative panels, learned patterns.

---

## 1. What was mismatched before

`scripts/run_bbbc022_segmentation.py` trained a plain **L1 reconstruction** model
and then computed Dice/IoU from `pred_mask = (x_recon > 0.3)` — post-hoc
thresholding of the reconstruction. There was **no segmentation head**, **no
segmentation task loss** in the training loop, and `segmentation_bce_weight` was
**dead config** (never read by any trainer). So the reported learned-vs-random
Dice gap came from better *reconstruction*, not from task-aware *illumination*
learning. (A prior RR-3 attempt added stages 2–3 but used a Sigmoid head with
`binary_cross_entropy` on probabilities, loaded Stage 1 from an external
checkpoint, had no Dice loss, and logged no freeze/gradient evidence.)

## 2. What code was added / changed

**Added**
- `src/training/segmentation_losses.py` — `BCEWithLogitsLoss` (main task loss),
  `soft_dice_loss`, and `task_aware_segmentation_loss` combining
  `seg_bce_weight` + `seg_dice_weight` + optional `reconstruction_l1_weight`.
  `TaskAwareLossWeights.from_config` reads `seg_bce_weight` **and** the historical
  `segmentation_bce_weight` alias (the previously-dead key is now honored).
- `scripts/figure04_segmentation/train_stage1_frozen.py` — staged matrix runner;
  saves per-stage checkpoints, metrics, qualitative panels, learned patterns,
  config snapshots, and a report; compares to the historical post-hoc numbers.
- `configs/figure04_segmentation/stage1_frozen.yaml` — explicit config
  (dataset, compression/T, seg-head architecture, per-stage step budgets,
  per-component LRs, loss weights, mask params, seed, output dir).

**Modified**
- `src/models/segmentation_head.py` — now emits **raw logits** `[B,1,H,W]`
  (Conv/ReLU blocks ending in a bare 1-channel Conv2d; no Sigmoid). Flexible depth.
- `src/models/task_aware_microscope.py` — forward returns `x_recon`,
  `seg_logits`, `seg_prob`; adds fine-grained freeze/unfreeze
  (`set_microscope/inverse/illumination/segmentation_trainable`) and a
  `trainable_parameter_report()` for stage evidence.
- `src/training/train_task_aware_segmentation.py` — rewritten as a **single
  three-stage trainer** (details below) with `requires_grad` reports, per-stage
  gradient-norm logging, illumination-pattern-change tracking, val-selected
  thresholding, qualitative panels, and per-stage checkpoints.
- `tests/test_task_aware_segmentation.py` — rewritten + expanded to 10 tests.

**Reused unchanged (deliberately)**
- `src/datasets/bbbc022_hoechst.py::make_pseudo_mask` — pseudo-GT masks per
  supplement **A.2.3**: normalize→[0,1], threshold 0.3, morphological closing
  with a (10,10) kernel, computed from the **ground-truth** image (not the
  reconstruction). Same split/data pipeline (split-by-well, 168/21/21,
  `paper_strict`, patch 256) as the existing experiment.
- `src/training/staged_hardening_train.py`, `train_reconstruction.py`, and the
  forward/inverse/pattern models — used to run Stage 1 content-aware pretraining.

## 3. The three training stages and evidence they ran

Each cell ran all three stages fresh (`stage1_mode: train`). See `run.log` and
each run's `metrics/stage_evidence.json`, `metrics/stage2_history.json`,
`metrics/stage3_history.json`, and `stage1_content_aware/` (with per-phase
checkpoints `inverse_warmup_m1`, `joint_soft_m1`, `harden_m2/4/8`).

- **Stage 1 — content-aware reconstruction pretraining.** L1 reconstruction.
  Learnable variants use the repo's staged-hardening schedule
  (inverse-warmup 1500 → joint-soft 3500 → harden m∈{2,4,8}×800); the fixed
  variant trains the inverse model only (4000 steps) with illumination fixed.
- **Stage 2 — segmentation-head-only.** Microscope frozen; train the head on
  pseudo-masks (1200 steps).
- **Stage 3 — end-to-end task-aware finetune.** Unfreeze seg head + inverse
  (+ illumination for the learnable variant); finetune on the segmentation loss
  with a soft→hard sigmoid schedule (m=1 for 60% of 2500 steps, then m∈{2,4,8}),
  so the task loss can flow into the frequency-domain pattern parameters before
  they re-binarize for evaluation.

## 4. Evidence Stage 2 froze the microscope / reconstruction

From `stage_evidence.json` (every learnable & fixed cell), Stage 2:
`trainable_report` shows `illumination.all_frozen = true` and
`inverse_model.all_frozen = true`, `segmentation_head.all_trainable = true`; and
`grad_norms` show **inverse = 0.0, illumination = 0.0**, seg-head > 0 (e.g.
×1024 learnable: seg-head max 0.906, inverse 0.0, illumination 0.0). The trainer
also `assert`s these conditions at runtime.

## 5. Evidence Stage 3 finetuned all components end-to-end

Stage 3 `trainable_report` shows all of illumination / inverse / seg-head
`all_trainable = true` (learnable variant), and nonzero gradient norms for all
finetuned groups. Inverse-model Stage-3 grad-norm max: ×64 3.80, ×256 8.20,
×1024 17.47.

## 6. Evidence the learnable illumination received the segmentation gradient

Stage-3 illumination gradient norms are **nonzero** for every learnable cell, and
— because `illumination_lr = 1.0` with Adam converts the (small-norm) saturated-
sigmoid gradients into real updates — the **sharpened (eval m=10) patterns
actually move**:

| compression | illumination grad-norm (Stage 3, max) | pattern Δ ‖H_t‖₂ | relative Δ |
|---|---:|---:|---:|
| ×64 learnable | 1.28e-05 | 71.67 | 20.8% |
| ×256 learnable | 1.59e-05 | 95.44 | 27.6% |
| ×1024 learnable | 2.93e-05 | 143.14 | 41.5% |

Fixed pseudo-random cells have **no** illumination parameters, so their grad-norm
and pattern Δ are exactly 0.0 (as expected). The raw grad-norm is small (the
binarizing sigmoid saturates); the pattern Δ is the meaningful signal that the
segmentation loss optimized the illumination.

## 7. Dice / IoU — fixed vs. learnable (test, val-selected threshold)

| compression | illumination | Dice | IoU | Dice@0.5 | thr |
|---|---|---:|---:|---:|---:|
| ×64  | fixed     | 0.8938 | 0.8095 | 0.8938 | 0.65 |
| ×64  | learnable | **0.9322** | **0.8732** | 0.9322 | 0.60 |
| ×256 | fixed     | 0.6958 | 0.5344 | 0.6707 | 0.15 |
| ×256 | learnable | **0.8757** | **0.7794** | 0.8754 | 0.35 |
| ×1024| fixed     | 0.4993 | 0.3335 | 0.4069 | 0.10 |
| ×1024| learnable | **0.5986** | **0.4284** | 0.5810 | 0.15 |

**Learnable illumination beats fixed pseudo-random at every compression**
(×64 +0.038, ×256 +0.180, ×1024 +0.099 Dice) — consistent with the paper's
Fig. 4 direction (the paper is qualitative; we do not claim its numbers).
Training uses logits + BCE/Dice; thresholds are only used for *reporting*
Dice/IoU and are selected on the validation split.

## 8. Qualitative outputs

Per-run 5-column panels (GT image · pseudo mask · reconstruction · seg prob ·
predicted mask) at `taskaware_<comp>_<mode>_seed42/figures/qualitative_panel.png`,
e.g. `taskaware_x64_learnable_frequency_seed42/figures/qualitative_panel.png`.
Learned illumination patterns (raw / binarized / FFT / histogram / stats) at
`taskaware_<comp>_<mode>_seed42/learned_patterns/H_t_final*`.

## 9. Comparison to the previous post-hoc diagnostic

Post-hoc numbers from `experiments/task_aware_segmentation/bbbc022_segmentation/`
(preserved as historical diagnostics; **not** deleted):

| compression | mode | post-hoc Dice (x_recon>0.3) | task-aware Dice (this work) | Δ |
|---|---|---:|---:|---:|
| ×64  | fixed     | 0.9018 | 0.8938 | −0.008 |
| ×64  | learnable | 0.9293 | 0.9322 | +0.003 |
| ×256 | fixed     | 0.7068 | 0.6958 | −0.011 |
| ×256 | learnable | 0.8658 | 0.8757 | +0.010 |
| ×1024| fixed     | 0.2665 | **0.4993** | **+0.233** |
| ×1024| learnable | 0.5669 | 0.5986 | +0.032 |

The point of AM-2 was **not** to maximize Dice but to implement the real
procedure. Still, the learned head most clearly helps where post-hoc thresholding
of a poor reconstruction fails (×1024 fixed: 0.267→0.499). At low compression the
fixed reconstruction is already near-binary, so a fixed-illumination learned head
adds little (small −Δ); this is expected and honestly reported.

## 10. Honest scope statement

This is a **BBBC022 Hoechst substitute proxy** for the paper's U2OS task-aware
segmentation experiment. The original U2OS dataset is unavailable, so absolute
numbers are **not** comparable to the paper and this is **not** an exact
reproduction. What is reproduced is the *method*: content-aware pretrain →
frozen segmentation-head training → end-to-end finetune of excitation patterns +
inverse model + segmentation head with the segmentation task loss.

## 11. Resolution criteria (all satisfied)

- ✅ Real segmentation head (raw-logit conv net) — `src/models/segmentation_head.py`.
- ✅ Training loop uses the segmentation task loss (BCEWithLogits + soft Dice).
- ✅ Staged training matching the paper (content-aware → frozen head → full finetune).
- ✅ Post-hoc `x_recon > threshold` is no longer the defining training method.
- ✅ Learnable illumination parameters are updated by the segmentation loss in
  Stage 3 (nonzero grads; 20.8–41.5% pattern change).
- ✅ Metrics + visuals for ×64, ×256, ×1024.
- ✅ Honestly labeled as a task-aware BBBC022 proxy (U2OS unavailable).

**Final status: `AM-2 — FULLY_RESOLVED_IMPLEMENTATION_PASS / RESULTS_PROXY_BBBC022`**
