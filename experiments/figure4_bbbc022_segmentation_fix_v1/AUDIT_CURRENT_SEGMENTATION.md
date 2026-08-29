# Audit — current Figure 4 (task-aware segmentation) replication

Isolated audit for `experiments/figure4_bbbc022_segmentation_fix_v1/`. Read-only
review of the existing task-aware segmentation implementation against the paper
(§5.3 "Task-aware sampling", supplement A.2.3 "Pseudo-ground Truths", supplement
B.0.1 "Segmentation-aware microscopy: Training procedure", and the Figure 4
caption). Nothing here is modified in place; the frozen `am2_task_aware_full`
run is preserved.

**Headline finding.** Unlike the Figure 3 SwinIR columns (which were genuinely
under-performing and had to be rebuilt), the Figure 4 segmentation experiment is
**already implemented faithfully** in
`experiments/task_aware_segmentation/am2_task_aware_full/`. It runs the paper's
real three-stage procedure with a real segmentation head and a segmentation
task loss (not post-hoc thresholding), and its results already show the paper's
qualitative direction (learnable illumination beats fixed pseudo-random at every
compression). "The same care" here therefore means: an independent audit, a
proper sanity-gate suite, a self-contained *verified* re-run in this isolated
directory, a clean **paper-layout** Figure 4, and an advisor-ready report — not
a rescue of a broken result.

> **Mask method updated (latest run).** The pseudo-ground-truth masks are now
> generated with a **TrackMate-style** raw-intensity detector (classify **raw
> MIP** pixels **> 506** as foreground, group with **4-connectivity**, resample
> each region contour at **~2 px**, **Douglas–Peucker** simplify at **ε 0.5**,
> then fill), replacing the paper's A.2.3 normalize→thr 0.3→closing(10) recipe.
> Rationale: the legacy masks (on the clipped/normalized image) produced inflated,
> merged blobs; the TrackMate masks follow individual nuclei, so absolute Dice is
> lower but the learnable-vs-fixed comparison is cleaner. Implemented in
> `src/datasets/bbbc022_hoechst.py::make_trackmate_mask` (`mask_mode: trackmate`).
> The legacy recipe is preserved (`mask_mode: threshold_closing`,
> `backup_thr0p3closing/`). Sections below mark the paper spec vs. this run.

---

## 1. What the paper specifies for Figure 4

| Item | Paper (U2OS) |
|---|---|
| Task | Binary segmentation as a representative downstream task (§5.3) |
| Pseudo-GT masks (A.2.3) | Normalize image → [0,1]; threshold at **0.3**; morphological **closing** with a **(10,10)** kernel *(this run substitutes a TrackMate detector — see banner)* |
| Training procedure (B.0.1) | **(1)** train the end-to-end model for *content-aware* (Fig. 1 procedure: learn illumination + reconstruction); **(2)** append and train a **small convolutional network** for segmentation with **all content-aware parameters fixed**; **(3)** finetune **all** components end-to-end (excitation pattern network + inverse model + segmentation network) |
| Compressions (Fig. 4) | **×64, ×256, ×1024** |
| Illuminations (Fig. 4) | **pseudo-random Hᵗ** (C1/D1/E1) vs **learnable Hᵗ (proposed)** (C2/D2/E2) |
| Figure 4 layout | **A)** GT test images; **B)** pseudo-GT segmentation maps; **C1–E2)** segmentation results for the 3 compressions × 2 illuminations; **F)** representative illumination pattern |
| Claim | "The proposed method consistently generates better segmentation maps." (qualitative; no numeric table for Fig. 4) |

## 2. Current implementation (canonical: `am2_task_aware_full`)

Audited files:
- `configs/task_aware/bbbc022_segmentation_task_aware.yaml`
- `scripts/run_bbbc022_task_aware_segmentation.py`
- `src/training/train_task_aware_segmentation.py`
- `src/training/segmentation_losses.py`
- `src/models/task_aware_microscope.py`, `src/models/segmentation_head.py`
- `src/datasets/bbbc022_hoechst.py` (`make_trackmate_mask`, `make_pseudo_mask`, split logic)

| Audit dimension | Finding | Verdict |
|---|---|---|
| **Substitute data** | U2OS unavailable → **BBBC022 Hoechst 33342** (`data/substitute_data`), MIP + `paper_strict` normalization (bias 134.28, clip 500, min–max → [0,1]) for the specimen; patch 256; well-disjoint split **168/21/21** | Proxy (honest) |
| **Pseudo-GT masks (this run)** | `make_trackmate_mask` on the **raw MIP** intensity: foreground `> 506` (original pixel values, no normalization), 4-connected regions, boundary resampled ~2 px, Douglas–Peucker (ε 0.5), filled. Computed from the **ground-truth image**, not the reconstruction. Full-image masks cached and cropped per sample (flips applied jointly) | ✅ TrackMate as specified |
| **Pseudo-GT masks (paper A.2.3)** | `make_pseudo_mask` = normalize→threshold `≥ 0.3`→closing (k=10→11 odd). Retained as `mask_mode: threshold_closing` + `backup_thr0p3closing/` for provenance | ⚠️ Substituted this run |
| **Stage 1 — content-aware pretrain** | Learnable variant uses the repo staged-hardening schedule (inverse-warmup 1500 → joint-soft 3500 → harden m∈{2,4,8}×800); fixed variant trains inverse only (4000 steps). L1 reconstruction loss | ✅ Faithful to B.0.1 step 1 |
| **Stage 2 — frozen head** | Microscope frozen (`.eval()` + `requires_grad=False`); train the seg head only (1200 steps). Runtime `assert`s: illumination & inverse fully frozen (grad-norm = 0), seg head trainable (grad-norm > 0) | ✅ Faithful to B.0.1 step 2 |
| **Stage 3 — end-to-end finetune** | Unfreeze seg head + inverse (+ illumination for learnable); soft→hard sigmoid schedule (m=1 for 60% of 2500 steps, then m∈{2,4,8}) so the task loss reaches the frequency-domain pattern params before they re-binarize for eval. Runtime `assert`s nonzero grads for all finetuned groups; illumination pattern Δ > 0 for learnable | ✅ Faithful to B.0.1 step 3 |
| **Segmentation network** | `SegmentationHead`: small conv stack `[16,16,1]`, 3×3, ReLU between blocks, **raw 1-channel logits** on the last layer | ✅ "small convolutional network" |
| **Segmentation loss** | `BCEWithLogitsLoss` (weight 1.0) + auxiliary **soft Dice** (weight 0.5); optional recon-L1 stabilizer (0.0 by default). Paper does not name the seg loss | ✅ Reasonable, defensible choice |
| **Compressions** | ×64 (downscale 16), ×256 (32), ×1024 (64), T=4 patterns (compression = downscale²/T) | ✅ Matches Fig. 4 |
| **Illuminations** | `random_fixed` (pseudo-random, m=10) vs `learnable_frequency` | ✅ Matches Fig. 4 |
| **Evaluation** | Per-sample Dice/IoU; threshold selected on **validation** (sweep 0.10–0.90) and applied to test; also reports Dice@0.5 | ✅ No test-set tuning |
| **Evidence logging** | Per-stage `requires_grad` reports + gradient norms; illumination pattern Δ (L2 + relative); per-stage checkpoints, qualitative panels, learned patterns | ✅ Auditable |

## 3. Current results (this isolated run, `runs/metrics_summary.json`, TrackMate masks, seed 42)

| Compression | Fixed Dice | Learnable Dice | Δ | Learnable wins? |
|---|---:|---:|---:|:--:|
| ×64  | 0.8480 | 0.9039 | +0.056 | ✅ |
| ×256 | 0.6283 | 0.8150 | +0.187 | ✅ |
| ×1024| 0.3959 | 0.4583 | +0.062 | ✅ |

Direction matches the paper's qualitative claim at every compression. Absolute
numbers are lower than the legacy thr0.3+closing run (e.g. ×64 learnable 0.93→0.90)
because the TrackMate masks are tighter, but the learnable advantage is preserved
and widens at ×256. (Absolute numbers are **not** comparable to the paper —
different dataset and mask recipe.)

## 4. Gaps / weaknesses this isolated fix addresses

1. **No paper-layout Figure 4.** The existing outputs are per-run 5-column
   qualitative strips + a Dice bar chart. There is no single figure matching the
   paper's A/B/C1–E2/F grid (shared GT row, shared pseudo-mask row, a 3×2
   segmentation grid, and a representative illumination pattern). → We build one.
2. **No standalone sanity-gate suite.** The trainer has runtime `assert`s, but
   there is no separate, inspectable suite (mask correctness, split
   no-leakage, tiny-overfit capability, degenerate baselines) analogous to the
   Figure 3 fix. → We add `sanity/`.
3. **Learnable-advantage attribution is implicit.** The learnable illumination
   is optimized in *two* places: Stage 1 (content-aware) and Stage 3 (task-aware
   finetune). The paper's Fig. 4 comparison (learnable-full vs fixed-random) is
   exactly what `am2` reports, but for the advisor it is worth explicitly
   separating the Stage-2 (frozen content-aware patterns) vs final (Stage-3
   task-finetuned patterns) Dice to show how much the *task-aware* finetuning
   adds on top of content-aware pretraining. → Reported from stage metrics.
4. **Reproducibility not isolated.** The `am2` numbers live in a single run dir.
   → We re-run Stage 2+3 into this isolated dir from the *frozen* Stage-1
   content-aware checkpoints and confirm the numbers reproduce.

## 5. Deviations from the paper (intentional, documented)

- **Dataset:** BBBC022 Hoechst substitute, not U2OS (data unavailable). Absolute
  metrics are a proxy; only the *direction* (learnable > fixed) is claimed.
- **Pseudo-GT mask recipe:** TrackMate-style raw-intensity detector (raw MIP > 506,
  4-connectivity, resample ~2 px, Douglas–Peucker ε 0.5, fill) instead of the
  paper's A.2.3 normalize→thr 0.3→closing(10). Both are rule-based masks from the
  GT image; the TrackMate variant tracks individual nuclei more tightly. The A.2.3
  recipe remains available (`mask_mode: threshold_closing`).
- **Noise:** `noise_free` for Fig. 4 (matches the frozen content-aware Stage-1
  base). The paper's content-aware experiments use a 10 000-photon model; the
  segmentation supplement does not specify a noise setting. Kept consistent with
  the Stage-1 base we reuse.
- **Seg loss = BCE + soft Dice:** the paper does not name the segmentation loss;
  BCE(+Dice) is the standard, stable choice for binary masks.
- **Training budget:** step counts are a compute-bounded proxy (the paper gives
  no segmentation step counts). Chosen to converge the head + finetune stably.
- **Stage-1 reuse:** we reuse the already-trained content-aware microscope
  (frozen) as B.0.1 step 1, exactly as the Figure 3 SwinIR fix reused the frozen
  content-aware base. B.0.1 step 1 *is* "train the end-to-end model for
  content-aware"; reusing that trained model (produced by the identical
  code/config/seed) is faithful and avoids redundant compute.

## 6. Constraints honored

- No fabrication; no test-set tuning (thresholds selected on val only).
- No post-hoc mask "cleanup" of predictions beyond the single val-selected
  threshold used consistently for reporting.
- Pseudo-GT masks come from the **ground-truth** raw MIP intensity (TrackMate
  detector, §banner), never from the reconstruction.
- Additive only: the frozen `am2_task_aware_full` run is not modified; all new
  artifacts live under `figure4_bbbc022_segmentation_fix_v1/`.
