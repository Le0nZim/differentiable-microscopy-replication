# Figure 4 — Task-aware (segmentation-aware) sampling: audit, verification & clean reproduction

**Status:** `FIG4_FIX_PASS / RESULTS_PROXY_BBBC022` — latest verified re-run complete (TrackMate pseudo-GT); tables below are live.

BBBC022 Hoechst **substitute** proxy for the paper's U2OS segmentation-aware
experiment (§5.3, supplement A.2.3 + B.0.1, Figure 4). U2OS data are
unavailable, so absolute numbers are **not** comparable to the paper; only the
qualitative direction (learnable Hᵗ > pseudo-random Hᵗ) is claimed.

This report documents the **latest and best** run of the experiment, which uses a
**TrackMate-style pseudo-ground-truth mask** (see §1). Earlier thr0.3+closing
artifacts are preserved under `backup_thr0p3closing/` and
`runs_backup_thr0p3closing/` for provenance.

---

## 0. Headline

The task-aware segmentation pipeline (real segmentation head + real segmentation
task loss + the paper's three-stage B.0.1 procedure) was already implemented
faithfully in `experiments/task_aware_segmentation/am2_task_aware_full/`. This
isolated experiment wraps it in full rigor **and** upgrades the pseudo-GT mask
generator to a TrackMate-style detector that follows nuclei far more tightly than
the legacy normalize→threshold-0.3→closing recipe. With the new masks the
qualitative claim holds *more* clearly than before: **learnable illumination
beats fixed pseudo-random at every compression**, with the gap widening at ×64
and ×256.

Deliverables:

1. an **independent audit** (`AUDIT_CURRENT_SEGMENTATION.md`) and a
   paper-vs-implementations **diff** (`PAPER_VS_CURRENT_FIG4_DIFF.md`);
2. a standalone **sanity-gate suite** (`sanity/`) — all gates pass on the new masks;
3. a **self-contained verified re-run** of the full ×64/×256/×1024 × {fixed,
   learnable} matrix (Stage 2+3 recomputed on the frozen content-aware Stage-1 base);
4. a clean **paper-layout Figure 4** (A / B / C1–E2 / F), a viridis composite, and
   **per-panel image tiles** for rebuilding the figure in PowerPoint;
5. this advisor-ready report with explicit **acceptance criteria** and a
   **task-aware vs. content-aware attribution**.

Nothing in the frozen `am2` run was modified.

## 1. Pseudo-GT mask method (TrackMate-style) — the change in this run

The pseudo-ground-truth segmentation maps (Figure 4 row B, and the training/eval
targets) are generated with a **TrackMate-style thresholded detector on the raw
MIP intensity** — not the legacy normalized threshold+closing recipe:

- **Threshold on raw intensity.** Classify pixels of the **raw maximum-intensity
  projection** (no bias subtraction, no clipping, no min–max normalization) with
  intensity **> 506** as foreground; ≤ 506 as background. Original pixel values
  are preserved for the decision.
- **4-connectivity grouping.** Foreground pixels are grouped into connected
  regions using 4-connectivity.
- **Simplified contours.** Each region's boundary is traced, resampled at
  **~2-pixel** intervals, then simplified with **Douglas–Peucker** (max deviation
  **0.5 px**) and filled. No extra normalization or morphological post-processing.

Implementation: `src/datasets/bbbc022_hoechst.py::make_trackmate_mask` (+
`BBBC022HoechstDataset` with `mask_mode: trackmate`). Full-image masks are
precomputed once (cached) and cropped per sample so the mask stays pixel-aligned
with the specimen crop (flips applied jointly). Config knobs (in
`configs/fig4_seg_fix_base.yaml`):

```yaml
dataset:
  mask_mode: trackmate
  mask_raw_threshold: 506.0
  mask_smooth_interval: 2.0
  mask_dp_epsilon: 0.5
```

Why this matters: the legacy thr0.3+closing masks (on the clipped/normalized
image) produced inflated, blocky blobs that merged adjacent nuclei — inflating
absolute Dice and muddying the learnable-vs-fixed comparison. The TrackMate masks
follow individual nuclei, so absolute Dice drops but the **method comparison is
cleaner and the learnable advantage is larger**.

## 2. What the paper does (§5.3 / A.2.3 / B.0.1, Fig. 4)

- **Task:** binary segmentation as a representative downstream task.
- **Pseudo-GT masks (paper A.2.3):** normalize image → [0,1]; threshold **0.3**;
  morphological **closing** with a **(10,10)** kernel. *(This run substitutes the
  TrackMate detector of §1; the paper's recipe is preserved under
  `backup_thr0p3closing/` and `mask_mode: threshold_closing`.)*
- **Training (B.0.1):** (1) train content-aware end-to-end (learn illumination +
  reconstruction); (2) append a **small conv network** for segmentation and train
  it with the content-aware parameters **fixed**; (3) **finetune all components**
  end-to-end (illumination + inverse + segmentation).
- **Figure 4:** A) GT test images; B) pseudo-GT masks; C1–E2) segmentation for
  **×64 / ×256 / ×1024** × **pseudo-random / learnable Hᵗ**; F) representative
  illumination pattern. Claim: "The proposed method consistently generates better
  segmentation maps." (qualitative — no numeric table).

## 3. Audit verdict (detail in `AUDIT_CURRENT_SEGMENTATION.md`)

| Item | Verdict |
|---|---|
| Pseudo-GT masks — **TrackMate** (raw MIP > 506, 4-conn, resample ~2 px, DP 0.5, fill) | ✅ as specified |
| Stage 1 content-aware pretrain (L1, staged-hardening / inverse-only) | ✅ faithful |
| Stage 2 frozen seg-head (microscope frozen; asserted) | ✅ faithful |
| Stage 3 end-to-end finetune (illum+inverse+head; grads asserted) | ✅ faithful |
| Small conv seg head, raw logits + BCE(+soft Dice) | ✅ reasonable |
| ×64/×256/×1024 × {pseudo-random, learnable} | ✅ matches Fig. 4 |
| Threshold selected on **val**, applied to test | ✅ no test tuning |
| Dataset = BBBC022 Hoechst (not U2OS) | ⚠️ substitute (honest) |
| Mask = TrackMate detector, not paper A.2.3 recipe | ⚠️ deliberate substitution (§1) |
| Noise-free (matches the reused Stage-1 base) | ⚠️ documented choice |

## 4. Sanity gates — `sanity/SANITY_REPORT.md` (all PASS ✅, TrackMate masks)

| gate | result |
|---|---|
| mask_correctness | mask binary; IoU vs. raw MIP>506 = **0.95** (contours only perturb boundary); fg ≈ 0.22 |
| no_leakage_split | wells 168/21/21, **zero** overlap; split-by-well |
| distribution | images in [0,1] all splits; mask fg ≈ 0.17–0.19 |
| mask_not_in_input | forward consumes specimen only; responds to specimen, not mask |
| degenerate_baselines | Dice zeros=0.00, ones=0.32, perfect=1.00; post-hoc(x_recon>0.3)=0.83 |
| tiny_overfit | seg head overfits 4 samples to train Dice **0.952** |

## 5. Method (this verified re-run)

- **Stage 1 reused frozen** from `am2_task_aware_full/.../stage1_content_aware/best.pt`
  via `stage1_mode=load` (B.0.1 step 1 = "train content-aware"; the already-trained
  model is reused — Stage 1 is content-aware only and does not depend on the masks).
- **Stage 2** (1200 steps): microscope frozen, train seg head only. Runtime
  asserts illumination & inverse grads = 0, seg-head grad > 0.
- **Stage 3** (2500 steps): unfreeze seg head + inverse (+ illumination for the
  learnable variant); soft→hard sigmoid schedule (m=1 for 60%, then m∈{2,4,8}).
  Runtime asserts nonzero grads for all finetuned groups + illumination pattern Δ>0.
- Loss: `BCEWithLogits` (1.0) + soft Dice (0.5). Eval at sharpened m=10, threshold
  selected on validation (sweep 0.10–0.90), applied to test.
- Config: `configs/fig4_seg_fix_base.yaml`. Seed 42.

## 6. LIVE RESULTS — Dice / IoU (test, val-selected threshold)

_Auto-filled from `runs/metrics_summary.json` / `metrics/fig4_metrics.csv`._

<!-- RESULTS_TABLE -->

| compression | illumination | Dice | IoU | Dice@0.5 | thr |
|---|---|---:|---:|---:|---:|
| x64 | fixed | 0.8480 | 0.7375 | 0.8425 | 0.1 |
| x64 | learnable | 0.9039 | 0.8256 | 0.9050 | 0.2 |
| x256 | fixed | 0.6283 | 0.4598 | 0.6024 | 0.1 |
| x256 | learnable | 0.8150 | 0.6890 | 0.8112 | 0.1 |
| x1024 | fixed | 0.3959 | 0.2476 | 0.3637 | 0.1 |
| x1024 | learnable | 0.4583 | 0.2984 | 0.4456 | 0.2 |

| compression | fixed Dice | learnable Dice | Δ | learnable wins? |
|---|---:|---:|---:|:--:|
| x64 | 0.8480 | 0.9039 | +0.0560 | ✅ |
| x256 | 0.6283 | 0.8150 | +0.1867 | ✅ |
| x1024 | 0.3959 | 0.4583 | +0.0625 | ✅ |

## 7. Task-aware vs. content-aware attribution

The learnable illumination is optimized in two places: Stage 1 (content-aware)
and Stage 3 (task-aware finetune). Column "stage2 (content-aware) val Dice"
isolates how well a seg head does on the *frozen content-aware* reconstruction;
the final test Dice adds the Stage-3 task-aware finetuning. See
`metrics/fig4_metrics.csv` (`task_aware_gain_vs_stage2`).

<!-- ATTRIBUTION_TABLE -->

| compression | illumination | stage2 (content-aware) val Dice | final test Dice |
|---|---|---:|---:|
| x64 | fixed | 0.8672 | 0.8480 |
| x64 | learnable | 0.9133 | 0.9039 |
| x256 | fixed | 0.6837 | 0.6283 |
| x256 | learnable | 0.8453 | 0.8150 |
| x1024 | fixed | 0.3580 | 0.3959 |
| x1024 | learnable | 0.5128 | 0.4583 |

## 8. Acceptance criteria

<!-- ACCEPTANCE -->

| criterion | result |
|---|:--:|
| Learnable Hᵗ beats pseudo-random at ×64, ×256, ×1024 | ✅ |
| Learnable Hᵗ updated by the segmentation loss in Stage 3 (nonzero grad + pattern Δ) | ✅ |
| Outputs produced for all three compressions | ✅ |

**Status: `FIG4_FIX_PASS / RESULTS_PROXY_BBBC022`**

## 9. Figures & per-panel tiles

- `figures/figure4_full.png` — grayscale paper-layout: A (GT) / B (pseudo-GT) /
  C1–E2 (segmentation, 3 compressions × 2 illuminations) / F (patterns).
- `figures/figure4_full_viridis.png` — the same composite in the paper's viridis
  style (matches the published Figure 4 look).
- `figures/figure4_paper_layout_viridis.png` — the A/B/C1–E2 viridis grid;
  `figures/figure4_panel_F_viridis.png` — the six illumination patterns.
- `figures/figure4_dice_iou_bars.png` — Dice/IoU bars (pseudo-random vs learnable).
- **`figures/plot_tiles/`** — every panel as a standalone PNG for PowerPoint:
  `A_col{0..4}.png` (GT, display-stretched; `A_col{j}_raw.png` unstretched),
  `B_col{0..4}.png` (pseudo-GT), `{C1,C2,D1,D2,E1,E2}_col{0..4}.png` (predictions),
  `F_{C1..E2}.png` (patterns), and `tiles_manifest.json` (grid layout + thresholds).

## 10. Reproduce

```bash
PY=.venv/bin/python   # run from the replication/ directory (device: cuda:0)
# 1) sanity gates (must pass before training) — validates the TrackMate masks
$PY scripts/fig4_seg_fix_sanity.py --device cuda:0
# 2) verified re-run of the 6-cell matrix (Stage 2+3; Stage 1 reused frozen)
$PY scripts/fig4_seg_fix_train.py --device cuda:0 --seed 42
# 3) metrics CSV + paper-layout Figure 4 + bars + REPORT tables
$PY scripts/fig4_seg_fix_report.py --device cuda:0 --seed 42 --k 5
# 4) per-panel tiles + viridis composites for PowerPoint
$PY scripts/fig4_seg_fix_export_tiles.py --device cuda:0 --seed 42 --k 5
```

## 11. Limitations / honest scope

- **Substitute data:** BBBC022 Hoechst, not U2OS — absolute Dice/IoU are a proxy;
  only the direction (learnable > pseudo-random) is claimed.
- **Pseudo-GT, not manual annotation:** masks are a TrackMate-style raw-intensity
  detector (§1), so Dice measures agreement with a rule-based target, not expert
  labels. (This substitutes the paper's A.2.3 threshold+closing recipe.)
- **Noise-free** forward for Fig. 4 (kept consistent with the reused Stage-1 base).
- **Compute-bounded budgets** (1200 + 2500 steps) — a proxy; the paper gives no
  segmentation step counts.
- **Stage-1 reused frozen** (not retrained here) — a deliberate, documented choice.

## 12. Conclusion

<!-- CONCLUSION -->

The Figure 4 task-aware segmentation experiment reproduces the paper's qualitative claim on the BBBC022 substitute: learnable illumination beats fixed pseudo-random at every compression (Dice Δ ×64 +0.056, ×256 +0.187, ×1024 +0.062), and the segmentation loss provably reaches the illumination parameters in Stage 3. With the TrackMate-style pseudo-GT masks (raw MIP > 506, 4-connectivity, Douglas–Peucker-simplified contours) the targets follow individual nuclei rather than inflated blobs, so absolute Dice is lower than the legacy thr0.3+closing masks but the learnable-vs-fixed advantage is clearer. Verified with an independent audit, a passing sanity-gate suite, a self-contained re-run, and a clean paper-layout Figure 4 (+ per-panel tiles). Absolute numbers remain a substitute-data proxy (not U2OS).
