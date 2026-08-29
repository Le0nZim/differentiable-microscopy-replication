# Figure 4 — Task-aware (segmentation-aware) sampling: audit, verification & clean reproduction

**Status:** `FIG4_FIX_PASS / RESULTS_PROXY_BBBC022` — verified re-run complete; tables below are live.

BBBC022 Hoechst **substitute** proxy for the paper's U2OS segmentation-aware
experiment (§5.3, supplement A.2.3 + B.0.1, Figure 4). U2OS data are
unavailable, so absolute numbers are **not** comparable to the paper; only the
qualitative direction (learnable Hᵗ > pseudo-random Hᵗ) is claimed.

---

## 0. Headline

Applying "the same care" as the Figure 3 SwinIR fix, but Figure 4 started from a
very different place: the task-aware segmentation experiment was **already
implemented faithfully** in `experiments/figure04_segmentation/stage1_frozen/`
(real segmentation head + real segmentation task loss + the paper's three-stage
B.0.1 procedure, with results already in the paper's direction). It was **not**
broken like the SwinIR columns were. So this fix does not rescue a wrong result;
it wraps the existing faithful method in the same rigor:

1. an **independent audit** against the paper (`AUDIT_CURRENT_SEGMENTATION.md`)
   and a paper-vs-implementations **diff** (`PAPER_VS_CURRENT_FIG4_DIFF.md`);
2. a standalone **sanity-gate suite** (`sanity/`) — all gates pass;
3. a **self-contained verified re-run** into this isolated directory (Stage 2+3
   recomputed on top of the frozen content-aware Stage-1 base);
4. a clean **paper-layout Figure 4** (A / B / C1–E2 / F) + quantitative CSV;
5. this advisor-ready report with explicit **acceptance criteria** and an
   explicit **task-aware vs. content-aware attribution**.

Nothing in the frozen `am2` run was modified.

## 1. What the paper does (§5.3 / A.2.3 / B.0.1, Fig. 4)

- **Task:** binary segmentation as a representative downstream task.
- **Pseudo-GT masks (A.2.3):** normalize image → [0,1]; threshold **0.3**;
  morphological **closing** with a **(10,10)** kernel.
- **Training (B.0.1):** (1) train content-aware end-to-end (learn illumination +
  reconstruction); (2) append a **small conv network** for segmentation and
  train it with the content-aware parameters **fixed**; (3) **finetune all
  components** end-to-end (illumination + inverse + segmentation).
- **Figure 4:** A) GT test images; B) pseudo-GT masks; C1–E2) segmentation for
  **×64 / ×256 / ×1024** × **pseudo-random / learnable Hᵗ**; F) representative
  illumination pattern. Claim: "The proposed method consistently generates
  better segmentation maps." (qualitative — no numeric table).

## 2. Audit verdict (detail in `AUDIT_CURRENT_SEGMENTATION.md`)

| Item | Verdict |
|---|---|
| Pseudo-GT masks (thr 0.3 + closing 10, from GT image) | ✅ faithful |
| Stage 1 content-aware pretrain (L1, staged-hardening / inverse-only) | ✅ faithful |
| Stage 2 frozen seg-head (microscope frozen; asserted) | ✅ faithful |
| Stage 3 end-to-end finetune (illum+inverse+head; grads asserted) | ✅ faithful |
| Small conv seg head, raw logits + BCE(+soft Dice) | ✅ reasonable |
| ×64/×256/×1024 × {pseudo-random, learnable} | ✅ matches Fig. 4 |
| Threshold selected on **val**, applied to test | ✅ no test tuning |
| Dataset = BBBC022 Hoechst (not U2OS) | ⚠️ substitute (honest) |
| Noise-free (matches the reused Stage-1 base) | ⚠️ documented choice |

## 3. Sanity gates — `sanity/SANITY_REPORT.md` (all PASS ✅)

| gate | result |
|---|---|
| mask_correctness | mask == threshold(0.3)+closing(10) of paired GT; binary; fg ≈ 0.22 |
| no_leakage_split | wells 168/21/21, **zero** overlap; split-by-well |
| distribution | images in [0,1] all splits; mask fg ≈ 0.24–0.27 |
| mask_not_in_input | forward consumes specimen only; responds to specimen, not mask |
| degenerate_baselines | Dice zeros=0.00, ones=0.42, perfect=1.00; post-hoc(x_recon>0.3)=0.93 |
| tiny_overfit | seg head overfits 4 samples to train Dice **0.956** |

## 4. Method (this verified re-run)

- **Stage 1 reused frozen** from `am2_task_aware_full/.../stage1_content_aware/best.pt`
  via `stage1_mode=load` (B.0.1 step 1 = "train content-aware"; the already-trained
  model is reused, exactly as the Fig. 3 SwinIR fix reused its frozen base).
- **Stage 2** (1200 steps): microscope frozen, train seg head only. Runtime
  asserts illumination & inverse grads = 0, seg-head grad > 0.
- **Stage 3** (2500 steps): unfreeze seg head + inverse (+ illumination for the
  learnable variant); soft→hard sigmoid schedule (m=1 for 60%, then m∈{2,4,8}).
  Runtime asserts nonzero grads for all finetuned groups + illumination pattern Δ>0.
- Loss: `BCEWithLogits` (1.0) + soft Dice (0.5). Eval at sharpened m=10, threshold
  selected on validation.
- Config: `configs/fig4_seg_fix_base.yaml`. Seed 42.

## 5. LIVE RESULTS — Dice / IoU (test, val-selected threshold)

_Auto-filled from `runs/metrics_summary.json` / `metrics/fig4_metrics.csv`._

<!-- RESULTS_TABLE -->

| compression | illumination | Dice | IoU | Dice@0.5 | thr |
|---|---|---:|---:|---:|---:|
| x64 | fixed | 0.9007 | 0.8201 | 0.9005 | 0.35 |
| x64 | learnable | 0.9272 | 0.8652 | 0.9250 | 0.9 |
| x256 | fixed | 0.7196 | 0.5627 | 0.6910 | 0.1 |
| x256 | learnable | 0.8358 | 0.7194 | 0.8303 | 0.9 |
| x1024 | fixed | 0.5000 | 0.3344 | 0.4233 | 0.1 |
| x1024 | learnable | 0.5245 | 0.3621 | 0.5062 | 0.1 |

| compression | fixed Dice | learnable Dice | Δ | learnable wins? |
|---|---:|---:|---:|:--:|
| x64 | 0.9007 | 0.9272 | +0.0265 | ✅ |
| x256 | 0.7196 | 0.8358 | +0.1162 | ✅ |
| x1024 | 0.5000 | 0.5245 | +0.0245 | ✅ |

## 6. Task-aware vs. content-aware attribution

The learnable illumination is optimized in two places: Stage 1 (content-aware)
and Stage 3 (task-aware finetune). Column "stage2 (content-aware) val Dice" in
`report_tables.md` isolates how well a seg head does on the *frozen
content-aware* reconstruction; the final test Dice adds the Stage-3 task-aware
finetuning. See `metrics/fig4_metrics.csv` (`task_aware_gain_vs_stage2`).

<!-- ATTRIBUTION_TABLE -->

| compression | illumination | stage2 (content-aware) val Dice | final test Dice |
|---|---|---:|---:|
| x64 | fixed | 0.9085 | 0.9007 |
| x64 | learnable | 0.9346 | 0.9272 |
| x256 | fixed | 0.7453 | 0.7196 |
| x256 | learnable | 0.8862 | 0.8358 |
| x1024 | fixed | 0.4500 | 0.5000 |
| x1024 | learnable | 0.5893 | 0.5245 |

## 7. Acceptance criteria

<!-- ACCEPTANCE -->

| criterion | result |
|---|:--:|
| Learnable Hᵗ beats pseudo-random at ×64, ×256, ×1024 | ✅ |
| Learnable Hᵗ updated by the segmentation loss in Stage 3 (nonzero grad + pattern Δ) | ✅ |
| Outputs produced for all three compressions | ✅ |

**Status: `FIG4_FIX_PASS / RESULTS_PROXY_BBBC022`**

## 8. Figures

- `figures/figure4_full.png` — paper-layout: A (GT) / B (pseudo-GT) / C1–E2
  (segmentation, 3 compressions × 2 illuminations) / F (illumination patterns).
- `figures/figure4_paper_layout.png` — the A/B/C1–E2 grid only.
- `figures/figure4_panel_F_illumination.png` — representative Hᵗ patterns.
- `figures/figure4_dice_iou_bars.png` — Dice/IoU bars (pseudo-random vs learnable).
- Per-cell 5-column panels: `runs/taskaware_*/figures/qualitative_panel.png`.

## 9. Reproduce

```bash
PY=python
# 1) sanity gates (must pass before training)
CUDA_VISIBLE_DEVICES=0 $PY scripts/figure04_segmentation/sanity.py --device cuda:0
# 2) verified re-run of the 6-cell matrix (Stage 2+3; Stage 1 reused frozen)
CUDA_VISIBLE_DEVICES=0 $PY scripts/figure04_segmentation/train.py --device cuda:0 --seed 42
# 3) metrics CSV + paper-layout Figure 4 + bars
CUDA_VISIBLE_DEVICES=0 $PY scripts/figure04_segmentation/report.py --device cuda:0 --seed 42
```

## 10. Limitations / honest scope

- **Substitute data:** BBBC022 Hoechst, not U2OS — absolute Dice/IoU are a proxy;
  only the direction (learnable > pseudo-random) is claimed.
- **Pseudo-GT, not manual annotation:** masks are threshold+closing of the GT
  image (paper's own A.2.3 recipe), so Dice measures agreement with a rule-based
  target, not expert labels.
- **Noise-free** forward for Fig. 4 (kept consistent with the reused Stage-1 base).
- **Compute-bounded budgets** (1200 + 2500 steps) — a proxy; the paper gives no
  segmentation step counts.
- **Stage-1 reused frozen** (not retrained here) — a deliberate, documented choice.

## 11. Conclusion

<!-- CONCLUSION -->

The Figure 4 task-aware segmentation experiment reproduces the paper's qualitative claim on the BBBC022 substitute: learnable illumination beats fixed pseudo-random at every compression (Dice Δ ×64 +0.027, ×256 +0.116, ×1024 +0.024), and the segmentation loss provably reaches the illumination parameters in Stage 3. The method was already faithful in `am2_task_aware_full`; this isolated fix confirms it with an independent audit, a passing sanity-gate suite, a self-contained verified re-run, and a clean paper-layout Figure 4. Absolute numbers remain a substitute-data proxy (not U2OS).
