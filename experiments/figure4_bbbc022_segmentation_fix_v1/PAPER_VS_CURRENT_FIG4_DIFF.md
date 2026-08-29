# Figure 4 — paper protocol vs. implementations (side-by-side)

Analogue of the Figure 3 `TABLE2_FIG7_VS_FIG3_SWINIR_DIFF.md`. Instead of
diffing a "working" pipeline against a "broken" one, Figure 4 has already been
migrated from a broken **post-hoc** approach to a faithful **task-aware** one.
This table makes the migration and the remaining fix scope explicit.

Columns:
- **Paper (B.0.1 / A.2.3 / §5.3)** — what the paper requires.
- **Legacy post-hoc** (`experiments/task_aware_segmentation/bbbc022_segmentation/`)
  — the original approach: train an L1 reconstruction and call `x_recon > 0.3`
  the "segmentation". **This was the broken analogue.**
- **Current faithful** (`experiments/task_aware_segmentation/am2_task_aware_full/`)
  — the real three-stage task-aware pipeline. **Already correct.**
- **This isolated fix** (`figure4_bbbc022_segmentation_fix_v1/`) — what we add.

| Dimension | Paper (B.0.1/A.2.3/§5.3) | Legacy post-hoc (broken) | Current faithful (`am2`) | This isolated fix |
|---|---|---|---|---|
| Segmentation predictor | small **conv network** appended to reconstruction | ❌ none — threshold `x_recon > 0.3` | ✅ `SegmentationHead` (conv, raw logits) | reuse (verified) |
| Training loss | supervised on pseudo-GT masks | ❌ L1 recon only; `segmentation_bce_weight` **dead** | ✅ BCEWithLogits + soft Dice | reuse (verified) |
| Pseudo-GT masks | A.2.3: normalize→thr 0.3→closing (10,10), from GT image | ✅ same recipe (eval only) | ✅ A.2.3 recipe, used as **training target** | ✅ **TrackMate detector** (raw MIP>506, 4-conn, DP ε0.5) replaces A.2.3; training+eval target; sanity-checked (A.2.3 kept as fallback) |
| Stage 1 (content-aware pretrain) | train end-to-end content-aware (Fig. 1) | partial (recon only, no seg) | ✅ staged-hardening (learnable) / inverse-only (fixed) | **reuse frozen ckpt** |
| Stage 2 (frozen head) | append + train head, content-aware **fixed** | ❌ absent | ✅ microscope frozen; head-only; freeze asserted | re-run (verified) |
| Stage 3 (end-to-end finetune) | finetune illumination + inverse + head | ❌ absent | ✅ all unfrozen; soft→hard schedule; grads asserted | re-run (verified) |
| Learnable illumination gets task gradient | implied by "finetune all components" | ❌ n/a | ✅ nonzero Stage-3 illum grad; pattern Δ 21–41% | re-verify |
| Compressions | ×64, ×256, ×1024 | ×64/×256/×1024 | ✅ ×64/×256/×1024 | same |
| Illuminations | pseudo-random vs learnable | pseudo-random vs learnable | ✅ `random_fixed` vs `learnable_frequency` | same |
| Threshold selection | (not specified) | fixed 0.3 | ✅ **val-selected**, applied to test | reuse (no test tuning) |
| Figure 4 (A/B/C1–E2/F) | shared GT + pseudo-mask rows, 3×2 seg grid, illum | ❌ per-run strips + bar chart | ⚠️ per-run strips + bar chart (no paper-layout grid) | ✅ **paper-layout panel (gray + viridis) + per-panel tiles for PowerPoint** |
| Sanity-gate suite | — | ❌ none | ⚠️ runtime asserts only | ✅ **standalone suite** |
| Advisor-ready report | — | `figure4_style_report.md` (post-hoc) | `report.md` (tables) | ✅ **narrative + acceptance** |

## What changed to get from "broken" → "faithful" (already done in `am2`)

1. Added a real segmentation head that emits **raw logits** (was: none / post-hoc threshold).
2. Added a **segmentation task loss** (BCEWithLogits + soft Dice) actually used in the loop (was: dead config).
3. Implemented the full **three-stage** B.0.1 schedule with freeze/gradient evidence (was: recon-only).
4. Made the pseudo-GT mask the **training target** from the GT image (was: only used for eval).
5. Selected the decision threshold on **validation** (was: fixed 0.3).

## What this isolated fix adds (the remaining "same care")

1. Independent audit (`AUDIT_CURRENT_SEGMENTATION.md`) + this diff.
2. A standalone **sanity-gate suite** (`sanity/`): mask correctness/pairing,
   split no-leakage, tiny-overfit capability, degenerate/identity baselines.
3. A **self-contained verified re-run** of Stage 2+3 (from the frozen Stage-1
   content-aware checkpoints) into this directory, confirming the numbers.
4. A clean **paper-layout Figure 4** (A/B/C1–E2/F), in both grayscale and the
   paper's viridis style, plus **per-panel image tiles** (`figures/plot_tiles/`)
   for rebuilding the figure in PowerPoint, and a quantitative Dice/IoU CSV + bars.
5. An upgraded **TrackMate-style pseudo-GT mask** (raw MIP>506, 4-connectivity,
   Douglas–Peucker ε0.5) that tracks individual nuclei, with the A.2.3 recipe kept
   as a fallback and preserved under `backup_thr0p3closing/`.
6. An advisor-ready **REPORT.md** with explicit acceptance criteria and an
   explicit Stage-2 (content-aware) vs final (task-aware) attribution.

## Fix strategy (one line)

Treat `am2_task_aware_full` as the trusted reference (like Table 2/Fig 7 was for
Fig 3), reuse its frozen content-aware Stage-1 base, and only rebuild the
missing rigor around it (sanity suite, isolated verified re-run, paper-layout
figure, advisor report) — changing no faithful behavior.
