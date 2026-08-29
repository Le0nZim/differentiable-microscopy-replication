# Figure 4 — preprocessing ablation: `paper_strict` vs `bbbc022_calibrated`

Same task-aware segmentation regime (paper §5.3 / A.2.3 / B.0.1), same architecture,
same seed (42), same step budgets. **The only change is dataset preprocessing.**
Because the preprocessing changed, all three stages (content-aware pretrain →
frozen seg-head → end-to-end task-aware finetune) were **retrained from scratch**
on calibrated data (`stage1_mode=train`) — the paper_strict am2 Stage-1 base was
NOT reused. BBBC022 Hoechst substitute proxy (not U2OS); absolute metrics are
illustrative, not a literal paper reproduction.

- `paper_strict`      = `(raw − 134.28)` clipped to `[0, 500]`, then min–max to `[0,1]`.
- `bbbc022_calibrated`= percentile background subtraction (p1) + p99.9 clip, then min–max.

## Test Dice / IoU (val-selected threshold)

| compression | illumination | Dice (calibrated) | Dice (paper_strict) | IoU (calibrated) | IoU (paper_strict) |
|---|---|---:|---:|---:|---:|
| x64   | pseudo-random | 0.8482 | 0.9007 | 0.7385 | 0.8201 |
| x64   | learnable     | 0.8932 | 0.9272 | 0.8085 | 0.8652 |
| x256  | pseudo-random | 0.5434 | 0.7196 | 0.3782 | 0.5627 |
| x256  | learnable     | 0.8211 | 0.8358 | 0.6985 | 0.7194 |
| x1024 | pseudo-random | 0.4087 | 0.5000 | 0.2576 | 0.3344 |
| x1024 | learnable     | 0.4516 | 0.5245 | 0.2942 | 0.3621 |

## Learnable illumination beats fixed pseudo-random at every compression (both preprocessings)

| compression | Δ Dice (calibrated) | Δ Dice (paper_strict) |
|---|---:|---:|
| x64   | +0.0450 | +0.0265 |
| x256  | **+0.2777** | +0.1162 |
| x1024 | +0.0429 | +0.0245 |

The paper's central claim (learned illumination > fixed pseudo-random) holds under
both preprocessings, and the margin is **larger** with calibrated preprocessing at
x256 (+0.28 vs +0.12).

## Why the calibrated Dice is numerically lower — and why the masks are still better

The pseudo-ground-truth mask is derived from the *preprocessed* image (normalize →
threshold 0.3 → morphological closing). The two preprocessings therefore produce
**different targets**, so absolute Dice is not directly comparable across them:

- Under `paper_strict`, Hoechst nuclei saturate to 1.0 and bloom together. Both the
  pseudo-GT target and the predictions become large merged blobs — numerically
  "easy" to overlap (high Dice) but poor instance delineation ("noisy"/blobby masks
  the advisor flagged).
- Under `bbbc022_calibrated`, nuclei keep internal texture and stay separated. The
  target is tighter and the masks delineate individual nuclei; overlap Dice is a bit
  lower, but the masks are visibly cleaner and less over-segmented (see
  `figures/compare_mask_quality.png`).

So calibrated preprocessing fixes the two issues raised: (1) the GT no longer looks
"too fluorescent" — nuclei show texture instead of saturating (`figures/compare_gt_appearance.png`);
(2) the x256 / x1024 masks are less flooded and better separated. The x1024 case
(4 measurements per 64×64 super-pixel) remains inherently ill-posed for both
preprocessings.

## Artifacts
- Runs: `runs/taskaware_{x64,x256,x1024}_{random_fixed,learnable_frequency}_seed42/`
- Metrics CSV: `metrics/fig4_metrics.csv`
- Visual comparisons: `figures/compare_gt_appearance.png`, `figures/compare_mask_quality.png`
- Constituent SVGs (PowerPoint): `final_components/1_constituent_components/figure04_segmentation_new_pre/`
