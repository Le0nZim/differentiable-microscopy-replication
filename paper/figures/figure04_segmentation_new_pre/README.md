# Figure 4 — Segmentation-aware sampling (calibrated preprocessing)

- **Paper section:** 5.3
- **Status:** **data_blocked** (U2OS unavailable). Same claim (learnable > fixed) as the paper-strict folder; masks look cleaner. Absolute Dice is **not** comparable across preprocessings.
- **Not a clean-split `_fixed` twin.** This is a separate retrain on `bbbc022_calibrated` (percentile background subtract + p99.9) vs `paper_strict`.

## Run tree

`experiments/figure4_bbbc022_segmentation_calibrated_v1/`

Stage 1 is retrained from scratch on calibrated data (the frozen AM-2 paper-strict checkpoints are not reused).

## Reproduce

```bash
python scripts/fig4_seg_calibrated_train.py --device cuda:0
python scripts/fig4_seg_calibrated_report.py --device cuda:0
python paper/_build_components.py --only fig4_new_pre
```

Comparison vs paper-strict: `experiments/figure4_bbbc022_segmentation_calibrated_v1/COMPARISON_paper_strict_vs_calibrated.md`
