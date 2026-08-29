# Figure 4 — Segmentation-aware sampling

- **Paper section:** 5.3
- **Status:** **data_blocked** (U2OS unavailable; BBBC022 Hoechst substitute). Learnable Dice > fixed at x64 / x256 / x1024 on the substitute.
- **Canonical experiment:** `experiments/figure4_bbbc022_segmentation_fix_v1/` (`paper_strict` preprocessing).

## Run trees

- Figure / task-aware stages: `experiments/figure4_bbbc022_segmentation_fix_v1/`
- Frozen Stage-1 base (required): `experiments/task_aware_segmentation/am2_task_aware_full/`

## Reproduce

```bash
python scripts/fig4_seg_fix_train.py --device cuda:0 --seed 42
python scripts/fig4_seg_fix_sanity.py --device cuda:0
python scripts/fig4_seg_fix_report.py --device cuda:0 --seed 42
python paper/_build_components.py --only fig4
```

Metrics: `experiments/figure4_bbbc022_segmentation_fix_v1/metrics/fig4_metrics.csv`
