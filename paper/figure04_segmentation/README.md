# Figure 4 — Segmentation-aware sampling

- **Paper section:** 5.3
- **Status:** **data_blocked** (U2OS unavailable; BBBC022 Hoechst substitute). Learnable Dice > fixed at x64 / x256 / x1024 on the substitute.
- **Canonical experiment:** `experiments/figure04_segmentation/task_aware/` (`paper_strict` preprocessing).

## Run trees

- Figure / task-aware stages: `experiments/figure04_segmentation/task_aware/`
- Frozen Stage-1 base (required): `experiments/figure04_segmentation/stage1_frozen/`

## Reproduce

```bash
python scripts/figure04_segmentation/train.py --device cuda:0 --seed 42
python scripts/figure04_segmentation/sanity.py --device cuda:0
python scripts/figure04_segmentation/report.py --device cuda:0 --seed 42
python paper/_build_components.py --only fig4
```

Metrics: `experiments/figure04_segmentation/task_aware/metrics/fig4_metrics.csv`
