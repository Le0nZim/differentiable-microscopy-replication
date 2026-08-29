# Table 3 — Ablation study A/B/C/D (U2OS x16, T=4)

- **Paper ID / folder:** `table03_ablation`
- **Paper section:** 5.7
- **Type:** table
- **Datasets:** U2OS (paper) — UNAVAILABLE; BBBC022 Cell-Painting Hoechst substitute (ours)
- **Settings:** x16 compression, T=4, downscaling 8x8:1
- **Metrics:** SSIM, MSE
- **Replication status:** **data_blocked**
- **Qualitative companion:** [Figure 10](../figure10_ablation/)

> Ablation over A (baseline fixed Ht + transpose-conv up), B (+learnable Ht), C (+proposed locality upsampling), D (-frequency-domain optimization). Paper: C is best.

## Run trees

- Multi-seed numbers: `experiments/table03_ablation/`
- Qualitative / newest ordering: `experiments/figure10_ablation/`

## Reproduce

```bash
python scripts/table03_ablation/run.py --device cuda:0
python scripts/table03_ablation/aggregate.py
```

## Files in this folder

- `paper_expected.md` — what the paper reports for this item
- `our_replication.md` — BBBC022 proxy numbers
- `comparison.md` — side-by-side paper vs ours
- `provenance.md` — exact source files used
- `missing_components.md` — U2OS unavailable
- `paper_values.csv` / `our_values.csv` / `comparison.csv`
- `rendered/` — paper-ready markdown/csv
- `components/` — root-cause writeup, per-seed CSV, data-status note

## Summary

The A/B/C/D wiring is proven faithful by 23 machine-checkable tests and a per-run audit (locality block, frequency path, custom-sigmoid schedule all verified). On a PatchMNIST sanity check the proposed locality block *wins* (C 0.00697 < B 0.00947). The paper's U2OS DAPI-confocal dataset is unavailable, so Table 3 cannot be numerically reproduced.

On the BBBC022 substitute proxy (3 seeds) the ordering inverts (best B, C worst), diagnosed as the high-capacity locality upsampler overfitting the lower-diversity widefield proxy.
