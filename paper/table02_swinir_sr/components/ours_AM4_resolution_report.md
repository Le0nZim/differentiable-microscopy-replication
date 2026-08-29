# AM-4 SwinIR Table-2 — `full` run summary

**Checkpoint:** best-val. **Effective batch:** 32. **All finite:** True.

| Dataset | w/o LI PSNR | with LI PSNR | LI gain PSNR | w/o LI SSIM | with LI SSIM | LI gain SSIM | paper LI gain PSNR |
|---|---|---|---|---|---|---|---|
| Set5 | 25.55 | 27.45 | +1.90 | 0.7201 | 0.7768 | +0.0567 | 12.71 |
| Set14 | 23.62 | 25.17 | +1.55 | 0.6379 | 0.6882 | +0.0503 | 9.96 |
| BSD100 | 23.91 | 25.12 | +1.21 | 0.5902 | 0.6279 | +0.0377 | 8.62 |
| Urban100 | 22.55 | 23.90 | +1.34 | 0.5833 | 0.6434 | +0.0601 | 8.00 |
| Manga109 | 24.21 | 26.09 | +1.88 | 0.7224 | 0.7799 | +0.0576 | 8.09 |

- **swinir_wo_li**: iters 20000/20000, best_val_L1 0.05730@20000, embed_dim 180, illum learnable=False (0 params)

- **swinir_with_li**: iters 20000/20000, best_val_L1 0.04814@15000, embed_dim 180, illum learnable=True (16384 params)

**Deviations / assumptions:** see `aggregate_summary.json` and the config `deviations_from_paper`.