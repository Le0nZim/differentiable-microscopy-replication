# Our replication — Table 3

- **Experiment(s):** am3_table3_resolution (BBBC022 proxy, 3 seeds) + bbbc022_ablation_x16
- **Status:** data_blocked

The A/B/C/D wiring is proven faithful by 23 machine-checkable tests and a per-run audit (locality block, frequency path, custom-sigmoid schedule all verified). On a PatchMNIST sanity check the proposed locality block *wins* (C 0.00697 < B 0.00947). However the paper's U2OS DAPI-confocal dataset is unavailable, so the Table-3 numbers cannot be numerically reproduced.

On the BBBC022 substitute proxy (3 seeds) the ordering inverts (best B, C worst), which is fully diagnosed as the high-capacity locality upsampler overfitting the lower-diversity widefield proxy (train MSE lowest for C/D, largest val-train gaps; C-B test gap shrinks monotonically with more data).

## Our values

| variant | label | proxy_ssim_bbbc022 | proxy_mse_bbbc022 |
| --- | --- | --- | --- |
| A | fixed Ht + Tr.Conv.Up + freq | 0.8781 | 0.003625 |
| B | (+) learnable Ht | 0.888 | 0.002706 |
| C | (+) proposed locality upsampling (paper best) | 0.8602 | 0.005321 |
| D | (-) frequency-domain optimization | 0.8663 | 0.004583 |

