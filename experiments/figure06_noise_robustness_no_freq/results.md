# Figure 6 / Table 1 noise robustness — no frequency-domain optimization

PatchMNIST ×8, T=8, d=8, batch=32, illum LR=1.0, inverse LR=0.001, gamma=10, `paper_v3` detector normalization — identical to Table 1.
`learnable_spatial` runs are new; `random_fixed` and `learnable_frequency` are read from the frozen `experiments/table01_noise_robustness/` runs.

| photon_count | sigma_read | fixed MSE | learnable+freq MSE | learnable NO-freq MSE | no-freq / freq | no-freq wins? |
|---|---:|---:|---:|---:|---:|---|
| 10 | 0.0 | 0.0208 | 0.0064 | 0.0049 | 0.767 | yes |
| 10 | 6.0 | 0.0230 | 0.0074 | 0.0057 | 0.771 | yes |
| 10000 | 0.0 | 0.0109 | 0.0030 | 0.0029 | 0.979 | yes |
| 10000 | 6.0 | 0.0110 | 0.0032 | 0.0029 | 0.929 | yes |

## Extreme cell pc=10, σ=6 — multi-seed (training) robustness

| method | per-seed MSE | mean | std | n |
|---|---|---:|---:|---:|
| fixed random | 0.0230, 0.0224, 0.0232 | 0.0229 | 0.0004 | 3 |
| learnable + freq (paper) | 0.0074, 0.0074, 0.0074 | 0.0074 | 0.0000 | 3 |
| learnable, NO freq | 0.0057, 0.0057, 0.0057 | 0.0057 | 0.0000 | 3 |

## Extreme cell pc=10, σ=6 — eval-noise robustness (seed-42 checkpoint, 5 noise draws)

| method | mean | std | n |
|---|---:|---:|---:|
| fixed random | 0.0229 | 0.0000 | 5 |
| learnable + freq (paper) | 0.0074 | 0.0000 | 5 |
| learnable, NO freq | 0.0057 | 0.0000 | 5 |

## Gate snapshot

- `cells_compared`: 4
- `spatial_beats_frequency_all_cells`: True
- `spatial_beats_random_all_cells`: True
- `spatial_beats_frequency_at_extreme_cell_meanseed`: True
- `median_ratio_spatial_over_frequency`: 0.8500163020415619
