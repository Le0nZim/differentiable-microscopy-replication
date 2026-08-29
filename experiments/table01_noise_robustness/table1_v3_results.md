# Table 1 v3 results (corrected paper-normalized noise, `paper_v3`)

PatchMNIST ×8, T=8, d=8, batch=32, illum LR=1.0, inverse LR=0.001, gamma=10.
Detector model: supplement A.2.2 eqs. S5–S10 with `alpha_norm = alpha_down` (no /d²).
Seed 42 grid below; the extreme cell (pc=10, σ=6) additionally has seeds 43, 44.

| photon_count | sigma_read | random MSE | learnable MSE | learnable/random | learnable wins? |
|---|---:|---:|---:|---:|---|
| 10 | 0.0 | 0.0208 | 0.0064 | 0.310 | yes |
| 10 | 2.7 | 0.0211 | 0.0066 | 0.312 | yes |
| 10 | 2.0 | 0.0207 | 0.0066 | 0.317 | yes |
| 10 | 6.0 | 0.0230 | 0.0074 | 0.322 | yes |
| 10000 | 0.0 | 0.0109 | 0.0030 | 0.276 | yes |
| 10000 | 2.7 | 0.0109 | 0.0031 | 0.283 | yes |
| 10000 | 2.0 | 0.0109 | 0.0032 | 0.292 | yes |
| 10000 | 6.0 | 0.0110 | 0.0032 | 0.288 | yes |

## Extreme cell pc=10, σ=6 — multi-seed (training) robustness

| method | per-seed MSE | mean | std | n |
|---|---|---:|---:|---:|
| random_fixed | 0.0230, 0.0224, 0.0232 | 0.0229 | 0.0004 | 3 |
| learnable_frequency | 0.0074, 0.0074, 0.0074 | 0.0074 | 0.0000 | 3 |

## Extreme cell pc=10, σ=6 — eval-noise robustness (seed-42 checkpoint, 5 noise draws)

| method | mean | std | n |
|---|---:|---:|---:|
| random_fixed | 0.0229 | 0.0000 | 5 |
| learnable_frequency | 0.0074 | 0.0000 | 5 |

## Gate snapshot

- `all_tests_pass`: None
- `normalized_path_used_in_train_and_eval`: True
- `learnable_beats_fixed_all_8_cells`: True
- `pc10000_flat`: True
- `pc10_materially_flatter_than_v2`: False
- `extreme_cell_no_reversal_meanseed`: True
- `spread_pc10`: 0.0009850718003387255
- `spread_pc10000`: 0.00017517131830876087
- `v2_spread_pc10`: None
- `v2_spread_pc10000`: None
