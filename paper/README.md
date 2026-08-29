# Paper-facing tables and figure parts

This tree is the published bookkeeping of the main-paper tables and constituent figure assets.

Where a `*_fixed` sibling existed (clean val/test split), **only the fixed folder is kept**.

| Path | Paper item |
|---|---|
| `tables/table01_noise_robustness_fixed/` | Table 1 (shared run with Fig. 6) |
| `tables/table02_swinir_sr_fixed/` | Table 2 (shared run with Fig. 7) |
| `tables/table03_ablation/` | Table 3 (companion Fig. 10) |
| `figures/figure03_content_aware/` | Figure 3 |
| `figures/figure04_segmentation/` | Figure 4 (`paper_strict`) |
| `figures/figure04_segmentation_new_pre/` | Figure 4 (calibrated preprocessing) |
| `figures/figure05_upsampling_fixed/` | Figure 5 |
| `figures/figure06_noise_robustness_fixed/` | Figure 6 |
| `figures/figure07_swinir_standard_sr_fixed/` | Figure 7 |
| `figures/figure08_mcf7_swinir/` | Figure 8 |
| `figures/figure09_mcf7_widefield/` | Figure 9 |
| `figures/figure10_ablation/` | Figure 10 |

Atomic figure parts can be rebuilt (GPU + data) with:

```bash
python paper/_build_components.py
python paper/_build_components.py --only fig4_new_pre
```

Checkpoints and logs live under `experiments/` at the repo root, not here.
