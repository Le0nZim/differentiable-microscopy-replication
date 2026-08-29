# Paper-facing tables and figure parts

Each folder is one paper item. The folder name is the **slug** used in
`scripts/`, `configs/`, and `experiments/` as well.

Atomic figure parts can be rebuilt (GPU + data) with:

```bash
python paper/_build_components.py
```

`--only fig3` (etc.) wipes and rebuilds only that figure slug. Table folders
are never deleted by a rebuild.

Checkpoints and logs live under `experiments/` at the repo root, not here.

## Catalog

| Path | Paper item | Shared run |
|---|---|---|
| [`table01_noise_robustness/`](table01_noise_robustness/) | Table 1 | Figure 6 |
| [`table02_swinir_sr/`](table02_swinir_sr/) | Table 2 | Figure 7 |
| [`table03_ablation/`](table03_ablation/) | Table 3 | companion Figure 10 (separate tree) |
| [`figure03_content_aware/`](figure03_content_aware/) | Figure 3 | |
| [`figure04_segmentation/`](figure04_segmentation/) | Figure 4 | |
| [`figure05_upsampling/`](figure05_upsampling/) | Figure 5 | |
| [`figure06_noise_robustness/`](figure06_noise_robustness/) | Figure 6 | Table 1 |
| [`figure07_swinir_sr/`](figure07_swinir_sr/) | Figure 7 | Table 2 |
| [`figure08_mcf7_swinir/`](figure08_mcf7_swinir/) | Figure 8 | Figure 9 |
| [`figure09_mcf7_widefield/`](figure09_mcf7_widefield/) | Figure 9 | Figure 8 |
| [`figure10_ablation/`](figure10_ablation/) | Figure 10 | companion Table 3 (separate tree) |

## Old → new paths

| Old | New |
|---|---|
| `paper/tables/table01_noise_robustness_fixed/` | `paper/table01_noise_robustness/` |
| `paper/tables/table02_swinir_sr_fixed/` | `paper/table02_swinir_sr/` |
| `paper/tables/table03_ablation/` | `paper/table03_ablation/` |
| `paper/figures/figure03_content_aware/` | `paper/figure03_content_aware/` |
| `paper/figures/figure04_segmentation/` | `paper/figure04_segmentation/` |
| `paper/figures/figure05_upsampling_fixed/` | `paper/figure05_upsampling/` |
| `paper/figures/figure06_noise_robustness_fixed/` | `paper/figure06_noise_robustness/` |
| `paper/figures/figure07_swinir_standard_sr_fixed/` | `paper/figure07_swinir_sr/` |
| `paper/figures/figure08_mcf7_swinir/` | `paper/figure08_mcf7_swinir/` |
| `paper/figures/figure09_mcf7_widefield/` | `paper/figure09_mcf7_widefield/` |
| `paper/figures/figure10_ablation/` | `paper/figure10_ablation/` |
| `experiments/noise_robustness/rr1_v3_normalized_full_fixed/` | `experiments/table01_noise_robustness/` |
| `experiments/swinir_or_highres/am4_swinir_table2_resolution_fixed/` | `experiments/table02_swinir_sr/` |
| `experiments/ablations/am3_table3_resolution/` | `experiments/table03_ablation/` |
| `experiments/ablations/bbbc022_content_aware_v2/` | `experiments/figure03_content_aware/base/` |
| `experiments/figure3_bbbc022_swinir_fix_v1/` | `experiments/figure03_content_aware/swinir/` |
| `experiments/figure4_bbbc022_segmentation_fix_v1/` | `experiments/figure04_segmentation/task_aware/` |
| `experiments/task_aware_segmentation/am2_task_aware_full/` | `experiments/figure04_segmentation/stage1_frozen/` |
| `experiments/upsampling_ablation/patchmnist_upsampling_analysis_fixed/` | `experiments/figure05_upsampling/` |
| `experiments/figure89_mcf7_swinir_highres_fix_v1/` | `experiments/figure08_mcf7/` |
| `experiments/figure10_bbbc022_ablation_v1/` | `experiments/figure10_ablation/` |
