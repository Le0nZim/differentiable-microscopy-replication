# Differentiable Microscopy — main tables and figures

Standalone reproduction package for the **main-paper tables and figures** of
*Differentiable Microscopy for Content and Task Aware Compressive Fluorescence Imaging*
([arXiv:2203.14945](https://arxiv.org/abs/2203.14945)).

Paper-facing numbers and SVG parts live under [`paper/`](paper/). Full run trees
(including checkpoints) live under [`experiments/`](experiments/), stored once and
cross-linked when two paper items share a run (Table 1 ↔ Fig. 6, Table 2 ↔ Fig. 7,
Fig. 8 ↔ Fig. 9).

Where a `*_fixed` variant existed, **only that folder is kept** (clean val/test
splits). `figure04_segmentation_new_pre` is a calibrated-preprocessing retrain, not
a `*_fixed` twin; both Fig. 4 folders are included.

## Catalog

| Item | Status | Paper-facing | Run tree |
|---|---|---|---|
| Table 1 | close | [`paper/tables/table01_noise_robustness_fixed`](paper/tables/table01_noise_robustness_fixed) | `experiments/noise_robustness/rr1_v3_normalized_full_fixed` |
| Table 2 | close | [`paper/tables/table02_swinir_sr_fixed`](paper/tables/table02_swinir_sr_fixed) | `experiments/swinir_or_highres/am4_swinir_table2_resolution_fixed` |
| Table 3 | data_blocked | [`paper/tables/table03_ablation`](paper/tables/table03_ablation) | `experiments/ablations/am3_table3_resolution` |
| Figure 3 | data_blocked | [`paper/figures/figure03_content_aware`](paper/figures/figure03_content_aware) | `experiments/ablations/bbbc022_content_aware_v2` + `experiments/figure3_bbbc022_swinir_fix_v1` |
| Figure 4 | data_blocked | [`paper/figures/figure04_segmentation`](paper/figures/figure04_segmentation) | `experiments/figure4_bbbc022_segmentation_fix_v1` + frozen `experiments/task_aware_segmentation/am2_task_aware_full` |
| Figure 4 (calibrated) | data_blocked | [`paper/figures/figure04_segmentation_new_pre`](paper/figures/figure04_segmentation_new_pre) | `experiments/figure4_bbbc022_segmentation_calibrated_v1` |
| Figure 5 | aligned | [`paper/figures/figure05_upsampling_fixed`](paper/figures/figure05_upsampling_fixed) | `experiments/upsampling_ablation/patchmnist_upsampling_analysis_fixed` |
| Figure 6 | close | [`paper/figures/figure06_noise_robustness_fixed`](paper/figures/figure06_noise_robustness_fixed) | same as Table 1 |
| Figure 7 | close | [`paper/figures/figure07_swinir_standard_sr_fixed`](paper/figures/figure07_swinir_standard_sr_fixed) | same as Table 2 |
| Figure 8 | close | [`paper/figures/figure08_mcf7_swinir`](paper/figures/figure08_mcf7_swinir) | `experiments/figure89_mcf7_swinir_highres_fix_v1` |
| Figure 9 | close | [`paper/figures/figure09_mcf7_widefield`](paper/figures/figure09_mcf7_widefield) | same as Figure 8 |
| Figure 10 | data_blocked | [`paper/figures/figure10_ablation`](paper/figures/figure10_ablation) | `experiments/figure10_bbbc022_ablation_v1` |

**data_blocked** means the paper's U2OS confocal dataset is unavailable. Those items use a BBBC022 Hoechst widefield substitute and are not numerically comparable to the paper.

Tables 1–2 and Figs 5–7 labelled `_fixed` use leakage-safer splits (disjoint MNIST-test val/test; Flickr2K HR-only + scene-level val). Claims are unchanged vs the discarded leaky-split runs.

## Install

Python ≥ 3.10, PyTorch 2.x, CUDA recommended for training.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

SwinIR experiments (Table 2, Figs 3/7/8/9) also need the upstream clone at the
commit recorded in [`SwinIR.COMMIT`](SwinIR.COMMIT):

```bash
git clone https://github.com/JingyunLiang/SwinIR.git SwinIR
git -C SwinIR checkout 6545850fbf8df298df73d81f3e8cba638787c8bd
```

`timm` is required by that vendor code.

## Data

Datasets are **not** in this repository (~100 GB). See [`data/README.md`](data/README.md).
Caveats and inferred paper details: [`ASSUMPTIONS.md`](ASSUMPTIONS.md).

## View results vs re-run

- **View:** open `paper/tables/*/comparison.md` and the SVG parts under `paper/figures/`. No GPU.
- **Re-run:** each `paper/.../README.md` has the train/eval command. Drivers live in `scripts/` and write under `experiments/`.

## Git LFS

Checkpoints (`*.pt`, `*.pth`, `*.ckpt`) are tracked with Git LFS. Several SwinIR
weights exceed GitHub's 100 MB blob limit. Cloning needs `git lfs`. GitHub Free
LFS quota (10 GB storage / 1 GB monthly bandwidth) may be too small for this tree
(~12 GB); use a data pack or another host with a higher quota.

## Layout

```
src/           library (models, training, datasets, SwinIR adapters)
scripts/       experiment drivers
configs/       YAML used by the kept experiments
tests/         unit tests
experiments/   full run trees (metrics + checkpoints), original relative paths
paper/         tables + constituent SVGs
data/          download instructions only
```
