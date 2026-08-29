# Differentiable Microscopy — main tables and figures

Standalone reproduction package for the **main-paper tables and figures** of
*Differentiable Microscopy for Content and Task Aware Compressive Fluorescence Imaging*
([arXiv:2203.14945](https://arxiv.org/abs/2203.14945)).

**The slug is the path.** Finding Table 1 means opening anything named
`table01_noise_robustness` under `paper/`, `scripts/`, `configs/`, and
`experiments/`. Shared runs are stored once (Table 1 ↔ Fig. 6, Table 2 ↔ Fig. 7,
Fig. 8 ↔ Fig. 9).

## Catalog

| Item | Status | Paper | Scripts | Configs | Runs |
|---|---|---|---|---|---|
| Table 1 | close | [`paper/table01_noise_robustness`](paper/table01_noise_robustness) | [`scripts/table01_noise_robustness`](scripts/table01_noise_robustness) | [`configs/table01_noise_robustness`](configs/table01_noise_robustness) | [`experiments/table01_noise_robustness`](experiments/table01_noise_robustness) |
| Table 2 | close | [`paper/table02_swinir_sr`](paper/table02_swinir_sr) | [`scripts/table02_swinir_sr`](scripts/table02_swinir_sr) | [`configs/table02_swinir_sr`](configs/table02_swinir_sr) | [`experiments/table02_swinir_sr`](experiments/table02_swinir_sr) |
| Table 3 | data_blocked | [`paper/table03_ablation`](paper/table03_ablation) | [`scripts/table03_ablation`](scripts/table03_ablation) | [`configs/table03_ablation`](configs/table03_ablation) | [`experiments/table03_ablation`](experiments/table03_ablation) |
| Figure 3 | data_blocked | [`paper/figure03_content_aware`](paper/figure03_content_aware) | [`scripts/figure03_content_aware`](scripts/figure03_content_aware) | [`configs/figure03_content_aware`](configs/figure03_content_aware) | [`experiments/figure03_content_aware`](experiments/figure03_content_aware) (`base/` + `swinir/`) |
| Figure 4 | data_blocked | [`paper/figure04_segmentation`](paper/figure04_segmentation) | [`scripts/figure04_segmentation`](scripts/figure04_segmentation) | [`configs/figure04_segmentation`](configs/figure04_segmentation) | [`experiments/figure04_segmentation`](experiments/figure04_segmentation) (`task_aware/` + `stage1_frozen/`) |
| Figure 5 | aligned | [`paper/figure05_upsampling`](paper/figure05_upsampling) | [`scripts/figure05_upsampling`](scripts/figure05_upsampling) | [`configs/figure05_upsampling`](configs/figure05_upsampling) | [`experiments/figure05_upsampling`](experiments/figure05_upsampling) |
| Figure 6 | close | [`paper/figure06_noise_robustness`](paper/figure06_noise_robustness) | same as Table 1 | same as Table 1 | same as Table 1 |
| Figure 7 | close | [`paper/figure07_swinir_sr`](paper/figure07_swinir_sr) | same as Table 2 | same as Table 2 | same as Table 2 |
| Figure 8 | close | [`paper/figure08_mcf7_swinir`](paper/figure08_mcf7_swinir) | [`scripts/figure08_mcf7`](scripts/figure08_mcf7) | [`configs/figure08_mcf7`](configs/figure08_mcf7) | [`experiments/figure08_mcf7`](experiments/figure08_mcf7) |
| Figure 9 | close | [`paper/figure09_mcf7_widefield`](paper/figure09_mcf7_widefield) | same as Figure 8 | same as Figure 8 | same as Figure 8 |
| Figure 10 | data_blocked | [`paper/figure10_ablation`](paper/figure10_ablation) | [`scripts/figure10_ablation`](scripts/figure10_ablation) | [`configs/figure10_ablation`](configs/figure10_ablation) | [`experiments/figure10_ablation`](experiments/figure10_ablation) |

**data_blocked** means the paper's U2OS confocal dataset is unavailable. Those items use a BBBC022 Hoechst widefield substitute and are not numerically comparable to the paper.

Tables 1–2 and Figs 5–7 use leakage-safer splits (disjoint MNIST-test val/test; Flickr2K HR-only + scene-level val).

Old path names (AM/RR codes, `_fixed` suffixes) are listed in [`paper/README.md`](paper/README.md).

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

- **View:** open `paper/<slug>/comparison.md` (tables) and the SVG parts under `paper/figure*/`. No GPU.
- **Re-run:** each `paper/<slug>/README.md` has the train/eval command. Drivers live in `scripts/<slug>/` and write under `experiments/<slug>/`.

## Git LFS

Checkpoints (`*.pt`, `*.pth`, `*.ckpt`) are tracked with Git LFS. Several SwinIR
weights exceed GitHub's 100 MB blob limit. Cloning needs `git lfs`. GitHub Free
LFS quota (10 GB storage / 1 GB monthly bandwidth) may be too small for this tree
(~12 GB); use a data pack or another host with a higher quota.

## Layout

```
src/           library (models, training, datasets, SwinIR adapters)
scripts/       drivers grouped by paper slug
configs/       YAML grouped by paper slug; shared bases in configs/_shared/
tests/         unit tests
experiments/   run trees (metrics + checkpoints), same slugs
paper/         tables + constituent SVGs, same slugs
data/          download instructions only
archive/       superseded scripts, configs, and backup runs
```
