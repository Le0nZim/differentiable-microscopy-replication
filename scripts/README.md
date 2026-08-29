# Experiment drivers

Each subdirectory is a paper slug. Shared launchers live in `_shared/`.

| Slug | Main entry |
|---|---|
| [`table01_noise_robustness/`](table01_noise_robustness/) | `run.py` |
| [`table02_swinir_sr/`](table02_swinir_sr/) | `run.py` |
| [`table03_ablation/`](table03_ablation/) | `run.py`, `aggregate.py` |
| [`figure03_content_aware/`](figure03_content_aware/) | `train_base.py`, `train_swinir.py` |
| [`figure04_segmentation/`](figure04_segmentation/) | `train.py` |
| [`figure05_upsampling/`](figure05_upsampling/) | `run.py`, `launch.sh` |
| [`figure08_mcf7/`](figure08_mcf7/) | `train.py`, `reproduce_fig8.py`, `reproduce_fig9.py` |
| [`figure10_ablation/`](figure10_ablation/) | `train.py`, `reproduce.py` |
| [`_shared/`](_shared/) | Table 1+2 two-GPU launcher, figure export |

Commands and configs: `paper/<slug>/README.md`.
Superseded drivers: [`archive/scripts/`](../archive/scripts/).
