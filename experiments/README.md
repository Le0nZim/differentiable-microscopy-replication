# Experiment run trees

Each folder matches a paper slug. Checkpoints, metrics, and logs live here.
Paper-facing numbers and SVGs are under `paper/<slug>/`.

| Slug | Contents |
|---|---|
| [`table01_noise_robustness/`](table01_noise_robustness/) | Table 1 + Figure 6 (PatchMNIST noise grid) |
| [`table02_swinir_sr/`](table02_swinir_sr/) | Table 2 + Figure 7 (SwinIR SR) |
| [`table03_ablation/`](table03_ablation/) | Table 3 multi-seed A/B/C/D |
| [`figure03_content_aware/base/`](figure03_content_aware/base/) | Figure 3 base microscopes |
| [`figure03_content_aware/swinir/`](figure03_content_aware/swinir/) | Figure 3 SwinIR refinement |
| [`figure04_segmentation/task_aware/`](figure04_segmentation/task_aware/) | Figure 4 stages 2–3 |
| [`figure04_segmentation/stage1_frozen/`](figure04_segmentation/stage1_frozen/) | Figure 4 frozen stage-1 bases |
| [`figure05_upsampling/`](figure05_upsampling/) | Figure 5 locality vs transpose grid |
| [`figure08_mcf7/`](figure08_mcf7/) | Figures 8 and 9 (MCF7) |
| [`figure10_ablation/`](figure10_ablation/) | Figure 10 qualitative A/B/C/D |

Train/eval commands are in `paper/<slug>/README.md`. Drivers: `scripts/<slug>/`.
