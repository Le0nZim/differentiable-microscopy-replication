# Fig4 task-aware segmentation fix — verified re-run tables (BBBC022 proxy)

Stage 1 content-aware pretrain **reused frozen** from am2_task_aware_full; Stage 2 (frozen seg-head) + Stage 3 (end-to-end task-aware finetune) recomputed in this isolated directory. NOT exact U2OS reproduction.

## Dice / IoU (test, val-selected threshold)

| compression | illumination | Dice | IoU | Dice@0.5 | thr | stage2 (content-aware) val Dice |
|---|---|---:|---:|---:|---:|---:|
| x64 | fixed | 0.9007 | 0.8201 | 0.9005 | 0.35 | 0.9085 |
| x64 | learnable | 0.9272 | 0.8652 | 0.9250 | 0.9 | 0.9346 |
| x256 | fixed | 0.7196 | 0.5627 | 0.6910 | 0.1 | 0.7453 |
| x256 | learnable | 0.8358 | 0.7194 | 0.8303 | 0.9 | 0.8862 |
| x1024 | fixed | 0.5000 | 0.3344 | 0.4233 | 0.1 | 0.4500 |
| x1024 | learnable | 0.5245 | 0.3621 | 0.5062 | 0.1 | 0.5893 |

## Learnable vs. fixed (test Dice)

| compression | fixed | learnable | Δ | learnable wins? |
|---|---:|---:|---:|---|
| x64 | 0.9007 | 0.9272 | +0.0265 | yes |
| x256 | 0.7196 | 0.8358 | +0.1162 | yes |
| x1024 | 0.5000 | 0.5245 | +0.0245 | yes |

## Stage-3 gradient evidence (illumination receives the seg-task gradient)

| run | seg-head gn (stage2) | inverse gn (stage3) | illumination gn (stage3) | illum pattern Δ (L2) | rel |
|---|---:|---:|---:|---:|---:|
| taskaware_x64_random_fixed | 1.285e+00 | 3.206e+00 | 0.000e+00 | 0.000e+00 | 0.0% |
| taskaware_x64_learnable_frequency | 1.290e+00 | 4.453e+00 | 2.105e-05 | 7.446e+01 | 21.6% |
| taskaware_x256_random_fixed | 1.133e+00 | 6.643e+00 | 0.000e+00 | 0.000e+00 | 0.0% |
| taskaware_x256_learnable_frequency | 1.270e+00 | 7.772e+00 | 2.585e-05 | 9.933e+01 | 28.7% |
| taskaware_x1024_random_fixed | 5.071e-01 | 9.170e+00 | 0.000e+00 | 0.000e+00 | 0.0% |
| taskaware_x1024_learnable_frequency | 8.887e-01 | 1.109e+01 | 6.213e-05 | 1.435e+02 | 41.5% |
