# Fig4 task-aware segmentation fix — verified re-run tables (BBBC022 proxy)

Pseudo-GT masks: **TrackMate** detector (raw MIP>506, 4-connectivity, Douglas–Peucker ε0.5). Stage 1 content-aware pretrain **reused frozen** from am2_task_aware_full; Stage 2 (frozen seg-head) + Stage 3 (end-to-end task-aware finetune) recomputed in this isolated directory. NOT exact U2OS reproduction.

## Dice / IoU (test, val-selected threshold)

| compression | illumination | Dice | IoU | Dice@0.5 | thr | stage2 (content-aware) val Dice |
|---|---|---:|---:|---:|---:|---:|
| x64 | fixed | 0.8480 | 0.7375 | 0.8425 | 0.1 | 0.8672 |
| x64 | learnable | 0.9039 | 0.8256 | 0.9050 | 0.2 | 0.9133 |
| x256 | fixed | 0.6283 | 0.4598 | 0.6024 | 0.1 | 0.6837 |
| x256 | learnable | 0.8150 | 0.6890 | 0.8112 | 0.1 | 0.8453 |
| x1024 | fixed | 0.3959 | 0.2476 | 0.3637 | 0.1 | 0.3580 |
| x1024 | learnable | 0.4583 | 0.2984 | 0.4456 | 0.2 | 0.5128 |

## Learnable vs. fixed (test Dice)

| compression | fixed | learnable | Δ | learnable wins? |
|---|---:|---:|---:|---|
| x64 | 0.8480 | 0.9039 | +0.0560 | yes |
| x256 | 0.6283 | 0.8150 | +0.1867 | yes |
| x1024 | 0.3959 | 0.4583 | +0.0625 | yes |

## Stage-3 gradient evidence (illumination receives the seg-task gradient)

| run | seg-head gn (stage2) | inverse gn (stage3) | illumination gn (stage3) | illum pattern Δ (L2) | rel |
|---|---:|---:|---:|---:|---:|
| taskaware_x64_random_fixed | 1.398e+00 | 3.325e+00 | 0.000e+00 | 0.000e+00 | 0.0% |
| taskaware_x64_learnable_frequency | 1.399e+00 | 2.784e+00 | 1.960e-05 | 7.648e+01 | 22.2% |
| taskaware_x256_random_fixed | 1.256e+00 | 5.890e+00 | 0.000e+00 | 0.000e+00 | 0.0% |
| taskaware_x256_learnable_frequency | 1.391e+00 | 4.205e+00 | 2.158e-05 | 1.008e+02 | 29.1% |
| taskaware_x1024_random_fixed | 6.447e-01 | 8.560e+00 | 0.000e+00 | 0.000e+00 | 0.0% |
| taskaware_x1024_learnable_frequency | 1.054e+00 | 5.540e+00 | 2.565e-05 | 1.415e+02 | 41.0% |
