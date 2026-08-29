# Task-aware segmentation report (BBBC022 substitute; paper B.0.1 proxy)

Staged procedure: content-aware pretrain -> frozen segmentation-head training -> end-to-end task-aware finetune. BBBC022 Hoechst SUBSTITUTE data; NOT exact U2OS reproduction.

## Dice / IoU (test, val-selected threshold)

| compression | illumination | Dice | IoU | Dice@0.5 | thr | post-hoc Dice |
|---|---|---:|---:|---:|---:|---:|
| x64 | fixed | 0.8938 | 0.8095 | 0.8938 | 0.65 | 0.9018 |
| x64 | learnable | 0.9322 | 0.8732 | 0.9322 | 0.6 | 0.9293 |
| x256 | fixed | 0.6958 | 0.5344 | 0.6707 | 0.15 | 0.7068 |
| x256 | learnable | 0.8757 | 0.7794 | 0.8754 | 0.35 | 0.8658 |
| x1024 | fixed | 0.4993 | 0.3335 | 0.4069 | 0.1 | 0.2665 |
| x1024 | learnable | 0.5986 | 0.4284 | 0.5810 | 0.15 | 0.5669 |

## Learnable vs. fixed (test Dice)

| compression | fixed | learnable | learnable wins? |
|---|---:|---:|---|
| x64 | 0.8938 | 0.9322 | yes |
| x256 | 0.6958 | 0.8757 | yes |
| x1024 | 0.4993 | 0.5986 | yes |

## Stage-3 gradient evidence (illumination receives the seg-task gradient)

| run | seg-head gn (stage2) | inverse gn (stage3) | illumination gn (stage3) | illum pattern Δ (L2) |
|---|---:|---:|---:|---:|
| taskaware_x64_random_fixed | 1.272e+00 | 3.983e+00 | 0.000e+00 | 0.000e+00 |
| taskaware_x64_learnable_frequency | 1.330e+00 | 3.799e+00 | 1.281e-05 | 7.167e+01 |
| taskaware_x256_random_fixed | 1.073e+00 | 5.799e+00 | 0.000e+00 | 0.000e+00 |
| taskaware_x256_learnable_frequency | 1.308e+00 | 8.195e+00 | 1.588e-05 | 9.544e+01 |
| taskaware_x1024_random_fixed | 5.688e-01 | 6.359e+00 | 0.000e+00 | 0.000e+00 |
| taskaware_x1024_learnable_frequency | 9.059e-01 | 1.747e+01 | 2.926e-05 | 1.431e+02 |
