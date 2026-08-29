# Figure 5 - original vs clean-split test MSE

Clean split = val and test drawn from disjoint halves of the MNIST test digit pool.
Train digits come from the MNIST train pool in both versions, so neither has train/test leakage.

| img size | # train | method | original MSE | clean-split MSE | delta |
| --- | --- | --- | --- | --- | --- |
| 64 | 600 | OurUpSampling | 0.02099 | 0.02070 | -0.00029 |
| 64 | 600 | TransposeConv | 0.03386 | 0.03324 | -0.00062 |
| 64 | 3000 | OurUpSampling | 0.01328 | 0.01415 | +0.00087 |
| 64 | 3000 | TransposeConv | 0.02758 | 0.02665 | -0.00093 |
| 64 | 6000 | OurUpSampling | 0.01269 | 0.01296 | +0.00027 |
| 64 | 6000 | TransposeConv | 0.02544 | 0.02462 | -0.00082 |
| 128 | 600 | OurUpSampling | 0.01771 | 0.01786 | +0.00016 |
| 128 | 600 | TransposeConv | 0.03858 | 0.03868 | +0.00011 |
| 128 | 3000 | OurUpSampling | 0.01238 | 0.01239 | +0.00001 |
| 128 | 3000 | TransposeConv | 0.03625 | 0.03604 | -0.00021 |
| 128 | 6000 | OurUpSampling | 0.01197 | 0.01173 | -0.00023 |
| 128 | 6000 | TransposeConv | 0.03646 | 0.03564 | -0.00081 |
| 256 | 600 | OurUpSampling | 0.01695 | 0.01670 | -0.00025 |
| 256 | 600 | TransposeConv | 0.03660 | 0.03675 | +0.00015 |
| 256 | 3000 | OurUpSampling | 0.01173 | 0.01195 | +0.00022 |
| 256 | 3000 | TransposeConv | 0.03640 | 0.03639 | -0.00000 |
| 256 | 6000 | OurUpSampling | 0.01105 | 0.01138 | +0.00034 |
| 256 | 6000 | TransposeConv | 0.03571 | 0.03608 | +0.00038 |
| 512 | 600 | OurUpSampling | 0.01633 | 0.01620 | -0.00013 |
| 512 | 600 | TransposeConv | 0.03535 | 0.03541 | +0.00007 |
| 512 | 3000 | OurUpSampling | 0.01178 | 0.01172 | -0.00006 |
| 512 | 3000 | TransposeConv | 0.03517 | 0.03501 | -0.00016 |
| 512 | 6000 | OurUpSampling | 0.01096 | 0.01121 | +0.00025 |
| 512 | 6000 | TransposeConv | 0.03548 | 0.03530 | -0.00017 |

**Winner check (clean split):** locality-aware wins every cell of the paper grid.
