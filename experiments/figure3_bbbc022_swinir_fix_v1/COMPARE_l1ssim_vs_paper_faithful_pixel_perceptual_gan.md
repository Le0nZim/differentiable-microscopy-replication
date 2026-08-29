# l1_ssim vs paper_faithful_pixel_perceptual_gan — shared cells

SwinIR-refinement SSIM/MSE by loss recipe (test split, n=60/cell). `base` is the frozen content-aware reconstruction (identical for both).

| comp | illum | base SSIM | l1_ssim SSIM | paper_faithful_pixel_perceptual_gan SSIM | base MSE | l1_ssim MSE | paper_faithful_pixel_perceptual_gan MSE |
|---|---|---|---|---|---|---|---|
| x16 | learnable | 0.9150 | 0.9399 | 0.9150 | 0.000937 | 0.000500 | 0.000729 |
| x16 | pseudo_random | 0.8974 | 0.9384 | 0.9107 | 0.001141 | 0.000539 | 0.000804 |
| x64 | learnable | 0.9034 | 0.9291 | 0.9005 | 0.001232 | 0.000756 | 0.001063 |
| x64 | pseudo_random | 0.8359 | 0.9008 | 0.8623 | 0.003616 | 0.001756 | 0.002577 |
| x256 | learnable | 0.8135 | 0.8746 | 0.8351 | 0.004945 | 0.002872 | 0.003888 |
| x256 | pseudo_random | 0.6755 | 0.7751 | 0.7339 | 0.015195 | 0.010559 | 0.013926 |
| x1024 | learnable | 0.6481 | 0.6954 | 0.6748 | 0.023334 | 0.024090 | 0.029205 |
| x1024 | pseudo_random | 0.5618 | 0.6419 | 0.6345 | 0.040170 | 0.045746 | 0.046230 |

_Note: l1_ssim optimises SSIM+MSE directly; the pixel+perceptual+GAN recipe optimises perceptual/texture realism and may trade a little pixel MSE for sharper, more paper-like restorations._
