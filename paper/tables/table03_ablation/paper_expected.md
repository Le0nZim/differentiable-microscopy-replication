# Paper expected — Table 3

**Caption / paraphrase:** Ablation over A (baseline fixed Ht + transpose-conv up), B (+learnable Ht), C (+proposed locality upsampling), D (-frequency-domain optimization). Paper: C is best.

- **Section:** 5.7
- **Type:** table
- **Expected panels/components:** SSIM/MSE for A, B, C, D
- **Expected metrics:** SSIM, MSE
- **Expected datasets:** U2OS (paper) -- UNAVAILABLE; BBBC022 Cell-Painting substitute (ours)
- **Compression / downscaling / pattern settings:** x16 compression, T=4, downscaling 8x8:1

## Paper values

| variant | label | paper_ssim | paper_mse |
| --- | --- | --- | --- |
| A | fixed Ht + Tr.Conv.Up + freq | 0.7872 | 0.0042 |
| B | (+) learnable Ht | 0.795 | 0.0038 |
| C | (+) proposed locality upsampling (paper best) | 0.8426 | 0.0029 |
| D | (-) frequency-domain optimization | 0.7857 | 0.0041 |

