# Paper expected — Table 2

**Caption / paraphrase:** PSNR/SSIM of SwinIR reconstruction with vs without learnable illumination (LI) at x16 compression, on five standard SR test sets.

- **Section:** 5.6
- **Type:** table
- **Expected panels/components:** PSNR table (w/o LI, with LI) x 5 datasets; SSIM table (w/o LI, with LI) x 5 datasets
- **Expected metrics:** PSNR (dB), SSIM
- **Expected datasets:** Div2K/Flickr2K (train); Set5, Set14, BSD100, Urban100, Manga109 (test)
- **Compression / downscaling / pattern settings:** x16 compression, 64x64 patches, SwinIR-M embed_dim=180, batch 32, illum LR 0.1; ours 20k/500k iters (compute-limited)

## Paper values

| dataset | condition | psnr | ssim |
| --- | --- | --- | --- |
| Set5 | SwinIR w/o LI | 14.03 | 0.3079 |
| Set5 | SwinIR with LI | 26.74 | 0.8113 |
| Set14 | SwinIR w/o LI | 13.64 | 0.2258 |
| Set14 | SwinIR with LI | 23.6 | 0.693 |
| BSD100 | SwinIR w/o LI | 14.28 | 0.2094 |
| BSD100 | SwinIR with LI | 22.9 | 0.6317 |
| Urban100 | SwinIR w/o LI | 13.51 | 0.2146 |
| Urban100 | SwinIR with LI | 21.51 | 0.6402 |
| Manga109 | SwinIR w/o LI | 12.09 | 0.1952 |
| Manga109 | SwinIR with LI | 20.18 | 0.6652 |

