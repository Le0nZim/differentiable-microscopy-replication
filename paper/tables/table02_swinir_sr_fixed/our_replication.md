# Our replication — Table 2 (clean split)

- **Experiment:** `experiments/swinir_or_highres/am4_swinir_table2_resolution_fixed/full`
- **Status:** close
- **Split:** Flickr2K HR-only + scene-level 2% val holdout.

Canonical numbers are from the AM-4 `full/` run (SwinIR-M embed_dim=180, effective batch 32, fair full deterministic tiling, 20k of ~500k paper iterations).

- Our with-LI matches or exceeds the paper's with-LI on every dataset.
- The LI-helps direction reproduces on all 5 datasets.
- SR datasets are fully available; this is compute-limited, not data-blocked.

## Our values

| dataset | condition | psnr | ssim |
| --- | --- | --- | --- |
| Set5 | SwinIR w/o LI | 25.55 | 0.7201 |
| Set5 | SwinIR with LI | 27.45 | 0.7768 |
| Set14 | SwinIR w/o LI | 23.62 | 0.6379 |
| Set14 | SwinIR with LI | 25.17 | 0.6882 |
| BSD100 | SwinIR w/o LI | 23.91 | 0.5902 |
| BSD100 | SwinIR with LI | 25.12 | 0.6279 |
| Urban100 | SwinIR w/o LI | 22.55 | 0.5833 |
| Urban100 | SwinIR with LI | 23.90 | 0.6434 |
| Manga109 | SwinIR w/o LI | 24.21 | 0.7224 |
| Manga109 | SwinIR with LI | 26.09 | 0.7799 |
