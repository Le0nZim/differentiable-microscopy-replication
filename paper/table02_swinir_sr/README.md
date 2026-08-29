# Table 2 — Learnable illumination + SwinIR, grayscale super-resolution (x16, clean split)

- **Paper section:** 5.6
- **Status:** **close** (LI helps on all five datasets; paper's +8–12 dB magnitude is not reproduced)
- **Variant:** clean-split rerun. Flickr2K **HR-only** (drop bicubic x2/x3/x4) and **scene-level** 2% val holdout.
- **Shared run:** also produces [Figure 7](../figure07_swinir_sr/).

> PSNR/SSIM of SwinIR reconstruction with vs without learnable illumination (LI) at x16 compression, on Set5 / Set14 / BSD100 / Urban100 / Manga109.

## Run tree

`experiments/table02_swinir_sr/`

Config: `configs/table02_swinir_sr/full.yaml`

SwinIR-M (`embed_dim=180`), effective batch 32, 20k of ~500k paper iterations (compute-limited). Re-runs need the pinned SwinIR vendor clone; see root `SwinIR.COMMIT`.

## Reproduce

```bash
python scripts/table02_swinir_sr/run.py \
  --config configs/table02_swinir_sr/full.yaml \
  --output-base experiments/table02_swinir_sr \
  --device cuda:0
```

## Results (this folder)

- `our_values.csv` / `paper_values.csv` / `comparison.csv` / `comparison.md`
- `rendered/table02_comparison.md`
- `components/ours_AM4_resolution_report.md`

LI-helps holds on all 5 datasets (PSNR +1.5–2.5 dB class). Our with-LI matches or exceeds the paper's with-LI. The paper's huge gap is dominated by a weak w/o-LI baseline (~12–14 dB vs our ~22–25 dB).

## Files

- `paper_expected.md` — paper values
- `our_replication.md` — clean-split numbers
- `missing_components.md` — magnitude / iteration caveats
- `provenance.md` — source run files
