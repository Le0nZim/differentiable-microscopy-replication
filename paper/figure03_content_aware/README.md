# Figure 3 — Content-aware reconstruction (BBBC022 substitute)

- **Paper section:** 5.2 / 5.6
- **Status:** **data_blocked** (U2OS unavailable; BBBC022 Hoechst widefield substitute). Learnable > pseudo-random at every compression on the substitute; +SwinIR improves SSIM in every cell of the paper-faithful recipe.
- **Assets:** atomic SVGs (images, patterns, plots, symbols, labels, scale bars). No composite panels.

## Run trees

- Base microscopes: `experiments/figure03_content_aware/base/` (large well-disjoint split + per-epoch crops)
- +SwinIR refinement: `experiments/figure03_content_aware/swinir/` (SwinIR-M, paper-faithful pixel+perceptual+GAN)

The superseded base-tree `swinir/` columns are **not** shipped; use `experiments/figure03_content_aware/swinir`.

Split: `configs/_shared/splits/split_fig03_large.json` (1980 / 40 / 60).

## Reproduce

```bash
# base 4×4 matrix (example; see scripts/figure03_content_aware/train_base.py)
python scripts/figure03_content_aware/train_base.py --device cuda:0

python scripts/figure03_content_aware/train_swinir.py \
  --config configs/figure03_content_aware/paper_faithful_pixel_perceptual_gan.yaml \
  --device cuda:0
python scripts/figure03_content_aware/report.py --name paper_faithful_pixel_perceptual_gan --full-panel
python paper/_build_components.py --only fig3
```

Needs BBBC022 substitute data under `data/substitute_data/` and a GPU for rebuilds. The SVGs in this folder are viewable offline.
