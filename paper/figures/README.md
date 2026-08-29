# Constituent figure parts (Figures 3–10)

Every item is a single smallest part — never a composite. Assemble in PowerPoint (or similar) from these folders.

- `images/` — clean tiles from checkpoints/tensors (no markers, SSIM/PSNR text, titles, or clipping)
- `patterns/` — illumination pattern tiles
- `plots/` — vector graphs (Arial)
- `symbols/` / `labels/` / `scale_bars/` — legend glyphs, text, unlabeled bar templates

Where a `*_fixed` folder exists, that is the published variant (clean split). Figure 4 ships both `paper_strict` and calibrated-preprocessing (`figure04_segmentation_new_pre`).

Rebuild:

```bash
python paper/_build_components.py
```
