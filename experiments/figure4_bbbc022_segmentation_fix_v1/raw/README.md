# Figure 4 plot images — pre-threshold form

These are the **same 5 test patches** shown in row A of `figures/figure4_paper_layout.png`
(seed 42, first 5 test samples — the report script default `--k 5`).

## What you should threshold in ImageJ

Use the `*_prethreshold_float32.tif` files.

They are **float32**, values in **[0, 1]**, exactly as the tensor that
`make_pseudo_mask` receives before applying threshold **0.3** (+ morphological
closing with kernel 10).

Pipeline applied before this state:
1. maximum-intensity projection (if multi-page TIFF)
2. subtract bias **134.28**
3. clip to **[0, 500]**
4. per-image min–max normalize → **[0, 1]**
5. center crop **256×256** (test split; no random flips)

In ImageJ: open the float TIFF, then try thresholds on the 0–1 scale
(e.g. 0.3). The experiment's binary pseudo-GT (after closing) is also saved
as `*_pseudoGT_mask_thr0p3_closing10.tif` for reference.

`*_prethreshold_u16.tif` is the same image scaled to 16-bit (value * 65535)
for viewing convenience; for threshold experiments prefer the float32 files
(or divide u16 by 65535).

See `manifest.json` for source BBBC022 filenames.


## Raw (unprocessed) images

Because the experiment's pre-threshold patches are min–max normalized to [0,1]
(and clipped at 500), they look saturated. Also included:

- `fig4_colXX_original_full_*.tif` — the **original BBBC022 TIFF** from disk (full FOV, native uint16).
- `fig4_colXX_original_center256.tif` — **same 256×256 FOV** as the plot column, but
  **MIP only** (no bias / clip / normalize). Use these to try thresholds in native intensity space.
