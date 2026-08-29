# Figure 7 — SwinIR standard SR qualitative tiles (clean split)

- **Paper section:** 5.6
- **Status:** **close** (same claims as Table 2)
- **Shared run:** [Table 2](../../tables/table02_swinir_sr_fixed/)
- **Variant:** Flickr2K HR-only + scene-level 2% val holdout. Tiles rebuilt from that AM-4 run (`best.pt`).

## Run tree

`experiments/swinir_or_highres/am4_swinir_table2_resolution_fixed/full/`

Full-image eval metadata: `.../full/full_image_eval/metadata.json`

## Reproduce

Same training as Table 2, then:

```bash
python scripts/finalize_clean_split_fixed.py
# optional last.pt (20k) sibling:
python scripts/render_fig07_fixed_last20k.py
```

See `SPLIT.md`. Needs DIV2K/Flickr2K HR and the five SR test sets under `data/sr/`.
