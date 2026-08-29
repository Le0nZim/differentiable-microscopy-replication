# Figure 7 — SwinIR standard SR qualitative tiles (clean split)

- **Paper section:** 5.6
- **Status:** **close** (same claims as Table 2)
- **Shared run:** [Table 2](../table02_swinir_sr/)
- **Variant:** Flickr2K HR-only + scene-level 2% val holdout. Tiles rebuilt from that AM-4 run (`best.pt`).

## Run tree

`experiments/table02_swinir_sr/full/`

Full-image eval metadata: `.../full/full_image_eval/metadata.json`

## Reproduce

Same training as Table 2, then:

```bash
python scripts/_shared/finalize_clean_split.py
# optional last.pt (20k) sibling:
python scripts/table02_swinir_sr/render_last20k.py
```

See `SPLIT.md`. Needs DIV2K/Flickr2K HR and the five SR test sets under `data/sr/`.
