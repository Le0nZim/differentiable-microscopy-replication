# Figure 5 — log(MSE) vs #training images (locality vs transpose, clean split)

- **Paper section:** 5.4
- **Status:** **aligned** on trend. Locality-aware upsampling wins every cell of the published grid.
- **Variant:** val and test from disjoint halves of the MNIST **test** digit pool. Train digits still come from the MNIST train pool.

Grid: image sizes {128, 256, 512} × train counts {600, 3000, 6000} at x8, random_fixed patterns, 4000 steps, seed 42. Extra size 64 is in the supplementary plots.

## Run tree

`experiments/figure05_upsampling/`

## Reproduce

```bash
bash scripts/figure05_upsampling/launch.sh
python scripts/figure05_upsampling/finalize.py
```

Or collect-only after shards:

```bash
python scripts/figure05_upsampling/run.py --collect-only \
  --output-root experiments/figure05_upsampling
```

## This folder

- `results.csv` — clean-split test MSE
- `COMPARISON.md` — vs the discarded leaky-split run
- `SPLIT.md`
- `plots/`, `symbols/`, `scale_bars/`
