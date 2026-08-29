# Figure 5 — log(MSE) vs #training images (locality vs transpose, clean split)

- **Paper section:** 5.4
- **Status:** **aligned** on trend. Locality-aware upsampling wins every cell of the published grid.
- **Variant:** val and test from disjoint halves of the MNIST **test** digit pool. Train digits still come from the MNIST train pool.

Grid: image sizes {128, 256, 512} × train counts {600, 3000, 6000} at x8, random_fixed patterns, 4000 steps, seed 42. Extra size 64 is in the supplementary plots.

## Run tree

`experiments/upsampling_ablation/patchmnist_upsampling_analysis_fixed/`

## Reproduce

```bash
bash scripts/launch_fig05_fixed.sh
python scripts/finalize_fig05_fixed.py
```

Or collect-only after shards:

```bash
python scripts/run_fig05_upsampling_fixed.py --collect-only \
  --output-root experiments/upsampling_ablation/patchmnist_upsampling_analysis_fixed
```

## This folder

- `results.csv` — clean-split test MSE
- `COMPARISON.md` — vs the discarded leaky-split run
- `SPLIT.md`
- `plots/`, `symbols/`, `scale_bars/`
