# Figure 6 — Extreme-noise reconstructions (clean split)

- **Paper section:** 5.5
- **Status:** **close** (same claims as Table 1)
- **Shared run:** [Table 1](../../tables/table01_noise_robustness_fixed/)
- **Variant:** disjoint MNIST-test val/test pools. Panels are the **first TEST batch** after the clean-split retrain.

Extreme cell: photon count 10, read-noise std 6.0, x8, T=8.

## Run tree

`experiments/noise_robustness/rr1_v3_normalized_full_fixed/`

## Reproduce

Same commands as Table 1, then:

```bash
python scripts/finalize_clean_split_fixed.py
```

See `SPLIT.md`.
