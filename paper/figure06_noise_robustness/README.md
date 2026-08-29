# Figure 6 — Extreme-noise reconstructions (clean split)

- **Paper section:** 5.5
- **Status:** **close** (same claims as Table 1)
- **Shared run:** [Table 1](../table01_noise_robustness/)
- **Variant:** disjoint MNIST-test val/test pools. Panels are the **first TEST batch** after the clean-split retrain.

Extreme cell: photon count 10, read-noise std 6.0, x8, T=8.

## Run tree

`experiments/table01_noise_robustness/`

## Reproduce

Same commands as Table 1, then:

```bash
python scripts/_shared/finalize_clean_split.py
```

See `SPLIT.md`.
