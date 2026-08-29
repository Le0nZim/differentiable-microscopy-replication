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

## Follow-up: same cells without frequency-domain optimization

Figure 6 / Table 1's learnable arm is `learnable_frequency`. The companion
`experiments/figure06_noise_robustness_no_freq/` retrains a subset of those
cells with `learnable_spatial` (the paper's Table-3 variant D) and compares
against the frozen Table-1 checkpoints.
