# Table 1 — Robustness to Poisson and read noise (PatchMNIST, clean split)

- **Paper section:** 5.5
- **Status:** **close** (headline claims reproduced; paper's inverted photon-count ordering is not)
- **Variant:** clean-split rerun (`disjoint_val_test: true`). Train digits from the MNIST train pool; val and test from **disjoint halves** of the MNIST test digit pool.
- **Shared run:** also produces [Figure 6](../figure06_noise_robustness/).

> Reconstruction MSE of learnable illumination vs fixed pseudo-random illumination, across photon counts {10, 10000} and read-noise std {0.0, 2.7, 2.0, 6.0} at x8 compression with T=8.

## Run tree

`experiments/table01_noise_robustness/`

Config: `configs/table01_noise_robustness/noise_table.yaml`

## Reproduce

```bash
python scripts/table01_noise_robustness/run.py \
  --config configs/table01_noise_robustness/noise_table.yaml \
  --output-root experiments/table01_noise_robustness \
  --device cuda:0 --shard 0 --num-shards 2
python scripts/table01_noise_robustness/run.py \
  --config configs/table01_noise_robustness/noise_table.yaml \
  --output-root experiments/table01_noise_robustness \
  --device cuda:1 --shard 1 --num-shards 2
python scripts/table01_noise_robustness/run.py \
  --config configs/table01_noise_robustness/noise_table.yaml \
  --output-root experiments/table01_noise_robustness \
  --aggregate-only --device cuda:0
```

Two-GPU launcher: `bash scripts/_shared/launch_clean_split.sh`

## Results (this folder)

- `our_values.csv` / `paper_values.csv` / `comparison.csv` / `comparison.md`
- `rendered/table01_comparison.md`
- `components/ours_table1_v3_results.md`, `components/ours_noise_formula_audit.md`

Learnable beats fixed in all 8 cells (~3× lower MSE). MSE is flat across read noise at both photon counts. Absolute MSEs are the same order of magnitude as the paper; the paper's pc=10 < pc=10000 ordering is not reproduced (the supplement's own noise equations predict the opposite, physically expected, ordering).

## Files

- `paper_expected.md` — paper values
- `our_replication.md` — clean-split numbers
- `missing_components.md` — photon-count ordering caveat
- `provenance.md` — source run files
