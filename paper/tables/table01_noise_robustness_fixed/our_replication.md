# Our replication — Table 1 (clean split)

- **Experiment:** `experiments/noise_robustness/rr1_v3_normalized_full_fixed`
- **Status:** close
- **Split:** MNIST train pool for train; disjoint halves of MNIST test for val vs test.

Implemented the paper's normalized stochastic detector model (supplement A.2.2, eqs. S5–S10, `noise_normalization: paper_v3`) and reran the full 8-cell grid plus 3 seeds on the extreme cell (pc=10, sigma=6).

- Learnable beats fixed in all 8 cells (~3× lower MSE).
- MSE is flat across read noise at both photon counts.
- PatchMNIST is fully available; this table is not data-blocked.

## Our values (seed 42)

| method | photon_count | sigma_read | our_mse |
| --- | --- | --- | --- |
| learnable (Our) | 10 | 0.0 | 0.0064 |
| fixed_random | 10 | 0.0 | 0.0208 |
| learnable (Our) | 10 | 2.7 | 0.0066 |
| fixed_random | 10 | 2.7 | 0.0211 |
| learnable (Our) | 10 | 2.0 | 0.0066 |
| fixed_random | 10 | 2.0 | 0.0207 |
| learnable (Our) | 10 | 6.0 | 0.0074 |
| fixed_random | 10 | 6.0 | 0.0230 |
| learnable (Our) | 10000 | 0.0 | 0.0030 |
| fixed_random | 10000 | 0.0 | 0.0109 |
| learnable (Our) | 10000 | 2.7 | 0.0031 |
| fixed_random | 10000 | 2.7 | 0.0109 |
| learnable (Our) | 10000 | 2.0 | 0.0032 |
| fixed_random | 10000 | 2.0 | 0.0109 |
| learnable (Our) | 10000 | 6.0 | 0.0032 |
| fixed_random | 10000 | 6.0 | 0.0110 |
