# Paper expected — Table 1

**Caption / paraphrase:** Reconstruction MSE of the proposed (learnable) illumination vs fixed pseudo-random illumination, across photon counts {10, 10000} and read-noise std {0.0, 2.7, 2.0, 6.0} at x8 compression with T=8.

- **Section:** 5.5
- **Type:** table
- **Expected panels/components:** MSE grid: learnable vs fixed x {pc10, pc10000} x 4 read-noise levels
- **Expected metrics:** MSE
- **Expected datasets:** PatchMNIST
- **Compression / downscaling / pattern settings:** x8 compression, T=8, d=8, photon_count {10, 10000}, sigma_read {0.0, 2.7, 2.0, 6.0}

## Paper values

| method | photon_count | sigma_read | paper_mse |
| --- | --- | --- | --- |
| learnable (Our) | 10 | 0.0 | 0.0025 |
| fixed_random | 10 | 0.0 | 0.0108 |
| learnable (Our) | 10 | 2.7 | 0.0024 |
| fixed_random | 10 | 2.7 | 0.0107 |
| learnable (Our) | 10 | 2.0 | 0.0024 |
| fixed_random | 10 | 2.0 | 0.0107 |
| learnable (Our) | 10 | 6.0 | 0.0024 |
| fixed_random | 10 | 6.0 | 0.0107 |
| learnable (Our) | 10000 | 0.0 | 0.0059 |
| fixed_random | 10000 | 0.0 | 0.0214 |
| learnable (Our) | 10000 | 2.7 | 0.0058 |
| fixed_random | 10000 | 2.7 | 0.021 |
| learnable (Our) | 10000 | 2.0 | 0.0061 |
| fixed_random | 10000 | 2.0 | 0.0213 |
| learnable (Our) | 10000 | 6.0 | 0.0069 |
| fixed_random | 10000 | 6.0 | 0.0235 |

