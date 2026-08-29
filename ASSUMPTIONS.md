# Assumptions and Discrepancies

This file records missing, ambiguous, or inferred details that are **not** explicitly specified in the paper sources. If `spec/REPRODUCTION_SPEC.md` disagrees with the paper PDF, Markdown, extracted figures, or `data/cell.tif`, the paper/data take precedence and the discrepancy is noted here.

## Repository setup

| Item | Assumption / note |
|------|-------------------|
| Missing PDF | `paper_sources/paper.pdf` was not present at restructuring time. Implementation will rely on `paper.md` and extracted figures until the PDF is added. |
| `*_meta.json` | Kept alongside paper sources as PDF-conversion metadata; not treated as a primary source. |
| `REPRODUCTION_SPEC.md` | Working plan only — not ground truth for paper details. |

## Model and experiments

### Pattern generator

| Item | Assumption / note |
|------|-------------------|
| Hadamard patterns | Paper mentions Hadamard basis but does not specify reshaping for arbitrary `H x W`. We use Sylvester Hadamard rows of size `2^ceil(log2(H*W))`, take the first `H*W` entries, reshape to `(H, W)`, and map `{-1, +1}` to `[0, 1]` via `(h + 1) / 2`. |
| `random_fixed` | Paper compares pseudo-random patterns without giving an exact generation recipe. We use `tau_0 ~ N(0, 1)` and `H_t = sigmoid(random_fixed_m * tau_0)` with `random_fixed_m = 10` by default to approximate binary patterns. |
| `learnable_spatial` | Supported as a direct spatial parameterization (`tau` learned) for ablations; the paper's primary method is frequency-domain optimization (`learnable_frequency`). |
| Complex `W` | `W` is stored as a complex `nn.Parameter` initialized by `fft2(tau_0)`. |
| Sigmoid schedule reset | Algorithm 1 sets `m = 1` on epochs that do not increment. We interpret this as resetting to `m_init` (default `1.0`), not preserving the accumulated maximum. |

### Forward model

| Item | Assumption / note |
|------|-------------------|
| Impulse PSF size | Paper states impulse kernels by default. We use a `3 x 3` discrete impulse (center pixel = 1) so convolution is well-defined in PyTorch. |
| Non-impulse PSFs | Optional box kernels normalized to sum 1 are provided for future experiments; not validated against the paper yet. |

### Detector noise

| Item | Assumption / note |
|------|-------------------|
| Normalized Poisson form | Paper eq. 11 gives `y = alpha + sqrt(alpha) * z`. For training-scale inputs we apply photon-count normalization with `gamma = 10` background as in the reproduction plan: `y = alpha + gamma/k + sqrt(alpha/k + gamma/k^2) * z`. |
| `noise_normalization: paper_v3` | **AM-1 FULLY RESOLVED (RR-1 v3) — authoritative.** Paper eq. S5 `alpha_down = k·alpha_down^norm` with `alpha_down^norm = ψ(X, H_t^norm)`, `H_t^norm ∈ [0,1]`. The repo forward model already computes `ψ` with binary patterns ∈ {0,1} and image ∈ [0,1] (sum-pool, eq. 8), so its output **is** `alpha_down^norm` (range `[0, d²]`). Therefore `alpha_norm = alpha_down` directly (no `÷k`, no `÷d²`); then eqs. S7 (`y_poiss_norm`) and S9 (`y_read_norm = sigma_read/k·z`), independent draws, `gamma=10`. Output: `experiments/table01_noise_robustness/`. |
| `noise_normalization: paper` (v2, superseded) | Earlier AM-1 attempt: `alpha_norm = alpha_down / d²`. This over-normalized the signal to `[0,1]` (the paper constrains `H_t^norm ∈ [0,1]`, not `alpha^norm`), so `sigma_read/k` swamped pc=10. Kept only for reproducing the frozen `noise_table_normalized_v2/` run; do not use for new work. |
| `sigma_read` | Treated as normalized read-noise standard deviation applied after Poisson approximation when mode is `differentiable_poisson_plus_read`. With `paper`/`paper_v3`, read term is `sigma_read/k * z`. |
| Input scale | With `legacy`, the noise module added read noise as `sigma_read·z` (no `÷k`) on the sum-pool scale. With `paper_v3`, the inverse model receives `y_norm` on the sum-pool scale `[0, d²]` (eq. S10) — this is the paper's normalized `alpha_down^norm` scale; the reconstruction CNN's final Sigmoid only bounds the reconstruction (target `[0,1]`), not the measurement. |

### Inverse model

| Item | Assumption / note |
|------|-------------------|
| Locality-aware weights init | Paper does not specify initialization for per-pixel projection weights `W_{i,j}`. We initialize with `N(0, 0.01^2)` so initial patches are small. |
| Mixing CNN | Paper Fig. 2 includes a small Conv/ReLU/MaxPool CNN after tiling. Disabled by default (`use_mixing_cnn: false`); the standalone `ReconCNN` handles reconstruction for the initial reproduction. |
| Recon block order | Spec lists `Conv2d, ReLU, BatchNorm2d` per block; we follow that order literally (not Conv-BN-ReLU). |
| Paper recon architecture | Paper mentions max-pooling in the reconstruction CNN; the reproduction spec uses only Conv/ReLU/BN for the initial 6-block CNN. We follow the spec for phase 2. |
| `LocalityUpsampling` spatial size | Locality-aware mode requires fixed `(H_down, W_down)` at construction because weights are `[T, H_down, W_down, n, n]`. Transpose-convolution mode accepts variable input sizes. |

### Datasets

| Item | Assumption / note |
|------|-------------------|
| PatchMNIST patch sampling | Paper specifies a `20 x 20` grid of `32 x 32` digits (`640 x 640` canvas) and `256 x 256` patches. We randomly crop patch origins within the valid range for each generated image. |
| PatchMNIST val/test digits | Validation and test patch generation uses MNIST test digits only; training uses MNIST train digits. |
| U2OS loader | Full U2OS training requires TIFF stacks under `data/u2os/`. `U2OSPreprocessor` is fully testable; `U2OSDataset` raises a clear error until data is present. |
| U2OS substitute data | Original paper U2OS stacks are unavailable. All U2OS experiments use substitute BBBC022 widefield fluorescence TIFFs from `data/substitute_data/` (3456 single-channel 2D fields, approx. `520 x 696`, uint16). These are **not** the paper's `60 x 2304 x 2304` stacks. Substitute preprocessing uses `clip_max: 3000`, `downscale_factor: 1.0` so `256 x 256` crops fit without over-shrinking. Results are for pipeline validation only, not paper metric reproduction. |
| Debug training LRs | Paper illumination LR is `1.0`. Unit-test smoke runs may use reduced epochs/LRs for speed. |
| Training device | Default experiment device is physical **GPU 1** (`cuda:1`). Override with `experiment.device`, `--device`, or `DIFF_MICROSCOPY_DEVICE`. The earlier CPU run happened because the Cursor sandbox cannot access the NVIDIA driver. |
| Exp1 step budget | Scaled from 512-sample validation: random `max_steps=7350`; learnable staged `3000` inverse warmup + `7350` joint @ m=1 + `1500` steps each at m∈{2,4,8}. Illumination LR `0.3`, inverse LR `0.001`. |

### Experiments

| Item | Assumption / note |
|------|-------------------|
| `cell.tif` | Local sanity-check file (`512 x 512`, uint16). Not a full U2OS `60 x 2304 x 2304` stack. Used only to validate preprocessing and patch extraction. |
| U2OS ablation schedules | Paper uses `12150` epochs per stage. Experiment configs use the scaled debug schedule (`epoch_baseline=20`) until the pipeline is validated. |
| Noise sweep matrix | Official noise grid: `configs/table01_noise_robustness/noise_table.yaml` → `experiments/table01_noise_robustness/`. Additional sweeps can be generated via `build_noise_sweep_experiments()`. |
| Exp1 / LR=0.3 historical runs | Superseded by `experiments/content_aware/` (LR=1.0 paper-aligned PatchMNIST). Step-budget notes below describe the earlier mechanism-validation recipe only. |
| Paper target metrics | U2OS x16 ablation target SSIM/MSE values are logged in the spec for ordering checks only; exact matching is not required initially. |
