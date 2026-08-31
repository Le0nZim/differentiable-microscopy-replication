# Udith legacy schedule: corrected C vs D on PatchMNIST

**Label:** Udith legacy schedule under the current controlled PatchMNIST optimizer/data recipe. This is **not** a full reproduction of Udith’s legacy training loop.

**Status:** schedule-only causal experiment. Isolated from `experiments/figure10_ablation_patchmnist/`.

## Checkpointing and data iterator (same for C and D)

- Phase-boundary evaluations **are eligible for global-best checkpoint selection**. They use the same `best.update` path as regular `log_every` rows. They are **not** diagnostic-only. This is not changed between C and D.
- Step **121,500** is always evaluated: it is the last step of the last phase (the five-step \(m=8\) tail). It is **not** on the `log_every=200` grid (`121500 % 200 == 100`); last-step logging is what includes it.
- The inherited runner uses `itertools.cycle(train_loader)`. It caches and repeats the first loader traversal. This remains fair because both schedules and both C/D arms use it, but differs from Udith’s original per-epoch loader recreation.
- Physical illumination displacement is \(\|	au-\tau_0\|_2\) and \(\|H_t-H_{t,0}\|_2\). Do **not** compare raw Fourier-parameter norms against spatial-parameter norms.

## Interpretation boundaries

- This tests whether the **legacy schedule** changes the corrected C/D ordering on PatchMNIST.
- It is **not** a U2OS reproduction.
- It does **not** validate the historical Table-3 D condition (historical D retained an IFFT; ours is direct spatial `learnable_spatial`).
- It does **not** test the full historical optimizer loop (still the current single Adam, simultaneous update, grad-clip 1.0).
- Do not edit the manuscript or replace any current paper figure from these numbers until the report is reviewed.

## What was held fixed vs the completed 8,500-step PatchMNIST run

Dataset (PatchMNIST 3000/375/375, disjoint val/test), ×16, T=4, d=8, impulse PSFs, noise-free, locality-aware upsampling, L1, batch 32, illum LR 1.0, inverse LR 0.001, single Adam with parameter groups, grad-clip 1.0, simultaneous update, global best-val-MSE checkpointing, validation cadence `log_every=200`.

The only causal change is the training schedule: 121,500 optimizer steps matching 24,300 legacy epochs × 5 minibatches/epoch, with accumulating m = 1,2,3,4,5,6,7,8 and a faithful 5-step m=8 tail. C and D share a paired τ₀ initialization (C: W₀=FFT2(τ₀), D: τ₀).

## Schedule (optimizer steps)

| Global steps | State |
| --- | --- |
| 1–60,750 | illumination frozen, m=1 |
| 60,751–97,195 | joint training, m=1 |
| 97,196–101,245 | m=2 |
| 101,246–105,295 | m=3 |
| 105,296–109,345 | m=4 |
| 109,346–113,395 | m=5 |
| 113,396–117,445 | m=6 |
| 117,446–121,495 | m=7 |
| 121,496–121,500 | m=8 |

## Per-seed paired differences (D − C)

| seed | C MSE | D MSE | D−C MSE | C SSIM | D SSIM | D−C SSIM | C best m/step | D best m/step | C bin.frac | D bin.frac | C ‖τ−τ₀‖ | D ‖τ−τ₀‖ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.005334 | 0.006745 | 0.001411 | 0.9525 | 0.9372 | -0.0153 | 7.0/118800 | 7.0/120400 | 0.633 | 1.000 | 428.88 | 6347.17 |

## Mean ± std

- **C (Udith):** MSE 0.005334 ± 0.000000, SSIM 0.9525 ± 0.0000 (n=1).
- **D (Udith):** MSE 0.006745 ± 0.000000, SSIM 0.9372 ± 0.0000 (n=1).
- **D − C (paired):** MSE 0.001411 ± 0.000000, SSIM -0.0153 ± 0.0000 (n=1).
- **C (current 8,500):** MSE 0.006948 ± 6.2e-05, SSIM 0.9357 ± 0.0012.
- **D (current 8,500):** MSE 0.006553 ± 2.1e-05, SSIM 0.9401 ± 0.0002.

## Pattern diagnostics (best checkpoint)

Physical displacements only (\(\|	au-\tau_0\|_2\), \(\|H_t-H_{t,0}\|_2\)). Raw Fourier-parameter norms are not compared to spatial-parameter norms.

| seed | C binary frac | D binary frac | C ‖τ−τ₀‖ | D ‖τ−τ₀‖ | C ‖H−H₀‖ | D ‖H−H₀‖ |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.633 | 1.000 | 428.88 | 6347.17 | 180.97 | 247.82 |

## Verdict

`C_beats_D_all_seeds`

C wins on every seed. The C/D ordering is schedule-sensitive, and the legacy schedule favors frequency-domain parameterization under this dataset.

## Figures

- `figures/cd_panel_udith_schedule.png`
- `figures/val_mse_curves.png`
- `figures/binary_fraction_curves.png`
- `figures/current_vs_udith_mse.png`

Reproduce:

```bash
PY=path/to/python python scripts/figure10_ablation_patchmnist_udith_schedule/train.py --protocol shared_warmup --device cuda:0 --allow-gpu0 --skip-gpu-check --seeds 42 43 44 --variants C D
```
