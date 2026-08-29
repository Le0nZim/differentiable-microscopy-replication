# Figure 6 / Table 1 noise robustness without frequency-domain optimization

**Status:** completed on a *subset* of the Table-1 grid. Rendered figures:

- `plots/fig06_style_extreme_cell.png` — Figure-6-style panel at the extreme cell (pc=10, σ=6), with the no-freq arm added.
- `plots/per_cell_mse.png` — per-cell test MSE, three arms.
- `plots/mse_vs_read_noise.png` — MSE vs read noise at both photon counts.

## What Figure 6 / Table 1 actually train

Yes — the published noise-robustness experiment uses frequency-domain optimization on the learnable arm. Table 1 / Figure 6 compare two illumination modes at ×8, T=8, on PatchMNIST:

| arm | `pattern_generator.mode` | frequency-domain opt? |
|---|---|---|
| fixed random | `random_fixed` | no (patterns are not trained) |
| learnable (paper) | `learnable_frequency` | **yes** (`tau = real(ifft(W))`) |

Figure 6 is a qualitative panel at the *extreme cell* (photon_count=10, σ_read=6.0) rendered from those same Table-1 checkpoints. There is no separate Figure-6 trainer.

## What this study changes

Adds the missing third arm — the paper's Table-3 variant D — and asks whether dropping frequency-domain optimization also helps *under detector noise*.

| | Table 1 / Figure 6 | this study |
|---|---|---|
| data | PatchMNIST 3000/375/375, disjoint val/test | identical |
| compression | ×8 (d=8, T=8) | identical |
| detector | `paper_v3` Poisson + read noise | identical |
| inverse | locality-aware + ReconCNN | identical |
| schedule | staged hardening 3000 + 7350 + 3×1500 | identical |
| learnable mode | `learnable_frequency` | **`learnable_spatial`** |
| cells trained | full 8-cell grid + extra seeds at extreme | **4 corners** {10, 10000} × {0.0, 6.0}, plus seeds 43/44 at the extreme cell |

The frozen Table-1 `random_fixed` and `learnable_frequency` checkpoints are read back read-only; they are never retrained. All six new runs pass through the same `train_staged_hardening` path Table 1 uses (`learn_patterns=True`, `use_staged_hardening=True`; `pattern_is_learnable` covers both learnable modes).

## Results

Seed 42 for all four corners; extreme cell also at seeds 43 and 44.

| photon_count | σ_read | fixed MSE | learnable + freq | learnable, NO freq | no-freq / freq |
|---|---:|---:|---:|---:|---:|
| 10 | 0.0 | 0.0208 | 0.0064 | **0.0049** | 0.767 |
| 10 | 6.0 | 0.0230 | 0.0074 | **0.0057** | 0.771 |
| 10000 | 0.0 | 0.0109 | 0.0030 | **0.0029** | 0.979 |
| 10000 | 6.0 | 0.0110 | 0.0032 | **0.0029** | 0.929 |

Extreme cell (pc=10, σ=6), three training seeds:

| method | per-seed MSE | mean |
|---|---|---:|
| fixed random | 0.0230, 0.0224, 0.0232 | 0.0229 |
| learnable + freq (paper) | 0.0074, 0.0074, 0.0074 | 0.0074 |
| learnable, NO freq | 0.0057, 0.0057, 0.0057 | **0.0057** |

### The headline answer

**Removing frequency-domain optimization helps on every cell we ran, and the benefit is much larger in the low-photon regime that Figure 6 is about.**

- At pc=10 (the Figure-6 operating point) the spatial parametrization cuts MSE by **~23%** relative to the paper's frequency arm (ratio 0.77), on both σ=0 and σ=6, and on all three extreme-cell seeds with no overlap.
- At pc=10000 the same swap is nearly a wash (2–7%): both learnable arms already sit at MSE ≈ 0.003, so there is little left to gain.
- Both learnable arms still crush fixed random (~3–4× at pc=10, ~3.7× at pc=10000). The Table-1 headline ("learnable beats fixed") is not an artifact of the Fourier parametrization.

This is a *larger* frequency-domain penalty than the noise-free Figure-10 PatchMNIST ablation, where dropping freq-domain opt only bought 5.7%. Two differences matter: Figure 10 is ×16 / T=4 / noise-free; this sweep is ×8 / T=8 / with the paper's detector. The extra per-pixel degrees of freedom in `learnable_spatial` appear to help more when the measurements are actually noisy.

Qualitatively, at the extreme cell both learnable reconstructions recover readable digits and the no-freq panel is not visually worse (if anything slightly crisper). The fixed-random panel is still the smear the paper reports.

## What this does *not* say

- It does not rerun the four interior cells (σ_read ∈ {2.0, 2.7}). Table 1 is almost flat in σ_read at each photon count, and both corners at each pc already agree, so a reversal in the middle is unlikely — but those cells were not trained.
- It does not claim the paper's U2OS Figure-6 would look the same. Same DATA_BLOCKED caveat as Table 3 / Figure 10: the paper's frequency-domain-optimization benefit is a U2OS-distribution claim we cannot test.

## Reproduce

```bash
# default: 4 corner cells + extra seeds at the extreme cell (two GPUs)
PY=path/to/python bash scripts/figure06_noise_robustness_no_freq/launch.sh corners

# optional: the whole 8-cell Table-1 grid
PY=path/to/python bash scripts/figure06_noise_robustness_no_freq/launch.sh full

# re-aggregate only (reads frozen Table-1 checkpoints + these runs)
python scripts/figure06_noise_robustness_no_freq/run.py --cells corners --aggregate-only --device cuda:0
```

Config: `configs/figure06_noise_robustness_no_freq/noise_table.yaml` (identical to
`configs/table01_noise_robustness/noise_table.yaml` except `pattern_generator.mode: learnable_spatial`).
Aggregates: `results.md`, `results.json`, `results.csv`.
