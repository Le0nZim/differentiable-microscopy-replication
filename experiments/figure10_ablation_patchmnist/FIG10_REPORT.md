# Figure 10 / Table 3 ablation (A/B/C/D) on PatchMNIST

**Status:** completed. This is a *dataset swap* of the BBBC022 Fig-10 ablation
(`experiments/figure10_ablation/`), run to test whether that run's surprising
"variant D wins" result was an artifact of the BBBC022 substitute data.

Rendered figures:
- `figures/figure10_paper_style.png` — paper-layout qualitative panel (3 rows a/b/c × 4 cols A/B/C/D).
- `figures/figure10_table3_comparison.png` — Table-3 metrics: ours (PatchMNIST) vs paper (U2OS).

## What changed vs `experiments/figure10_ablation/`

**Only the dataset.** The protocol, the A/B/C/D knobs, the optimizer, the phase
schedule and the checkpoint rule all run through the exact same code path
(`run_am3_table3.run_one` + `apply_variant` + `default_phases`, reached via the
Fig-10 driver), so the two experiments are directly comparable.

| | `figure10_ablation` | `figure10_ablation_patchmnist` (this) |
|---|---|---|
| data | BBBC022 Hoechst nuclei substitute | **PatchMNIST** |
| source of the data block | `configs/figure10_ablation/ablation.yaml` | `configs/table01_noise_robustness/noise_table.yaml` (the Fig-6 / Table-1 data) |
| split | `split_fig03_large`, 1980 / 40 / 60 images | 3000 / 375 / 375 canvases, disjoint val/test digit pools |
| preprocessing | `minimal_percentile` (per-image q0.1–99.9 clip → min-max) | none beyond PatchMNIST's native `[0,1]` scaling (synthetic data) |
| everything else | ×16 (d=8, T=4), impulse PSFs, noise-free, L1, illum lr 1.0 / inverse lr 0.001, grad-clip 1.0, batch 32, Algorithm-1 8500 steps, global best-val-MSE checkpoint | identical |

PatchMNIST canvases are 256 px crops of a 20×20 grid of 32 px MNIST digits, generated
by `src/datasets/patchmnist.py` and loaded through the same `build_dataloader` the
noise-robustness experiments use. Fig-6 and Table-1 share one data pipeline
(Fig-6's panels are rendered from the Table-1 runs), so "the Fig-6 loader" and
"the Table-1 loader" are the same thing.

**Noise stays off.** Table 1 / Figure 6 exist to sweep detector noise, but noise
lives in the forward model, not in the data loading. Only the data was swapped, so
the forward model remains noise-free exactly as in Fig-10 — otherwise the A/B/C/D
comparison would not be apples-to-apples with the BBBC022 run.

All four variants pass the machine-checkable `variant_audit.check_variant()`
wiring gate at ×16 / T=4 (`wiring_problems: []` in each
`runs/<L>_seed42/metrics/variant_metadata.json`).

## Results

Seed 42 for all four variants; C and D additionally at seeds 43 and 44, because
the C-vs-D margin is small enough to need a seed spread
(`runs/aggregate_multiseed.json`).

| variant | SSIM ↑ | MSE ↓ | seeds |
|---|---|---|---|
| A: fixed Hₜ + Tr.Conv.Up + freq | 0.7025 | 0.03611 | 42 |
| B: (+) learnable Hₜ | 0.8939 | 0.01161 | 42 |
| C: (+) locality upsampling *(paper best)* | 0.9357 ± 0.0012 | 0.00695 ± 0.00006 | 42, 43, 44 |
| **D: (−) freq-domain optimization** | **0.9401 ± 0.0002** | **0.00655 ± 0.00002** | 42, 43, 44 |

### The headline answer

**Removing frequency-domain optimization still helps on PatchMNIST, so the
paper's claim does not reproduce here either — but the effect is far smaller
than the BBBC022 run made it look, and the rest of the ablation reproduces much
more convincingly.**

| | BBBC022 substitute | PatchMNIST |
|---|---|---|
| A → C MSE improvement (does the ablation discriminate at all?) | 1.25× | **5.2×** |
| D vs C relative MSE gap (how much does dropping freq-domain opt buy?) | **37%** | **5.7%** |
| D beats C on every seed | 1 seed only | yes (3/3) |

On BBBC022 every variant landed at SSIM 0.90–0.93 and the qualitative panel
(`experiments/figure10_ablation/figures/figure10_paper_style.png`) shows four
essentially indistinguishable reconstructions: Hoechst nuclei are large, smooth
and low-frequency, so ×16 compression barely hurts and the ablation has almost no
dynamic range to measure. PatchMNIST's sparse high-contrast strokes are genuinely
hard at ×16, and the panel separates the variants clearly (A is visibly
degraded, B is smeared, C and D are crisp).

### What reproduces

- **Learnable > fixed:** B and C beat A by a wide margin (MSE 0.0116 and 0.0070 vs 0.0361;
  SSIM 0.894 and 0.936 vs 0.702). ✔ Much stronger than on BBBC022 (0.00108 / 0.00100 vs 0.00125).
- **Locality-aware > transpose conv:** C beats B (MSE 0.00695 vs 0.01161, a 40% reduction;
  SSIM 0.936 vs 0.894). ✔ On BBBC022 this held by only 7%.
- **A → B → C is strictly monotonically improving**, matching the paper.

### What does not reproduce

- **"Frequency-domain optimization is important":** D is still the best variant, on all
  three seeds, so the paper's finding that D is *worst* does not reproduce on PatchMNIST.
  The margin, however, collapses from 37% to 5.7% — C and D are near-equivalent here,
  whereas on BBBC022 D looked decisively better.

## Why the BBBC022 run overstated the D advantage

The per-run diagnostics point at an optimization failure on BBBC022 rather than a
property of frequency-domain parametrization. Comparing `illum_delta_final` (how
far the illumination actually moved from its initialization during training):

| variant | BBBC022 illum Δ | PatchMNIST illum Δ |
|---|---|---|
| B (learnable frequency) | 125 | 4816 |
| C (learnable frequency) | 266 | 3924 |
| D (learnable spatial) | 3791 | 4227 |

On BBBC022 the frequency-parametrized variants barely trained their illumination
(B and C moved 14–30× less than D), and B's final patterns were nowhere near binary
(binary fraction 0.27, best checkpoint at m=2 — it never survived hardening). So
the BBBC022 "D wins by 37%" is substantially **"B and C failed to learn their
patterns"**, not "the spatial parametrization is better". On PatchMNIST all three
learnable variants move comparably far, B and C harden properly (binary fractions
0.77 and 0.78, both best at m=8), and once they actually train, C closes almost the
entire gap to D.

The residual 5.7% should be interpreted as an optimizer-coordinate effect under
the evaluated recipe. No Fourier coefficients are truncated, so the Fourier and
spatial branches span the same attainable spatial-logit family; neither branch has
more per-pixel degrees of freedom. Adam is not invariant to this coordinate change,
and a shared nominal illumination learning rate does not equalize physical pattern
updates. A parameterization-specific learning-rate sweep would be needed to claim
that one coordinate system is intrinsically better. Table 3 remains numerically
**DATA_BLOCKED** without the paper's U2OS data (see
`experiments/table03_ablation/U2OS_DATA_STATUS.md`).

The same `learnable_spatial` swap under the Figure-6 / Table-1 *noise* protocol
(×8, T=8, `paper_v3` detector) is in
`experiments/figure06_noise_robustness_no_freq/`. There the no-freq arm wins
every cell too, and by more (~23% MSE at pc=10) than the 5.7% seen in this
noise-free ×16 ablation.

## Reproduce

```bash
# train all four variants (GPU0: A,B ; GPU1: C,D) + auto-render
PY=path/to/python bash scripts/figure10_ablation_patchmnist/train_all_and_render.sh

# or per-variant / per-seed
python scripts/figure10_ablation_patchmnist/train.py --variants A B C D --seed 42 --device cuda:0

# multi-seed C-vs-D aggregate
python scripts/figure10_ablation_patchmnist/aggregate_seeds.py

# re-render only (from saved run artifacts)
python scripts/figure10_ablation/reproduce.py \
    --runs experiments/figure10_ablation_patchmnist/runs \
    --out-dir experiments/figure10_ablation_patchmnist/figures \
    --data-label "PatchMNIST (Fig-6 / Table-1 data)" \
    --table-label "PatchMNIST, Fig-6 / Table-1 data" \
    --short-label PatchMNIST \
    --mirror-dir ""
```

Config: `configs/figure10_ablation_patchmnist/ablation.yaml`.
Artifacts per variant: `runs/<L>_seed<N>/{checkpoints_best.pt, learned_patterns/H_t.pt,
figures/qualitative_tensors.pt, metrics/{run_summary,diagnostics,variant_metadata,step_log}.*}`.
Aggregates: `runs/aggregate_summary.json` (seed 42, paper-comparison format) and
`runs/aggregate_multiseed.json` (per-variant mean/std over all seeds present).
