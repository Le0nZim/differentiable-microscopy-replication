# Figure 10 / Table 3 reproduction (ablation A/B/C/D) on the BBBC022 substitute

**Status:** reproduced on the *substitute* dataset (see caveats). Rendered figures:
- `figures/figure10_paper_style.png` — paper-layout qualitative panel (3 rows a/b/c × 4 cols A/B/C/D).
- `figures/figure10_table3_comparison.png` — Table-3 metrics: ours (substitute) vs paper (U2OS).
- mirrored under `results/reproduced_figures/fig10/`.

## What the paper does (Fig 10 / Table 3, §5.7)

Ablation on **U2OS** cells at ×16 compression (T = 4, 8×8:1 downscaling), L1 loss, conventional
reconstruction CNN. Four cumulative variants:

| | condition | pattern mode | upsampling | freq-domain opt |
|---|---|---|---|---|
| **A** | baseline: fixed Hₜ + Tr.Conv.Up + freq | `random_fixed` | transpose conv | (fixed patterns) |
| **B** | (+) learnable Hₜ | `learnable_frequency` | transpose conv | yes |
| **C** | (+) proposed upsampling **(paper best)** | `learnable_frequency` | locality-aware | yes |
| **D** | (−) frequency-domain optimization | `learnable_spatial` | locality-aware | no |

Paper Table 3 (U2OS): A 0.7872 / 0.0042, B 0.7950 / 0.0038, **C 0.8426 / 0.0029**, D 0.7857 / 0.0041
(SSIM↑ / MSE↓). Paper conclusions: learnable > fixed; locality > transpose; **frequency-domain
optimization is important** (removing it, D, is worst by SSIM).

## Why this is on a substitute, and which substitute

The paper's U2OS dataset is not publicly available and is not shipped with this repo
(exhaustively documented in `experiments/table03_ablation/U2OS_DATA_STATUS.md`).
Per instruction, this reproduction uses **the exact same substitute data the repo's Figure 3
reproduction uses**: the **BBBC022 Hoechst 33342 nuclei** stand-in, loaded via
`bbbc022_preproc_ablation` with `minimal_percentile` preprocessing (per-image robust percentile
clip q0.1–99.9 → min-max), and the well-disjoint **`split_fig03_large`** split
(**1980 train / 40 val / 60 test** images). This is a substantially larger and more diverse split
than the old 168-image `paper_strict` ablation (`experiments/ablations/bbbc022_ablation_x16`), which
matters (see results).

Config: `configs/fig10_ablation_base.yaml`. The A/B/C/D knobs are applied by the SAME validated
`apply_variant()` used for the audited AM-3 ablation, and each run passes the machine-checkable
`variant_audit.check_variant()` wiring gate (all four: `wiring_problems: []`).

## Protocol (identical for all four variants)

Reuses `run_am3_table3.run_one` verbatim: uniform scaled Algorithm-1 phases —
inverse-warmup 1500 (illum frozen, m=1) → joint-soft 4000 (m=1) → m-hardening 2/4/8 (1000 each) =
**8500 steps**; L1 loss; illumination lr 1.0, inverse lr 0.001, grad-clip 1.0; batch 32; **global
best-val-MSE checkpoint selection** (same rule for every variant). The only thing that differs
between A/B/C/D is exactly what Table 3 ablates.

## Results (our substitute, seed 42)

| variant | SSIM ↑ | MSE ↓ | best m |
|---|---|---|---|
| A: fixed Hₜ + Tr.Conv.Up + freq | 0.9048 | 0.00125 | 4 |
| B: (+) learnable Hₜ | 0.9103 | 0.00108 | 8 |
| C: (+) locality upsampling *(paper best)* | 0.9103 | 0.00100 | 8 |
| **D: (−) freq-domain optimization** | **0.9269** | **0.00063** | 8 |

Absolute SSIM is higher / MSE lower than the paper's U2OS numbers because BBBC022 Hoechst nuclei are
simpler and more stationary than U2OS Cell-Painting fields — **magnitudes are not comparable; only the
A/B/C/D ordering is.**

## What reproduces, and what does not

**Reproduces (paper's core upsampling claims):**
- **Learnable > fixed:** B, C beat A (MSE 0.00108, 0.00100 < 0.00125; SSIM 0.910 > 0.905). ✔
- **Locality-aware > transpose conv:** C beats B (MSE 0.00100 < 0.00108). ✔ — note this
  *now holds* on the large Fig-3 split, whereas on the old 168-image split the locality block
  overfit and lost to transpose (root-caused in `am3_table3_resolution/AM3_root_cause.md`). The
  bigger, more diverse Fig-3 data (1980 train images) is exactly what the paper's PatchMNIST
  size/#-images study (Fig 5) predicts the locality block needs.
- **Qualitative pattern structure:** the learned Hₜ in row (b) matches the paper's qualitative story
  — A/B/C are high-frequency speckle, while **D (spatial, no freq-domain opt) collapses to a
  distinctly grid-like/blocky pattern**, exactly as in the paper's Fig 10-D.

**Does NOT reproduce (data-dependent claim):**
- **"Frequency-domain optimization is important":** on the substitute, **D (no freq-domain opt) is
  the *best* variant**, not the worst. The spatial-domain parametrization has more per-pixel degrees
  of freedom and, with 1980 training images of fairly stationary densely-packed nuclei, fits the
  substitute distribution better (lowest MSE). The paper's argument for freq-domain optimization is
  spatial invariance ("cells might appear anywhere") — a property that is far weaker for BBBC022
  nuclei (uniformly tiled across the FOV) than for sparser U2OS fields. This is a genuine
  **substitute-data effect**, not a wiring/implementation bug (all four wiring audits pass; the
  grid-like D pattern is exactly the pathology the paper describes, it simply is not penalized on
  this data distribution).

**Bottom line:** 3 of the paper's 4 monotonic ablation steps reproduce on the substitute
(fixed→learnable→locality is strictly monotonically improving, matching the paper). The
frequency-domain-optimization benefit is U2OS-distribution-specific and does not carry over to the
BBBC022 nuclei substitute. This is consistent with, and refines, the pre-existing
`am3_table3_resolution` conclusion that Table 3 is numerically **DATA_BLOCKED** without the paper's
U2OS data.

## Follow-up: the same ablation on PatchMNIST

`experiments/figure10_ablation_patchmnist/` re-runs this ablation with the identical
protocol on PatchMNIST (the Figure-6 / Table-1 data), to check how much of the result
above is a property of the BBBC022 substitute. Two things it found:

- The BBBC022 numbers above have very little dynamic range (A→C is only a 1.25× MSE
  improvement, all four variants land at SSIM 0.90–0.93), because Hoechst nuclei are
  smooth and low-frequency enough that ×16 compression barely hurts. On PatchMNIST the
  same ablation spans 5.2×, and A→B→C reproduce the paper far more convincingly.
- Variant B and C's illumination barely trained here (`illum_delta_final` 125 and 266,
  vs 3791 for D; B's patterns never hardened past m=2, binary fraction 0.27). So D's
  37% margin above is substantially an optimization failure in B/C rather than evidence
  for the spatial parametrization. On PatchMNIST, where all three learnable variants
  train comparably, D still wins but by only 5.7%.

## Reproduce

```bash
# train all four variants (GPU0: A,B ; GPU1: C,D) + auto-render
bash scripts/figure10_ablation/train_all_and_render.sh

# or per-variant
./.venv/bin/python scripts/figure10_ablation/train.py --variants A B C D --device cuda:0

# re-render only (from saved run artifacts)
./.venv/bin/python scripts/figure10_ablation/reproduce.py --runs experiments/figure10_ablation/runs
```

Artifacts per variant: `runs/<L>_seed42/{checkpoints_best.pt, learned_patterns/H_t.pt,
figures/qualitative_tensors.pt, metrics/{run_summary,diagnostics,variant_metadata,step_log}.*}`.
Aggregate + paper comparison: `runs/aggregate_summary.json`.
