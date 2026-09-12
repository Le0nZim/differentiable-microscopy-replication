# Rerun protocol after the scientific audit

For the integrated workstation setup and sequential paper queue, start with
[`START_HERE.md`](../START_HERE.md). The controlled protocol below remains a
separate scientific experiment, available as `python paper.py run --stages controlled`.

Read [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md) before interpreting historical
tables. The new runner is a controlled CNN experiment, not a replacement for
every SwinIR or segmentation driver and not a numerical U2OS reproduction.
Run commands from the repository root. Use a fresh output directory per
configuration; the runner refuses to reuse nonempty per-run folders.

## 1. Software checks

Install `pip install -e '.[dev]'` and the pinned SwinIR clone described in the
root README if running its tests. Then:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -m 'not slow'

python scripts/audit/check_original.py \
  --original-root ../CompressiveDabbaMu \
  --output experiments/audit_checks/original_diagnostics.json

python scripts/audit/run_controlled.py \
  --config configs/audit/controlled_patchmnist.yaml \
  --device cpu --seeds 42 --smoke \
  --output experiments/audit_checks/synthetic_smoke
```

The upstream check requires commit
`f8cc74b51847bd31b320a7f153dcf3eedecc3e7a`. The smoke uses random 16×16 tensors,
small networks and eight updates. Its scores are software evidence only.
It deliberately has short final batches and tests all five illumination modes.

## 2. Declare the protocol before examining new test scores

The checked-in controlled configuration defines:

| Quantity | Setting / interpretation |
|---|---|
| Training seeds | 42, 43, 44; compare methods as paired seeds |
| Data seed | Fixed at 42 across methods and training seeds |
| PatchMNIST splits | 3,000 / 375 / 375; globally disjoint MNIST-test digit pools for validation/test |
| BBBC022 | Existing well-disjoint 1,980/40/60 split, per-image 0.1%/99.9% normalization; widefield substitute |
| Updates | 10,100 per method, including 1,500 inverse-only warmup updates |
| Loss / inverse optimizer | L1; Adam, inverse LR 0.001 |
| Illumination optimizer | Adam LR 1.0, one declared point rather than a tuned optimum |
| Masks | Exact binary hard forward with sigmoid STE gradients; full spatial resolution |
| m | 1 during warmup, then linear surrogate hardening to 8; binary forward throughout |
| Shared initialization | Same inverse weights within a training seed; spatial/Fourier begin from the same random physical mask |
| Data iteration | Independent loader RNG; fresh iterator/crops on each pass |
| Dose | `equal_mean`, sequence dose 2 full-on equivalents by default; run `fixed_peak` sensitivity too |
| Training detector | Gaussian approximation in explicit `paper_v3` units, k=10,000, gamma=10, sigma=2 |
| Model selection | Minimum validation MSE at a fixed logging interval; exact Poisson plus read noise, fixed validation noise seed 201 |
| Primary test | Selected checkpoint only; exact Poisson/read draws 301,302,303 |
| Diagnostic test | Gaussian-approximation draw 301, separately labeled |
| Metrics | Image-mean MSE, image-mean PSNR, 11×11 Gaussian SSIM with reflect padding and range 1 |

Same validation noise realizations make checkpoint comparisons repeatable but
do not eliminate selection uncertainty. Additional validation noise draws and
learning-rate choices must be decided on validation, not on test winners.
Test random draws are nested measurement uncertainty, not independent training
replicates. Three training seeds do not supply a precise confidence interval.

The existing test sets have already been used during earlier design work.
Results on them remain exploratory even when held out from gradient training.
For confirmatory claims, allocate a new sealed well/specimen/plate holdout
appropriate to the desired generalization setting. Audit source identities,
not only image crops. Do not claim unseen treatments from a well-only split.

## 3. Smallest useful reconstruction comparisons

Prepare MNIST under `data/mnist` (both train and test) as described in
`data/README.md`; the controlled configuration sets `download: false` to make
data preparation explicit. BBBC022 source images must match the split manifest.

```bash
# Five methods, same inverse network and acquisition budget: C=16, d=8, T=4.
python scripts/audit/run_controlled.py \
  --config configs/audit/controlled_patchmnist.yaml \
  --device cuda:1 --seeds 42 43 44 \
  --output experiments/controlled_v1/patchmnist_c16_equal_mean

# The alternative physical question: fixed on-pixel illumination scale.
python scripts/audit/run_controlled.py \
  --config configs/audit/controlled_patchmnist.yaml \
  --dose-mode fixed_peak --device cuda:1 --seeds 42 43 44 \
  --output experiments/controlled_v1/patchmnist_c16_fixed_peak

# Original custom_v2 projection/channel lift and original decoder widths.
python scripts/audit/run_controlled.py \
  --config configs/audit/original_architecture_patchmnist.yaml \
  --device cuda:1 --seeds 42 43 44 \
  --output experiments/controlled_v1/patchmnist_original_locality

# Change only the original upsampler while retaining those decoder widths.
python scripts/audit/run_controlled.py \
  --config configs/audit/original_architecture_patchmnist.yaml \
  --upsampling original_transpose --device cuda:1 --seeds 42 43 44 \
  --output experiments/controlled_v1/patchmnist_original_transpose

# BBBC022 substitute at C=16. Repeat d=16,32,64 for C=64,256,1024.
python scripts/audit/run_controlled.py \
  --config configs/audit/controlled_bbbc022.yaml \
  --downscale 8 --num-patterns 4 --device cuda:1 --seeds 42 43 44 \
  --output experiments/controlled_v1/bbbc022_c16_equal_mean
```

`--modes` can restrict to a comparison, e.g.
`--modes learnable_spatial learnable_frequency`. Use the same seeds and
configuration on both sides. Different upsamplers generally cannot share
inverse weights; pair data and seeds and report this limitation.

## 4. Noise robustness and compression sweeps

For a controlled counterpart to Table 1 use T=8, d=8 (C=8), and cross
k∈{10,10000} with sigma∈{0,2.7,2,6}. Train all prescribed seeds in every cell.
For example:

```bash
python scripts/audit/run_controlled.py \
  --config configs/audit/controlled_patchmnist.yaml \
  --num-patterns 8 --downscale 8 --photon-count 10 --sigma-read 6 \
  --modes random_fixed learnable_spatial learnable_frequency \
  --device cuda:1 --seeds 42 43 44 \
  --output experiments/controlled_v1/noise_c8_k10_sigma6
```

The default equal-mean sequence dose stays fixed when T changes, so each
exposure's mean dose changes. If the desired experiment instead keeps peak
power/exposure fixed, use and report `--dose-mode fixed_peak`. A sample-count
compression sweep and a fixed-dose sweep answer different questions.

Historical Figure 3 and Table 1 drivers now equalize update counts and write
to `experiments/audited_v1/`. They retain soft-mask legacy protocol choices;
their outputs must not be merged with this binary controlled experiment.
Historical renderers still point to historical artifacts until deliberately
rebound and regenerated from the new selected checkpoints.

## 5. Check evidence, then analyze

Each run writes its complete configuration, distinct `best.pt`/`last.pt`,
evaluated masks, per-image test metrics for each noise draw, mask/dose statistics,
validation/test tensor hashes, inverse-initialization hash, checkpoint hash,
selected m and step. Check that paired methods have:

- Identical data fingerprints, prescribed updates, detector geometry, T and C.
- Identical shared inverse initialization within a seed for a fixed architecture.
- Exact binary status and the intended mean-dose or fixed-peak rule.
- Finite metrics and gradients; selected checkpoints that replay strictly.

Average noise draws within each trained model first. Report per-training-seed
paired differences, mean and spread across seeds, and raw results. Bootstrap
biological units (such as wells/specimens) for specimen uncertainty; treating
multiple crops or sites from one well as independent replicates inflates n.
Keep MSE-averaged-then-log PSNR distinct from mean per-image PSNR. Report effect
sizes even if a preferred method loses. Prespecify qualitative image identities
and shared display normalization; display the masks actually evaluated.

## 6. Figure 9 and downstream tasks

For a 256×1280 field, tile=256, d=8, T=4, nonoverlap acquires five tiles and
20,480 detector scalar measurements: C=16. Overlap=64 acquires seven tiles
and 28,672 measurements: effective C=80/7≈11.43. The overlap helper reruns
the full optical model; smoothing is not free decoder postprocessing. With
tile=64 and overlap=16, the implemented boundary-aligned grid uses 135 tiles
instead of 80, giving effective C≈9.48 at the same per-tile C=16. The fresh
workstation renderer smoke verifies these counts against its acquisition metadata.

Use the updated Figure 9 metadata to report both acquisition totals. A future
decoder-only overlap method must first acquire one shared global measurement
tensor and crop that tensor for reconstruction; it cannot call the microscope
on overlapping ground-truth crops and claim a fixed acquisition budget.

For SwinIR, recheck full training/loss settings, loss-weight search, scene
splits, equal budgets, actual deployed masks and photon constraints. For
segmentation, keep the pseudo-mask task distinct from independent biological
annotation, preserve frozen BatchNorm state in the head-only phase, and select
thresholds using validation. These downstream experiments need dedicated full
reruns; the CNN smoke does not validate their scientific conclusions.
