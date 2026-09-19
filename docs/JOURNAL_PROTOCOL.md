# Finalized experiment protocol: journal_v2

This is the prospective protocol for the next manuscript, built on audit commit
`10f79d74ce3fdb3dd3b748adcf612505d4cc5e48`. Its runnable source is the
`journal/final-protocol` branch. All earlier fixes are included. Historical
results remain historical; they are not inputs to this campaign. This revision
replaces the journal_v1 pseudo-mask segmentation target with real manual
annotations at the user's request, and restores the original pseudo-mask
definition for the explicit fallback. Other experiment recipes are unchanged.

The scientific question is whether jointly optimized illumination improves
reconstruction or nucleus segmentation under explicitly specified simulated
acquisitions, and whether that result depends on the inverse architecture and
illumination coordinates. A learned method is allowed to lose. No completion
gate requires a favorable ranking.

The original U2OS confocal data and exact original SwinIR execution recipes are
unavailable. Consequently this supports a corrected computational replication
and sensitivity study, not exact numerical reproduction or an experimental
hardware validation. The paper's claim that the IFFT parameterization itself
introduces optical convolution/translation invariance is not retained.

## Decisions by experiment

| Experiment | Final scientific choice | Reason and interpretation |
|---|---|---|
| Fig. 3, content-aware | Original locality block and CNN widths; BBBC022 native-resolution percentile-normalized specimens; simulated photon scale 10,000. All four illumination types at T=4; additional random/learned pairs at T=1 and T=16 for each compression. | Original inverse architecture removes a reproduction confound. Additional geometry conditions distinguish compression from the number of exposures. BBBC022 is explicitly a substitute. |
| Fig. 3 / Table S1, refinement | Frozen freshly trained T=4 bases followed by the declared SwinIR refinement recipe. Maximum validation SSIM selects a checkpoint, with the initial identity candidate eligible. The configured MSE gate is recorded. | SwinIR can trade pixel error for structural similarity. Report MSE and SSIM together and retain a no-improvement outcome. This is an explicit independent recipe because the original executable is unavailable. |
| Fig. 4, segmentation | Three-stage training on BBBC039's paired images and manual nucleus masks, with its official 100/50/50 split. Original inverse CNN; aligned native-resolution crops/flips. | Uses available manual annotations instead of intensity-derived targets. Positive annotation labels form binary nucleus foreground; no threshold or morphology is applied to image intensities to generate labels. BCE + 0.5 soft Dice and validation-selected output thresholds remain unchanged. This is semantic foreground segmentation, not an instance-separation benchmark. |
| Fig. 5, upsampling | Original locality and original multistage transpose blocks with the same original decoder; fixed random illumination, C=8; sizes 128/256/512 and train counts 600/3000/6000. | Reproduces the architectural question. These blocks have different parameter counts and weight sharing; the result cannot isolate locality independently of capacity. |
| Table 1 / Fig. 6, noise | Preserve the correct normalized `paper_v3` Gaussian approximation. Original inverse architecture; k=10/10,000 and read SD=0/2.7/2/6. Fixed/learned arms share budgets and phase/selection rules. | The original and corrected replication equations agree. Mean test scores use five explicitly seeded detector-noise draws; exact Poisson is checked separately in the binary controls. |
| Table 2 / Fig. 7, natural images | Original locality block, declared 1x1 channel adapter and SwinIR-M; identical learned/fixed recipe; scene-separated validation. Primary scores average full-image metrics equally over images, covering every pixel. | Avoids weighting larger images more through their tile counts. Border tiles use zero padding; their measurements and effective compression are counted. SwinIR hyperparameters remain disclosed assumptions. |
| Fig. 8, MCF7 | Original locality for both CNN256 and SwinIR256; same pixel/perceptual/GAN loss, reconstruction LR, effective batch, sharpness schedule, and minimum-validation-MSE selection. Original CNN widths. | Tests the inverse architecture within a jointly optimized pipeline without changing upsampler or objective. The learned masks can adapt separately to each architecture; this is not a fixed-mask backbone test. |
| Fig. 9, MCF7 fields | Add CNN64 with the same matched objective. Primary reconstructions use nonoverlapping acquisitions. Additional overlapping acquisitions remain separately accounted diagnostics. | Patch context differs, so this evaluates complete field-reconstruction pipelines. It does not establish a free stitching gain at unchanged measurement count. |
| Table 3 / Fig. 10, A-D | Original architectures for A/B/C/D. Four-point, validation-only LR searches for C/D; transfer the selected Fourier LR to B to hold that optimizer setting fixed for B/C. Repeat C/D using the rewritten architecture as a sensitivity analysis. | Separates a representation comparison from an arbitrary same-number LR comparison. Results are conditional on the bounded search, not a proof of global optimizer superiority. |
| Additional PatchMNIST A-D | Same original/rewritten comparison and validation search on clean, disjoint source-digit pools. | Independent controlled diagnostic; it is not U2OS Table 3. |
| Supplement Fig. S3 family | Dedicated PatchMNIST content-aware sweep: fixed versus learned, T=4, d=8/16/32/64, photon scale 10,000. | Closes the missing experiment-family coverage. This declared fixed-T sweep is not a digitized recreation of every original plotted geometry. |
| Binary/dose controls | Original inverse CNN; binary forward masks with an STE for training; equal mean incident sequence dose; exact Poisson/read-noise test evaluation and Gaussian-approximation diagnostic. | Tests physical assumptions separately from soft-mask reproduction. Soft-validation-selected LRs transfer to the learned arms; binary coordinate rankings are secondary, not separately tuned claims. |

## Architecture and acquisition

Original CNN widths are `[24,12,8,4,2,1]`. `original_locality` mixes the detector
site's T measurements into a spatial patch and uses the original two Conv-BN
channel-lift layers. `original_transpose` uses the original stack of stride-2
transpose-convolution/ReLU/BN blocks. Their copied-weight forward maps and input
gradients were checked against the original source in the earlier architecture
audit. The rewritten C/D sensitivity uses the previous larger decoder and
per-pattern locality block; it is an explicit inverse-architecture family.

Compression is `C=d²/T`, a count of scalar detector measurements relative to
object pixels. It is not a dose or phototoxicity guarantee. For Fig. 3:

| C | Primary (T,d) | Additional (T,d) |
|---:|---|---|
| 16 | (4,8) | (1,4), (16,16) |
| 64 | (4,16) | (1,8), (16,32) |
| 256 | (4,32) | (1,16), (16,64) |
| 1024 | (4,64) | (1,32), (16,128) |

Primary legacy comparisons use soft masks and their declared intensity scale.
Uniform, fixed random and learned masks can have different mean transmission.
The binary control explicitly normalizes sequence mean incident dose to 2
full-on exposure equivalents. This does not also match peak power, detected
photons, or local phototoxicity. Impulse PSFs retain the original simplified
optical setting; these experiments do not establish performance under unknown
real PSFs.

BBBC022 normalization defines the simulated specimen. Photon scale 10,000 is a
simulation parameter applied to that normalized object, not a calibration of
the BBBC022 camera. The active noise equation remains unchanged.

## Data and evaluation controls

- Keep the existing source datasets in place. No original images are rewritten.
- PatchMNIST uses separated train/validation/test source pools and a reusable
  cache; generation is deterministic for a frozen configuration.
- BBBC022 retains the 220 training and 40 validation wells. The new 60 test wells
  come from the previously unused final 64 wells in the deterministic ordering.
  For the same `data_seed`, they are disjoint from all wells used by the recorded
  220/40/60 campaign. The previous 60 test wells are excluded from this campaign.
  The prepared manifests record the policy and identities. The explicit
  pseudo_trackmate fallback uses a nested 168/21/21-image subset of this partition.
- Default segmentation instead uses [BBBC039v1](https://bbbc.broadinstitute.org/BBBC039):
  200 fields from BBBC022 with manually annotated nuclei. Use the published
  100 training / 50 validation / 50 test lists. The loader checks matching image
  identities, native 520x696 dimensions and plate+well separation; masks are not
  resized, smoothed or rethresholded from specimen intensities. Following the
  [dataset's decoding example](https://gist.github.com/jccaicedo/15e811722fca51e3ae90e8b43057f075),
  use positive red-channel labels and ignore alpha, uniting instances for the
  binary task. Train on random 256x256 crops and evaluate one deterministic
  center 256x256 crop per field; these are not full-field instance AP scores.
- All three segmentation stages use that same BBBC039 split. Reconstruction
  pretraining starts fresh from its training images; no BBBC022 content-stage
  checkpoint is transferred. BBBC039 is drawn from BBBC022 and is not an
  independent acquisition cohort. The two studies' results must not be pooled
  as independent datasets.
- The explicit `run.segmentation_labels: pseudo_trackmate` fallback preserves
  `mask_mode: trackmate`, `mask_raw_threshold: 506.0`,
  `mask_smooth_interval: 2.0`, and `mask_dp_epsilon: 0.5` from the user's original
  template, including raw-MIP contour generation. The journal_v1 override to
  normalized threshold 0.3 plus closing has been removed. Those two target
  generators are different definitions; changing their values is not a mere
  preprocessing fix. Missing manual masks fail preflight instead of falling back.
- This reserve-well statement is relative to the recorded campaign, not proof
  that no earlier external analysis ever viewed those images. Natural-image
  benchmarks and MNIST are reused public datasets; disclose their role honestly.
- MCF7 uses tubulin w2 only, split by well. SR uses HR originals with parent-scene
  validation separation and no bicubic copies.
- Default final model seeds are 42, 43 and 44. The dedicated LR-search model seed
  is 314159 and cannot also be a final seed. Data splits remain fixed across arms.
- Fourier/spatial C/D begin with paired physical masks and identical inverse
  weights within each architecture family. Loader and crop randomness are
  separated from model randomness. Raw global gradient clipping is disabled in
  this ablation because its norm is not comparable across coordinate systems.
- Validation noise is fixed for the staged reconstruction and A-D comparisons;
  these draws do not advance the training RNG. No test result selects a learning
  rate, checkpoint, segmentation threshold or architecture.
- Noisy content, PatchMNIST compression and noise-grid jobs reload their selected
  checkpoint and report the mean of noise seeds 101,202,303,404,505. Raw draws and
  their SD are saved in `metrics/test_noise_draws.json`; SD over detector draws is
  distinct from SD over trained models. Earlier single-draw exports are labeled
  diagnostics in the final summary. Use the campaign report and updated
  `run_summary.json` as primary outputs; older trainer-level CSV exports retain
  their original single-draw values.
- SR primary metrics cover the complete image after nonoverlapping tiling.
  Zero-padded border tiles count as acquisitions; per-image effective compression
  can be lower than nominal C=16. Per-image metrics and counts are saved.
- Legacy SSIM uses the repository's common implementation throughout the new
  comparisons; do not equate its values with original Ignite SSIM numerically.

## Optimization, selection and compute budgets

Every LR candidate receives 8,500 updates: 1,500 inverse-only at m=1, 4,000 joint
at m=1, then 1,000 each at m=2,4,8. Only m=8 validation checkpoints are eligible
for the final score. The final C/D runs use the same schedule. This is a declared
step-budget adaptation, not an assertion that the original epoch schedule was
reproduced exactly.

Each dataset and inverse architecture gets the same four-candidate budget per
coordinate system. Both datasets use the same simulated photon scale 10,000,
background 10 and zero read-noise SD for these A-D comparisons:

| Coordinates | Illumination LR candidates |
|---|---|
| Fourier | 0.01, 0.1, 1, 10 |
| Spatial | 0.001, 0.01, 0.1, 1 |

The grids differ because the unnormalized inverse FFT changes parameter scale.
The original LR=1 point is included in both. Candidates never construct a test
loader. Minimum validation MSE wins; exact ties choose the smaller LR. The
selection artifact contains all scores and flags a boundary winner. A boundary
winner does not justify claiming a global optimum outside the grid. The chosen
LRs are fixed before the final three-seed jobs. The A-D runner now respects the
architecture family instead of overwriting it with rewritten blocks.

| Stage | Jobs, default three final seeds | Budget per trained model |
|---|---:|---|
| `patchmnist_tune` | 18 (16 models, 2 selection jobs) | 8,500 updates |
| `patchmnist` | 18 | 8,500 updates |
| `ablation_tune` | 18 (16 models, 2 selection jobs) | 8,500 updates |
| `ablation` | 18 | 8,500 updates |
| `upsampling` | 54 | 4,000 updates |
| `noise` | 48 | 14,850 updates |
| `patchmnist_content` | 24 | 10,100 updates |
| `content` | 96 | 10,100 updates |
| `content_swinir` | 24 | 50,000 updates |
| `segmentation` | 18 | 7,400 reconstruction + 1,200 head + 2,500 joint updates |
| `sr` | 6 | 20,000 updates |
| `mcf7` | 12 (9 models, 3 rendering jobs) | 230 full epochs |
| `controlled` | 15 | 10,100 updates |
| **Total** | **369** | Full declared budgets; no wall-time estimate |

Fixed and learned content/noise arms and segmentation pretraining use the same phase boundaries,
optimizer-state restoration and final-phase checkpoint eligibility, as well as
the same total update budget. Fixed masks stay fixed; their configured sharpness
is 8. Upsampling uses fixed random masks for every architecture condition.

MCF7 uses the original cumulative sharpness rule: m=1 until the increment at
one-based epoch 160, followed by m=2,3,4,5 at epochs 160,180,200,220. Illumination
unfreezes after the 150 baseline epochs. All three conditions use the same
pixel/perceptual/GAN objective, Adam settings, reconstruction LR 0.0002 and
effective batch 32. All retain remainder images and normalize partial
accumulation by actual image counts (3,000 images, 94 updates per full epoch).
Validation is evaluated at deployment m=5; only epochs 220-230 are eligible for
selection. GAN warmup is 57 epochs in all arms. SwinIR and CNN still have
different capacities and normalization layers, as intended for this comparison.

The available SwinIR recipes retain their declared architectural/loss assumptions
and budgets, including a 1x1 adapter for the T upsampled channels. They are not
represented as recovered original recipes. MCF7 is noiseless simulated sensing
of real source images; the noisy optical controls are separate.

## Running and interpreting the campaign

Follow [START_HERE.md](../START_HERE.md). Use a fresh checkout of
`journal/final-protocol` and a new `runs/journal_v2` output directory. The runner
freezes this protocol, source fingerprint, exact resolved configs, environment
and manifests. An old campaign cannot silently resume with the new recipe.

`python paper.py run` executes the 369 jobs sequentially with prerequisites.
The Fourier questions run first. `--stages patchmnist ablation` runs the complete
72-job tuning/final-comparison subset; a later unrestricted `run` continues the
same campaign. No manual learning-rate choice is required. Completed jobs are
skipped only after artifact verification; interrupted jobs restart in a new
attempt directory. This is not an exact optimizer-state resume claim.

`python paper.py report` writes per-seed scores, means/SDs, matched within-seed
differences, a separate LR-search table, coverage and plots. Tuning trials never
enter final test aggregates. `C - D` is Fourier minus spatial, so a negative MSE
difference favors Fourier. Training-seed SD is not a population confidence
interval over biological specimens. Report all prescribed arms and secondary
metrics; do not select the manuscript's architecture after inspecting the test
winner.

Implementation checks and their limits are recorded in
[JOURNAL_VALIDATION.md](audit/JOURNAL_VALIDATION.md); the manual-label revision is
validated separately in [MANUAL_MASK_VALIDATION.md](audit/MANUAL_MASK_VALIDATION.md).

The run produces evidence for a manuscript, not an automatic acceptance or
confirmation of the original rankings. Conclusions must follow those results.
