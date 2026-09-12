# Scientific and implementation audit

Audit date: 2026-09-12. This report concerns the paper's main reconstruction,
segmentation, upsampling, noise, SwinIR, and spatial/Fourier experiments.

**The existing results are exploratory and need selective reevaluation or
retraining. A different result from the paper is not evidence of a failed
implementation. There are confirmed errors in both repositories, and several
comparisons change more than the quantity they are supposed to isolate.**

The replication is better on globally separated PatchMNIST validation/test
digits, independent gradient updates, whole-model evaluation mode, and its
current `paper_v3` detector units. The upstream code is the better reference
for the particular upsampling architectures and cumulative hardening schedule
it actually ran. Neither repository establishes that one illumination
parameterization must win. The unavailable U2OS confocal dataset prevents a
numerical reproduction of those experiments; BBBC022 is a new experiment on
a different specimen/image distribution.

This change fixes executable errors, adds a controlled protocol, and identifies
claims that must wait for experiments. It does **not** replace historical scores
with guessed corrections or claim to have retrained full microscopy models.

## Scope and evidence

Reviewed sources are pinned so later changes do not alter the comparison:

| Source | Version / evidence |
|---|---|
| [Paper and supplement](https://arxiv.org/pdf/2203.14945) | Equations 1–12, Algorithm 1, §§5.1–5.7, supplement A.2 and B |
| [Original repository](https://github.com/udithhaputhanthri/CompressiveDabbaMu/tree/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a) | `f8cc74b51847bd31b320a7f153dcf3eedecc3e7a` |
| [Replication before this audit](https://github.com/Le0nZim/differentiable-microscopy-replication/tree/b71d8441e7e321c54df89f814cebbd587d4bc9b2) | `b71d8441e7e321c54df89f814cebbd587d4bc9b2` |
| [SwinIR dependency](https://github.com/JingyunLiang/SwinIR/tree/6545850fbf8df298df73d81f3e8cba638787c8bd) | The commit already specified by this repository |
| [Original-code diagnostics](audit/original_diagnostics.json) | Executes selected, reviewed upstream definitions on tiny tensors; includes source SHA-256 hashes |
| [Historical inventory](audit/historical_inventory.json) | 102 saved `config.yaml` files across listed experiment families; includes hashes, budgets, preprocessing, noise and acquisition settings |
| [Validation record](audit/VALIDATION.md) | Tests and synthetic execution evidence; limitations stated separately |

The inventory is not a claim to have replayed every artifact. JSON-only run
families are assessed through their drivers/configuration/report paths, rather
than represented as nonexistent experiments. Full historical checkpoints were
not downloaded and reevaluated. No BBBC022 source images or CUDA device were
available in this environment. Published performance bias from an individual
bug cannot be quantified by a toy diagnostic.

## What should be optimized and compared?

For the simplified two-dimensional simulation, the noiseless measurement is

\[
a_t=\operatorname{sum\_pool}_{d\times d}
\{\mathrm{emPSF} * [X\circ(\mathrm{exPSF}*H_t)]\}.
\]

With impulse PSFs this is the sum of pattern-weighted specimen pixels inside
each detector bin. Multiplication happens **before** integration. An excitation
mask can encode differences between pixels inside a bin, even though the
detector cannot resolve those pixels directly. A pattern that is constant
within a bin loses this encoding freedom; it supplies only a scaled bin sum.

For T displayed patterns and complete square bins, the sample-count compression
is `C = d²/T`. It is neither `d/T` nor an acquisition-time speedup. Multiple
uniform exposures provide repeated measurements of the same bin sum; they can
reduce independent noise but do not increase noiseless sensing rank. Binary
Hadamard rows here are single nonnegative exposures. Recovering a signed
Hadamard coefficient by complementary subtraction would require accounting
for both exposures and their noise. [Paper, §§4.1–4.3](https://arxiv.org/pdf/2203.14945)

The falsifiable question is whether optimizing the optical encoder improves
held-out reconstruction or task loss under a declared acquisition budget,
specimen distribution, and training budget. It is not whether a plot looks
like the paper. Test these comparisons separately:

1. Fixed versus learned patterns with the same decoder, update budget,
   initialization of shared parameters, data order, measurement count, and
   explicit dose rule.
2. Original locality versus original transpose upsampling, followed by a
   separate comparison with the rewritten architectures.
3. Spatial versus full-spectrum Fourier coordinates with paired initial
   physical masks and inverse weights. A declared same-LR comparison is one
   optimization setting; it is not a guarantee of equally tuned optimizers.
4. Soft simulation versus binary deployment, and Gaussian training noise
   versus exact Poisson/read-noise evaluation.

Neither successful training nor a tiny overfit test guarantees recovery of
unmeasured biological detail. At high compression, a reconstruction network
can use its learned image prior to produce plausible structures. Assess
task errors and out-of-distribution specimens before making biological or
hardware conclusions.

## Original implementation versus the replication

| Component | Original implementation | Replication before audit | Assessment and correction |
|---|---|---|---|
| Optical integration | Repeated 2× sum pooling; `lambda_scale_factor=q` implies `d=2^(q-1)` | Direct d×d sum pooling | Equivalent for complete power-of-two bins and matching PSFs. Keep sum pooling; reject incomplete bins. |
| PSFs | Simplified kernels / impulse experiments | Lazy buffers and impulse defaults | Scientific simplification is shared; checkpoint behavior was broken. Use stable buffers and true convolution for asymmetric kernels. |
| Detector units | Normalized form of supplement S5–S10 | `legacy`, `paper`, and `paper_v3` coexist | Current `paper_v3` is correct; retain explicitly named old modes for replay. Add exact Poisson evaluation. |
| Learned masks | Sigmoid of spatial logits or real IFFT of complex parameters | Same general construction, default soft | Finite sigmoid output is not binary. Add opt-in hard-forward STE for deployment-consistent experiments. |
| Fourier transform | `fftshift(ifft2(W)).real` | `ifft2(W).real` | Fixed spatial permutation, not a difference in representable mask space. Neither operation convolves the specimen. |
| Hardening | Accumulates m at prescribed boundaries | Generic schedule reset m on intervening epochs | Make generic schedule monotone and stateless. Retain the separately specified Udith step schedule. |
| Optimization | Decoder backward/step, then illumination backward/step with accumulated gradients | One zero/backward/step for joint parameter groups | Replication is a coherent joint optimizer. Do not copy the upstream accumulation error. |
| Evaluation mode | Decoder and illumination set to eval; upsampler omitted | Whole microscope set to eval | Replication is correct. Include a regression for upsampler BatchNorm behavior. |
| Locality block | `custom_v2`: site-specific T-vector → one d² patch, bias, then two Conv–BN channel-lift blocks | T separate scalar-to-patch maps; no local bias or default channel-lift CNN | Both are possible models; they are not the same ablation. Add `original_locality`. |
| Transpose baseline | Repeated kernel-4/stride-2/pad-1 ConvTranspose–ReLU–BN blocks | Single kernel-d/stride-d transpose convolution | Substantially different baseline. Add `original_transpose`. |
| Reconstruction widths | Default hidden channels 24,12,8,4,2 | 64,64,32,32,16 in main configurations | Capacity differs. Supply an original-architecture sensitivity configuration. Defaults are not proof of every published run's exact configuration. |
| Hadamard masks | Detector-bin-tiled signed rows passed through a sigmoid | Global Sylvester rows, exactly 0/1, huge full-matrix allocation | Avoid the allocation and expose tiled ordering. For the actual Figure 3 T=4 sweep, both row orderings coincide; this is not its performance explanation. Soft versus binary remains different. |
| PatchMNIST generation | Repeated seeded grid construction and reshuffled validation/test source pools | Independently generated crops; optional global validation/test digit separation | Active clean configurations improve split integrity. Grid construction, crops, JPEG path and sample counts still differ. |
| Metrics | Ignite SSIM, reflect padding; mean over loader batches | Custom SSIM, zero padding; mean over loader batches | Weight image-level MSE/SSIM by image count. Preserve old SSIM default for replay; corrected controlled protocol explicitly uses reflect padding. |
| Cancer data | Original U2OS confocal data | BBBC022 Hoechst widefield substitute; MCF7 is a separate experiment | No numerical interchangeability. Do not transfer camera offset/normalization assumptions without calibration. |

Upstream implementation evidence: [training loop](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/train_utils.py),
[upsamplers](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/models/decoder_upsampling_nets.py),
[model construction](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/train.py),
[defaults](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/defaults.py),
[mask initialization](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/models/initialize_H_weights.py),
[grid generation](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/create_patchMNIST_dataset.py).

## Confirmed upstream errors

**O1 — accumulated illumination gradients.** `loop` clears both optimizers
once per batch. Decoder-phase backward also differentiates illumination; after
updating the decoder, illumination-phase backward adds another gradient
without clearing it. The illumination optimizer therefore receives gradients
from two different decoder states. With more inner iterations, accumulation
also occurs inside each phase. A legitimate alternating implementation would
clear each phase's gradients and freeze/detach the other parameter block as
appropriate. A legitimate joint implementation takes one backward pass for
both groups. The diagnostic scalar example observes H=0.65693748, matching
the summed-gradient prediction 0.6569375; correctly cleared alternating SGD
would give H=0.5819375. This demonstrates the bug, not the magnitude of its
effect on the paper's results. [Executable evidence](audit/original_diagnostics.json)

**O2 — validation changes upsampler state.** The upstream loop omits
`model_decoder_upsample.eval()`. BatchNorm in both learned upsampling variants
therefore uses validation-batch statistics and updates its running buffers.
The diagnostic increments `num_batches_tracked` and changes the running mean
from 0 to 0.05 during validation. This is validation-dependent model state;
its performance effect may go in either direction.

**O3 — PatchMNIST data generation does not provide the implied independent
samples.** `save_grids` resets NumPy's seed to 500 on every call. The same
training tensor is supplied on 400 repetitions, so the same 150 full grids
are repeatedly saved under new names. Within-grid duplicate assertions cannot
detect duplicates across saved grids. The validation/test source digit pool
is reshuffled across outer repetitions, allowing the same source digit to
occur in both datasets across repetitions. Also, 5000 digits at 400/grid create
13 outputs, but the next start offset advances by 12: the last partial grid
can be overwritten. Random subsequent crops may still vary; it would be
incorrect to infer that every training batch is identical. The replication's
globally disjoint digit pools are a methodological improvement.

**O4 — Algorithm 1 conflicts with the implementation and hardening prose.**
The pseudocode resets m=1 in an else branch; upstream `inc_m_class` retains m.
Cumulative hardening is the coherent interpretation of the stated objective.
This ambiguity should be disclosed, not treated as a reason to reproduce a
resetting schedule. [Upstream schedule](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/m_inc_procs.py)

## Replication findings and fixes

Severity describes impact on interpretation, not a measured score correction.

| ID / severity | Finding, evidence path | Action and results affected |
|---|---|---|
| R1 / high | Generic `SigmoidSchedule.step` reset sharpness between boundaries. | Cumulative epoch formula; repeated calls/resume do not advance it. Retrain runs using that generic schedule. Explicit fixed-m and Udith schedule runs are different paths. |
| R2 / high | `ForwardModel` changed PSF buffer shape after its first forward; a fresh strict checkpoint load could fail. A later initialization could overwrite loaded kernels. | Eager persistent kernels, narrow migration for old zero placeholders, no lazy reset. Reevaluate affected checkpoints; no retraining solely for this loader fix. |
| R3 / medium | `_conv2d_same` performed cross-correlation for asymmetric PSFs. | Flip kernels for mathematical convolution. Existing symmetric impulse/box runs are unaffected. |
| R4 / high | Generic checkpoints omitted m, and some used `epoch` for another quantity. Default evaluation could use m=1 instead of the selected m. | Save explicit m/step/schema; ambiguous legacy evaluation requires an explicit `--sigmoid-m`. Figure 3 refiner reads saved m or recorded final-training m instead of hardcoded 10. Reevaluate and retrain downstream refiners whose acquisition changed. |
| R5 / high | Step and hardening trainers restored best weights but left final-step Adam moments; best/last artifacts could be duplicates. | Capture matching model/Adam/m/step; write distinct snapshots. Later hardening phases start from a consistent selected state. Retrain affected multi-phase experiments. No claim of exact RNG/data-loader resume. |
| R6 / high | `itertools.cycle(loader)` caches the first pass. Fresh crops/flips and reshuffling did not occur as advertised. | Fresh iterator repetition in generic step, hardening, and task trainers. Retrain augmentation-dependent comparisons. The expressly defined historical Table 3/Udith runner retains its cached iterator for that diagnostic; use the new runner for corrected comparisons. |
| R7 / high | Figure 3 learned runs received 10,100 inverse updates versus 4,000 fixed; Table 1 learned received 14,850 versus 7,350 fixed. | Equalize future driver budgets and route their outputs under `experiments/audited_v1/`. New controlled runner also has equal budgets. Historical winners cannot be attributed solely to illumination learning. |
| R8 / high | Soft and sharpened masks were discussed/displayed as binary acquisition. Finite m is not binarization. | Add hard-forward STE, use it in every controlled training/evaluation pass, save actual evaluated masks and binary status. Figure 9 displays actual grayscale values. Legacy soft experiments remain labeled as such. |
| R9 / high | Equal T and photon scale k do not imply equal incident dose when mask fill factors differ. | Add explicit `fixed_peak` and `equal_mean` dose modes and report mean/max sequence exposure, on-pixel scale, T, detector measurements and C. New experiments required for dose-controlled conclusions. |
| R10 / high | Generic reconstruction and hardening evaluation gave a short final batch the weight of a full batch. | Aggregate MSE/SSIM by sample count; likewise segmentation BCE. Reevaluate historic scores before quantitative comparison. Legacy Table 3 aggregation remains a replay path. |
| R11 / medium | SSIM border convention differed from upstream. | Add explicit zero/reflect/valid choices; controlled evaluation uses 11×11 Gaussian reflect-padded SSIM, data range 1. Historical default stays zero padded. Do not mix reported conventions. |
| R12 / high | Rewritten locality and transpose networks differ substantially from upstream. | Add explicit original variants and a sensitivity config. Projection layout is checked against upstream `custom_v2`. Repeat the architectural ablation; do not reinterpret old Figure 5 as the original ablation. |
| R13 / medium | Hadamard constructed an entire `(HW)×(HW)` matrix: at 256², the final float matrix alone is 16 GiB. | Compute only needed rows using bit parity, O(T·HW) output memory; expose detector-bin tiling. Regression confirms old and tiled T=4 main-sweep masks coincide. |
| R14 / high | Loader RNG could depend on earlier model construction. | Opt-in independent `loader_seed`; controlled comparisons pair inverse initialization and training order while holding the dataset seed fixed across training seeds. Historical global-RNG behavior remains available. |
| R15 / high | Staged trainer exposed test variants after each phase; a named split is not protection from iterative human tuning. | Use validation for phase diagnostics and test only final prescribed variants. Previously consulted test sets remain exploratory; code cannot undo that exposure. |
| R16 / high | BBBC split logic checked file count, not sufficient distinct wells; manifest `disjoint` metadata was trusted. | Require sufficient wells, nonempty splits, and actual disjoint well identities. Well separation does not establish treatment/plate/batch separation. |
| R17 / high | SwinIR split helper could substitute a training image for missing validation. | Fail on an impossible held-out split. No silent train/validation overlap fallback. |
| R18 / medium | SwinIR empty evaluation returned zero-like scores; capped tile evaluation could score black unfilled canvas regions. | Reject empty evaluation; stitched full-image metrics require a complete uncapped acquisition. |
| R19 / high | Table 2 exported masks/examples before loading the checkpoint selected for table metrics. | Preserve final-step diagnostics with `_final`; export principal examples/masks after selected checkpoint load. Regenerate existing figures from matching checkpoints. |
| R20 / high | GAN resume restored discriminator optimizer state without discriminator weights. | Save/load D weights, reject incomplete GAN resumes, explicitly label non-exact resume. Retrain continuations from incomplete old checkpoints when relevant. |
| R21 / high | Figure 9 overlap helper calls the full microscope for every crop. It obtains additional optical measurements. | Primary figure now uses nonoverlapping acquisition; overlap diagnostics report their measurement count and lower effective compression. Existing overlap image cannot support the original fixed-budget claim. |
| R22 / medium | Figure 8/9 scripts referenced removed script paths; checkpoint loading was permissive. | Fix imports and use strict state loading, with the explicit legacy PSF migration. Missing pattern state no longer silently leaves random illumination. |
| R23 / high | Figure 3/4 and other diagnostic prose used a required winning method as correctness evidence. Overfit ranking was asserted to rule out bugs. | Remove ranking from segmentation acceptance; report observed trends separately. Revise split/training and overfit interpretations. A worse result is allowed. |
| R24 / medium | Invalid noise mode/units could silently fall through; zero signal and zero background produced a singular sqrt gradient. | Validate parameters/modes; retain exact zero variance with a finite surrogate boundary derivative. Regression tests check noise moments and boundary behavior. |
| R25 / medium | Segmentation `best.pt` was actually the final prescribed phase, without selected threshold/m metadata. | Keep compatibility filename but store the true selection rule, m, and validation-selected threshold. Do not claim a best-validation checkpoint search that did not happen. |
| R26 / high | Manuscript/catalog text presents historical exploratory results as settled evidence. | Add visible audit status and neutralize central unverified superiority statements. Remaining numeric tables/figures are historical pending the rerun decisions below. |

**An illustrative existing soft/binary discrepancy:** the stored Figure 3 ×64
summary reports soft test MSE 0.00124144 and thresholded-pattern MSE 0.00135992,
about 9.5% higher. Its near-binary fraction is about 0.743. These are read from
historical summaries, not recomputed here, and use the old evaluation protocol.
They show why soft and binary results cannot be silently identified. They do
not establish the corrected binary model's eventual performance.

## Detector noise and physical interpretation

Let k be the photon-scale parameter, gamma the background count at the
detector, and sigma the read-noise standard deviation in detector-count units.
For the code's normalized noiseless signal a, the exact evaluation model is

\[
Y = [\operatorname{Poisson}(ka+\gamma)+\sigma Z_r]/k.
\]

The differentiable Gaussian approximation used for training is

\[
Y\approx a+\gamma/k+
\sqrt{a/k+\gamma/k^2}\,Z_p+(\sigma/k)Z_r,
\qquad Z_p\perp Z_r.
\]

Consequently E[Y]=a+gamma/k and Var[Y]=(ka+gamma+sigma²)/k². The new moment
test checks these against exact samples. The forward model already uses
normalized masks and sum pooling; dividing a again by d² changes signal/noise
units. Current Table 1 uses `paper_v3`, so the older normalization error is
**not** an explanation that can be reapplied to its current runs. `legacy`
uses different read units and `paper` divides the signal by d²; both remain
explicit legacy modes. [Supplement A.2.2](https://arxiv.org/pdf/2203.14945)

The Gaussian approximation has continuous support and can generate negative
measurements, particularly at low counts. Do not clip them casually: clipping
changes moments. Evaluate on the exact distribution and quantify the mismatch.
Adding a digital constant after readout does not reproduce optically generated
background photons, which also contribute Poisson variance.

The controlled runner's equal-mean mode keeps H exactly binary and applies one
scalar exposure/power multiplier `D / sum_t mean(H_t)`. It equalizes mean
**incident sequence** exposure, not detected photons for every specimen, peak
power, local phototoxicity, or acquisition time. It can exceed the old on-pixel
power; report that multiplier and the maximum sequence dose. Fixed-peak mode
is also provided because a peak-power-limited instrument asks a different
question. Neither choice is universally the only correct constraint.

The impulse-PSF model is internally consistent for a simplified experiment.
It does not establish diffraction-limited DMD projection, scattering behavior,
detector full well, readout timing, photobleaching, or calibrated photon yields.
Fine binary DMD pixels can produce gray specimen-plane illumination through
the excitation PSF. Hardware claims need measured PSFs and acquisition limits.

## Fourier reasoning

Writing `tau = real(ifft2(W))` changes optimization coordinates. A full Fourier
basis can represent every real spatial logit map. IFFT of a parameter tensor
does not multiply the specimen spectrum or convolve the specimen with a
learned filter. Thus the paper's §5.7 explanation invoking induced convolution
and translation invariance is not supported by this implementation. A shift
in upstream logits is a fixed permutation, not extra model capacity.

Unconstrained complex W is redundant for a real logit map. FFT normalization
and the real/complex coordinate representation change gradient scales; Adam
is not invariant to a general Fourier change of basis. Equal numerical LR is
not equal physical mask-update size. A spatial win at LR=1 does not prove
Fourier coordinates are intrinsically useless, nor does the paper's Fourier
win prove an optical mechanism. Pair initial tau, inverse weights, data order,
and noise; compare equal budgets and validation-selected learning-rate grids
of equal search size. Report logit/mask displacement and hard-mask flips.
The provided config is one declared starting point, not the result of such
a hyperparameter study.

## Which old results remain usable?

| Item / paths | Status after this audit | Required before a strong paper claim |
|---|---|---|
| Table 1 / Figure 6, `experiments/table01_noise_robustness` | Correct current noise units, but unequal training budgets, soft acquisition, historical metrics. | Equal-budget training; binary and dose sensitivity; exact-noise reevaluation; matched metric conventions and training seeds for all cells, not only the extreme cell. |
| Table 2 / Figure 7, `scripts/table02_swinir_sr` | Shared architecture/optimizer checks are useful but not a complete optical fairness proof. Loss/discriminator/perceptual details are partly inferred. | Full declared training budget and benchmarks; actual dose/binary accounting; selected-checkpoint figure export; valid held-out split; no incomplete GAN resume. |
| Figure 3, `experiments/figure03_content_aware/base` | BBBC022 substitute, all 16 saved base configs noise-free; learned/fixed update budgets differ. | Retrain equal budgets with fresh augmentation; reevaluate real binary masks; evaluate detector noise and dose separately. Any claim about original U2OS remains blocked. |
| Figure 3 SwinIR refiner | Downstream of the base encoder and susceptible to m mismatch and phase changes. | Retrain after base protocol is frozen; compare selected checkpoint with the same evaluated acquisition. Identity initialization does not guarantee improvement on unseen data. |
| Figure 4, `scripts/figure04_segmentation` | Legitimate segmentation head/task loss and validation-selected threshold exist. Labels are image-derived pseudo-masks, not independent cell annotation. | Fresh-iterator retraining; matched budgets and binary/dose acquisition; evaluate against independent annotations for biological segmentation claims. |
| Figure 5, `experiments/figure05_upsampling` | 24 saved configurations use fixed random patterns and 4,000 updates; architecture differs from upstream. | Repeat with both original and rewritten architecture families; equal budgets/seeds; corrected image-weighted metrics. Describe capacity/locality tradeoffs rather than treating one ranking as a software gate. |
| Figures 8/9, `scripts/figure08_mcf7` | Distinct MCF7 experiment; wide-field overlap image acquired more measurements. | Strict checkpoint replay, regenerated selected-pattern images, field-wide acquisition accounting, nonoverlap primary comparison and calibrated spatial scale if shown. |
| Table 3 / Figure 10 BBBC022 | Dataset substitute and architecture/schedule differences prevent literal table reproduction. | Controlled factorial ablation using declared architecture and data; no desired ranking. |
| Figure 10 PatchMNIST / Udith C–D diagnostic | Historical schedule comparison, not a corrected acquisition study. | Preserve the 121,500-step protocol for that narrowly defined diagnostic; use new controlled runs for fresh loaders, binary deployment, dose and corrected evaluation. |
| Manuscript numerical tables/figures | Historical evidence, not regenerated in this change. | Update from a frozen protocol and verified selected checkpoints; remove or qualify claims not supported by corrected runs. |

Current Figure 3 uses per-image `minimal_percentile` preprocessing with
0.1%/99.9% quantiles and the recorded 1,980/40/60-image well-disjoint split.
It does not currently use the old fixed 134.28 camera-bias subtraction.
Per-image normalization changes absolute intensity/dose interpretation. A
camera offset from one instrument is not a transferable constant for another.

Original segmentation targets were also pseudo-ground truth (supplement
A.2.3: normalized intensity threshold and morphological closing). Current
BBBC022 TrackMate-style raw threshold/contour targets are a different labeling
procedure. Agreement with them measures reproduction of that image-derived
labeling rule; it is not by itself proof of nucleus/cell accuracy.

The stored Udith schedule is deliberately preserved: 60,750 frozen updates,
36,445 joint m=1 updates, 4,050 each at m=2…7, and five final updates at m=8,
total 121,500. Its paired initialization/shared spatial warmup and inverse
Adam-state handling remain intact. That compatibility choice does not make
cached augmentation or historical metrics the recommended new protocol.

## What cannot be concluded yet

- The relative contributions of missing U2OS data, preprocessing, architecture,
  optimizer settings, and bugs to any numerical discrepancy. Factorial reruns
  are needed; this audit does not invent a causal decomposition.
- Whether the paper's counterintuitive low/high-photon Table 1 ordering is a
  column error, a training/selection effect, or something else. The released
  code/configurations do not establish a unique explanation.
- Whether learned binary patterns beat all baselines after equal-budget and
  dose controls, or whether spatial beats tuned Fourier optimization.
- Independent biological segmentation accuracy from pseudo-mask overlap.
- A throughput or phototoxicity advantage from sample-count compression alone.
- Confirmatory generalization from a test set already consulted during earlier
  design iterations. A fresh sealed specimen/well/plate test set is needed
  for that claim; a new evaluation script cannot make an old test set unseen.

Use the [rerun protocol](RERUN_PROTOCOL.md) to turn these unresolved questions
into concrete experiments. Report unfavorable outcomes with the same rules
as favorable ones.
