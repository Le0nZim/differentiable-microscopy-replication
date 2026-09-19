# Original versus replication: experiment-level comparison

The table below audits the pinned versions listed here. The prospective fixes
and final decisions are now implemented in the [journal protocol](JOURNAL_PROTOCOL.md)
on `journal/final-protocol`; follow [START_HERE.md](../START_HERE.md) for new runs.

Audit date: 2026-09-19. **Use the original CNN architectures inside the corrected
replication training/evaluation infrastructure for the primary reproduction.**
Keep the rewritten architectures as an explicitly separate sensitivity study.
An architecture can be a valid model even when it is not the architecture that
the paper evaluated. A favorable ranking alone does not decide correctness.

## Versions and meaning of the verdicts

- **Original code:** [CompressiveDabbaMu at f8cc74b](https://github.com/udithhaputhanthri/CompressiveDabbaMu/tree/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a).
- **Paper:** [arXiv:2203.14945v2, including the supplement](https://arxiv.org/pdf/2203.14945v2).
- **Pushed results:** [replication snapshot c74c30a](https://github.com/Le0nZim/differentiable-microscopy-replication/tree/c74c30ae98ad4561800e6987f3146fbf3f57c636), whose campaigns record source commit `8b228f1`.
- **Corrected replication:** [1f59226](https://github.com/Le0nZim/differentiable-microscopy-replication/tree/1f59226af61a750c4a4a2259659773ee54bb120e), published in draft PR #3. At this audit it is a separate branch, not the source used for the 194 completed jobs.

The small MCF7 gradient-accumulation correction accompanying this report is
identified separately below; it was not already present in `1f59226`.

The verdicts distinguish **implementation correctness** (equations, gradients,
state and metrics), **fidelity** (matching the original experimental choices),
and **support for a scientific claim** (controlling competing explanations).
These can favor different implementations. A code default is not proof of the
exact configuration used to generate a published table. Where the release is
incomplete, the table says so rather than reconstructing undocumented choices.

The [earlier audit](SCIENTIFIC_AUDIT.md) documents the original gradient and
validation bugs and the first set of replication fixes. The
[progress audit](PROGRESS_AUDIT_2026-09-19.md) documents the additional
BatchNorm, augmentation and checkpoint-selection defects in the pushed runs.
Those defects are not silently transferred to corrected source, and corrected
source is not treated as having retroactively repaired old numerical results.

## Experiment-by-experiment verdict

| Experiment / claim | Original paper or released implementation | Replication differences, including current limits | Whose implementation is preferable, and for what? |
|---|---|---|---|
| **Figure 3: content-aware reconstruction** | Paper: U2OS confocal, photon scale 10,000, multiple pattern-count/downscale combinations. Released CNN uses the original locality block. | BBBC022 substitute with per-image percentile scaling; **noise disabled**. The workflow uses T=4 and one downscale per compression. Rewritten upsampler and wider decoder; 10,100 updates for every condition. Pushed learned runs suffered validation-state contamination, fixed in PR #3. | **Original conditions/architecture for reproduction; corrected replication for training/evaluation.** Current BBBC022 results are a new dataset/protocol experiment, not numerical U2OS reproduction. Restore the declared noise and geometry coverage when testing the original claim. [O1, O2, Y1, Y2] |
| **Figure 3 / Table S1: SwinIR refinement** | Paper: image-to-image SwinIR after a trained content-aware model. No corresponding SwinIR implementation or exact recipe was found in the pinned original release. | Explicit frozen-base caches, 50,000 refinement updates and assumed SwinIR/loss details. Base acquisition differs as above. PR #3 makes the initial identity candidate eligible for validation selection and fixes GAN resume state. | **No defensible original-versus-yours executable verdict.** Yours is inspectable, but exact fidelity is unverified. Use corrected base checkpoints and report the declared refinement/selection protocol. A gain in SSIM need not be a gain in MSE. [Y3] |
| **Figure 4: task-aware segmentation** | Supplement: reconstruction pretraining, head-only training, then joint finetuning. Released `train_segmentation.py` instead builds a fresh decoder and trains against generated masks; it does not implement that three-stage orchestration. | Yours implements all three stages with BCE plus soft Dice and a validation-selected output threshold. Different dataset and raw-intensity/contour pseudo-labels. Frozen training crops and learned-stage-1 BatchNorm contamination affect the pushed runs; both are addressed by PR #3. | **Corrected yours is closer to the supplement's training procedure.** The different label generator/loss is a new segmentation experiment. Neither image-derived target constitutes independent biological ground truth. The transferred U2OS camera offset in this recipe remains uncalibrated for BBBC022. [O3, Y4] |
| **Figure 5: locality versus transpose upsampling** | Released `custom_v2` combines all T measurements at each detector site; transpose baseline is a stack of 2x ConvTranspose-ReLU-BN blocks. | Default workflow uses independent per-pattern local projections and a single stride-d transpose layer, with a wider common decoder. Sizes 128/256/512 and counts 600/3000/6000 are swept at C=8; illumination is explicitly fixed and each run has 4,000 updates. | **Original architectures are the right primary comparison for reproducing this figure.** Both rewrites are legitimate networks, but they test different architectures. Use corrected evaluation for both originals; keep the rewrites as a separate result. Exact original per-cell training settings are not released. [O1, O4, Y5] |
| **Table 1 / Figure 6: noise robustness** | Released normalized Poisson-approximation/read-noise equation agrees with the supplement. | Active `paper_v3` has the same units. Cleaner PatchMNIST source splits; rewritten inverse network; both methods now receive 14,850 updates. Main runs still use Gaussian-approximation noise and soft masks. Learned pushed runs use the affected staged trainer. | **Tie on the active noise equation; corrected yours is preferable for data separation and training controls.** Original networks are needed for architectural fidelity. Exact-Poisson testing and declared dose/binary conditions are separate checks, not established by the legacy sweep. The old high/low-photon table ordering has no proven explanation. [O2, O5, Y6] |
| **Table 2 / Figure 7: natural-image SwinIR** | Paper: grayscale SR datasets, 64px patches, C=16, pixel/perceptual/adversarial losses. Exact executable SwinIR implementation is absent from this original release. | Explicit SwinIR-M, rewritten locality block, a learned 1x1 channel fusion, assumed loss/discriminator details, scene holdout, 20,000 updates, tile-based evaluation. Learned and fixed conditions share the declared recipe. | **Yours is an auditable independent implementation; exact replication is unverified.** Do not identify its 20k schedule with an undocumented original schedule. Declare tile versus image averaging and excluded border pixels. Only one paired seed was complete in the inspected snapshot. [Y7] |
| **Figure 8: MCF7 reconstruction** | Paper: channel 2, 256px reconstruction, SwinIR versus a conventional reconstruction pipeline. The released repo does not provide its MCF7/SwinIR execution path. | Same intended channel, well-grouped patch splits. Current conditions compare locality+SwinIR/full loss against **transpose+CNN/L1**, changing the upsampler, backbone and loss. The schedule uses m=1,2,4,8, not the original accumulating-m rule. An incomplete accumulation group was underweighted; the fix accompanying this report corrects this and retains remainder images. | **Neither a verified original implementation nor a clean SwinIR-only ablation is established.** Yours can compare complete pipelines if labeled accordingly. A claim specifically about the backbone requires the same upsampler and controlled losses. The numerical correction does not resolve those design differences. [Y8] |
| **Figure 9: MCF7 full-field rendering** | Paper: full-field qualitative comparison; exact released acquisition/stitching implementation is unavailable. | Earlier overlapping rendering reacquired every crop. Corrected rendering reports additional measurements and uses nonoverlap as the primary acquisition. CNN64 and SwinIR256 still differ in patch size/context and loss. | **Corrected yours has the defensible measurement accounting.** Overlap is not a free improvement at unchanged compression. The original stitching procedure cannot be adjudicated from missing code. This remains a whole-pipeline comparison rather than an isolated backbone test. [Y9] |
| **Table 3 / Figure 10: A-D ablation** | Paper: U2OS, C=16; learned masks, locality block and Fourier coordinates varied. Exact per-run configurations/checkpoints are not in the released repo. | BBBC022 substitution, rewritten architectures, noise-free acquisition, 8,500-update schedule. C/D use full-spectrum Fourier versus spatial logits and the same numeric illumination LR. Original architecture ports exist but are **not selected by the default A-D runner**. | **Original architecture plus corrected yours is the preferred reproduction.** Neither parameterization is inherently more correct. Equal-LR results establish performance under that recipe, not superiority after equally thorough tuning. Preserve the result whichever wins. [O6, Y10] |
| **Additional PatchMNIST A-D diagnostic** | Not the paper's U2OS Table-3 experiment. Original PatchMNIST generator has repeated grids and overlapping validation/test source-digit pools. | Globally separated digit pools, cached generated data, paired C/D initial masks/decoder seeds, 8,500 updates. Completed three-seed results favor spatial coordinates. This uses the custom A-D runner, not the newly affected staged trainer. | **Yours is preferable for a clean additional diagnostic.** It supports the observed ranking for that recipe, not a claim about unavailable U2OS data or optimally tuned Fourier coordinates. [O7, Y10] |
| **Supplement Figure S3: PatchMNIST content-aware compression sweep** | A separate compression-sweep result is reported in the supplement. | The current default plan contains a PatchMNIST A-D diagnostic, noise sweep and upsampling sweep, but no dedicated equivalent S3 content-aware compression stage. | **Coverage gap.** The extra A-D experiment does not replace the supplemental sweep. This needs an explicit recipe if full supplement reproduction is part of the intended scope. [Y1] |
| **Optional binary/equal-dose controls** | Released pattern model uses a finite sigmoid; no directly corresponding controlled suite is provided. | Hard binary forward with an STE surrogate, explicit fixed-peak/equal-mean dose options and exact-noise evaluation. This stage is optional and has no results in the pushed snapshot. | **Yours supplies stronger controls for the corresponding hardware claims.** It is an extension, not a literal replication. Equal-mean incident dose does not also equalize peak power, detected photons or local phototoxicity. [O6, Y11] |

## Architecture: what changed, and what has now been checked

| Component | Original released architecture | Default rewritten architecture | Verdict |
|---|---|---|---|
| Locality upsampler | Each detector site's T-vector maps to one d-by-d patch, with a site bias; two shared Conv-BN layers lift one channel back to T. | Each of T detector values independently scales its own d-by-d patch; retains T channels; no site bias or channel-lift network by default. | Different functions and information mixing. Use `original_locality` for fidelity. Neither form is intrinsically an invalid reconstruction model. |
| Transpose upsampler | log2(d) stages of kernel-4/stride-2/padding-1 ConvTranspose, ReLU and BatchNorm. | One kernel-d/stride-d ConvTranspose, with no intervening activation/normalization. | Material change in depth, receptive field and nonlinear processing. Use `original_transpose` in the original-architecture comparison. |
| Reconstruction CNN | Default hidden widths 24,12,8,4,2, followed by one output channel; Conv-ReLU-BN then final sigmoid. | Hidden widths 64,64,32,32,16, followed by one output channel; same block ordering. | Different capacity. `[24,12,8,4,2,1]` recreates the default original widths in `ReconCNN`. Defaults do not identify all historical publication runs. |

Sources: [O1], [O4], [Y5]. For H=W=256, T=4 and d=8, measured parameter counts
excluding the illumination generator are:

| Block | Original | Rewritten |
|---|---:|---:|
| Locality upsampler | 327,788 | 262,144 |
| Transpose upsampler | 804 | 1,028 |
| Reconstruction CNN | 4,849 | 72,193 |
| Locality + reconstruction total | 332,637 | 334,337 |
| Transpose + reconstruction total | 5,653 | 73,221 |

The locality pipelines have nearly equal total parameter counts while placing
those parameters in very different operations. Conversely, the original
locality/transpose comparison has a much larger parameter-count imbalance than
the rewritten comparison. Neither “the rewrite is simply larger” nor “this
isolates locality independently of capacity” accurately describes all arms.

I copied weights from the released original modules into the ports and checked
24 combinations across train/eval modes, T=1/4/8 and d=4/8 on small tensors.
All forward outputs matched exactly in this CPU check. Maximum absolute input
gradient difference was 9.54e-7. The decoder with the original widths also
matched. This checks the implemented maps and input derivatives at matched
weights; it does not replay GPU training, establish identical random
initialization streams, or certify published performance.

Reproducible evidence:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python docs/audit/check_architecture_equivalence.py \
  --original-root /path/to/CompressiveDabbaMu \
  --output /tmp/architecture-equivalence.json
```

The original checkout must be clean and at the pinned commit. See the
[recorded result](audit/architecture_equivalence_2026-09-19.json) and
[checker](audit/check_architecture_equivalence.py).

## Shared correctness verdicts

| Issue | Verdict |
|---|---|
| Forward optical integration | Repeated 2x sum pooling and a single d-by-d sum are equivalent for matching power-of-two bins and PSFs. **Tie** in the impulse-PSF setting. Corrected replication handles convolution orientation for asymmetric PSFs and stable checkpoint buffers. |
| Optimizer updates | Original backward passes accumulate illumination gradients across two decoder states. Corrected joint updates are coherent. **Corrected replication wins**; a correctly implemented alternating optimizer would also be legitimate. |
| Validation state | Original omits upsampler eval mode. The replication also had a separate export-induced BN mutation, repaired in PR #3. **Corrected replication wins; the pushed results are not repaired retroactively.** |
| Hardening | Original cumulative increments are coherent; Algorithm-1 resetting prose/pseudocode is ambiguous. The corrected generic replication schedule is cumulative too. Shortened/doubling schedules are deliberate or inherited protocol changes, not exact replays. |
| Metrics | Corrected replication uses appropriate image weighting in audited paths. Original uses batch means and can drop remainder images. SSIM padding still differs in legacy recipes (original Ignite reflect versus replication zero padding). **Replication wins on aggregation; use one metric definition for numerical comparison.** |
| Fourier representation | Original has an additional spatial `fftshift`; both full-spectrum forms span all real logit maps. **Tie as representations.** The paper's convolution/translation-invariance explanation is not established by these operations. Optimization/tuning can still favor either coordinate system. |
| Binary and dose claims | Neither legacy soft-mask pipeline establishes a binary, dose-matched advantage. **The optional controlled replication is the appropriate extension**, with its physical assumptions disclosed. |
| Data identity | Cleaner source-digit/well separation is preferable. A different cell dataset, calibration or pseudo-label rule remains a different experiment regardless of clean implementation. |

## Additional MCF7 issues identified in this comparison

These were present in the compared `1f59226` source. The first two are unresolved
protocol differences; the third is fixed in the code accompanying this report.

1. **Attribution is confounded.** `CONDITIONS` selects `transpose_conv` for the
   256px CNN and `locality_aware` for SwinIR. SwinIR also adds perceptual/GAN
   losses and selects by validation SSIM, whereas the CNN selects by MSE.
   This can compare declared full pipelines, but cannot isolate a SwinIR
   architecture effect. An additional CNN256 arm using the same locality
   block is necessary for that attribution; loss and selection policies must
   be controlled or separately varied.
2. **The “faithful Algorithm-1” label is too strong.** At zero-based epochs
   150,170,190, the current code switches to m=2,4,8 and evaluates at m=8.
   Applying the original cumulative increment rule with threshold 150 and
   interval 20 instead gives m=2,3,4,5 at one-based epochs 160,180,200,220.
   Matching 230 total epochs does not make these the same schedule. The
   original rule is the better reference for a faithful schedule comparison,
   while noting the paper's pseudocode ambiguity.
3. **The final accumulation group was underweighted; corrected now.** With 3,000 examples,
   microbatch 4 and accumulation 8, there are 750 microbatches: 93 full groups
   and a final group of six. The code divides each loss by eight even in that
   final group, scaling its mean gradient by 6/8. The CNN path instead drops
   its final 24 examples. The accompanying correction weights microbatch losses
   by their actual image counts within each optimizer step, for generator and
   discriminator, and retains the final batch in both pipelines. With the
   declared batch sizes, both now process all 3,000 images in 94 updates per
   epoch. Logged `train_images` makes that count inspectable.
   With Adam, a 0.75 gradient factor does not imply a 0.75 parameter update.

   Two focused regressions exercise the actual trainer, with and without GAN
   loss, against an unaccumulated reference. Nine images form groups of six and
   three; microbatch size two forces both an incomplete accumulation group and
   a one-image microbatch. Generator and discriminator gradients agree with
   the reference within floating-point tolerance. These tiny models have no
   batch-dependent layers: this tests normalization, not an equivalence between
   physical large batches and microbatches for networks with BatchNorm.

The focused MCF7 checks passed **4 tests**, including those two gradient
regressions, evaluation weighting, and the workstation CNN path retaining a
one-image remainder batch:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  tests/test_progress_audit.py tests/test_workstation.py -q \
  -k 'mcf7 or worker_executes_mcf'
```

No completed MCF7 jobs were present in the inspected snapshot. The unresolved
protocol differences limit attribution of the pending comparison; they do not
invalidate the earlier PatchMNIST, upsampling or completed natural-image SR
jobs. The script's introductory claim of exact paper fidelity is also corrected.

## Recommended comparison for the next manuscript

1. **Primary reproduction:** original locality/transpose architectures and
   original decoder widths, implemented in the corrected training framework.
   Retain repaired gradient clearing, evaluation modes, checkpoints and data
   separation. Report corrections rather than copying confirmed defects.
2. **Architecture sensitivity:** repeat the same question with the rewritten
   networks. For the Fourier claim, use the 2-by-2 comparison of original versus
   rewritten inverse architecture and spatial versus Fourier illumination.
   Pair initial physical masks and shared inverse weights within each
   architecture. Give both coordinate systems equal validation-search budgets,
   not merely an identical numeric learning rate.
3. **Original-protocol and physical-control results:** keep soft-mask
   reproduction separate from binary/dose/exact-noise tests. Maintain the same
   acquisition and data assumptions within each comparison.
4. **New data:** label BBBC022 as a substitute, report calibration and
   preprocessing, and avoid claiming numerical U2OS reproduction. Previously
   inspected test sets support exploratory analysis; a fresh holdout is needed
   for a confirmatory claim after iterative method selection.
5. **Choose conclusions after the comparison:** using the original architecture
   removes a confound. It does not guarantee Fourier will win, and a win would
   still not prove the rejected convolution explanation.

The original architecture ports are available now, but the default
`paper.py` plan still schedules the rewritten architectures. In particular,
`scripts/table03_ablation/run.py::apply_variant` overwrites the upsampling mode
from its A-D map. Merely changing a YAML mode does not select original
architectures for that runner. The existing
`configs/audit/original_architecture_patchmnist.yaml` is a binary/equal-dose
sensitivity recipe, not an exact soft-mask Table-3 recipe. An explicit new
architecture-family experiment recipe is needed before launching that matrix.

## Evidence index

Original paths are pinned to `f8cc74b`; replication paths below are relative to
this repository and were read at `1f59226`, with the MCF7 correction above added
alongside this report.

- **O1:** [model assembly and decoder defaults](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/train.py), [defaults](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/defaults.py).
- **O2:** [training/evaluation loop](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/train_utils.py).
- **O3:** [segmentation entry point](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/train_segmentation.py), [segmentation loop](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/train_utils_segmentation.py), [pseudo-target generator](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/tasks.py). The code's threshold `(310-134.28)/500 = 0.35144` also differs from the supplement's 0.3.
- **O4:** [upsamplers](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/models/decoder_upsampling_nets.py), [support blocks](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/models/decoder_support_blocks.py), [decoder](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/models/decoder.py).
- **O5:** [forward/noise model](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/models/forward_model.py).
- **O6:** [mask model](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/models/forward_H.py), [IFFT preprocessing](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/models/preprocess_H_weights.py), [cumulative schedule](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/m_inc_procs.py).
- **O7:** [PatchMNIST generator](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/create_patchMNIST_dataset.py), [data loading](https://github.com/udithhaputhanthri/CompressiveDabbaMu/blob/f8cc74b51847bd31b320a7f153dcf3eedecc3e7a/modules/data_utils.py).
- **Y1:** [workflow matrix](../workflow/catalog.py), [resolved recipe construction](../workflow/worker.py).
- **Y2:** [content reconstruction driver](../scripts/figure03_content_aware/train_base.py), [base config](../configs/_shared/base_bbbc022_substitute.yaml).
- **Y3:** [refinement driver](../scripts/figure03_content_aware/train_swinir.py), [recipe](../configs/figure03_content_aware/paper_faithful_pixel_perceptual_gan.yaml).
- **Y4:** [three-stage task trainer](../src/training/train_task_aware_segmentation.py), [segmentation template](../configs/figure04_segmentation/task_aware.yaml); workflow overrides make stage 1 fresh and equal-budget and enable advancing crops.
- **Y5:** [upsamplers and ports](../src/models/locality_upsampling.py), [decoder](../src/models/recon_cnn.py), [upsampling template](../configs/figure05_upsampling/base.yaml); workflow overrides use fixed patterns and 4,000 updates.
- **Y6:** [noise model](../src/models/detector_noise.py), [noise driver](../scripts/table01_noise_robustness/run.py), [noise template](../configs/table01_noise_robustness/noise_table.yaml).
- **Y7:** [SR recipe](../configs/table02_swinir_sr/full.yaml), [pipeline](../src/baselines/swinir/table2_pipeline.py), [training/evaluation](../src/baselines/swinir/am4_table2.py).
- **Y8:** [MCF7 driver](../scripts/figure08_mcf7/train.py), [MCF7 recipe](../configs/figure08_mcf7/swinir_fix.yaml).
- **Y9:** [full-field renderer](../scripts/figure08_mcf7/reproduce_fig9.py), [progress audit](PROGRESS_AUDIT_2026-09-19.md).
- **Y10:** [A-D runner](../scripts/table03_ablation/run.py), [BBBC022 recipe](../configs/figure10_ablation/ablation.yaml), [PatchMNIST recipe](../configs/figure10_ablation_patchmnist/ablation.yaml).
- **Y11:** [controlled runner](../scripts/audit/run_controlled.py), [protocol](RERUN_PROTOCOL.md).

This is a source-and-protocol comparison with small architecture checks. No
original U2OS data or full trained GPU checkpoints were available for replay.
