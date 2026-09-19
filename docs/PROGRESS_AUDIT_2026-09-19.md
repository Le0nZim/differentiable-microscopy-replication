# Workstation progress audit — 2026-09-19

**The pushed results are internally consistent as files, but several comparisons
need corrected training.** The most serious new defect is validation-image
export changing reconstruction BatchNorm state between training phases. A
separate dataset defect freezes segmentation augmentation. Neither a completed
job nor a matching artifact hash establishes a valid scientific comparison.

This audit inspects snapshot `c74c30ae98ad4561800e6987f3146fbf3f57c636`
on `workflow/clean-workstation`. Both campaigns record training source commit
`8b228f1794e655159eec162b4e56c01f86afc54d`. Independently recomputing the source
fingerprint before these fixes gave the recorded value:

```text
db3f0ba9729dc17d8261f8f125605bd0b59801fc554f1004080288a6cec75c0b
```

The snapshot is a record of progress, not access to the workstation's live
processes. No historical metrics, checkpoints, configs or state ledgers were
rewritten for this audit. The earlier [scientific audit](SCIENTIFIC_AUDIT.md)
and [workstation protocol](WORKSTATION_PROTOCOL.md) remain relevant.

## What was actually checked

| Stage | Completed jobs in this snapshot | Assessment |
|---|---:|---|
| PatchMNIST A–D | 12 | No newly identified defect in this execution path; retain as exploratory evidence. |
| Upsampling | 54 | No newly identified defect in this execution path; architecture and capacity interpretation still matters. |
| Noise | 48 | 24 learned runs use the affected staged trainer. Rerun both methods for corrected comparisons. |
| Content reconstruction | 48 | 12 learned runs use the affected staged trainer. Rerun all conditions for corrected comparisons. |
| Content SwinIR | 11 | Five completed learned-base refiners inherit affected bases; all need the declared checkpoint-selection policy applied consistently. |
| Segmentation | 18 | All have frozen train crops; nine learned stage-1 models also use the affected staged trainer. |
| Natural-image SR | 3 | Only seed 42 has both conditions completed. No three-seed comparison yet. |
| MCF7 / BBBC022 ablation | 0 | Apply these fixes before starting. |

There are **194 distinct completed job IDs**, with no completed-ID duplication
between the two campaigns. `paper_v1` records 173 completed jobs and one running
job, `content_swinir/x64_learnable_frequency_seed43`. `paper_gpu0` records 21
completed jobs and one interrupted job, `sr/swinir_with_li_seed43`. These labels
describe the snapshot; they do not establish current process status.

Integrity checks verified:

- All **367 recorded non-checkpoint artifacts** against their actual SHA-256
  digests and sizes.
- **194 checkpoint LFS pointers** against recorded checkpoint SHA-256 and size.
  The weight bytes were unavailable locally; this is pointer verification,
  **not checkpoint-byte verification or real-result checkpoint replay**.
- All **1,158 per-seed metric rows** against verified result JSONs, plus all
  **478 aggregate rows** against the per-seed means, sample SDs and counts.

The machine-readable record is
[progress_2026-09-19_integrity.json](audit/progress_2026-09-19_integrity.json).
To repeat the read-only check on a relocated snapshot:

```bash
python scripts/audit/inspect_campaign_snapshot.py \
  --campaign runs/paper_v1 --campaign runs/paper_gpu0 \
  --output /tmp/campaign-integrity.json
```

The inspector never edits the ledger or makes a relocated campaign eligible
for training resume. Normal workflow provenance checks remain strict.

## Confirmed implementation defects and fixes

### 1. Diagnostic exports changed the model using validation images

`capture_detector_snapshot()` unconditionally returned the model to training
mode. `evaluate_all_pattern_variants()` called it at the end of evaluation.
The subsequent `save_variant_artifacts()` used `no_grad`, but did not set eval
mode. Its three reconstruction passes updated BatchNorm running statistics
using the held-out reference specimen. Disabling gradients does not freeze
BatchNorm statistics.

A small reproduction on the pre-fix implementation changed all five
reconstruction BatchNorm layers three times per export. In staged training,
this happened after phase-best selection and before the next training phase.
At the end, the copied best checkpoint could therefore differ from the model
used for the final reported metrics. This affects training itself; reevaluating
the old checkpoint alone cannot remove the validation influence from its
training history. The size and direction of the effect on real microscopy
scores have not been measured here.

**Fix:** evaluation and diagnostic helpers now preserve every module's prior
training/eval flag, run in eval mode without gradients, and restore flags even
on exceptions. Detector snapshots run only the acquisition path. Regression
tests verify unchanged buffers, mixed-mode preservation, exception handling,
and equality between a tiny staged run's reported test metrics and its reloaded
`best.pt` metrics.

### 2. Segmentation kept sampling the same training patch

`BBBC022HoechstDataset.__getitem__()` reseeded its generator from the sample
index on every access. The advertised random training crops and flips were
therefore one fixed crop/augmentation per image. Rebuilding the dataloader did
not fix this. This applies to segmentation pretraining and task training.

**Fix:** the workstation segmentation recipe explicitly enables advancing crop
generators. Both microscopy dataset implementations now support dedicated
training RNGs independent of model and detector-noise random draws. Validation
and test crops remain deterministic. The historical fixed-crop option remains
available for replay, with its meaning documented. Paired image/mask transforms
are tested. The default corrected reconstruction crop sequence also changes
because its previous implementation used the model's global RNG; rerun the
fixed baselines alongside learned models for a consistent corrected protocol.

The raw pseudo-label threshold and image preprocessing were not retuned in
response to these test results.

### 3. Refinement excluded its identity baseline from checkpoint selection

The refiner is initialized to return the frozen base reconstruction, but its
initial candidate was ineligible for best-checkpoint selection. In **five of
the 11 completed jobs**, every recorded trained validation SSIM is lower than
the initial/base validation SSIM: all four completed x16 jobs and learned
x64 seed 42. This does not establish a test-set winner; it shows the selection
procedure could discard its best available validation candidate.

**Fix:** include the initial candidate by default, record whether it was
selected, and select by validation metrics only. The option is explicit as
`selection.include_initial_checkpoint`. Resuming across a changed eligibility
rule is rejected. When the optional MSE non-regression gate is enabled, it is
now a real feasibility constraint; previously it only subtracted a score
penalty and could still choose an infeasible candidate. The pushed primary
recipe has this gate disabled, so that gate bug does not explain these runs.

The GAN refinement resume path also saved the discriminator optimizer without
the discriminator weights. It now saves/restores both, plus crop and torch RNG
states, and rejects incomplete GAN resumes. Fresh workflow attempts did not
use this resume path; there is no evidence it affected completed jobs. This
does not turn workstation restart into exact training-state resume.

### 4. Pending MCF7 evaluation and checkpoint metadata

MCF7 evaluation averaged batch means, overweighting a short final batch, and
counted batches for an argument named `max_items`. Its selected checkpoint's
reported validation MSE could also be the minimum from an earlier SSIM-selected
checkpoint rather than the MSE of the actual saved model.

**Fix:** image-weighted MSE/SSIM, mean per-image PSNR, a true image cap, explicit
empty-set errors, preservation of model modes, paired metrics from the selected
checkpoint, and saved evaluation sigmoid sharpness. Corrected comments no
longer claim that minimizing MSE and maximizing SSIM are equivalent for an L1
model. No MCF7 result in the pushed snapshot needs replacing.

SR result metadata now also declares its existing aggregation: primary scores
average nonoverlapping tiles, so images contribute according to tile count;
the stitched alternative averages reconstructed images. Both omit right/bottom
remainders outside the tile grid. This is a reporting clarification, not a
change to existing SR numerical results.

## What the numbers suggest, and what they do not establish

The following are observations from the archived run, not corrected results.

- **Learned illumination looks promising in some regimes.** Mean content MSE
  is approximately 58%, 64% and 37% lower than random at x64, x256 and x1024.
  At x16 the difference is only about 6%, with substantial seed variation.
  The learned branch's BatchNorm defect prevents attributing these differences
  cleanly to the illumination method until the controlled reruns.
- **Soft-mask performance does not establish binary acquisition performance.**
  At k=10,000 and read SD=0, mean learned MSE rises from about 0.003006 (soft)
  to 0.004592 (thresholded), a 53% increase. This is an inference-time
  thresholding diagnostic, not a comparison to a properly trained binary model.
  The optional controlled binary/equal-dose stage has no results in this push.
- **Fourier parameterization is not a guaranteed improvement.** PatchMNIST D
  (spatial/locality) has mean MSE 0.006282, versus C (Fourier/locality) 0.006804,
  about 7.7% lower under the declared recipe. Preserve this outcome. It does
  not by itself show all Fourier parameterizations fail, or supply the optical
  convolution absent from the parameterization discussed in the earlier audit.
- **SwinIR can trade pixel error for SSIM.** For random x1024 seed 42, test MSE
  rises from about 0.038980 to 0.045800 while SSIM rises from 0.5954 to 0.6400.
  The declared primary recipe selects maximum validation SSIM without an MSE
  gate. A higher MSE is therefore not automatically a coding failure, and this
  run cannot be described as an improvement in both metrics.
- **Segmentation remains a pseudo-label experiment.** Its encouraging scores
  are affected by the above defects and quantify agreement with generated
  targets, not biological annotation accuracy.
- **The upsampling gap needs its architectural context.** A shared transpose
  operator and a position-dependent locality operator have different capacity
  and inductive assumptions. Their observed ranking alone does not establish
  an implementation error or isolate parameter count as a cause.

Both main microscopy branches now have equal update counts within the declared
workstation comparisons. That earlier training-budget defect is not the new
explanation for every disagreement. Dose still differs across illumination
conditions: the content masks have mean transmission about 1.0 (uniform),
0.625 (Hadamard), 0.5 (random), and 0.486–0.496 (learned). Equal updates and
equal compression alone do not establish equal specimen dose. BBBC022 also
remains a substitute for the unavailable original U2OS dataset.

## How to proceed on the workstation

1. Pause affected microscopy/refinement work before spending more GPU time on
   it. Let a separate SR job continue if desired: this audit did not establish
   a new training defect in that completed SR path. A snapshot cannot tell
   which job is currently active.
2. Preserve both existing campaign folders. Use a separate checkout of
   `fix/progress-audit-2026-09-19` and a new `run.output_root`, such as
   `runs/paper_v2`. Reuse the original datasets. Do not change source underneath
   a running worker, edit provenance checks, or copy old completion records
   into the new campaign.
3. Start with corrected `noise`, `content`, `content_swinir` and `segmentation`
   stages. Rerun fixed conditions with the corrected augmentation protocol too.
   The workflow automatically rebuilds content prerequisites for refiners.
   Run the still-pending `ablation` and `mcf7` stages from the corrected source.
4. Retain the existing PatchMNIST, upsampling and completed SR evidence with
   its original provenance. These newly found defects alone do not require
   discarding it. A full fresh default campaign will rerun those stages too;
   choose stages explicitly if avoiding that cost.
5. Before making physical binary-illumination claims, run the separate
   `controlled` stage and report its acquisition assumptions. Do not pool its
   results into the soft-mask tables.

After setup in the **separate corrected checkout**, one example is:

Until these changes are merged, use `--branch fix/progress-audit-2026-09-19`
in place of `--branch workflow/clean-workstation` in the clean-clone command in
[START_HERE.md](../START_HERE.md).

```bash
python paper.py --config workstation-v2.yaml init \
  --data-root /absolute/path/to/your/data --output-root /absolute/path/to/runs/paper_v2
# Check the GPU, worker count and paths in workstation-v2.yaml before running.
python paper.py --config workstation-v2.yaml check --stages noise content_swinir segmentation ablation mcf7 controlled
python paper.py --config workstation-v2.yaml run --stages noise content_swinir segmentation ablation mcf7 controlled
```

For two GPUs, assign disjoint stage sets and distinct campaign roots/configs.
Both pushed plans contain some of the same future stages. Completion is tracked
per campaign, so unrestricted `run` in both campaigns can duplicate work even
though this snapshot has no duplicated completed job IDs.

## Validation and limits

The broad CPU non-slow suite passed 200 tests, skipped 14 and deselected one
slow test. Its sole failure was the historical-byte-preservation test because
the sparse source checkout omitted its seven tracked archive fixtures. Those
exact files were restored from the audited commit; no reference hash or
archive content was changed. The final targeted run, including that check and
the new audit regressions, passed **11 tests**. Commands:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m pytest -q -m 'not slow' --disable-warnings
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m pytest -q tests/test_progress_audit.py \
  tests/test_udith_legacy_schedule.py::test_existing_patchmnist_fig10_outputs_byte_for_byte \
  --disable-warnings
```

The new regressions cover evaluation-state preservation, staged-checkpoint
replay, reproducible changing crops with mask alignment, initial refinement
selection, GAN state, MCF7 image weighting, and independent snapshot/table
integrity verification. The existing scientific-audit and workstation tests
also passed (42 checks).

There is no local CUDA device or real microscopy/MNIST training data. Only
small synthetic models were trained here. GPU checkpoints from these campaigns
have not been replayed. These repairs improve the implementation's validity;
corrected microscopy performance rankings and a publishable conclusion still
require the experiments and their scientific analysis.
