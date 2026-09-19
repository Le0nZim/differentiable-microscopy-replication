# Manual segmentation labels: correction and validation

Date: 2026-09-19. Parent commit: `c39a4164401b5a83bd6cbe05f1c30cfe62a0441c`.

The journal_v1 override changed the user's target definition from raw-intensity
TrackMate-style masks to normalized threshold/closing masks. Calling this only
"corrected preprocessing" understated that change. The override is removed.
The original template and `make_trackmate_mask` implementation are unchanged:
raw threshold **506**, contour spacing **2**, simplification epsilon **0.5**.
This is the explicitly selected `pseudo_trackmate` fallback.

Default segmentation instead uses the real annotations in
[BBBC039v1](https://bbbc.broadinstitute.org/BBBC039), a manually annotated subset
of BBBC022. Download its paired TIFF images, PNG masks and official split lists
with `python paper.py download-segmentation-data`. Missing manual masks fail
preflight; they are never filled in with generated labels. No old user data or
result files were rewritten.

## Validation performed

- Retrieved all three official archives and pinned their observed SHA256 hashes
  in `workflow/bbbc039.py`. These are observed archive hashes, not a claim that
  Broad separately publishes checksums. CLI extraction with `--archives-dir`
  verified the hashes and preserved existing matching files.
- Decoded **all 200 image/mask pairs**: matching filenames and native 520x696
  shapes, 16-bit source TIFFs, official **100/50/50** split, and no repeated
  image or plate+well identity across splits. Image and mask hashes are frozen
  in preparation and input inventories.
- **35 focused tests passed** across `test_manual_segmentation.py`,
  `test_journal_protocol.py` and `test_workstation.py`. Checks include opaque
  RGBA handling, aligned advancing crops/flips, unchanged pseudo parameters,
  manual-only preflight failure, three-stage execution without pseudo-mask
  generation, all 369 resolved jobs and existing downstream worker paths.
  The first run had 32 passes and three missing-SwinIR dependency failures in
  this separate checkout; the pinned source was made available and the same
  checks then passed. No experiment recipe was changed to resolve those failures.
- An additional real-data CPU integration used two official training images,
  one validation image and one test image with their actual manual annotations.
  A deliberately small model/crop and 3 reconstruction + 1 head + 1 joint
  updates completed, wrote stage checkpoints and a correctly labeled manual
  mask figure. This is a software smoke only; its scores carry no scientific
  performance claim.
- The public-archive CLI preflight passed; the segmentation-only plan still
  contains **18** jobs and the full default plan still contains **369** jobs.
- `git diff --check` passed. The original pseudo-mask template and implementation
  have no diff against the parent commit.

## Interpretation and running

The existing model predicts binary nucleus foreground. Positive red-channel
annotation labels are united; opaque alpha is ignored. Manual masks are not
resized or processed with the pseudo-mask threshold/contour algorithm. Training
uses aligned 256x256 crops and evaluation uses one center crop per field, so
these scores are not full-field instance-separation AP benchmarks.

All segmentation pretraining starts from the BBBC039 training split. Other
experiment recipes are unchanged, and no BBBC022 content checkpoint is reused
for segmentation. The datasets share an acquisition source; their study results
must not be counted as independent cohorts.

Use a fresh `run.output_root`, default `runs/journal_v2`, for the revised source.
An existing journal_v1 campaign is preserved and cannot silently accept changed
labels/configuration. Full GPU experiments have not been run here.
