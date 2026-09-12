# Validation evidence

Validated on 2026-09-12 in the audit working tree, based on replication commit
`b71d8441e7e321c54df89f814cebbd587d4bc9b2`.

| Check | Observed result |
|---|---|
| Full suite: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -m 'not slow'` | **175 passed, 14 skipped, 2 deselected**, 135.82 s. Collected before the old epoch smoke was converted to a small CPU test. |
| Repaired epoch smoke: `python -m pytest -q tests/test_training_smoke.py` | **1 passed**, 5.94 s. Six synthetic training images; real epoch trainer, checkpoint selection and artifact export. |
| Targeted scientific, segmentation, schedule and Udith checks | **63 passed**; the extra old slow smoke failed due to missing `run_id`, then was repaired and passed separately as above. |
| SwinIR test file with pinned vendor | **14 passed, 1 skipped**; architecture, optimizer, deterministic evaluation checks. |
| Five-mode controlled smoke | Five modes × eight updates; seven synthetic test images and three exact-noise draws per mode. All evaluated masks binary; identical inverse initialization and validation/test tensors; paired spatial/Fourier initial binary masks; equal mean incident sequence dose. |
| Original-architecture smoke | `original_locality` + spatial masks completed eight updates and selected step 8. Synthetic data only. |
| Upstream diagnostics | Demonstrated accumulated illumination gradients, validation BatchNorm mutation and repeated training source order. New original-locality projection agrees with upstream `custom_v2`. |
| Historical artifacts | No changes under `experiments/`; frozen artifact tests pass. The stale expected report hash was corrected to the unchanged baseline file. |
| Patch hygiene | `git diff --check` passed. |

See [controlled_smoke.json](controlled_smoke.json) for environment, source hash,
per-mode synthetic metrics and acquisition accounting, and
[original_diagnostics.json](original_diagnostics.json) for numerical upstream
proofs and source hashes. Synthetic scores are **not** microscopy or PatchMNIST
performance estimates.

## Test issues resolved

The initial full suite had seven missing-vendor failures plus a preexisting
stale artifact hash. Installing the prescribed SwinIR commit resolved the
dependency failures. The report hash did not match
`git show b71d844:experiments/figure10_ablation_patchmnist/FIG10_REPORT.md`;
the expected hash now matches that unchanged file.

A new zero-rate gradient test initially required a particular clamp
subgradient. Installed PyTorch chooses zero at the boundary. The final test
checks finite boundary gradients and the expected positive-signal gradient,
rather than an arbitrary boundary convention.

The old epoch smoke read an incomplete base configuration, requested CUDA and
3,000 MNIST images, and failed before training. It now supplies a run ID and
derived dimensions, uses deterministic small CPU tensors, and exercises the
real trainer. No expected method ranking is part of the test.

## Limits

- No CUDA device was available; CUDA-specific checks were skipped.
- BBBC022, MCF7, original U2OS, `cell.tif`, Set5 images and some historical
  binary artifacts were unavailable; dependent tests were skipped.
- The separate slow experiment-matrix integration test was not run. Full
  microscopy, SwinIR, GAN/perceptual and segmentation experiments were not
  retrained. Their scientific performance remains unvalidated.
- Historical large checkpoints were not downloaded for full reevaluation.
  There is no claim of byte-for-byte numerical reproduction.
- TeX editorial notices were reviewed as source; a manuscript PDF was not
  compiled from its external figure/style dependencies.
- Passing tests verifies the exercised behavior, not every configuration or
  absence of every scientific or implementation error.

Environment: Python 3.12.14; PyTorch 2.14.0+cu130; torchvision 0.29.0+cu130;
NumPy 2.3.5; SciPy 1.17.0; timm 1.0.29; CPU execution. SwinIR source commit
`6545850fbf8df298df73d81f3e8cba638787c8bd`. Dependency deprecation warnings
did not fail the successful checks.
