# Workstation workflow validation

Validated on CPU on 2026-09-12/13, starting from audit commit `8c27b7b`.

| Check | Evidence |
|---|---|
| New workstation tests | 16 passed: all 234 job configurations, seed-specific dependencies, dataset rebasing/channel checks, HR filtering, nested well splits, cache parity, stale-artifact detection, interrupted-job retry, and actual small trainer integrations |
| Broad non-slow regression suite | 191 passed, 14 skipped, 1 deselected; one legacy literal-source assertion failed because its assignment was expressed as a conditional expression |
| Final focused regression | **37 passed** after changing the conditional expression into explicit fresh/legacy branches, preserving behavior and the historical `itertools.cycle` assignment; covers the legacy schedule and every workstation test |
| Controlled smoke through `paper.py smoke` | All five modes completed 8 synthetic CPU updates, checkpoint selection, three exact-noise test draws and acquisition metadata |
| MCF7 renderer integration | Figures 8 and 9 plus both JSON metadata files generated from synthetic TIFFs and temporary, untrained checkpoints; full output-path/dependency route exercised |
| CLI/setup hygiene | Help, init, plan, dry configuration materialization, missing-data failure, shell syntax and `git diff --check` checked |

The actual small training integrations cover fixed CNN training, learned staged
hardening, A–D ablation, three-stage segmentation with newly trained stage 1,
SwinIR SR using a fixed manifest, MCF7 CNN training with separate data/model
seeds, and SwinIR refinement loading an explicit seed-43 base checkpoint.
The refinement test substitutes a small pair cache and L1 loss; the SR test
uses tiny SwinIR capacity and disables perceptual/GAN loss. Those tests exercise
orchestration and interfaces, not full-budget scientific performance.

The full-model five-mode smoke is synthetic and intentionally tiny. The MCF7
renderer check uses synthetic 768×1280 TIFFs and a small SwinIR architecture;
its images and scores have no biological interpretation. It verifies that
figure generation uses the supplied fresh checkpoints and writes the promised
artifacts, including the nonoverlap/overlap acquisition distinction.

This execution environment has no CUDA device or the user's downloaded image
collections. It also blocks the Unix socket IPC used by multiprocess PyTorch
dataloaders; the small SR integration uses zero workers. A preflight check now
detects that restriction. The normal workstation recipe uses spawn workers,
configurable through `run.num_workers`.

No full PatchMNIST, BBBC022, MCF7, GAN/perceptual or paper matrix was trained
here. No original U2OS data was recovered. The existing missing-data/CUDA test
skips remain. Before full training, run `python paper.py check` against the
actual workstation paths. This is a workflow/code validation, not evidence of
corrected numerical rankings.

Tested software: Python 3.12.14, PyTorch 2.14.0+cu130 (CPU execution),
torchvision 0.29.0+cu130, NumPy 2.3.5, SciPy 1.17.0, timm 1.0.29; SwinIR
`6545850fbf8df298df73d81f3e8cba638787c8bd`. Source, configuration and input
inventories are frozen per campaign. Deprecation warnings did not fail checks.
