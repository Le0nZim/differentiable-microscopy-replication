# Workstation experiment protocol

This workflow starts from audit commit `8c27b7b`. It reorganizes execution and
data preparation; it does not certify numerical reproduction. The source paper
is [arXiv:2203.14945](https://arxiv.org/abs/2203.14945). Consult the existing
[scientific audit](SCIENTIFIC_AUDIT.md) for the remaining interpretation limits.

| Stage | Paper item | Conditions per seed | Training budget per condition |
|---|---|---:|---|
| `patchmnist` | Additional A–D diagnostic | 4 | Uniform 8,500-step ablation schedule |
| `upsampling` | Figure 5 | 18 | Sizes 128/256/512 × train counts 600/3000/6000 × two upsamplers; 4,000 steps |
| `noise` | Table 1, Figure 6 | 16 | k=10/10,000 × read SD=0/2.7/2/6 × fixed/learned; 14,850 steps for both methods |
| `content` | Figure 3 | 16 | C=16/64/256/1024 × uniform/random/Hadamard/learned; 10,100 steps for all methods |
| `content_swinir` | Figure 3 refinement | 8 | Four compression levels × random/learned; corresponding freshly trained base, 50,000 refinement steps |
| `segmentation` | Figure 4 | 6 | C=64/256/1024 × random/learned; 7,400 stage-1 steps for both, 1,200 head-only + 2,500 joint task steps |
| `sr` | Table 2, Figure 7 | 2 | SwinIR with/without learned illumination; existing compute-limited 20,000-step recipe |
| `mcf7` | Figures 8–9 | 3 + 1 render | SwinIR-256, transpose-CNN-256, locality-CNN-64; 230 epochs / 150 baseline / 20-step hardening schedule |
| `ablation` | Table 3, Figure 10, BBBC022 | 4 | Uniform A–D 8,500-step schedule |
| `controlled` (optional) | Audit extension | 5 | STE binary patterns, equal mean sequence dose, 10,100 updates; exact-noise test evaluation |

The default queue contains 78 jobs per seed × 3 seeds = 234. Every family now
uses the same prescribed set of training seeds, rather than adding extra seeds
only to a favorable/extreme cell. Model seed and data seed are independent.
The spatial/Fourier ablation remains an empirical comparison with no target
ranking. These recipes use the existing reconstruction implementations; they
do not substitute the original student's architecture silently.

## Data identity and changes from historical execution

- PatchMNIST uses 32px digits on a 20×20 canvas, with 256px crops and 3,000/375/375
  samples except the explicitly varied Figure-5 settings. Validation/test draw
  from disjoint halves of MNIST-test; training draws from MNIST-train. Data seed
  stays fixed across all model seeds. A memory-mapped float32 cache is generated
  on demand and keyed by input hash and generation recipe.
- BBBC022 reconstruction/ablation/refinement share a 220/40/60 well partition;
  training uses nine sites/well, evaluation one. Segmentation takes the first
  168/21/21 wells **within those partitions** and one site/well. This prevents a
  well held out in one family appearing in another family's training data.
  It changes the old independently generated small split. All file identities
  are recorded; the dataset remains an exploratory substitute for unavailable
  U2OS confocal data, not a sealed confirmatory holdout.
- BBBC022 reconstruction keeps minimal per-image percentile normalization.
  Segmentation retains its documented paper-strict intensity preprocessing and
  TrackMate-style raw threshold 506 pseudo-labels. Those assumptions are not
  newly validated biological annotations or camera calibration.
- MCF7 uses the tubulin manifest and well grouping, with fixed data seed across
  model seeds and figure rendering. DAPI and actin are excluded. No manifest
  image may be missing or ambiguously identified.
- SR freezes a 2% validation holdout from the unique HR originals, then shares
  it across both conditions and all model seeds. Flat test views use symlinks.
  Exact repeated paths and repeated scene filenames in a collection fail
  preflight; there is no claim of a full perceptual duplicate-image audit.

## Fixes necessary to run from a clean slate

- Figure-3 refinement uses the actual dependency checkpoint and its model seed;
  it no longer assumes seed 42 or an old experiment directory.
- Segmentation trains its own stage 1 on its declared nested small split. Fixed
  and learned stage-1 update counts are equalized to 7,400.
- Fresh A–D runs request renewed dataloader passes and image-weighted metrics.
  The optional historical shared-warmup replay continues to use its explicitly
  legacy cached ordering; it is not part of this workflow.
- MCF7 `last.pt` is saved before selecting the best checkpoint; figure data seed
  matches the training partition. Selected-best and final artifacts have distinct
  meanings. This does not add full optimizer/RNG resume to the MCF7 trainer.
- All machine paths, output roots, seeds and derived compression values are
  materialized before each job. Old learned weights and historical result files
  are never input dependencies of the new queue.

## Resume and evidence

The campaign records a source/config fingerprint, input size/mtime inventory,
manifest hashes, exact per-job configs and output hashes. Completion requires
both expected metrics and checkpoint artifacts. A changed completed artifact
halts reuse so downstream results cannot silently reference a different model.
Normal resume checks checkpoint size/mtime and rehashes changed files; it hashes
small summaries every time. It is not a continuous file-integrity monitor.

Jobs run in fresh processes. Interrupted jobs restart as new attempts because
the existing trainers do not all persist the optimizer, data-loader and RNG
state required for exact replay. Previous attempts remain inspectable.
Use a new campaign directory for recipe/source changes. No automatic deletion,
dataset download, checkpoint selection by test score, or desired-ranking gate
is performed by this orchestrator.
