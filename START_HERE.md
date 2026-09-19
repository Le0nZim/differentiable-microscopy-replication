# Run from a clean workstation

Use Linux (or WSL2 with working CUDA). The only file you normally edit is
`workstation.yaml`. Keep your existing `data/` folder wherever it is. The workflow
reads its images in place and writes new experiments to a separate campaign.

**Existing September 2026 campaigns:** read the
[progress audit and rerun guide](docs/PROGRESS_AUDIT_2026-09-19.md) before
continuing microscopy training. It identifies affected results and corrected
stages. Use a new campaign for these source changes; preserve previous results.

## 1. Get a clean source checkout

Run these from the parent folder where you want the **new** checkout. Historical
experiment trees, old checkpoints and figure archives are excluded; no LFS
downloads are needed. Use a new folder, not your existing working directory.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --single-branch \
  --branch workflow/clean-workstation --filter=blob:none --sparse \
  https://github.com/Le0nZim/differentiable-microscopy-replication.git microscopy-clean
cd microscopy-clean
git sparse-checkout set src workflow scripts configs tests docs data
bash scripts/workstation/setup.sh
source .venv/bin/activate
python paper.py download-weights
```

Setup installs this project and the pinned SwinIR source. `download-weights`
caches the ImageNet VGG19 weights used by the perceptual loss (~548 MB); it is
the only model-weight download. Once software, weights and datasets are in
place, training does not need internet. If PyTorch does not detect your GPU,
install the CUDA-enabled PyTorch build appropriate for your workstation before
continuing. `check` refuses an unavailable GPU instead of silently using CPU.

## 2. Point at the data you already have

Replace **only** the path in this command with your actual existing data folder:

```bash
python paper.py init --data-root /absolute/path/to/differentiable_micoscopy_replication/data
```

This generates `workstation.yaml` from exactly the directory layout you supplied.
Absolute paths and spaces are supported. Inspect the paths in that one file;
your original directory spelling does not matter. The default device is
`cuda:1`. Indices respect `CUDA_VISIBLE_DEVICES`; when exposing only physical GPU
1 as `CUDA_VISIBLE_DEVICES=1`, set the config device to `cuda:0`.
`run.num_workers` controls SR dataloading processes (default 8); set it to 0 in
environments that do not permit worker IPC. It does not change batch size or training budgets.

| Your folder/file | What the workflow uses |
|---|---|
| `mnist/MNIST/raw/` | All four IDX files: train/t10k images and labels. Plain files or `.gz` accepted. PatchMNIST is generated automatically, not downloaded. |
| `substitute_data/BBBC022_v1_images_20585w1/` | One w1 plate, 3,456 TIFFs / 384 wells / 9 sites. No channel mixing. |
| `mcf7_bbbc021/channel2_selected/` | **Tubulin w2 only**. If empty/missing, init uses `channel2_tubulin/` when available. It never combines the two folders. |
| `mcf7_bbbc021/manifests/mcf7_channel2_manifest.csv` | `image_file` and `well` columns. Missing referenced images or duplicate rows fail early. Old absolute image paths are rebased by unique basename. If the manifest is absent, well/plate identities must be recoverable from filenames. |
| `sr/train/DIV2K/` | 800 training HR originals. Nested `HR/` or `DIV2K_train_HR/` works. If validation images are also present, point the config directly at the training HR directory. |
| `sr/train/Flickr2K/` | 2,650 HR originals. `FLICK2R/` is an alternative spelling used only if `Flickr2K/` has no images. No duplicate collection is added. |
| `sr/test/Set5,Set14,BSD100,Urban100,Manga109` | Respectively 5, 14, 100, 100, 109 HR targets. Nested folders are supported; LR/bicubic copies are excluded. |
| `cell.tif` | Optional single-image diagnostic; not required or sufficient for any training dataset. |
| MCF7 DAPI, actin and `raw_zips/` | Not read for the tubulin experiment. Leave them in place. |

No image collections are copied. The preparation step writes small manifests
and flat symlink views for SR test images. Gzip-only MNIST is extracted into the
campaign's prepared folder. Preflight checks filenames, identity overlaps,
expected counts, all source file sizes/timestamps, and decodes representative
images. It does not claim to decode every microscopy image in advance.

## 3. Check, inspect the queue, run

```bash
python paper.py check
python paper.py plan
python paper.py run
```

`run` automatically prepares the datasets, freezes the configuration and split
manifests, and starts **one job at a time**. Three model seeds (42, 43, 44) are
used throughout. The full default queue has **234 jobs**, including three MCF7
figure-rendering jobs. These are full existing recipe budgets, not a quick demo;
SwinIR stages can take substantial time. No wall-time estimate is inferred from
this CPU-only implementation environment. The exact recipes and budgets are in
[WORKSTATION_PROTOCOL.md](docs/WORKSTATION_PROTOCOL.md).

For a short software check that needs no datasets:

```bash
python paper.py smoke
```

For a gradual launch, start with PatchMNIST and add the remaining stages later:

```bash
python paper.py check --stages patchmnist
python paper.py run --stages patchmnist
python paper.py run
```

The last command skips the already completed PatchMNIST jobs. A stage can be
run independently, for example `--stages noise` or `--stages sr`.
`--stages content_swinir` automatically includes its `content` prerequisites.
To see every job ID without writing or training: `python paper.py run --dry-run`.
`prepare` is also available separately; it does not start training.

## 4. Watch progress and resume

```bash
python paper.py status
python paper.py report
```

The runner prints each active log path and a heartbeat every 30 seconds. Read
that log in another terminal. Keep the runner in a persistent terminal session
when disconnecting from the workstation.

To continue after interruption, rerun **the identical command**. A completed job
is skipped only when its recorded output artifacts still match. Failed or
interrupted jobs restart from step zero in `attempt_002`, etc.; previous attempts
are preserved. The workflow does **not** claim exact training-state resume.
Ctrl+C/SIGTERM stops the active process group. After a forced SIGKILL of the
runner, first stop any orphan worker before restarting.

Source edits, path changes and modified input inventories cannot silently reuse
completed work. For a genuinely new run, set `run.output_root` to a new empty
folder such as `runs/paper_v2`. **Do not delete your datasets.** A disk floor of
20 GiB is checked before each job; it is a guard, not an estimate of total disk
needs. PatchMNIST caches require about 14 GiB for the full size/count grid,
and checkpoints add more. You can place `cache_root` and `output_root` on larger
disks. Do not edit experiment recipes simply to make a preferred method win.

## 5. Where the outputs are

| Path under `run.output_root` | Contents |
|---|---|
| `campaign.json`, `plan.json` | Frozen settings, source fingerprint, commit and ordered job specifications |
| `input_inventory.json`, `prepared/` | Actual inputs, fixed splits and prepared dataset views |
| `state.json` | Completed/running/failed job ledger |
| `jobs/<stage>/<condition_seed>/attempt_001/` | Exact config, console log, checkpoints, metrics and per-job qualitative outputs |
| `report/README.md` | Progress, comparison plots and links to verified outputs |
| `report/metrics_by_seed.csv`, `aggregate.csv`, `coverage.csv` | Raw per-seed metrics, means/SDs and paper-item coverage |

The original U2OS confocal dataset is still unavailable. BBBC022-based results
are explicitly labeled substitutes, and segmentation uses pseudo-labels.
This workflow runs the audited repository recipes, which retain soft masks and
documented budget/architecture deviations. For the separate binary-mask,
equal-dose audit comparison use `python paper.py run --stages controlled`.
That experiment is not silently merged into the paper's legacy-protocol tables.
