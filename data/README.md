# Datasets (not vendored)

**Fresh workstation:** use [`START_HERE.md`](../START_HERE.md) and
`python paper.py init --data-root /your/existing/data`. Keep datasets wherever
they already live. The generated `workstation.yaml` is the single source of
machine paths for the new workflow. The legacy relative paths below describe
older individual drivers.

Place data under this directory. Nothing here is committed except this README.

Configs resolve paths relative to the **repository root**.

| Path | Used by | Notes |
|---|---|---|
| `data/mnist/` | Tables 1, Figs 5–6 (PatchMNIST) | torchvision can download MNIST if the loader is allowed network access |
| `data/substitute_data/` | Figs 3, 4, 10 and Table 3 | BBBC022 Hoechst widefield substitute for unavailable U2OS confocal |
| `data/mcf7_bbbc021/` | Figs 8–9 | BBBC021 Human MCF7, channel-2 tubulin |
| `data/sr/train/DIV2K/HR` | Table 2 / Fig 7 | optional if you only eval existing checkpoints |
| `data/sr/train/Flickr2K/HR` | Table 2 / Fig 7 train | HR-only (no bicubic x2/x3/x4 copies) |
| `data/sr/test/{Set5,Set14,BSD100,Urban100,Manga109}/HR` | Table 2 / Fig 7 eval | standard SR benches |

The original paper's **U2OS DAPI spinning-disk confocal** volumes are not available. Substitute numbers are pipeline/trend evidence only and are **not** numerically comparable to the paper's U2OS tables and figures.

See `ASSUMPTIONS.md` for detector-noise, pattern, and substitute-data caveats.
