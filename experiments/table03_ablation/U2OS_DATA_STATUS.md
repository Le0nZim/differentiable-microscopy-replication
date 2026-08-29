# AM-3 Phase 4 — U2OS data availability status

**Verdict: the paper's U2OS cell dataset is NOT available.** Table 3 / Fig. 10
cannot be reproduced on the paper's data. BBBC022 is used as a substitute proxy
and is explicitly labelled as such.

## Where I searched (and what was found)

| Location | Result |
|---|---|
| `data/u2os/` | **does not exist** |
| `data/` (repo) | `cell.tif`, `mcf7_bbbc021/`, `mnist/`, `sr/`, `substitute_data/` — **no U2OS** |
| `find data -iname "*u2os*"` | nothing |
| `find / -iname "*u2os*"` (full filesystem) | only `/home/leonidas/workspace/microscopy_tasks/3d_sim/OMX_U2OS_Actin_ex488_oil1518.tif` |
| `find /home/leonidas/workspace -iname "*dapi*"/"*2304*"/"*confocal*"` | nothing relevant (only an unrelated `hidapi` library dir) |
| author-adjacent dirs (`DEEP2/`, `DEEP_the_road_so_far/`, `dushan_codes/`, `microscopy_tasks/`) | no DAPI/confocal U2OS cell stacks |
| `src/datasets/u2os.py` | a **loader + documented preprocessor only**; it raises `FileNotFoundError` if `data/u2os` is absent (it is). No data ships with it. |

The single `*u2os*` hit (`OMX_U2OS_Actin_ex488_oil1518.tif`) is a 3D-SIM **actin**
image (wrong stain, wrong modality, single image) — not the paper's DAPI
spinning-disk confocal cell dataset. It is not usable for Table 3.

## What the paper's U2OS dataset is (and why BBBC022 ≠ it)

Paper §5.1: U2OS (bone osteosarcoma) cells, fixed 4% PFA, **DAPI** stained,
**spinning-disk confocal at 63×/1.4NA**, stacks `60 × 2304 × 2304`; MIP → subtract
bias 134.28 → clip 500 → min-max → downscale by 63/20 → `731×731`; split
168/21/21; random 256×256 crops + flips for train, fixed 256×256 patches for
val/test.

The local substitute (`data/substitute_data/BBBC022_v1_images_20585w1/`):

- BBBC022 **is** a U2OS Cell-Painting dataset, and the `w1` channel is **Hoechst
  33342** (a DNA stain, like DAPI) — so the *cell line and stained structure are
  the same family* as the paper. This is a closer proxy than a generic cell set.
- **But the acquisition differs fundamentally**: BBBC022 was imaged on an
  **ImageXpress Micro widefield** system; files are **520×696 uint16** single 2D
  fields (verified), not `2304×2304` confocal z-stacks. There is no z-stack
  (MIP is a no-op), and the paper's `63/20` downscale is not applicable.
- Preprocessing actually used (`experiments/ablations/preprocessing_report.json`,
  mode `paper_strict`): bias 134.28, clip 500, min-max. Documented deviations:
  no z-stack/MIP, no 63/20 downscale (native resolution, 256 crops), Hoechst not
  DAPI, clip 500 may be harsh on the BBBC022 uint16 scale.

## What the U2OS path *would* be if the data were dropped in

`src/datasets/u2os.py::U2OSPreprocessor` already implements the paper-faithful
chain (MIP → bias 134.28 → clip 500 → min-max → 63/20 downscale → 168/21/21 split
→ random 256 crops + flips train / fixed 256 crops val,test) and is covered by
`tests/test_u2os_preprocessor.py`. To run Track 1 (the only track allowed to
claim paper reproduction): place the U2OS stacks under `data/u2os/` and run the
ablation against a `u2os` dataset config. **This was not possible here because the
data is absent.**

## Conclusions allowed / not allowed

- **NOT allowed:** claiming Table 3 / Fig. 10 is numerically reproduced. The BBBC022
  numbers are a different microscope/resolution and must not be compared to the
  paper's U2OS MSE/SSIM.
- **Allowed:** using BBBC022 (a U2OS Cell-Painting proxy) to test whether the
  *implementation* is faithful and bug-free, and to study the *mechanism*
  (locality vs transpose vs data/regime). See `AM3_variant_audit.md` and
  `AM3_root_cause.md`.
- Because real U2OS is unavailable, AM-3 is **data-blocked for the Table-3
  numbers**; the resolution rests on proving implementation fidelity + diagnosing
  the proxy ordering (Phases 1–3, 5–7).
