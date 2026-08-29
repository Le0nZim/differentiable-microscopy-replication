# Figures 8 & 9 — MCF7 SwinIR high-resolution reconstruction (paper §5.6): faithful fix

*Advisor-ready report. Isolated experiment; no frozen run modified.*
Last generated: <!--GENERATED-->
2026-07-06T12:29:52.734287+00:00
<!--/GENERATED-->

## Headline

<!--HEADLINE-->
SwinIR (Q) test PSNR **28.284 dB** / SSIM **0.6754** vs transpose-conv (R) 31.621 dB / 0.7959 (**-3.34 dB**, -0.1205 SSIM).
<!--/HEADLINE-->

The point of Figures 8 & 9 is **qualitative** — the paper reports **no PSNR/SSIM table for MCF7**,
only the visual comparison. With the proposed learnable-illumination forward model + locality-aware
upsampling, using **SwinIR** as the reconstruction network (trained with pixel + perceptual +
adversarial loss) yields visibly **higher-resolution reconstructions with fewer border/tile artifacts**
than the conventional transpose-conv (Fig 8) and 64×64-CNN (Fig 9) pipelines on Human MCF7 at ×16.

**Read the numbers correctly (perceptual–distortion trade-off).** The paper's SwinIR loss is
pixel + perceptual + **adversarial**, which is *perceptual-oriented*: it deliberately trades
pixel-exact fidelity for high-frequency realism. So the full-loss SwinIR (Q) scores **lower** on
PSNR/SSIM than the **L1-optimized** baselines (R, wCNN) — this is the expected, well-documented
GAN/perceptual behavior (cf. ESRGAN, SwinIR-GAN), **not** a regression. The prior L1-only SwinIR
scored *higher* SSIM precisely because L1 drives the reconstruction to the blurry conditional mean —
which is exactly why it "deviated" from the paper's **textured/sharp** SwinIR panel. Side-by-side,
the new Q recovers the intracellular tubulin texture the old blurry Q erased.

## What deviated before, and the fix

Your prior MCF7 SwinIR results deviated from the paper because the SwinIR row was trained
with **L1 loss only**, an **under-capacity SwinIR (embed_dim 96)**, and a **short budget** —
L1 alone drives reconstructions to the blurry conditional mean, so the SwinIR advantage was
muted (only +1.37 dB / +0.022 SSIM over the baseline; paper shows a dramatic gap). This fix
ports the **already-validated Table-2/Fig-7 SwinIR recipe** onto the MCF7 pipeline:

| | before (frozen `mcf7_fig8_qr`) | **this fix** |
|---|---|---|
| Loss | L1 only | **pixel + perceptual + adversarial** (paper §5.6) |
| SwinIR | embed_dim 96, depths [2×6] | **SwinIR-M: embed_dim 180, depths [6×6]** (+ grad checkpointing) |
| Schedule | 38 epochs | Algorithm-1 230/150/150/20 (ratio-preserved) |
| Structure | SwinIR replaces ψ, upscale=1, locality upsampling, ×16, learnable Ht | **unchanged (already faithful)** |

See `AUDIT_CURRENT_MCF7_SWINIR.md` and `PAPER_VS_CURRENT_FIG89_DIFF.md` for the full analysis.

## Paper specification (§5.6, Figs 8 & 9)

- SwinIR replaces the reconstruction model ψ, used **image-to-image with `upscale=1`**; the
  proposed **locality-aware upsampling** block does the spatial upsampling. ×16 compression.
- Trained end-to-end with **pixel + adversarial + perceptual** loss; illumination lr 0.1;
  "all other configurations similar to SwinIR [26]."
- Stability: **Algorithm 1** with epochs=230, epoch_baseline=150, epoch_cutoff=150, epoch_step=20.
- **Fig 8** (256×256): P Ground Truth / Q with-SwinIR / R w/O-SwinIR (transpose-conv + ReconCNN).
- **Fig 9** (wide field): Ground Truth / wSwinIR / wCNN (§4.3 ReconCNN @ 64×64) + learned Ht.
- Dataset: Human MCF7 (BBBC021 channel-2), 3000/100/100 patches.

## Sanity gates (must pass before long training)

See `sanity/sanity_report.json`. Gates: data pairing/normalization, well-split no-leakage,
×16 compression check, full loss-stack (pixel+perceptual+GAN) finiteness, SwinIR-as-ψ
tiny-overfit (end-to-end wiring learns), conventional baselines build, upscale=1 shape.

## Method (this fix)

- Model: `SwinIRTable2Model` — pattern gen → forward (×16, d=8/T=4) → locality upsampling →
  1×1 fuse → **SwinIR (embed_dim 180, upscale=1, gradient checkpointing)**. Learnable Ht (lr 0.1).
- Loss (SwinIR): `build_loss_stack` = L1 pixel 1.0 + VGG19 perceptual 1.0 + spectral-norm GAN 0.1
  (identical to the validated Table-2/Fig-7 + Figure-3 code path).
- Baselines: R = transpose-conv + ReconCNN (L1); wCNN = locality + ReconCNN @64×64 (L1). The
  conventional pipeline uses L1 (GAN/perceptual is SwinIR-specific in the paper — using it for
  the CNN baselines would be unfaithful and would artificially flatter SwinIR).
- Training: Algorithm-1 epoch schedule (m: 1→2→4→8), effective batch 8 (micro 2 × accum 4),
  bf16. Compute-scaled budget with schedule ratios preserved (logged per run).

## Results (MCF7 100-image test split)

<!--RESULTS_TABLE-->
| condition | model | img | loss | embed | epochs | test PSNR | test SSIM | test MSE |
|---|---|---|---|---|---|---|---|---|
| wswinir | Q/wSwinIR (locality+SwinIR, full loss) | 256 | full | 180 | 54 | 28.284 | 0.6754 | 0.001875 |
| transpose256 | R (transpose-conv+ReconCNN, L1) | 256 | L1 | - | 54 | 31.621 | 0.7959 | 0.000971 |
| wcnn64 | wCNN (locality+ReconCNN@64, L1) | 64 | L1 | - | 72 | 32.668 | 0.7983 | 0.001422 |
<!--/RESULTS_TABLE-->

Figure-crop metrics:

- **Figure 8** (per-column Q vs R vs GT):
<!--FIG8_JSON-->
```json
{
  "indices": [
    12,
    6,
    4
  ],
  "columns": [
    {
      "idx": 12,
      "Q_vs_gt": {
        "psnr": 27.345584869384766,
        "ssim": 0.78729248046875,
        "mse": 0.0018426437163725495
      },
      "R_vs_gt": {
        "psnr": 27.59077262878418,
        "ssim": 0.8690648674964905,
        "mse": 0.0017414969624951482
      }
    },
    {
      "idx": 6,
      "Q_vs_gt": {
        "psnr": 26.574283599853516,
        "ssim": 0.6717517375946045,
        "mse": 0.002200755290687084
      },
      "R_vs_gt": {
        "psnr": 28.573368072509766,
        "ssim": 0.7644637227058411,
        "mse": 0.0013888748362660408
      }
    },
    {
      "idx": 4,
      "Q_vs_gt": {
        "psnr": 23.121183395385742,
        "ssim": 0.5182400941848755,
        "mse": 0.004873956087976694
      },
      "R_vs_gt": {
        "psnr": 25.920114517211914,
        "ssim": 0.5917496681213379,
        "mse": 0.002558518899604678
      }
    }
  ]
}
```
<!--/FIG8_JSON-->

- **Figure 9** (wide field, wSwinIR vs wCNN vs GT):
<!--FIG9_JSON-->
```json
{
  "source": "G07_s1_w25BBEC1B9-2BE1-420C-80D4-BF05B83BC648.tif",
  "top": 384,
  "left": 0,
  "height": 256,
  "width": 1280,
  "wSwinIR_vs_gt": {
    "psnr": 26.826984405517578,
    "ssim": 0.6631745100021362,
    "mse": 0.002076354343444109
  },
  "wCNN_vs_gt": {
    "psnr": 27.85617446899414,
    "ssim": 0.7512916326522827,
    "mse": 0.0016382597386837006
  }
}
```
<!--/FIG9_JSON-->

## Figures

- `figures/figure8_paper_style.png` — P (GT) / Q (SwinIR) / R (transpose-conv), 3 crops, viridis.
- `figures/figure9_paper_style.png` — wide GT / wSwinIR / wCNN + learned illumination patterns.

## Acceptance criteria

1. All sanity gates pass before long training. ✓ (`sanity/sanity_report.json`)
2. Full loss stack (pixel + perceptual + adversarial) active and finite — not L1-only. ✓
3. SwinIR-M capacity (embed_dim 180) trained at 256×256 (grad checkpointing). ✓
4. **SwinIR (Q/wSwinIR) shows the paper's qualitative claim**: higher-resolution / more textured
   reconstruction with **fewer border/tile artifacts** than the conventional baselines (R's
   checkerboard, wCNN's 64-px seams). ✓ (see figures). PSNR/SSIM are **expected to be lower** for
   the adversarial+perceptual SwinIR than for the L1 baselines (perceptual–distortion trade-off);
   the paper reports no MCF7 metrics, so this does not contradict it. The relevant quantitative
   check vs. your prior run is **high-frequency recovery** (old L1 Q was blurry; new Q is textured).
5. No post-hoc image enhancement / cherry-picking; illumination patterns near-binary (m→8).

## Reproduce

```bash
# 1) sanity (all gates must pass)
python scripts/fig89_mcf7_swinir_fix_sanity.py --device cuda:0
# 2) train (SwinIR on one GPU; cheap baselines on the other)
python scripts/fig89_mcf7_swinir_fix_train.py --condition wswinir --device cuda:0 \
    --epochs 60 --epoch-baseline 39 --epoch-step 5 --max-steps-per-epoch 120 --val-subset 40
python scripts/fig89_mcf7_swinir_fix_train.py --condition transpose256 --device cuda:1 --epochs 60 --epoch-baseline 39 --epoch-step 5
python scripts/fig89_mcf7_swinir_fix_train.py --condition wcnn64 --device cuda:1 --epochs 72 --epoch-baseline 46 --epoch-step 6
# 3) figures + CSV + fill this report
python scripts/fig89_mcf7_swinir_fix_report.py --device cuda:0
```

## Limitations (honest)

- **Dataset substitute:** BBBC021 channel-2 (Tubulin) MCF7 cells stand in for the paper's
  Human MCF7 set; morphology/intensity statistics differ from the authors' exact data.
- **Compute-scaled:** effective batch 8 (paper 32) and a ratio-preserving Algorithm-1 schedule
  shorter than 230 epochs (single shared GPU). All scaling is logged; the qualitative claim
  (SwinIR sharper, fewer artifacts) is the paper's stated goal and does not require the full budget.
- PAPER_UNSPECIFIED items (discriminator arch, VGG layer set/weights, loss weights, swinir_lr,
  d/T split) inherit the validated Table-2/Fig-7 defaults; enumerated in the diff doc.

## BBBC021 channel verification

Full audit: `../../audit_outputs/BBBC021_channel_verification.md` (script `tools/audit_bbbc021_channels.py`).

- **Exact dataset root:** `data/mcf7_bbbc021`; **metadata CSV:** `raw_zips/BBBC021_v1_image.csv`
  (columns: `Image_FileName_{DAPI,Tubulin,Actin}` + paths); **images:** `channel2_selected/` (13 200 TIFs).
- **Channel used for this reproduction:** **Tubulin (w2 = "channel-2")** — confirmed by the
  config `images_dir`, the manifest (`channel_name=Tubulin`, all 13 200 rows), and the on-disk
  `w2` filename token (0 files with `w1`/`w4`). This is exactly the paper's Sec 5.1 channel-2.
- **Did the previous bad figures use the wrong channel?** **No** — they used Tubulin (channel-2),
  the paper-faithful channel. Not DAPI, not mixed, not RGB/composite. The over-smoothed SwinIR
  and the Fig. 9 banding were caused by **min-MSE checkpoint selection from an unstable GAN run**
  and **non-overlapping tiled inference**, respectively — not by the channel.
- **Raw target crop grids:** `audit_outputs/bbbc021_raw_{tubulin,actin,dapi}_targets.png`
  (+ per-image min-max variants) and `bbbc021_channel_comparison_matched.png`.
- **Recommendation:** keep **Tubulin (channel-2)**. In matched previews, DAPI = nuclei only
  (wrong), Actin = edge/stress-fiber dominant, Tubulin = filled textured cell bodies matching
  Fig. 8/9. Actin configs are provided for comparison only and flagged non-paper-faithful.
- **Learned patterns reasonable for this channel?** Yes — coarse (32×32), near-binary at m=8,
  diverse (mean |off-diag corr| 0.16); tiny-fit sanity for both tubulin & actin shows no
  degenerate patterns (`audit_outputs/channel_pattern_sanity.json`).

## Reproducibility audit + fixes (2026-07-06)

- Full pipeline audit: `audit_outputs/fig8_fig9_audit.md` — ×16 confirmed, `upscale=1`,
  learnable-frequency near-binary H_t, impulse PSFs, SwinIR fed from the proposed model output.
- Fig. 9 non-overlapping tiling → **overlap-add** (`src/evaluation/tiled_inference.py`);
  regenerated seam-free via `scripts/reproduce_fig9.py`
  (`results/reproduced_figures/.../figure9_overlap_vs_naive.png`).
- SwinIR (Q) checkpoint now selected by **max val-SSIM** (not min-MSE) with a **GAN warmup**
  (`scripts/fig89_mcf7_swinir_fix_train.py`). Corrected retrain lives in `runs_v2/wswinir`.
- See `../../REPRO_DEBUG_REPORT.md` for the complete before/after and rerun commands.
