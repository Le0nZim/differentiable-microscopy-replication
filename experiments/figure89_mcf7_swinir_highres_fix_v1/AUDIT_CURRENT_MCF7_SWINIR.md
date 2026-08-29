# Audit — current MCF7 SwinIR high-resolution reconstruction (paper Figures 8 & 9, §5.6)

**Scope.** Figures 8 and 9 (paper §5.6, *second* part — the Human MCF7 experiments; the
*first* part, Table 2 / Fig 7, is the classical SR benchmark on Div2K/Flickr2K→Set5/… and
is already replicated and validated separately). This audit inspects the **existing** MCF7
Fig 8/9 implementation and explains why "the resolution with SwinIR deviates too much from
the paper." It does **not** modify any frozen run.

---

## 1. What the paper actually specifies for Figs 8 & 9

From `paper_sources/paper.md` §5.6:

- **Task.** Replace the reconstruction model ψ (Fig. 1) with **SwinIR used as an
  image-to-image network *without upscaling*** (`upscale = 1`); the proposed
  **locality-aware upsampling block** still does the spatial upsampling. Dataset: **Human
  MCF7 cells** (BBBC021 channel-2), **×16 compression**.
- **Loss (explicit).** *"Similar to SwinIR real-world image super-resolution task, we
  utilize **pixel loss, adversarial loss, and perceptual loss** to train the network
  end-to-end."* Illumination learning rate **0.1**; "all other configurations similar to
  SwinIR [26]."
- **Training stability (explicit).** *"To improve the stability of SwinIR-based training
  corresponding to Fig. 8, 9, we utilize algorithm 1 with **epochs = 230, epoch_step = 20,
  epoch_cutoff = 150, epoch_baseline = 150**."*
- **Fig. 8** (256×256): rows **P** = Ground Truth, **Q** = *with SwinIR* (locality-aware
  upsampling + SwinIR), **R** = *w/O SwinIR* (**transpose-convolution** upsampling +
  conventional ReconCNN of §4.3). Claim: Q sharper, fewer artifacts.
- **Fig. 9** (wide field): **Ground Truth / wSwinIR / wCNN**, where **wCNN** is the classical
  §4.3 convolutional reconstruction trained at **64×64** patches "for a fairer comparison,"
  with the learned illumination patterns Ht shown on the right. Claim: SwinIR has **fewer
  border artifacts and higher visual image quality**.

Reference images: `paper_sources/figures/_page_10_Figure_4.jpeg` (Fig 8),
`_page_11_Figure_0.jpeg` (Fig 9).

---

## 2. What the current implementation does

**Code paths.**
- Trainer: `scripts/run_mcf7_fig8_qr.py` (Fig 8 Q vs R), `scripts/run_mcf7_fig9_wcnn.py` (Fig 9 wCNN).
- Config: `configs/mcf7_li_swinir_paper_direct.yaml`.
- Model: `SwinIRTable2Model` (`src/baselines/swinir/table2_pipeline.py`) — pattern gen →
  forward → **locality upsampling → 1×1 fuse → SwinIR(upscale=1)**. Structurally correct.
- Frozen runs: `experiments/swinir_or_highres/mcf7_fig8_qr/`,
  `.../mcf7_fig9_superpixel/`, `.../mcf7_fig9_wcnn/`.
  Rendered: `paper_ready_results/02_main_figures/fig08_swinir_mcf7/`, `.../fig09_more_swinir_mcf7/`.

**Current frozen results** (`mcf7_fig8_qr/aggregate_summary.json`, 100-image MCF7 test):

| condition | backbone | epochs | MSE | SSIM | PSNR (dB) |
|---|---|---|---|---|---|
| Q: with SwinIR | locality + SwinIR | 38 | 0.000694 | **0.8559** | **33.44** |
| R: w/O SwinIR  | transpose-conv + CNN | 72 | 0.000935 | 0.8340 | 32.07 |

SwinIR advantage over the conventional baseline is only **+1.37 dB / +0.022 SSIM** — a
*mild* improvement. The paper's Fig 8 shows a *dramatic* sharpness/detail gap.

---

## 3. Root-cause analysis — why the SwinIR resolution deviates

The structure (SwinIR replaces ψ, `upscale=1`, locality upsampling, ×16, Algorithm-1) is
faithful. The **training fidelity** is not, on exactly the axes that govern perceptual
sharpness. Three deviations, in order of impact:

### (D1) Loss is L1-only — the paper uses pixel + perceptual + adversarial  ← dominant
`run_mcf7_fig8_qr.py:256` → `loss = F.l1_loss(rec, x)`; config `training.loss: l1`.
The paper's §5.6 SwinIR loss is **pixel + adversarial + perceptual**. Pure L1 is
mean-reverting and provably drives reconstructions toward the blurry conditional mean —
it *cannot* synthesize high-frequency texture. Perceptual (VGG) + adversarial (GAN) losses
are precisely what make SwinIR-family outputs look sharp/"high-resolution." **This is the
same defect we already diagnosed and fixed for Figure 3** (see
`experiments/figure3_bbbc022_swinir_fix_v1/`). It is the primary reason the SwinIR row
looks low-resolution vs the paper.

### (D2) Under-capacity SwinIR — embed_dim 96, depths [2×6] vs SwinIR-M (180, [6×6])
Config `mcf7_li_swinir_paper_direct.yaml:68-70`: `embed_dim 96`, `depths [2,2,2,2,2,2]`,
`num_heads [3,3,3,3,3,3]`, with the note *"vendor default 180 infeasible at 256."* The
paper's "similar to SwinIR configurations" = **SwinIR-M** (embed_dim 180, depths [6×6],
heads [6×6]) — what the validated Table-2/Fig-7 run uses. ~½ the transformer depth and
~⅓ the width materially reduces detail-restoration capacity.

### (D3) Short compute budget with no full-loss schedule
Q ran **38 epochs**, batch 8, 220 steps/epoch (~8.4k steps, ~2 h) vs the paper's
**230-epoch** Algorithm-1 schedule (baseline 150 / step 20) at batch 32. Because 38 « the
150-epoch baseline, the illumination sharpening (m → 8) barely engages, so the learned
patterns and the recon never reach the paper's operating point.

### (D-ok) Correctly faithful already
- SwinIR **replaces** ψ with `upscale=1` + locality-aware upsampling (not appended). ✓
- ×16 via `downscale_factor 8`, `num_patterns 4` (d²/T = 64/4 = 16). ✓
- Learnable-frequency illumination, `illumination_lr 0.1` (paper §5.6). ✓
- Fig 8 axis is **Q vs R = SwinIR vs transpose-conv** (a prior wrong LI-vs-noLI attempt was
  already superseded). ✓
- Fig 9 wCNN trained at 64×64; super-pixel (coarse) learned patterns match the paper. ✓
- MCF7 data is real BBBC021 channel-2, well-disjoint 3000/100/100 split. ✓

---

## 4. Fix strategy (mirrors the Figure-3 SwinIR fix)

Port the **already-validated** Table-2/Fig-7 SwinIR training recipe into the MCF7
end-to-end pipeline, changing only training fidelity — never post-hoc enhancement:

1. **Full loss stack** via `baselines.swinir.losses.build_loss_stack`:
   pixel(L1) 1.0 + perceptual(VGG19) 1.0 + adversarial(GAN) 0.1 — the exact stack used by
   the working `run_am4_swinir_table2.py` and the Figure-3 fix.
2. **SwinIR-M capacity** (embed_dim 180, depths/heads [6×6]) with **gradient checkpointing**
   (`use_checkpoint=True`) + grad accumulation so it fits at 256×256 within the free GPU
   memory (the "180 infeasible at 256" note is resolved by checkpointing).
3. **Algorithm-1 schedule** (epochs/baseline/cutoff/step = 230/150/150/20), ratio-preserved
   if compute-scaled, with the scaling **documented** exactly like Fig 3 and the prior runs.
4. **Baselines** R (transpose-conv + ReconCNN, L1) and wCNN (64×64, L1) retrained in the
   isolated dir for a clean, self-contained comparison (conventional pipeline → L1 is
   faithful; GAN/perceptual is SwinIR-specific per the paper).
5. **Sanity gates before long training** (data pairing/normalization, well-split
   no-leakage, ×16 forward compression check, loss-stack + GAN reachability + finiteness,
   SwinIR-as-ψ tiny-overfit, baseline builds, memory/speed smoke).

**Acceptance target.** After the fix, the MCF7 SwinIR row should qualitatively match the
paper: visibly sharper / higher-frequency detail than the transpose-conv (R) and 64×64-CNN
(wCNN) baselines, with fewer border/tile artifacts — a clear (not marginal) gap — while
remaining faithful and auditable (no cherry-picking, no post-hoc sharpening).
