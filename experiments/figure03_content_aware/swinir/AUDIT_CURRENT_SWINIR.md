# AUDIT — Current Figure-3 BBBC022 "+SwinIR" pipeline

Scope: the scripts/configs/checkpoints that produced the current Fig-3 `+SwinIR`
columns (the ones the advisor considers "only mildly improved"), audited against
the paper (§5.6, Fig 3 / Table S1) and against the *working* Table-2 / Fig-7
SwinIR codepath (which is treated as the strongest local source of truth).

All file paths are relative to `replication/`.

---

## 1. What produces the current Fig-3 SwinIR columns

| Artifact | Path |
|---|---|
| Trainer | `scripts/train_fig03_swinir_columns.py` |
| SwinIR refinement wrapper | `src/baselines/swinir/refinement_model.py` (`MicroscopeSwinIRRefinement`, `OfflineSwinIRRefinement`) |
| SwinIR builder (shared) | `src/baselines/swinir/model_wrapper.py` (`build_swinir_from_config`) |
| Frozen base microscopes | `experiments/figure03_content_aware/base/bbbc022_{comp}_{pattern}_seed42/checkpoints/best.pt` |
| Output | `experiments/figure03_content_aware/base/swinir/{comp}_{pattern}/` and `swinir/swinir_results.csv` |

The `+SwinIR` columns in the rendered Fig-3 come from these refiner checkpoints.

---

## 2. Answers to the audit checklist

**Where are the CNN/locality base models trained?**
Not in the SwinIR script — they are the pre-existing `bbbc022_content_aware_v2`
runs (locality-aware upsampling + 6-layer reconstruction CNN, trained to L1 with the
staged-hardening schedule for the learnable case). The SwinIR script only loads
`best.pt` for each `{comp,pattern}` cell.

**Where is SwinIR created?**
`MicroscopeSwinIRRefinement(microscope, SWINIR_CFG, {"mode":"direct"})` →
`OfflineSwinIRRefinement` → `build_swinir_from_config(SWINIR_CFG)`. It **does**
reuse the same builder as Table-2/Fig-7. ✅ (builder is shared)

**What does SwinIR receive as input?**
`x_base = frozen_microscope(x)["x_recon"]` — the **256×256 CNN/locality
reconstruction** (not the raw measurements). ✅ This is the paper-faithful wiring
("append SwinIR at the end of the trained end-to-end model").

**What is SwinIR trained to predict?**
The ground-truth specimen `x` (256×256). Loss `F.l1_loss(out["x_recon"], x)`.
Mapping is `x_base -> x_gt`. ✅ correct target, ❌ loss is L1-only (see below).

**Is the base (forward model, Ht, upsampling, CNN) frozen?**
Yes. `ref.set_freeze_base(True)` sets `requires_grad=False` on all microscope
params, and the script keeps `ref.microscope.eval()` throughout (a real bug was
previously fixed here: the recon CNN has BatchNorm whose running stats were being
polluted in train mode). Only `ref.offline_refiner` (the SwinIR) trains. ✅

**Trained separately for pseudo-random vs learnable?** Yes — `PATTERNS =
["random_fixed","learnable_frequency"]`, one refiner per pattern. ✅

**Trained separately per compression?** Yes — one refiner per
`{x16,x64,x256,x1024}`. ✅

**upscale = 1?** Yes (`SWINIR_CFG["upscale"] = 1`, `upsampler = ""`). ✅ SwinIR
is used as an image-to-image restoration net, not a spatial upscaler.

**Input/target normalization paired & consistent?** Yes. GT is produced once by
the `minimal_percentile` (per-image q0.001/q0.999 → [0,1]) preprocessing; `x_base`
is the frozen model's reconstruction of that same GT, so both live in the same
[0,1]-ish range. The base was trained with L1 against that exact GT. ✅

**Any independent per-image/per-patch min-max after reconstruction?** No. Neither
`x_base` nor GT is re-normalized before the loss; eval only `clamp(0,1)`. ✅ (no
prettifying normalization)

**SwinIR architecture size used?**
```
upscale=1, in_chans=1, img_size=256, window_size=8, upsampler="",
embed_dim=96, depths=[2,2,2,2,2,2], num_heads=[3,3,3,3,3,3],
mlp_ratio=2, resi_connection="1conv", img_range=1.0
```
❌ **This is a ~1/4-capacity SwinIR, not SwinIR-M.** `depths=[2]*6` and
`embed_dim=96` (vs SwinIR-M `depths=[6]*6`, `embed_dim=180`). This is the single
biggest deviation from the working Table-2/Fig-7 setup.

**How many iterations/epochs?**
`--steps 4000` per cell (default). ❌ **~125× shorter** than the SwinIR-scale
schedule used for Table-2/Fig-7 (500k in the config; SwinIR README batch32/500k).

**Loss = L1/MSE only, or pixel+perceptual+adversarial?**
❌ **L1 only.** The paper (§5.6) explicitly says the SwinIR experiments — *including
the U2OS/Fig-3 ones* ("we conduct a similar set of experiments") — use **pixel +
adversarial + perceptual** loss. The Table-2/Fig-7 codepath already implements
this stack (`src/baselines/swinir/losses.py`: `pixel_loss` + `VGGPerceptualLoss` +
`VGGStyleDiscriminator`/`GANLoss`). The Fig-3 script does not use it at all.

**Batch / effective batch size?**
`--batch 8`, no gradient accumulation → **effective batch 8**. ❌ The paper/
SwinIR use batch 32; Table-2 reaches effective 32 via grad-accum.

**Validation model-selection logic?**
Max validation SSIM with an MSE-no-regression gate (`score = ref_ssim if
ref_mse<=base_mse else ref_ssim-1`), best state = highest score. This is a
reasonable, defensible selection and matches the paper's "both metrics improve"
framing. ✅ (kept in the fix)

**Optimizer/schedule:** Adam(lr 2e-4, betas 0.9/0.99), linear warmup(200) + cosine
decay to 2%. Matches SwinIR-family LR; the warmup/cosine is a minor extra vs
Table-2 (constant LR) but not harmful. ⚠️ minor.

---

## 3. Root-cause summary (ranked by expected impact)

The wiring is **correct** (frozen base → SwinIR upscale=1 → refine 256×256; correct
target; paired normalization; base frozen with BN handled). The weak result is a
**capacity + training-recipe** problem, not a plumbing problem:

1. **Under-capacity SwinIR** — `embed_dim 96, depths [2]*6, heads [3]*6` instead of
   SwinIR-M `180 / [6]*6 / [6]*6`. ~11.5M → ~3M params. A shallow SwinIR cannot do
   the aggressive restoration seen in the paper.
2. **Loss too weak** — L1 only. Paper §5.6 uses **pixel + perceptual + adversarial**.
   Perceptual+GAN are exactly what create the visually strong, texture-restoring
   behaviour ("random+SwinIR looks better than learned-CNN").
3. **Training far too short** — 4000 steps vs a SwinIR-scale schedule (≥50k). At
   4k steps a from-scratch SwinIR barely departs from a smoothing operator.
4. **Effective batch 8** vs 32.

None of these is "SwinIR can't work" — Table-2/Fig-7 prove SwinIR works here. They
are all deviations of the Fig-3 script *away from* the proven Table-2/Fig-7 recipe.

## 4. What is already correct and must be preserved in the fix

- Frozen base, BN kept in eval mode (do not regress this).
- SwinIR fed the **CNN/locality reconstruction** (not measurements).
- `upscale=1`, `in_chans=1`, `window_size=8`.
- Paired [0,1] normalization; no post-hoc per-image renormalization.
- Separate refiner per (compression, illumination).
- SSIM-primary checkpoint selection with MSE-no-regression gate.
- Shared `build_swinir_from_config` builder.

The fix (see `TABLE2_FIG7_VS_FIG3_SWINIR_DIFF.md` and the new
`fig3_refine_stage.py`) keeps all of the above and only upgrades capacity, loss,
batch and training length to the proven Table-2/Fig-7 recipe.
