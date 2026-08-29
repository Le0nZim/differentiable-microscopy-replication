# Side-by-side: working Table-2/Fig-7 SwinIR vs current Fig-3 SwinIR

The advisor's guidance: Table-2/Fig-7 SwinIR already reproduces paper behaviour, so
SwinIR itself is **validated**. The Fig-3 bug is therefore a *divergence* of the
Fig-3 codepath from the Table-2/Fig-7 recipe. This document is that diff.

Sources of truth:
- Table-2/Fig-7 recipe: `archive/configs/swinir/am4_table2_full.yaml` + `scripts/table02_swinir_sr/run.py` + `src/baselines/swinir/{am4_table2,losses,table2_pipeline,model_wrapper}.py`.
- Current Fig-3 recipe: `scripts/train_fig03_swinir_columns.py` + `src/baselines/swinir/refinement_model.py`.

---

## A. SwinIR architecture

| Parameter | Table-2/Fig-7 (works) | Current Fig-3 (weak) | Match? |
|---|---|---|---|
| builder | `build_swinir_from_config` | `build_swinir_from_config` | ✅ same |
| `upscale` | 1 | 1 | ✅ |
| `in_chans` | 1 | 1 | ✅ |
| `window_size` | 8 | 8 | ✅ |
| `img_range` | 1.0 | 1.0 | ✅ |
| `upsampler` | "" (image-to-image) | "" | ✅ |
| `resi_connection` | "1conv" | "1conv" | ✅ |
| `mlp_ratio` | 2 | 2 | ✅ |
| **`embed_dim`** | **180 (SwinIR-M)** | **96** | ❌ |
| **`depths`** | **[6,6,6,6,6,6]** | **[2,2,2,2,2,2]** | ❌ |
| **`num_heads`** | **[6,6,6,6,6,6]** | **[3,3,3,3,3,3]** | ❌ |
| ≈ SwinIR params | **11.5 M** | **~3 M** | ❌ |

→ **Fig-3 used a ~1/4-capacity SwinIR.** This alone caps restoration quality.

## B. What SwinIR consumes / predicts (wiring)

| Aspect | Table-2/Fig-7 | Current Fig-3 | Notes |
|---|---|---|---|
| SwinIR input | fused **upsampled measurements** (`fuse(upsample(y_down))`) | frozen base **CNN reconstruction** `x_base` | **Intentionally different** — paper says append SwinIR *after the trained end-to-end model* for Fig-3. ✅ both are paper-faithful for their section |
| SwinIR role | *replaces* ψ (recon net), trained jointly (+Ht) | *refines* frozen ψ output | paper-faithful per §5.6 |
| target | GT patch | GT (256×256) | ✅ |
| upscale | 1 | 1 | ✅ not a spatial SR |
| base frozen? | n/a (trained jointly) | yes, `requires_grad=False` + `eval()` (BN safe) | ✅ correct for Fig-3 |
| grayscale | 1-chan; VGG repeats to 3-chan + ImageNet norm | 1-chan (no perceptual used) | see losses |

→ The wiring difference (measurements-in vs reconstruction-in) is **correct and
intended** — it's the paper's own distinction between §5.6 Table-2 and §5.6 U2OS.
The Fig-3 refiner is fed the right tensor.

## C. Losses

| Loss term | Table-2/Fig-7 | Current Fig-3 | Match? |
|---|---|---|---|
| pixel | L1, weight 1.0 | L1, weight 1.0 | ✅ |
| **perceptual (VGG19)** | **weight 1.0** (`VGGPerceptualLoss`) | **absent** | ❌ |
| **adversarial (GAN)** | **weight 0.1** (`VGGStyleDiscriminator`+`GANLoss`) | **absent** | ❌ |
| paper says | pixel+adversarial+perceptual (§5.6) | L1 only | ❌ |

→ **Fig-3 dropped the perceptual + adversarial losses** that the paper explicitly
lists for these experiments and that Table-2/Fig-7 implements. These are precisely
the terms that produce texture/detail restoration.

## D. Optimisation / schedule / precision

| Setting | Table-2/Fig-7 | Current Fig-3 | Match? |
|---|---|---|---|
| optimizer | Adam(betas 0.9,0.99) | Adam(betas 0.9,0.99) | ✅ |
| swinir_lr | 2e-4 | 2e-4 | ✅ |
| Ht lr | 0.1 (learnable, joint) | n/a (base frozen) | ✅ (Fig-3 doesn't train Ht here) |
| LR schedule | constant | warmup(200)+cosine→2% | ⚠️ minor (harmless) |
| **effective batch** | **32** (micro 8 × accum 4) | **8** (no accum) | ❌ |
| amp | bf16 autocast | fp32 | ⚠️ speed only |
| **iterations** | **≤5e5 (SwinIR-scale)** | **4000** | ❌ ~125× short |
| val metric / selection | best val L1 | best val SSIM + MSE-gate | ⚠️ different but fine |

## E. Data / normalization / eval scaling

| Aspect | Table-2/Fig-7 | Current Fig-3 | Match? |
|---|---|---|---|
| patch training | 64×64 patches | 256×256 full | different sizes but both valid; SwinIR is size-agnostic (window masks recomputed per input) |
| normalization | grayscale /255 → [0,1] | per-image percentile → [0,1] | dataset-appropriate; paired input/target ✅ |
| independent per-method renorm | none | none | ✅ |
| clamp for metrics | clamp(0,1) | clamp(0,1) | ✅ |

---

## Conclusion — the exact deltas the fix must close

Keep the Fig-3 wiring (frozen base → SwinIR upscale=1 → refined 256×256, fed the
CNN reconstruction, paired [0,1] normalization, per-cell refiners, BN-safe freeze,
SSIM-primary selection). **Change only the four items that diverge from the proven
Table-2/Fig-7 recipe:**

1. **Capacity:** `embed_dim 96→180`, `depths [2]*6→[6]*6`, `num_heads [3]*6→[6]*6` (SwinIR-M, the exact Table-2/Fig-7 arch).
2. **Loss:** add the Table-2/Fig-7 loss stack → `pixel + perceptual + adversarial` (paper §5.6), with `l1_only`/`l1_ssim` also selectable for metric-driven runs.
3. **Effective batch:** 8 → 32 via gradient accumulation.
4. **Training length:** 4000 → ≥50k (config capable of 200k+).

Fig-3-specific deviation that is **retained and justified** (documented in REPORT):
train the refiner on **64×64 crops** of the frozen 256×256 reconstruction (exactly
how the paper trains Table-2 SwinIR — "training is performed for 64×64 image
patches"), then apply/evaluate the trained SwinIR at full 256×256. This makes the
SwinIR-M schedule computationally feasible (256×256 SwinIR-M is ~2.5 img/s;
64×64 is ~58 img/s) without changing the network or the frozen base.
