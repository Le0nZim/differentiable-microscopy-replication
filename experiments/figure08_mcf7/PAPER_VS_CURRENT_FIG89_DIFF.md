# Side-by-side diff — paper §5.6 vs working Table-2/Fig-7 SwinIR vs current MCF7 Fig 8/9 vs this fix

The **Table-2/Fig-7 SwinIR pipeline is the validated reference** (`run_am4_swinir_table2.py`
+ `archive/configs/swinir/am4_table2_full.yaml`): it reproduces the paper's learnable-illumination
SwinIR SR gains. The current MCF7 Fig 8/9 runs use the *same model class*
(`SwinIRTable2Model`) but a *degraded training recipe*. This fix ports the reference recipe
onto MCF7.

| Axis | Paper §5.6 | Working Table-2/Fig-7 (reference) | Current MCF7 Fig 8/9 (deviating) | **This fix (Fig 8/9)** |
|---|---|---|---|---|
| Role of SwinIR | replaces ψ, `upscale=1`, image-to-image | replaces ψ, `upscale=1` ✓ | replaces ψ, `upscale=1` ✓ | replaces ψ, `upscale=1` ✓ (unchanged) |
| Upsampling | locality-aware block | locality-aware ✓ | locality-aware ✓ | locality-aware ✓ (unchanged) |
| **Loss** | **pixel + adversarial + perceptual** | pixel(L1) 1.0 + perceptual 1.0 + GAN 0.1 ✓ | **L1 only** ✗ | **pixel 1.0 + perceptual 1.0 + GAN 0.1** ✓ |
| **SwinIR capacity** | "similar to SwinIR" = SwinIR-M | **embed_dim 180**, depths/heads [6×6] ✓ | **embed_dim 96**, depths [2×6] ✗ | **embed_dim 180**, [6×6] + grad-checkpoint ✓ |
| Compression | ×16 | ×16 (d=8, T=4) ✓ | ×16 (d=8, T=4) ✓ | ×16 (d=8, T=4) ✓ |
| Illumination | learnable, lr 0.1 | learnable, lr 0.1 ✓ | learnable, lr 0.1 ✓ | learnable, lr 0.1 ✓ |
| Image size | 256×256 (Fig 8); 64×64 CNN (Fig 9 wCNN) | 64×64 patches (classical SR) | 256 (Q/R), 64 (wCNN) ✓ | 256 (Q/R), 64 (wCNN) ✓ |
| Schedule | Algorithm 1, 230/150/150/20 | fixed m=8 (SR task) | Algorithm 1, compute-scaled | Algorithm 1, 230/150/150/20 ratio-preserved |
| swinir_lr / betas | "similar to SwinIR" | 2e-4 / (0.9,0.99) | 2e-4 | 2e-4 / (0.9,0.99) (as reference) |
| Fig 8 R baseline | transpose-conv + ReconCNN, L1 | n/a | transpose-conv + ReconCNN, L1 ✓ | transpose-conv + ReconCNN, L1 ✓ |
| Fig 9 wCNN baseline | 64×64 §4.3 ReconCNN, L1 | n/a | locality + ReconCNN @64, L1 ✓ | locality + ReconCNN @64, L1 ✓ |

## The exact changes this fix makes vs the current MCF7 runs

1. **L1 → full loss stack** (pixel + perceptual + adversarial). *Dominant* change; restores
   the high-frequency texture that makes SwinIR look "high-resolution." Reuses
   `baselines.swinir.losses.build_loss_stack` + the GAN discriminator/perceptual path from
   `run_am4_swinir_table2.py` — the identical, already-validated code from the Figure-3 fix.
2. **embed_dim 96 → 180 (SwinIR-M)** with gradient checkpointing (`use_checkpoint=True`) so
   the full-capacity model fits at 256×256 (this dissolves the old "180 infeasible at 256"
   constraint), + grad accumulation for an effective batch.
3. **Longer, schedule-faithful training** (Algorithm-1 230/150/150/20, ratio-preserved if
   compute-scaled) so illumination hardening (m → 8) fully engages.

## What is deliberately kept identical (already faithful)

- `SwinIRTable2Model` wiring (forward → locality upsample → fuse → SwinIR).
- ×16 factorisation (d=8, T=4), learnable-frequency illumination, `illumination_lr=0.1`.
- Fig 8 = SwinIR (Q) vs transpose-conv (R); Fig 9 = wSwinIR vs 64×64 wCNN + learned Ht.
- Real BBBC021 channel-2 MCF7 data, well-disjoint 3000/100/100 split.
- Conventional baselines (R, wCNN) trained with L1 (GAN/perceptual is SwinIR-specific in the
  paper, so using it for the CNN baselines would be *unfaithful* and would flatter SwinIR).

## PAPER_UNSPECIFIED (inherited from the reference; logged, not invented here)

- Discriminator architecture ("similar to SwinIR"): compact spectral-norm VGG-style.
- VGG19 perceptual layer set/weights: conv1_2…conv5_4, ESRGAN weights.
- Loss weights pixel 1.0 / perceptual 1.0 / gan 0.1 (ESRGAN/Real-ESRGAN default).
- swinir_lr 2e-4, betas (0.9, 0.99), no LR warmup/decay.
- d=8, T=4 factorisation of the stated ×16.
