# AM-3 Phase 1–3 — Variant / implementation fidelity audit

Scope: prove that the Table 3 / Fig. 10 A–D ablation (U2OS, ×16 compression,
`T = 4`, 8×8 downscaling) is **wired and implemented exactly as the paper
specifies**, so that any result-level mismatch is attributable to data/regime
and not to a coding bug.

All claims below are backed by machine-checkable tests:

- `tests/test_am3_variant_audit.py` — Phase 1 (wiring) + Phase 2 (locality block)
- `tests/test_am3_frequency_schedule.py` — Phase 3 (frequency-domain + schedule)
- audit helper: `src/evaluation/variant_audit.py`

Run: `pytest tests/test_am3_variant_audit.py tests/test_am3_frequency_schedule.py`
→ **23 passed**.

Per-run wiring is also re-verified at training time: `scripts/table03_ablation/run.py`
calls `audit_microscope()` + `check_variant()` on the *built* model and aborts
the run if any variant is mis-wired (`metrics/variant_metadata.json`).

---

## Phase 1 — Table 3 variant matrix

Source config: `configs/table03_ablation/bbbc022_x16_matrix.yaml`
(base `configs/_shared/base_bbbc022_substitute.yaml`).

Introspected from the **built** `DifferentiableMicroscope` (not just the YAML):

| Variant | Illumination mode | H_t learnable | Freq-domain opt (W trained) | Upsampling module (actual) | Trainable illum params | Upsampler params | Recon params |
|---|---|:--:|:--:|---|--:|--:|--:|
| A | `random_fixed` (fixed pseudo-random) | no | no | `transpose_conv` | 0 | 1,028 | 72,193 |
| B | `learnable_frequency` | yes | yes | `transpose_conv` | 262,144 | 1,028 | 72,193 |
| C | `learnable_frequency` | yes | yes | `locality_aware` | 262,144 | **262,144** | 72,193 |
| D | `learnable_spatial` | yes | **no** | `locality_aware` | 262,144 | 262,144 | 72,193 |

Geometry (all variants identical): image 256×256, `downscale_factor = 8`
→ detector grid 32×32, `T = 4` patterns, compression `8² / 4 = ×16`.

Checked invariants (each is an assertion in the test suite):

- **downscale = 8, T = 4, compression = ×16** for all four variants.
- **A illumination mode** = fixed pseudo-random; A is the *only* variant with
  **0 trainable illumination parameters** (no accidental learning in A).
- **B/C/D** all expose **262,144 trainable illumination parameters** (no
  accidental freezing of H_t in B/C/D).
- **Upsampling, no leakage**: A,B use `transpose_conv`; C,D use `locality_aware`
  (no accidental locality in A/B, no accidental transpose in C/D).
- **Frequency-domain optimization** is operationally active only where a Fourier
  weight `W` is trained: B and C (`learnable_frequency`). **D uses
  `learnable_spatial`** → optimizes `τ` directly in the spatial domain, i.e. the
  frequency-domain path is removed while H_t stays learnable end-to-end (exactly
  the "(−) frequency domain optimization" ablation). A is fixed so there is no
  `W` to train.
  - Note on Fig. 10's caption: it lists A under "frequency domain optimization",
    but A's H_t is *fixed*, so there is no `W` being optimized. We record the
    operational fact (no W trained for A); this does not change A's role as the
    fixed baseline.
- **Same reconstruction backbone** across all variants: identical 72,193-param
  `ReconCNN` (6 `conv_relu_bn` blocks, final Conv+Sigmoid; supplement A.1). The
  only differences between variants are the three things Table 3 ablates.
- **Pattern parameterization per variant**: A has a fixed buffer (no `W`/`τ`);
  B,C carry a complex Fourier `W` and no spatial `τ`; D carries a spatial `τ`
  and no `W`.

Conclusion: **the A/B/C/D wiring is correct and matches Table 3 / Fig. 10.**
This reproduces (and hardens, with executable tests) the prior hand audit in
`bbbc022_ablation_x16/AM3_diagnosis.md`.

---

## Phase 2 — Locality-aware upsampling (Fig. 2 / §4.3)

Implementation: `src/models/locality_upsampling.py::LocalityAwareUpsampling`.

Paper spec: each detector location `(i,j)` has its own learnable weight matrix
`W_{i,j}` sized by the upsampling factor; the scalar `y_down(i,j)` is projected
through `W_{i,j}`, reshaped into an `n×n` patch, and all patches are tiled.

Verified (tests in `test_am3_variant_audit.py`):

- **One matrix per detector location, not a shared kernel.** Weight tensor shape
  is `[T, H_down, W_down, n, n] = [4, 32, 32, 8, 8]`. For 256×256 / downscale 8
  the detector grid is 32×32, so the parameters are location-specific over the
  32×32 measurement pixels. **Parameter count = 4·32·32·8·8 = 262,144** (logged
  per run and asserted in tests).
- **Output patch is 8×8, tiled into 256×256.** A single nonzero detector pixel
  at channel 2, location (5,7) changes **only** the output block
  `[:, 2, 40:48, 56:64]` and nothing else (`test_single_detector_pixel_activates_only_its_patch`).
- **Different locations use different weights** (`test_different_locations_use_different_weights`).
- **Correct tensor layout** including a deliberate non-square `H_down ≠ W_down`
  case to catch any height/width transpose bug (`test_layout_is_consistent_for_non_square_grid`).
- **Gradients flow** into all locality parameters with dense positive input; with
  a single activated detector pixel, gradient is nonzero **only** for that
  location/channel and zero elsewhere (`test_gradient_nonzero_only_for_activated_location`).
- **T = 4 pattern channels** handled independently and consistently.
- **Initialization** is `randn · 0.01`, identical across variants C and D.

Capacity note (key to AM-3): the locality upsampler has **262,144** parameters
vs the transpose upsampler's **1,028** (a 255× increase). This is the paper's
intended extra capacity, and it is exactly what overfits under the small/short
proxy regime (see Phase 5 and `AM3_root_cause.md`).

---

## Phase 3 — Frequency-domain optimization + custom-sigmoid schedule (§4.2)

Implementation: `src/models/pattern_generator.py`. Verified in
`tests/test_am3_frequency_schedule.py`:

- **eq. 6** — `τ = real(ifft2(W))` for `learnable_frequency`.
- **eq. 7 (sigmoid-friendly init)** — `W = fft2(τ₀)` with `τ₀ ~ N(0,1)`; the
  round-trip `real(ifft2(W))` is verified to be standard-normal (mean ≈ 0, std
  ≈ 1), **not** arbitrary tiny constants, so the custom sigmoid sees a usable
  input spread at init.
- **eq. 4/5 (custom sigmoid)** — `H_t = sigmoid(m·τ)`; higher `m` ⇒ more values
  pushed toward {0,1} (binary-fraction increases with `m`).
- **W receives gradient** (frequency-domain optimization is trainable).
- **D removes frequency-domain optimization** — `learnable_spatial` has no `W`
  and learns `τ` directly; `H_t = sigmoid(m·τ)` is still trained end-to-end.
- **Algorithm 1 schedule** (`SigmoidSchedule`) — inverse-only warmup (illumination
  frozen) while `epoch ≤ baseline`; after `cutoff`, `m` increments on the step
  schedule.

### Scaled Algorithm 1 used in the controlled rerun

The paper's U2OS schedule (`baseline = 12150`, `cutoff = 18630`, `step = 810`
epochs; ~24,300 epochs total) is far beyond this compute budget. The AM-3 runner
(`scripts/table03_ablation/run.py`) uses a **phase-preserving scaled** schedule, applied
**identically to all variants**:

1. `inverse_warmup` — train inverse only (illumination frozen), `m = 1` — 1,500 steps
2. `joint_soft`    — train illumination + inverse end-to-end, `m = 1` — 4,000 steps
3. `harden_m2 → m4 → m8` — increase `m` on schedule — 1,000 steps each

This keeps the three Algorithm-1 phases (inverse-only → joint soft → progressive
hardening) rather than silently ignoring the schedule; it is documented and
scale-controllable via `--scale`.

### Per-run diagnostics logged (every variant)

`metrics/step_log.csv` + `metrics/diagnostics.json` record, over training:
`m` value, pattern min/max/mean/std and binary fraction, illumination L2 norm and
its change from initialization, and **group gradient norms for illumination /
upsampler / reconstruction** separately, plus a trainable-parameter report
(`metrics/variant_metadata.json`). These satisfy the Phase-3 logging requirements
and are used as evidence in `AM3_root_cause.md`.

---

## Bottom line

Phases 1–3 establish that the A/B/C/D variants, the locality-aware upsampling
block, the frequency-domain optimization, and the (scaled) Algorithm-1 schedule
are **faithful to the paper and bug-free** under machine-checkable tests. The
remaining question — why C does not win on the substitute data — is therefore a
**data/training-regime** question, addressed in Phases 5–7.
