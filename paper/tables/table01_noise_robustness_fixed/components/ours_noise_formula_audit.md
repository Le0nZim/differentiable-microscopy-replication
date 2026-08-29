# Noise-formula audit (AM-1 / RR-1 v3)

Audit of the detector noise model against paper supplement A.2.2 (eqs. S5–S10),
and the precise fix applied in v3.

## 1. Paper equations (source of truth)

Main text (§4.4.2, eqs. 9–11):

- `y_down = y_poiss + y_normal` with `y_poiss ~ Poiss(alpha_down)`,
  `y_normal ~ N(0, sigma_read)` (eqs. 9–10).
- Normal approx to Poisson: `y_poiss = alpha_down + sqrt(alpha_down) * z` (eq. 11),
  plus a `gamma = 10` photon background so the approximation holds.

Optical demagnification (eq. 8): `sumpool_{n×n}(X) = AveragePool_{n×n}(X) * n²`.
So the forward output is on the **sum-pool scale `[0, d²]`** (d = downscale factor).

Supplement A.2.2 normalization (eqs. S5–S10):

- `alpha_down,t = k · alpha_down,t^norm`, where `alpha_down,t^norm = ψ(X, H_t^norm)`
  and `H_t^norm ∈ [0, 1]` is the **normalized excitation pattern** (eq. S5).
- `y_poiss_norm = alpha_norm + gamma/k + sqrt(alpha_norm/k + gamma/k²) · z_poiss` (eq. S7).
- `y_read_norm  = sigma_read/k · z_read` (eqs. S8–S9).
- `y_norm = y_poiss_norm + y_read_norm` (eq. S10), and **`y_norm` is fed to the
  inverse model instead of `y_down`**.

`z_poiss, z_read ~ N(0,1)` independent. `gamma = 10`.

## 2. What `alpha_norm` is, given this repo's forward model

The paper says `H_t^norm ∈ [0, 1]` and `alpha_down,t^norm = ψ(X, H_t^norm)`. The
repo's forward model (`src/models/forward_model.py`) computes `ψ` with **binary
patterns `∈ {0,1} = H_t^norm`** and **image `∈ [0,1]`**, using sum-pooling. Hence:

> The repo's forward output `alpha_down` **is already** `alpha_down,t^norm`
> (range `[0, d²]`). It never carried the physical photon factor `k`.

Therefore the value to substitute for `alpha_norm` in eqs. S7/S9 is **`alpha_down`
used directly** — no division by `d²`, no division by `k`.

The paper constrains `H_t^norm ∈ [0,1]`, **not** `alpha_down^norm`. With sum-pooling,
`alpha_down^norm ∈ [0, d²]` (≈ `[0,64]` at d=8). The earlier assumption
"`alpha_norm ∈ [0,1]`" (used by v2) conflated the pattern range with the signal range.

## 3. The three implemented modes (`src/models/detector_noise.py`)

| mode | `alpha_norm` | Poisson std | read term | status |
|---|---|---|---|---|
| `legacy` | `alpha_down` | `sqrt(alpha/k + gamma/k²)` | `sigma_read · z` | pre-AM-1 bug: read **not** /k |
| `paper` (v2) | `alpha_down / d²` | `sqrt(alpha_norm/k + gamma/k²)` | `sigma_read/k · z` | over-normalized signal → residual pc=10 spread |
| `paper_v3` | `alpha_down` | `sqrt(alpha_norm/k + gamma/k²)` | `sigma_read/k · z` | **correct S5–S10** |

The only difference between `paper` and `paper_v3` is the `alpha_divisor`
(`d²` vs `1`). The only difference between `legacy` and `paper_v3` is the read
term (`sigma_read · z` vs `sigma_read/k · z`).

## 4. Audit checklist (task 1)

Evaluated against `noise_normalization: paper_v3` (the v3 path) in
`src/models/detector_noise.py::_apply_noise_paper`:

- Is `alpha_down` divided by `d²` exactly once? **No (correct).** v3 uses
  `alpha_divisor = 1`, i.e. `alpha_norm = alpha_down`. (v2 divided by `d²` — the bug.)
- Is `gamma/k` added to the normalized mean? **Yes** (`poisson_mean = alpha_norm + gamma/k`).
- Is the Poisson std exactly `sqrt(alpha_norm/k + gamma/k²)`? **Yes.**
- Is read noise exactly `sigma_read/k · z_read`? **Yes** (`y_read_norm = (sigma_read/k) * read_noise`).
- Are Poisson and read noise independent? **Yes** — separate `torch.randn_like`
  draws (`poisson_noise`, `read_noise`).
- Any clamp / min-max / per-batch / max-normalization / denorm after noise? **No.**
  The only clamp is `alpha_norm.clamp_min(0.0)` *before* `sqrt` for numerical
  safety; it is a no-op for the non-negative forward output and does **not** clip
  `y_norm`. The inverse model consumes `y_norm` directly; the reconstruction CNN's
  final `Sigmoid` only constrains the *reconstruction* `x_recon ∈ [0,1]` (target
  scale), not the measurement.
- Is the normalized path used in both training and evaluation? **Yes.** Both
  `train_steps` / `train_fixed_m_phase` (training) and `evaluate_reconstruction` /
  `evaluate_all_pattern_variants` (eval) call the same `DifferentiableMicroscope`
  forward with `apply_noise=True`, which routes through `DetectorNoise`.
- Do all fixed and learnable Table-1 runs use the same normalized path? **Yes** —
  both `random_fixed` and `learnable_frequency` configs set
  `detector_noise.noise_normalization: paper_v3` (via the v3 base config).
- Are configs accidentally falling back to legacy? **No** — the v3 config sets
  `noise_normalization: paper_v3` explicitly, and `MicroscopeConfig.from_run_config`
  injects the `downscale_factor` into the detector config (only used by the `paper`
  mode; `paper_v3` ignores it because `alpha_divisor = 1`).

## 5. Verdict

v2 (`paper`) correctly implements the *form* of eqs. S7/S9 but uses the wrong
`alpha_norm` (`alpha_down / d²`), which shrinks the signal to `[0,1]` while the
read term `sigma_read/k` is unchanged. At low photon count (e.g. k=10, σ=6 →
read std `0.6`) this swamps a `[0,1]` signal — the residual pc=10 read-noise
spread reported in v2. v3 (`paper_v3`) uses `alpha_norm = alpha_down` (faithful to
eq. S5 given the repo's already-normalized forward output), restoring the
`[0,d²]` signal scale so the read term is small relative to the signal.
