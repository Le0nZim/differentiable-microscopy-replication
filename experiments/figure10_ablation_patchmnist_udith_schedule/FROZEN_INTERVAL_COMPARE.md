# Frozen-interval C vs D check (independent warmup)

**Label:** Udith legacy schedule under the current controlled PatchMNIST optimizer/data recipe. Not a full reproduction of Udith's legacy training loop.

## Verdict

Independent C and D **construction hashes match**, and initial physical patterns agree within FFT round-trip tolerance, but **logged warmup losses/validation metrics are not numerically equivalent**. Maximum |C-D| on seed 42 overlap is loss 1.813889e-03 (step 28800) and val MSE 2.155822e-03 (step 200). That is larger than the 8,500-step C/D test-MSE gap (~4e-4), so the frozen trajectories **diverged materially**.

Seeds **43/44 were stopped**. Scientific continuation: one shared warmup checkpoint (spatial/D arm) including inverse Adam, then fork C and D under `runs_shared_warmup/`. Independent D42 is left running as a diagnostic only.

## Construction hashes (audit_seed, seed 42)

- inverse C=D: `29a36bd9df404eda22a1962974acd55f0e73b96030b6cf991111961695de54c2` equal=True
- upsampler C=D: `84aca57e2c82bb2d91c73a0b26bd42bf2691b9c17e14666dc481262ae3a70252` equal=True
- first-batch C=D: `630e9e43edb2d1b3041d69136232bd58137f84725a5bf5f83997c8319e757f59` equal=True
- max |dtau| = 1.907349e-06; max |dH_t| = 2.980232e-07

CPU reconstruction also matches C vs D first-batch and RNG after model build. `audit_seed` hashes the first `next(iter(loader))`, not every `cycle` batch during `run_one`. The GPU warmup *loss* series is the trajectory check.

## Initial physical patterns (saved H_t0 / tau0 tensors)

- seed 42 `H_t0.pt`: max|C-D|=2.980232e-07, mean|C-D|=3.732100e-08
- seed 42 `tau_0.pt`: max|C-D|=1.907349e-06, mean|C-D|=1.963149e-07
- seed 43 `H_t0.pt`: max|C-D|=2.682209e-07, mean|C-D|=3.751783e-08
- seed 43 `tau_0.pt`: max|C-D|=2.145767e-06, mean|C-D|=1.972236e-07

## Frozen-interval logged metrics (not merely illum_delta == 0)

illum_delta, tau displacement, and H_t displacement are **exactly 0** on both arms throughout warmup. The split is in **loss / val MSE / SSIM / inverse grads**.

| seed | overlap steps | last overlap | max|dloss| | max|dval MSE| | max|dSSIM| | material? |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 210 | 42000 | 1.813889e-03 | 2.155822e-03 | 2.988535e-02 | True |
| 43 | 146 | 29200 | 5.612966e-03 | 5.305713e-03 | 3.731582e-02 | True |

Seed 42 first loss split: step 200, C=0.10314010, D=0.10453261, |d|=1.393e-03.

## Checkpointing / eval (unchanged between C and D)

- Phase-boundary evaluations **are eligible for global-best** (same `best.update` path).
- Step 121,500 is evaluated as the last step of the last phase (m=8, 5 steps). It is not on the log_every=200 grid (`121500 % 200 == 100`).
- `itertools.cycle(train_loader)` is still present.

## Independent-warmup C (diagnostic; not the scientific C vs D)

- C seed 42: test MSE=0.005388, SSIM=0.9524, best m=7.0 at step 121000, binary frac=0.626, ||tau-tau0||=425.90
- C seed 43: test MSE=0.005296, SSIM=0.9526, best m=7.0 at step 121495, binary frac=0.604, ||tau-tau0||=415.87

Paired D-C test metrics: **not reported** from this independent-warmup protocol.
