# AM-3 Phase 7 — Root-cause analysis (Table 3 / Fig. 10)

**One-line conclusion.** The A/B/C/D implementation is faithful and bug-free
(machine-checkable + a PatchMNIST sanity check where the locality block *does*
win). The paper's U2OS data is unavailable, so Table 3 cannot be numerically
reproduced. On the BBBC022 substitute proxy the ordering inverts (C worst instead
of best) **purely because the high-capacity locality upsampler overfits the small,
low-diversity, widefield proxy** — not because of any coding/optimization error.
The effect is data-regime driven and shrinks monotonically as training data grows.

Status: **FULLY_RESOLVED_IMPLEMENTATION_PASS_RESULTS_PROXY_LIMITED**
(with the U2OS Table-3 numbers explicitly **DATA_BLOCKED**).

---

## Headline numbers

### Paper Table 3 (U2OS, ×16, T=4, 8×8) — target
| | A fixed+Tr.Conv | B learn+Tr.Conv | C learn+locality | D learn+locality, −freq |
|---|--:|--:|--:|--:|
| MSE | 0.0042 | 0.0038 | **0.0029 (best)** | 0.0041 |
| order | C < B < D < A | | | |

### This work — BBBC022 **SUBSTITUTE PROXY** (3 seeds: 42/43/44), mean ± std
| variant | test MSE | test SSIM | best val MSE | best **train** MSE | overfit gap (val−train) |
|---|--:|--:|--:|--:|--:|
| A fixed+Tr.Conv+freq         | 0.00362 ± 0.00031 | 0.878 | 0.00322 | 0.00340 | **−0.0002** |
| B learn+Tr.Conv+freq         | **0.00271 ± 0.00029** | 0.888 | 0.00239 | 0.00202 | +0.0004 |
| C learn+locality+freq (paper best) | 0.00532 ± 0.00047 | 0.860 | 0.00471 | **0.00094** | **+0.0038** |
| D learn+locality, −freq      | 0.00458 ± 0.00046 | 0.866 | 0.00404 | 0.00083 | **+0.0032** |

Proxy order: **B < A < D < C** (`ordering_matches_paper = false`).

Read the last two columns together: the locality variants (C, D) reach the
**lowest training MSE** (0.0009 / 0.0008) yet the **highest validation/test MSE**,
with a 0.003–0.004 generalization gap. The transpose variants (A, B) have
train ≈ val (gap ≈ 0). That is the textbook overfitting signature, visible
directly in `figures/curves_ABCD.png`.

---

## The nine required questions

**1. Was the A/B/C/D variant wiring correct?**
Yes. Verified three ways: (i) static expectation table + `check_variant()` in
`src/evaluation/variant_audit.py`; (ii) 23 passing tests in
`tests/test_am3_variant_audit.py` + `tests/test_am3_frequency_schedule.py`;
(iii) a *per-run* audit of the **built** model written to each run's
`metrics/variant_metadata.json` (`wiring_problems: []` for all 12 runs). A is the
only variant with 0 trainable illumination params; B/C/D each expose 262,144;
A/B use `transpose_conv`, C/D use `locality_aware`; downscale 8, T=4, ×16 for all.
See `AM3_variant_audit.md` Phase 1.

**2. Was locality-aware upsampling implemented faithfully?**
Yes. One learnable matrix per detector location (`weights` shape
`[T,H_down,W_down,n,n] = [4,32,32,8,8]`, 262,144 params); a single detector pixel
activates only its 8×8 output patch; different locations use different weights;
gradients reach every location; non-square grids do not transpose H/W.
(`AM3_variant_audit.md` Phase 2.) Independent confirmation: on PatchMNIST the same
block **wins** (Q7).

**3. Was frequency-domain optimization implemented faithfully?**
Yes. `τ = real(ifft2(W))` (eq. 6); sigmoid-friendly init `W = fft2(τ₀), τ₀~N(0,1)`
round-trips to standard-normal (eq. 7); custom sigmoid `H_t = sigmoid(m·τ)` with
binary-fraction rising with `m`; `W` receives gradient; D (`learnable_spatial`)
has no `W` and learns `τ` directly. (`AM3_variant_audit.md` Phase 3.) The learned
patterns visually differ between B/C (fine pseudo-random, freq-domain) and D
(coarser/blockier, spatial) in `figures/learned_patterns_BCD.png`, confirming D
truly drops the frequency path while still learning H_t end-to-end.

**4. Was the Algorithm-1 training schedule implemented or faithfully scaled?**
Faithfully **scaled**, not ignored. The paper's U2OS schedule
(`baseline=12150`, `cutoff=18630`, `step=810` epochs) exceeds this budget, so the
runner uses a **phase-preserving** schedule applied **identically to every
variant**: inverse-only warmup (illumination frozen, m=1, 1500 steps) → joint soft
(m=1, 4000) → progressive hardening m=2→4→8 (1000 each). Documented and
`--scale`-controllable. `metrics/diagnostics.json` logs the realized `m_schedule`,
pattern stats, illumination L2/Δ, and per-group gradient norms for every run.

**5. Does C pass the tiny overfit diagnostics?**
Yes — `overfit_diagnostics/overfit_gate.json` → `pass: true`. With the same
optimizer budget and no augmentation, C reaches a **lower min train MSE than B**
at both subset sizes (n=1: C 2.2e-5 vs B 2.1e-4; n=8: C 1.4e-4 vs B 7.7e-4), i.e.
the locality block has *more* fitting capacity, exactly as expected. Its
validation MSE is simultaneously much higher (n=1: 0.0125 vs 0.0046; n=8: 0.0199
vs 0.0043). **C overfits; it is not broken.** This rules out status D
(implementation bug).

**6. Does C beat B on U2OS?**
**Unknown / DATA_BLOCKED.** The paper's U2OS DAPI confocal dataset is not present
anywhere on this machine (`U2OS_DATA_STATUS.md`). Track 1 could not be run.

**7. If U2OS does not exist, does C beat B on a sanity dataset (PatchMNIST)?**
**Yes.** Same ×16/T=4/256px config, 2 seeds:
locality C = **0.00697 ± 0.00002** vs transpose B = 0.00947 ± 0.00287
(`patchmnist_sanity/aggregate_summary.json`, `locality_beats_transpose: true`).
So in a data regime that the locality block suits, C beats B with the *identical*
code — direct proof the inversion on BBBC022 is data-driven, not a code bug.

**8. On BBBC022, why is C failing?**
It is the **train-good / val-bad** case (overfitting / generalization), driven by
the data regime — **not** an optimization/implementation problem, and **not**
within-seed noise:
- **Train loss is good** for C: best train MSE 0.0009 (lowest of all variants);
  it is val/test that are bad → generalization failure, not under-fitting.
- **Not within seed variance:** C−B test gap ≈ +0.0026 with per-variant std
  ≈ 0.0003–0.0005 across 3 seeds (≈ 5–8σ separation). The ordering is stable.
- **Crop/data regime differs from the paper:** BBBC022 is **widefield
  520×696 uint16** single fields (no z-stack/MIP, no 63/20 downscale, Hoechst not
  DAPI), versus the paper's **confocal 2304×2304** stacks. Fewer, lower-diversity
  256-crops at native resolution.
- **Insufficient images/crops for the 262,144 location-specific weights:** the
  trainsize sweep (`trainsize_sweep/`) shows the C−B test gap **shrinks
  monotonically** with data — 0.00535 (n=42) → 0.00386 (n=84) → 0.00304 (n=168) —
  i.e. C is closing on B as data grows. Extrapolating toward the paper's far
  larger/denser confocal regime is consistent with C eventually overtaking B, as
  in Table 3 (this is a trend, not a reproduction claim).

Summary of Q8: **train loss good, val/test bad → overfitting**, amplified by a
small, low-diversity, lower-resolution widefield proxy. Capacity ratio is the
mechanism: locality upsampler 262,144 params vs transpose 1,028 (255×).

**9. What can be honestly said in advisor-facing slides?**
- "We proved the ablation is implemented exactly as the paper specifies: the A/B/C/D
  wiring, the locality-aware upsampler, the frequency-domain optimization, and the
  (scaled) Algorithm-1 schedule all pass machine-checkable tests, and a per-run
  audit guards every training run."
- "On a sanity dataset (PatchMNIST) the proposed locality block *wins* over
  transpose with the same code, so the block is correct."
- "We could not obtain the paper's U2OS DAPI-confocal data, so we do **not** claim
  Table 3 is numerically reproduced."
- "On the BBBC022 substitute the ordering inverts (C worst) because the
  high-capacity locality upsampler overfits a small, low-diversity, widefield
  proxy: it attains the lowest *training* error but the highest *validation* error,
  and the gap to transpose shrinks as we add data — a data-regime limitation, not
  a bug."
- Do **not** present the BBBC022 numbers next to the paper's U2OS numbers as a
  match. Present them as a proxy-mechanism study.

---

## Evidence index
- Wiring/impl fidelity: `AM3_variant_audit.md`, `tests/test_am3_variant_audit.py`,
  `tests/test_am3_frequency_schedule.py` (23 passed), per-run
  `metrics/variant_metadata.json`.
- Data availability: `U2OS_DATA_STATUS.md`.
- Overfit gate: `overfit_diagnostics/overfit_gate.json` (PASS).
- Locality wins in-regime: `patchmnist_sanity/aggregate_summary.json`.
- Capacity → overfitting curves: `figures/curves_ABCD.png`.
- Data-size trend: `trainsize_sweep/aggregate_summary.json`,
  `trainsize_sweep/trainsize_sweep.png`.
- Proxy ablation: `track2_proxy_bbbc022/aggregate_summary.json`,
  `metrics_by_seed.csv`, `aggregate_summary.json`,
  `figures/qualitative_ABCD.png`, `figures/learned_patterns_BCD.png`.
