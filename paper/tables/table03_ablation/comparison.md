# Comparison — Table 3

Overall status: **data_blocked**

| Metric | Paper value | Our value | Abs diff | Rel diff | Status | Note |
|---|---|---|---|---|---|---|
| MSE variant A (U2OS) | 0.0042 |  |  |  | data_blocked | U2OS unavailable. BBBC022 proxy MSE=0.003625 (not comparable). |
| SSIM variant A (U2OS) | 0.7872 |  |  |  | data_blocked | U2OS unavailable. BBBC022 proxy SSIM=0.8781 (not comparable). |
| MSE variant B (U2OS) | 0.0038 |  |  |  | data_blocked | U2OS unavailable. BBBC022 proxy MSE=0.002706 (not comparable). |
| SSIM variant B (U2OS) | 0.795 |  |  |  | data_blocked | U2OS unavailable. BBBC022 proxy SSIM=0.888 (not comparable). |
| MSE variant C (U2OS) | 0.0029 |  |  |  | data_blocked | U2OS unavailable. BBBC022 proxy MSE=0.005321 (not comparable). |
| SSIM variant C (U2OS) | 0.8426 |  |  |  | data_blocked | U2OS unavailable. BBBC022 proxy SSIM=0.8602 (not comparable). |
| MSE variant D (U2OS) | 0.0041 |  |  |  | data_blocked | U2OS unavailable. BBBC022 proxy MSE=0.004583 (not comparable). |
| SSIM variant D (U2OS) | 0.7857 |  |  |  | data_blocked | U2OS unavailable. BBBC022 proxy SSIM=0.8663 (not comparable). |
| Best variant (ordering) | C (proposed) | B on BBBC022 proxy (C worst) |  |  | data_blocked | Ordering not reproduced on substitute; diagnosed as locality overfitting the low-diversity widefield proxy. |

Status rule (when not explicitly set): rel-diff ≤5% → `aligned`, ≤100% → `close`, else `mismatch`; non-numeric/blocked rows carry their explicit catalog status.

The same comparison (paper-ready) is rendered under `rendered/`.
