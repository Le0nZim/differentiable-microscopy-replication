# Table 1 comparison — Robustness to Poisson and read noise (PatchMNIST, clean split)

Status: **clean-split rerun**

| Metric | Paper | Ours | Abs diff | Rel diff | Status | Note |
|---|---|---|---|---|---|---|
| MSE learnable pc=10 sigma=0.0 | 0.0025 | 0.0064 | 0.0039 | 1.5726 | mismatch | clean split |
| MSE fixed_random pc=10 sigma=0.0 | 0.0108 | 0.0208 | 0.0100 | 0.9228 | close | clean split |
| MSE learnable pc=10 sigma=2.7 | 0.0024 | 0.0066 | 0.0042 | 1.7494 | mismatch | clean split |
| MSE fixed_random pc=10 sigma=2.7 | 0.0107 | 0.0211 | 0.0104 | 0.9748 | close | clean split |
| MSE learnable pc=10 sigma=2.0 | 0.0024 | 0.0066 | 0.0042 | 1.7370 | mismatch | clean split |
| MSE fixed_random pc=10 sigma=2.0 | 0.0107 | 0.0207 | 0.0100 | 0.9343 | close | clean split |
| MSE learnable pc=10 sigma=6.0 | 0.0024 | 0.0074 | 0.0050 | 2.0902 | mismatch | clean split |
| MSE fixed_random pc=10 sigma=6.0 | 0.0107 | 0.0230 | 0.0123 | 1.1524 | mismatch | clean split |
| MSE learnable pc=10000 sigma=0.0 | 0.0059 | 0.0030 | -0.0029 | -0.4906 | close | clean split |
| MSE fixed_random pc=10000 sigma=0.0 | 0.0214 | 0.0109 | -0.0105 | -0.4914 | close | clean split |
| MSE learnable pc=10000 sigma=2.7 | 0.0058 | 0.0031 | -0.0027 | -0.4660 | close | clean split |
| MSE fixed_random pc=10000 sigma=2.7 | 0.021 | 0.0109 | -0.0101 | -0.4788 | close | clean split |
| MSE learnable pc=10000 sigma=2.0 | 0.0061 | 0.0032 | -0.0029 | -0.4786 | close | clean split |
| MSE fixed_random pc=10000 sigma=2.0 | 0.0213 | 0.0109 | -0.0104 | -0.4882 | close | clean split |
| MSE learnable pc=10000 sigma=6.0 | 0.0069 | 0.0032 | -0.0037 | -0.5424 | close | clean split |
| MSE fixed_random pc=10000 sigma=6.0 | 0.0235 | 0.0110 | -0.0125 | -0.5331 | close | clean split |
| Claim N6: learnable beats fixed at every cell | yes | yes (all 8 cells) |  |  | aligned | Clean-split rerun. |
