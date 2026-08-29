# Fig4 fix vs. frozen am2 (reproducibility check)

| compression | illumination | fix Dice | am2 Dice | Δ |
|---|---|---:|---:|---:|
| x64 | fixed | 0.9007 | 0.8938 | +0.0069 |
| x64 | learnable | 0.9272 | 0.9322 | -0.0049 |
| x256 | fixed | 0.7196 | 0.6958 | +0.0237 |
| x256 | learnable | 0.8358 | 0.8757 | -0.0400 |
| x1024 | fixed | 0.5000 | 0.4993 | +0.0007 |
| x1024 | learnable | 0.5245 | 0.5986 | -0.0742 |
