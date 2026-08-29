# Fig4 fix vs. frozen am2 (cross-check)

> **Not a like-for-like reproducibility check.** This isolated run uses the **TrackMate** pseudo-GT masks (raw MIP>506, 4-conn, DP ε0.5) while the frozen `am2_task_aware_full` run used the legacy A.2.3 (normalize→thr 0.3→closing 10) masks. Absolute Dice is therefore expected to be lower here because the TrackMate targets are tighter; the meaningful, method-invariant result is that **learnable > fixed at every compression in both runs**.

| compression | illumination | fix Dice (TrackMate) | am2 Dice (thr0.3+closing) | Δ |
|---|---|---:|---:|---:|
| x64 | fixed | 0.8480 | 0.8938 | -0.0459 |
| x64 | learnable | 0.9039 | 0.9322 | -0.0282 |
| x256 | fixed | 0.6283 | 0.6958 | -0.0676 |
| x256 | learnable | 0.8150 | 0.8757 | -0.0607 |
| x1024 | fixed | 0.3959 | 0.4993 | -0.1034 |
| x1024 | learnable | 0.4583 | 0.5986 | -0.1403 |
