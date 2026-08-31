# Training status

**Label:** Udith legacy schedule under the current controlled PatchMNIST optimizer/data recipe.
This is **not** a full reproduction of Udith’s legacy training loop.

Independent-warmup C vs D frozen trajectories **diverged materially** (seed 42 max
|Δloss| ≈ 1.81e-3, max |Δval MSE| ≈ 2.16e-3; construction hashes still matched).
Seeds **43/44 were stopped**. See `FROZEN_INTERVAL_COMPARE.md`.

| Job | GPU | Status |
| --- | --- | --- |
| Independent C42 | cuda:1 | **done** (diagnostic only; divergent warmup) |
| Independent D42 | cuda:1 | still running (PID 462543); diagnostic only; watcher will SIGTERM this driver after `D_seed42` summary so it cannot start 43/44 |
| Independent C43 | cuda:0 | **done** (diagnostic only) |
| Independent D43 | cuda:0 | **stopped** (incomplete) |
| Independent 44 | — | **will not start** |
| Shared-warmup seed 42 (D warmup → fork C and D) | **cuda:0** | running PID **502334**; log `train_shared_warmup_seed42.log`; outputs `runs_shared_warmup/` |

Watcher that blocks 43/44: `scripts/figure10_ablation_patchmnist_udith_schedule/prevent_seed44.sh`  
Status: `prevent_seed44_STATUS.txt`

Phase-boundary evals **are eligible for global-best** (same `best.update` path). Step 121,500 is the last step of the last phase (m=8); it is not on the `log_every=200` grid (`121500 % 200 == 100`). `itertools.cycle(train_loader)` is still used.
