# Figure 4 task-aware segmentation — sanity gates

**Overall: PASS ✅**

| gate | passed | key numbers |
|---|:--:|---|
| mask_correctness | ✅ | trackmate raw>506: fg=0.2181, binary=True, IoU_vs_raw_thr=0.9509 |
| no_leakage_split | ✅ | wells 168/21/21, overlaps=0 |
| distribution | ✅ | train:[0.0,1.0] fg=0.1768, val:[0.0134,1.0] fg=0.1727, test:[0.0114,1.0] fg=0.1907 |
| mask_not_in_input | ✅ | forward_excludes_mask=True, responds_to_input=True |
| degenerate_baselines | ✅ | zeros=0.0, ones=0.3159, perfect=1.0, posthoc=0.8263 |
| tiny_overfit | ✅ | best_train_dice=0.9523 (4 samples, 300 steps) |

Mask examples: `mask_examples.png`
