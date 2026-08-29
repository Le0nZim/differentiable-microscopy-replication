# Missing / data-blocked components — Table 2

- **Paper's ~8-12 dB LI magnitude is NOT reproduced.** Our fair protocol gives the w/o-LI condition the same powerful trainable decoder, so our w/o-LI baseline (~22-25 dB PSNR) is far stronger than the paper's (~12-14 dB). The paper's dramatic gap is dominated by its weak w/o-LI baseline.
- **Iteration count is compute-limited** (20000 run vs ~500000 faithful = ~4%). Resumable; LI gain *grew* with more training (6k->20k), so longer training widens the gap rather than closing it. Magnitude affected, not sign.

