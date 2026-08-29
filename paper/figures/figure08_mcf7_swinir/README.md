# Figure 8 — SwinIR vs transpose-conv on Human MCF7 (x16)

- **Paper section:** 5.6
- **Status:** **close**. SwinIR sits at a data/capacity ceiling (~0.82); qualitatively fewer border artifacts than transpose-conv.
- **Shared run:** also produces [Figure 9](../figure09_mcf7_widefield/).

Dataset: BBBC021 MCF7 channel-2 tubulin, split 3000/100/100, x16, superpixel_factor=1, paper-faithful Alg. 1 budget.

## Run tree

`experiments/figure89_mcf7_swinir_highres_fix_v1/runs/` — conditions `wswinir`, `transpose256`, `wcnn64`.

## Reproduce

```bash
python scripts/fig89_mcf7_swinir_fix_train.py --condition wswinir --device cuda:0 --full-budget --gan-warmup-epochs 50
python scripts/reproduce_fig8.py --config configs/reproduce_fig8_bbbc021_tubulin_x16_swinir.yaml --device cuda:0
python paper/_build_components.py --only fig8
```

Needs `data/mcf7_bbbc021` and the SwinIR vendor clone.
