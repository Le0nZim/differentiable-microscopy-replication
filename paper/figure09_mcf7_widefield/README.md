# Figure 9 — Wide-field MCF7: SwinIR vs CNN (x16)

- **Paper section:** 5.6
- **Status:** **close**
- **Shared run:** [Figure 8](../figure08_mcf7_swinir/) (`wswinir` and `wcnn64`, overlap-add tiling, superpixel_factor=1)

## Run tree

`experiments/figure08_mcf7/runs/`

## Reproduce

```bash
python scripts/figure08_mcf7/reproduce_fig9.py \
  --config configs/figure08_mcf7/reproduce_fig9_tubulin.yaml \
  --device cuda:0
python paper/_build_components.py --only fig9
```
