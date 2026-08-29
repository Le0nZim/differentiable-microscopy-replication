# Figure 10 — Ablation qualitative A/B/C/D (BBBC022 substitute)

- **Paper section:** 5.7
- **Status:** **data_blocked** (paper U2OS x16). Same substitute and split as Figure 3.
- **Quantitative companion:** [Table 3](../../tables/table03_ablation/) (`experiments/ablations/am3_table3_resolution/`, seeds 42/43/44)

Variants: A fixed Ht + transpose + freq; B learnable Ht + transpose + freq; C learnable + locality + freq (paper best); D learnable + locality, no freq.

## Run tree

`experiments/figure10_bbbc022_ablation_v1/runs/{A,B,C,D}_seed42/`

## Reproduce

```bash
bash scripts/_fig10_train_all_and_render.sh
python scripts/reproduce_fig10.py
python paper/_build_components.py --only fig10
```

On the substitute, ordering does not match the paper (B best, C worst); see Table 3 root-cause notes.
