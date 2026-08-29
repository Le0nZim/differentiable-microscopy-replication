# Figure 10 — Ablation qualitative A/B/C/D (BBBC022 substitute)

- **Paper section:** 5.7
- **Status:** **data_blocked** (paper U2OS x16). Same substitute and split as Figure 3.
- **Quantitative companion:** [Table 3](../table03_ablation/) (`experiments/table03_ablation/`, seeds 42/43/44)

Variants: A fixed Ht + transpose + freq; B learnable Ht + transpose + freq; C learnable + locality + freq (paper best); D learnable + locality, no freq.

## Run tree

`experiments/figure10_ablation/runs/{A,B,C,D}_seed42/`

## Reproduce

```bash
bash scripts/figure10_ablation/train_all_and_render.sh
python scripts/figure10_ablation/reproduce.py
python paper/_build_components.py --only fig10
```

On the substitute, ordering does not match the paper (B best, C worst); see Table 3 root-cause notes.
