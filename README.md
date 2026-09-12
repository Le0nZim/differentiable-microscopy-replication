# Differentiable Microscopy — fresh workstation experiments

Reproduction code for [Differentiable Microscopy for Content and Task Aware
Compressive Fluorescence Imaging](https://arxiv.org/abs/2203.14945).

**Start with [START_HERE.md](START_HERE.md).** It includes the clean-clone commands
and maps your existing dataset folders to one editable `workstation.yaml`.

After installing the environment and VGG19 dependency:

```bash
python paper.py init --data-root /absolute/path/to/your/data
python paper.py check
python paper.py plan
python paper.py run
```

One sequential queue covers all main-paper experiment families. PatchMNIST is
generated and cached automatically. Downstream stages use newly trained
checkpoints. Repeating `run` skips verified completed jobs and restarts incomplete
jobs as separate attempts. `python paper.py status` shows progress;
`python paper.py report` collects this campaign's metrics and figure artifacts.

| Location | Purpose |
|---|---|
| `paper.py`, `workflow/` | Dataset setup, job order, execution, resume and reports |
| `workstation.yaml` (local) | Your paths, GPU, output/cache directories and seeds |
| `configs/` | Existing scientific recipes, separate from machine paths |
| `src/`, `scripts/` | Models, datasets and individual experiment implementations |
| `runs/` (local) | Fresh campaign configs, manifests, logs, checkpoints and reports |
| `docs/` | Audit, protocol, validation and historical catalog |

The [workstation protocol](docs/WORKSTATION_PROTOCOL.md) lists all stages, budgets
and data-split decisions. The [scientific audit](docs/SCIENTIFIC_AUDIT.md) explains
why historical scores remain exploratory. BBBC022 substitutes for the missing
original U2OS data, and the paper workflow retains documented soft-mask recipes
and deviations. The optional `--stages controlled` queue runs the separate
binary-mask/equal-dose audit experiment.

Historical outputs remain in Git history and in full checkouts, with their
[previous catalog](docs/LEGACY_CATALOG.md). The clean sparse checkout described in
START_HERE excludes them and does not fetch old LFS weights. No old results are
used as inputs to the fresh queue.
