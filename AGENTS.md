# Workstation workflow

Read START_HERE.md first. For a fresh setup use paper.py, not archived launchers.
The finalized branch is journal/final-protocol. Read docs/JOURNAL_PROTOCOL.md;
scientific choices are already fixed in configs/journal/protocol.yaml. Preserve
the declared LR searches, selection criteria, original primary architectures,
held-out reserve wells and distinct binary-control reporting. Do not choose an
architecture or tune a recipe because it produces the desired test ranking.

1. Keep the user's source datasets in place. Do not delete, rename or merge data
   folders. Do not download old Git LFS checkpoints for a clean rerun.
2. Run setup, then `paper.py init --data-root <the supplied data directory>`.
   Change machine paths/device/output directory only in workstation.yaml.
3. Run `paper.py check` and fix the reported input-path/extraction issues. Never
   bypass a missing input with random tensors, another fluorescence channel,
   downsampled SR images or a historical checkpoint.
4. Run `paper.py plan`. When the user requests training, `paper.py run` executes
   sequentially and includes prerequisites. Resume by repeating the command.
5. Use `paper.py status` and `paper.py report` for progress and output evidence.
   A failed job gets a new attempt, not a guessed completed status. Never edit
   state.json or experiment recipes to force completion or a method ranking.
6. Report BBBC022 results as substitutes. Default segmentation uses BBBC039
   manual masks and official splits; download with `paper.py download-segmentation-data`.
   Never synthesize missing manual masks. The explicitly selected `pseudo_trackmate`
   fallback must retain raw threshold 506, spacing 2 and epsilon 0.5 exactly.
   Report the actual label source in each experiment; do not call manual masks pseudo-labels.
   Preserve the distinction between the paper recipes and optional `controlled`
   binary/equal-dose experiments. A synthetic smoke is a software check only.

Implementation map: workflow/catalog.py declares the job matrix; workflow/data.py
prepares sources; workflow/worker.py binds existing trainers; workflow/cli.py
handles state; workflow/report.py reads only verified campaign outputs.
