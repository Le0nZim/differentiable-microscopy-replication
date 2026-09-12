# Workstation workflow

Read START_HERE.md first. For a fresh setup use paper.py, not archived launchers.

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
6. Report BBBC022 results as substitutes and segmentation masks as pseudo-labels.
   Preserve the distinction between the paper recipes and optional `controlled`
   binary/equal-dose experiments. A synthetic smoke is a software check only.

Implementation map: workflow/catalog.py declares the job matrix; workflow/data.py
prepares sources; workflow/worker.py binds existing trainers; workflow/cli.py
handles state; workflow/report.py reads only verified campaign outputs.
