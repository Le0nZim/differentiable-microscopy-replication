# Journal protocol implementation validation

Date: 2026-09-19. Parent: `10f79d74ce3fdb3dd3b748adcf612505d4cc5e48`.
This verifies the prospective implementation, not microscopy performance.

## Environment

- CPU only; no CUDA device or source microscopy collections available.
- Python 3.12.14, PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu.
- SwinIR source pinned to `6545850fbf8df298df73d81f3e8cba638787c8bd`.
- Synthetic MCF7 integration uses a small SwinIR and substitute differentiable
  perceptual/adversarial components. It checks training and checkpoint/render
  compatibility, not pretrained VGG features or full-size GPU memory needs.

## Executed checks

1. Broader non-slow suite: **216 passed, 14 skipped, 1 deselected**; two initial
   failures were investigated. One required unchanged historical fixtures
   excluded by sparse checkout; they were materialized directly from the
   parent commit. The other was a source-format assertion about checkpoint
   eligibility; its compatible syntax was restored without changing the new
   final-phase selection rule.
2. After those fixes and the final segmentation-pretraining change:
   **44 passed** in the focused journal/workstation/progress checks, including
   both formerly failing checks:

   ```bash
   OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
     tests/test_journal_protocol.py tests/test_workstation.py \
     tests/test_progress_audit.py \
     tests/test_udith_legacy_schedule.py::test_existing_patchmnist_fig10_outputs_byte_for_byte \
     tests/test_udith_legacy_schedule.py::test_phase_boundary_evals_share_best_update_path
   ```

   The segmentation smoke uses short explicit phase budgets. One earlier focused
   attempt was stopped after that fixture inadvertently inherited the complete
   7,400-step pretraining schedule; no scientific recipe was shortened.
3. The journal test module was rerun after sharing the seeded evaluation-RNG
   helper with the ablation runner: **14 passed**. It covers matched original/rewritten C/D
   initialization, validation-only tuning with no test-dataset construction,
   final-phase checkpoint eligibility, complete-grid LR selection, even and odd
   morphology against SciPy, matched MCF7 training and renderer reload, image
   metrics and measurement counts, and segmentation input assumptions.
4. All five controlled acquisition modes completed the actual eight-update
   synthetic CPU driver with `--upsampling original_locality`, seed 42.
   This exercises binary acquisition, equal mean dose, gradients, checkpoint
   selection and exact-noise evaluation. Scores are software diagnostics.
5. CLI `init`, `plan` and `run --dry-run` succeeded with a fresh config and no
   datasets. All **369** jobs have unique IDs and precede their dependants; all
   resolved configurations are covered by the workstation test. The
   `patchmnist ablation` subset contains **72** jobs.
6. Paired report differences were checked numerically for B-A, C-B and C-D.
   Source whitespace checks passed. Historical result fixtures remain unchanged.

The broader and focused counts overlap; they must not be added together.
A sparse checkout intentionally excludes historical artifacts, so its archival
preservation test needs those tracked fixtures; ordinary new-workflow tests do not.

## Limits and the workstation gate

Run `python paper.py check` on the actual workstation before training. It must
validate the real input inventories, CUDA device, worker IPC and required model
weights. CPU checks do not establish GPU memory sufficiency, numerical rankings,
convergence, exact U2OS replication, biological accuracy or journal acceptance.
The full source datasets and GPU experiments still need to be run.
