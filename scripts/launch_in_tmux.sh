#!/usr/bin/env bash
# Launch a long-running command in a detached tmux session.
# Survives closing Cursor/SSH disconnects. Re-attach with: tmux attach -t SESSION
#
# Usage:
#   scripts/launch_in_tmux.sh SESSION_NAME GPU_ID LOG_FILE COMMAND...
#
# Example:
#   scripts/launch_in_tmux.sh mcf7_li_swinir 1 experiments/swinir_or_highres/mcf7_paper_direct_full/run.log \
#     .venv/bin/python scripts/run_mcf7_li_swinir_paper_direct.py --device cuda:0 --seed 42

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 SESSION_NAME GPU_ID LOG_FILE COMMAND..." >&2
  exit 1
fi

SESSION="$1"
GPU="$2"
LOG_FILE="$3"
shift 3

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$(dirname "$LOG_FILE")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION" >&2
  echo "Or kill it first: tmux kill-session -t $SESSION" >&2
  exit 1
fi

CMD=$(printf '%q ' "$@")
LOG_Q=$(printf '%q' "$LOG_FILE")

tmux new-session -d -s "$SESSION" "cd $(printf '%q' "$REPO") && export PYTHONUNBUFFERED=1 && echo \"[$(date -Is)] starting: $CMD\" | tee -a $LOG_Q && CUDA_VISIBLE_DEVICES=$GPU $CMD 2>&1 | tee -a $LOG_Q; echo EXIT_CODE=\$? | tee -a $LOG_Q"

echo "Launched tmux session: $SESSION (GPU $GPU)"
echo "Log: $LOG_FILE"
echo "Attach: tmux attach -t $SESSION"
echo "Tail log: tail -f $LOG_FILE"
