#!/usr/bin/env bash
set -euo pipefail
destination="${1:-microscopy-clean}"
if [[ -e "$destination" ]]; then
  echo "Destination already exists: $destination. Choose a new directory; nothing was deleted." >&2
  exit 1
fi
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --single-branch --branch workflow/clean-workstation \
  --filter=blob:none --sparse https://github.com/Le0nZim/differentiable-microscopy-replication.git "$destination"
git -C "$destination" sparse-checkout set src workflow scripts configs tests docs data
echo "Clean source checkout: $destination. Next: cd into it and follow START_HERE.md."
