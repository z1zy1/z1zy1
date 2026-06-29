#!/usr/bin/env bash
set -euo pipefail
# Reuse the new required runner with training skipped. It still selects checkpoints,
# tests paper_best, and updates the paper summary.
bash scripts/run_new_required_train_experiments.sh --skip_train "$@"
