#!/usr/bin/env bash
set -euo pipefail
# Reuses the dataset-specific required runners with training skipped. The runners evaluate all snapshots and refresh validation CSVs.
bash scripts/run_levir_mci_required_experiments.sh --skip_train "$@"
bash scripts/run_second_cc_required_experiments.sh --skip_train "$@"