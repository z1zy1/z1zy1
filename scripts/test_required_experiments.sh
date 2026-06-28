#!/usr/bin/env bash
set -euo pipefail
# Reuses the dataset-specific required runners with training skipped. The runners test the selected checkpoint for each experiment.
bash scripts/run_levir_mci_required_experiments.sh --skip_train "$@"
bash scripts/run_second_cc_required_experiments.sh --skip_train "$@"