#!/usr/bin/env bash
set -euo pipefail
EXP_NAME="${LEVIR_MCI_BASELINE_EXP:-levir_mci_card_baseline}"
DATASET=levir_mci
DATA_ROOT="${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}"
BASE_CFG=configs/dynamic/transformer_levir_mci_baseline.yaml
BASELINE_FEATURE_ROOT="${LEVIR_MCI_FEATURE_ROOT:-}"
export EXP_NAME DATASET DATA_ROOT BASE_CFG BASELINE_FEATURE_ROOT
source scripts/_card_baseline_env.sh
bash scripts/_run_paper_training.sh
