#!/usr/bin/env bash
set -euo pipefail
EXP_NAME="${LEVIR_CC_BASELINE_EXP:-card_levir_cc_baseline}"
DATASET=levir_cc
DATA_ROOT="${LEVIR_CC_ROOT:-./Levir-CC}"
BASE_CFG=configs/dynamic/transformer_levir_cc_baseline.yaml
BASELINE_FEATURE_ROOT="${LEVIR_CC_FEATURE_ROOT:-}"
export EXP_NAME DATASET DATA_ROOT BASE_CFG BASELINE_FEATURE_ROOT
source scripts/_card_baseline_env.sh
bash scripts/_run_paper_training.sh
