#!/usr/bin/env bash
set -euo pipefail
EXP_NAME="${SECOND_CC_BASELINE_EXP:-second_cc_card_rgb_baseline}"
DATASET=second_cc
DATA_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
BASE_CFG=configs/dynamic/transformer_second_cc_aug_baseline.yaml
BASELINE_FEATURE_ROOT="${SECOND_CC_FEATURE_ROOT:-}"
export EXP_NAME DATASET DATA_ROOT BASE_CFG BASELINE_FEATURE_ROOT
source scripts/_card_baseline_env.sh
bash scripts/_run_paper_training.sh
