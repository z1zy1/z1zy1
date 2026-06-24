#!/usr/bin/env bash
set -euo pipefail
EXP_NAME="${EXP_NAME:-second_cc_ours_weak_coupled_final}"
SEMANTIC_INPUT_MODE="weak_coupled"
USE_SEMANTIC_PARTIAL_DETACH=1
SEMANTIC_DETACH_RATIO="${SEMANTIC_DETACH_RATIO:-0.5}"
export EXP_NAME SEMANTIC_INPUT_MODE USE_SEMANTIC_PARTIAL_DETACH SEMANTIC_DETACH_RATIO
source scripts/train_second_cc_card_semantic_aux.sh
