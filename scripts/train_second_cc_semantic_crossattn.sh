#!/usr/bin/env bash
set -euo pipefail
EXP_NAME="${EXP_NAME:-second_cc_semantic_crossattn}"
SEMANTIC_INPUT_MODE="cross_attention"
export EXP_NAME SEMANTIC_INPUT_MODE
source scripts/train_second_cc_card_semantic_aux.sh