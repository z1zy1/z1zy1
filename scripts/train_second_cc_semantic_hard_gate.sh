#!/usr/bin/env bash
set -euo pipefail
EXP_NAME="${EXP_NAME:-second_cc_semantic_hard_gate}"
SEMANTIC_INPUT_MODE="hard_gate"
export EXP_NAME SEMANTIC_INPUT_MODE
source scripts/train_second_cc_card_semantic_aux.sh