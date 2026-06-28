#!/usr/bin/env bash
set -euo pipefail
EXP_NAME="${EXP_NAME:-second_cc_semantic_aux}"
export EXP_NAME
source scripts/train_second_cc_card_semantic_aux.sh