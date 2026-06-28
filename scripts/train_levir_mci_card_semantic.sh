#!/usr/bin/env bash
set -euo pipefail
EXP_NAME="${EXP_NAME:-levir_mci_card_semantic}"
export EXP_NAME
source scripts/train_levir_mci_card_semantic_loss.sh