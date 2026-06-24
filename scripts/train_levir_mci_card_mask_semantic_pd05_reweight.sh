#!/usr/bin/env bash
set -euo pipefail
EXP_NAME="${EXP_NAME:-levir_mci_card_mask_semantic_pd05_reweight}"
USE_FEATURE_REWEIGHT=1
REWEIGHT_ALPHA="${REWEIGHT_ALPHA:-0.2}"
DETACH_REWEIGHT_MASK="${DETACH_REWEIGHT_MASK:-1}"
export EXP_NAME USE_FEATURE_REWEIGHT REWEIGHT_ALPHA DETACH_REWEIGHT_MASK
source scripts/train_levir_mci_card_mask_semantic_pd05.sh
