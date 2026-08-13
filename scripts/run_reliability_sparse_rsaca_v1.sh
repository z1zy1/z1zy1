#!/usr/bin/env bash
set -euo pipefail

# V1 candidate: CARD + RSACA with sparse change K/V tokens and a continuous
# per-sample reliability gate. Results are isolated from the locked legacy run.
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export RUN_ROOT="${RUN_ROOT:-./experiments/reliability_sparse_rsaca_v1}"
export SEMANTIC_FUSION_SPARSE_CHANGE_TOKENS=1
export SEMANTIC_FUSION_RELIABILITY_GATE=1
export SEMANTIC_FUSION_RELIABILITY_GATE_BIAS="${SEMANTIC_FUSION_RELIABILITY_GATE_BIAS:--1.5}"

# Keep the locked RSACA normalization, learning schedule, gamma, partial
# detach, and semantic sources unchanged. V1 therefore isolates sparse tokens
# and reliability gating as the only architectural difference.
export SEMANTIC_FUSION_NORM_MODE="${SEMANTIC_FUSION_NORM_MODE:-legacy_post_norm}"

printf '%s\n' 'V1 RSACA: SEMANTIC_FUSION_SPARSE_CHANGE_TOKENS=1 SEMANTIC_FUSION_RELIABILITY_GATE=1'
printf '%s\n' "V1 RSACA: RUN_ROOT=$RUN_ROOT gate_bias=$SEMANTIC_FUSION_RELIABILITY_GATE_BIAS"

exec bash scripts/run_unified_rsaca_experiments.sh "$@"
