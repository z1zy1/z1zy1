#!/usr/bin/env bash
set -euo pipefail

# V1 follow-up: sparse changed tokens plus an input-conditioned global token,
# with the reliability gate applied to the complete RSACA adapter output.
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export RUN_ROOT="${RUN_ROOT:-./experiments/reliability_sparse_rsaca_v1_whole_gate}"
export SEMANTIC_FUSION_SPARSE_CHANGE_TOKENS=1
export SEMANTIC_FUSION_RELIABILITY_GATE=1
export SEMANTIC_FUSION_GLOBAL_TOKEN=1
export SEMANTIC_FUSION_GATE_WHOLE_ADAPTER=1
export SEMANTIC_FUSION_RELIABILITY_GATE_BIAS="${SEMANTIC_FUSION_RELIABILITY_GATE_BIAS:--1.5}"
export SEMANTIC_FUSION_NORM_MODE="${SEMANTIC_FUSION_NORM_MODE:-legacy_post_norm}"

printf '%s\n' 'V1 whole-adapter RSACA: sparse=1 reliability_gate=1 global_token=1 whole_adapter_gate=1'
printf '%s\n' "V1 whole-adapter RSACA: RUN_ROOT=$RUN_ROOT gate_bias=$SEMANTIC_FUSION_RELIABILITY_GATE_BIAS"

exec bash scripts/run_unified_rsaca_experiments.sh "$@"
