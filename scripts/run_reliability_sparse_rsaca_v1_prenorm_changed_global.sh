#!/usr/bin/env bash
set -euo pipefail

# V1 follow-up: strict identity-preserving pre-norm fusion, conservative
# semantic scaling, and a global token pooled from changed locations only.
# Outputs are isolated and checkpoint selection remains validation-only.
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export RUN_ROOT="${RUN_ROOT:-./experiments/reliability_sparse_rsaca_v1_prenorm_changed_global}"
export SEMANTIC_FUSION_SPARSE_CHANGE_TOKENS=1
export SEMANTIC_FUSION_RELIABILITY_GATE=1
export SEMANTIC_FUSION_GLOBAL_TOKEN=1
export SEMANTIC_FUSION_GLOBAL_TOKEN_MODE=changed_mean
export SEMANTIC_FUSION_GATE_WHOLE_ADAPTER=1
export SEMANTIC_FUSION_NORM_MODE="${SEMANTIC_FUSION_NORM_MODE:-context_pre_norm}"
export SEMANTIC_FUSION_GAMMA_INIT="${SEMANTIC_FUSION_GAMMA_INIT:-0.01}"
export SEMANTIC_FUSION_GAMMA_MAX="${SEMANTIC_FUSION_GAMMA_MAX:-0.1}"
export SEMANTIC_FUSION_RELIABILITY_GATE_BIAS="${SEMANTIC_FUSION_RELIABILITY_GATE_BIAS:--2.5}"
export SELECTION_STRATEGY="${SELECTION_STRATEGY:-paper_balanced}"
export SELECTION_METRIC="${SELECTION_METRIC:-paper_balanced}"

printf '%s\n' 'V1 pre-norm RSACA: sparse=1 reliability_gate=1 global_token=changed_mean whole_adapter_gate=1'
printf '%s\n' "V1 pre-norm RSACA: RUN_ROOT=$RUN_ROOT norm=$SEMANTIC_FUSION_NORM_MODE gamma_max=$SEMANTIC_FUSION_GAMMA_MAX gate_bias=$SEMANTIC_FUSION_RELIABILITY_GATE_BIAS selection=$SELECTION_METRIC"

exec bash scripts/run_unified_rsaca_experiments.sh "$@"
