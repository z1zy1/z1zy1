#!/usr/bin/env bash
set -euo pipefail

# LEVIR-CC V2 candidate. This isolates gate calibration from the CARD feature
# branches while retaining the strongest pre-norm sparse RSACA core.
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export RUN_ROOT="${RUN_ROOT:-./experiments/reliability_sparse_rsaca_v2_detached_gate}"
export SEMANTIC_FUSION_SPARSE_CHANGE_TOKENS=1
export SEMANTIC_FUSION_RELIABILITY_GATE=1
export SEMANTIC_FUSION_RELIABILITY_GATE_BIAS="${SEMANTIC_FUSION_RELIABILITY_GATE_BIAS:--2.5}"
export SEMANTIC_FUSION_GLOBAL_TOKEN=1
export SEMANTIC_FUSION_GLOBAL_TOKEN_MODE=changed_mean
export SEMANTIC_FUSION_GATE_WHOLE_ADAPTER=1
export SEMANTIC_FUSION_DETACH_RELIABILITY_INPUTS=1
export SEMANTIC_FUSION_NORM_MODE=context_pre_norm
export SEMANTIC_FUSION_GAMMA_INIT=0.01
export SEMANTIC_FUSION_GAMMA_MAX=0.1

# Keep this candidate attributable to the detached gate rather than the
# rejected confidence, visual fallback, or warmup follow-ups.
export SEMANTIC_FUSION_VISUAL_CONSISTENCY_GATE=0
export SEMANTIC_FUSION_VISUAL_FALLBACK=0
export SEMANTIC_FUSION_WARMUP_STEPS=0
unset LEVIR_CC_SEMANTIC_DIFF_CONFIDENCE_ROOT
unset SEMANTIC_DIFF_CONFIDENCE_ROOT

export SELECTION_STRATEGY="${SELECTION_STRATEGY:-paper_balanced_no_spice}"
export SELECTION_METRIC="${SELECTION_METRIC:-paper_balanced_no_spice}"

printf '%s\n' 'V2 RSACA: sparse=1 reliability_gate=1 changed_mean=1 whole_adapter_gate=1 detached_gate_inputs=1'
printf '%s\n' "V2 RSACA: RUN_ROOT=$RUN_ROOT norm=$SEMANTIC_FUSION_NORM_MODE gamma_max=$SEMANTIC_FUSION_GAMMA_MAX gate_bias=$SEMANTIC_FUSION_RELIABILITY_GATE_BIAS visual_gate=0 fallback=0 warmup=0 selection=$SELECTION_METRIC"

exec bash scripts/run_unified_rsaca_experiments.sh "$@"
