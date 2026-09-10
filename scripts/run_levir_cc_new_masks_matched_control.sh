#!/usr/bin/env bash
set -euo pipefail

# Isolate the LEVIR-CC pseudo-mask change from the later confidence/gating
# follow-ups. This keeps the pre-norm changed-mean RSACA core and restores the
# settings used by the prior LEVIR-CC run, while reading the active new masks.
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export RUN_ROOT="${RUN_ROOT:-./experiments/reliability_sparse_rsaca_v1_levir_cc_new_masks_matched_control}"
export SEMANTIC_FUSION_VISUAL_CONSISTENCY_GATE=0
export SEMANTIC_FUSION_VISUAL_FALLBACK=0
export SEMANTIC_FUSION_WARMUP_STEPS=0
export SELECTION_STRATEGY=paper_balanced
export SELECTION_METRIC=paper_balanced
# Avoid persistent DataLoader workers in memory-constrained containers.
export NUM_WORKERS="${NUM_WORKERS:-0}"
export REQUIRE_CUDA=1

# Confidence maps are deliberately excluded from this mask-only comparison.
# The dataset-scoped isolation in the unified runner remains enabled; removing
# it would have no effect for LEVIR-CC and would reintroduce cross-dataset risk.
unset LEVIR_CC_SEMANTIC_DIFF_CONFIDENCE_ROOT
unset SEMANTIC_DIFF_CONFIDENCE_ROOT

printf '%s\n' "LEVIR-CC new-mask matched control: confidence=0 visual_gate=0 fallback=0 warmup=0 selection=paper_balanced workers=$NUM_WORKERS require_cuda=1"
printf '%s\n' "LEVIR-CC new-mask matched control: RUN_ROOT=$RUN_ROOT"

exec bash scripts/run_reliability_sparse_rsaca_v1_prenorm_changed_global.sh "$@"
