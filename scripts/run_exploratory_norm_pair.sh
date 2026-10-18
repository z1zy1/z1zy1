#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PROTOCOL_ID="exploratory_norm_context_pre_20261015"
OUTPUT_ROOT="${EXPLORATORY_OUTPUT_ROOT:-$(cd "$PROJECT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-/root/miniconda3/envs/card/bin/python}"
export PYTHON

usage() { echo "Usage: $0 C0|N1 levir_cc|levir_mci 1111|2222|3333 smoke|train|select" >&2; }
if [ "$#" -ne 4 ]; then usage; exit 2; fi
ARM="$1" DATASET="$2" SEED="$3" ACTION="$4"
case "$ARM" in C0|N1) ;; *) usage; exit 2 ;; esac
case "$DATASET" in levir_cc|levir_mci) ;; *) usage; exit 2 ;; esac
case "$SEED" in 1111|2222|3333) ;; *) usage; exit 2 ;; esac
case "$ACTION" in smoke|train|select) ;; *) echo "Formal test is prohibited by $PROTOCOL_ID" >&2; exit 2 ;; esac

if [ "$ACTION" = smoke ] || [ "$ACTION" = train ]; then
  "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 3)' || {
    echo "RESOURCE_BLOCKED: CUDA is required; no training was started." >&2
    exit 3
  }
fi

if [ "$ACTION" = smoke ]; then
  RUN_ROOT="${EXPLORATORY_OUTPUT_ROOT:-$(cd "$PROJECT_DIR/.." && pwd)}/smoke/$ARM/$DATASET"
  TRAIN_STEPS=100
  SNAPSHOT_STEPS=100
  WARMUP_STEPS=50
else
  RUN_ROOT="${EXPLORATORY_OUTPUT_ROOT:-$(cd "$PROJECT_DIR/.." && pwd)}/runs/$ARM/$DATASET"
  TRAIN_STEPS=10000
  SNAPSHOT_STEPS=1000
  WARMUP_STEPS=500
fi
EXP_NAME="norm_${ARM}_${DATASET}_seed${SEED}"
export PROJECT_DIR RUN_ROOT EXP_DIR="$RUN_ROOT" EXP_NAME DATASET SEED
export MODEL_TYPE=sgc_card REQUIRE_CUDA=1 PYTORCH_GPU=0 PROTOCOL_ID
export ISOLATE_LOADER_RNG=1 SEED_WORKERS=1 VALIDATION_GREEDY=1
export MAX_ITER="$TRAIN_STEPS" SNAPSHOT_INTERVAL="$SNAPSHOT_STEPS"
export SAVE_INTERVAL="$SNAPSHOT_STEPS" EVAL_INTERVAL="$SNAPSHOT_STEPS"
export BATCH_SIZE=32 NUM_WORKERS=8 LR=0.0002 LR_SCHEDULER=warmup_cosine
export LR_WARMUP_STEPS="$WARMUP_STEPS" MIN_LR_RATIO=0.05
export USE_CHANGE_MASK=0 ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0
export USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7
export USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5
export USE_FEATURE_REWEIGHT=0 LMASK=0 LSEM=0 INIT_CHECKPOINT=
export SEMANTIC_FUSION_GAMMA_INIT=0.01 SEMANTIC_FUSION_GAMMA_MAX=0.5
export SEMANTIC_FUSION_NORM_MODE=legacy_post_norm
export SEMANTIC_FUSION_SPARSE_CHANGE_TOKENS=1 SEMANTIC_FUSION_RELIABILITY_GATE=1
export SEMANTIC_FUSION_GLOBAL_TOKEN=1 SEMANTIC_FUSION_GLOBAL_TOKEN_MODE=all_mean
export SEMANTIC_FUSION_GATE_WHOLE_ADAPTER=1 SEMANTIC_FUSION_RELIABILITY_GATE_BIAS=-1.5
export SEMANTIC_FUSION_DETACH_RELIABILITY_INPUTS=0
export SEMANTIC_FUSION_VISUAL_CONSISTENCY_GATE=0 SEMANTIC_FUSION_VISUAL_FALLBACK=0
export SEMANTIC_FUSION_WARMUP_STEPS=0
export SELECTION_STRATEGY=five_metric_equal_weight_log SELECTION_METRIC=five_metric_equal_weight_log
export PAPER_SELECTION_MODE=1
if [ "$ARM" = N1 ]; then export SEMANTIC_FUSION_NORM_MODE=context_pre_norm; fi

case "$DATASET" in
  levir_cc)
    export LEVIR_CC_ROOT="${LEVIR_CC_ROOT:-/root/autodl-tmp/z1zy1/Levir-CC}"
    export DATA_ROOT="$LEVIR_CC_ROOT"
    export FEATURE_ROOT="${LEVIR_CC_FEATURE_ROOT:-$DATA_ROOT/features}"
    export BASE_CFG=configs/dynamic/transformer_levir_cc_sgc_card.yaml
    export SEMANTIC_DIFF_ROOT="$DATA_ROOT/pseudo_masks" SEMANTIC_DIFF_ONLY=1 SEMANTIC_DIFF_BINARY=1
    export SEMANTIC_UNKNOWN_CHANGE_CLASS=6 ALLOW_MISSING_PSEUDO_MASK=1
    ;;
  levir_mci)
    export LEVIR_MCI_ROOT="${LEVIR_MCI_ROOT:-/root/autodl-tmp/z1zy1/LEVIR-MCI-dataset}"
    export DATA_ROOT="$LEVIR_MCI_ROOT"
    export FEATURE_ROOT="${LEVIR_MCI_FEATURE_ROOT:-$DATA_ROOT/features}"
    export BASE_CFG=configs/dynamic/transformer_levir_mci_sgc_card.yaml
    export SEMANTIC_DIFF_ROOT="$DATA_ROOT/images" SEMANTIC_DIFF_PHASE=label SEMANTIC_DIFF_ONLY=1
    export SEMANTIC_DIFF_BINARY=0 ALLOW_MISSING_PSEUDO_MASK=0
    ;;
esac

if [ "$ACTION" = smoke ] || [ "$ACTION" = train ]; then exec bash scripts/_run_paper_training.sh; fi
exec "$PYTHON" scripts/select_best_snapshot_p1.py \
  --exp-dir "$RUN_ROOT/$EXP_NAME" --protocol-id "$PROTOCOL_ID" \
  --output-json "$RUN_ROOT/$EXP_NAME/selection.json" \
  --copy-path "$RUN_ROOT/$EXP_NAME/selected_checkpoint.pth"
