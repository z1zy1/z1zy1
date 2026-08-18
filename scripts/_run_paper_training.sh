#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
case "${OMP_NUM_THREADS:-}" in
  ''|*[!0-9]*|0) export OMP_NUM_THREADS=1 ;;
esac
PYTORCH_GPU="${PYTORCH_GPU:-0}"
PYTHON="${PYTHON:-python}"
EXP_DIR="${EXP_DIR:-./experiments}"
EXP_NAME="${EXP_NAME:?EXP_NAME is required}"
DATASET="${DATASET:?DATASET is required}"
MODEL_TYPE="${MODEL_TYPE:-sgc_card}"
BASE_CFG="${BASE_CFG:?BASE_CFG is required}"

if [ -z "${DATA_ROOT:-}" ]; then
  case "$DATASET" in
    levir_cc) DATA_ROOT="${LEVIR_CC_ROOT:-}" ;;
    levir_mci) DATA_ROOT="${LEVIR_MCI_ROOT:-}" ;;
    second_cc) DATA_ROOT="${SECOND_CC_ROOT:-}" ;;
    *) DATA_ROOT="" ;;
  esac
fi
if [ -z "$DATA_ROOT" ]; then
  echo "DATA_ROOT is empty. Set DATA_ROOT or the matching LEVIR_CC_ROOT/LEVIR_MCI_ROOT/SECOND_CC_ROOT." >&2
  exit 2
fi

FEATURE_ROOT="${FEATURE_ROOT:-}"
USE_CHANGE_MASK="${USE_CHANGE_MASK:-0}"
MASK_TYPE="${MASK_TYPE:-binary}"
NUM_MASK_CLASSES="${NUM_MASK_CLASSES:-}"
USE_SEMANTIC_MAPS="${USE_SEMANTIC_MAPS:-0}"
SEMANTIC_INPUT_MODE="${SEMANTIC_INPUT_MODE:-none}"
NUM_SEMANTIC_CLASSES="${NUM_SEMANTIC_CLASSES:-}"
SEMANTIC_MAP_ROOT="${SEMANTIC_MAP_ROOT:-}"
SEMANTIC_BEFORE_PHASE="${SEMANTIC_BEFORE_PHASE:-}"
SEMANTIC_AFTER_PHASE="${SEMANTIC_AFTER_PHASE:-}"
SEMANTIC_DIFF_ROOT="${SEMANTIC_DIFF_ROOT:-}"
SEMANTIC_DIFF_PHASE="${SEMANTIC_DIFF_PHASE:-}"
SEMANTIC_DIFF_ONLY="${SEMANTIC_DIFF_ONLY:-0}"
SEMANTIC_DIFF_BINARY="${SEMANTIC_DIFF_BINARY:-0}"
SEMANTIC_UNKNOWN_CHANGE_CLASS="${SEMANTIC_UNKNOWN_CHANGE_CLASS:-6}"
ENABLE_AUX_MASK="${ENABLE_AUX_MASK:-0}"
USE_AUX_SEMANTIC="${USE_AUX_SEMANTIC:-0}"
USE_SEMANTIC_PARTIAL_DETACH="${USE_SEMANTIC_PARTIAL_DETACH:-0}"
USE_AUX_WARMUP="${USE_AUX_WARMUP:-1}"
AUX_WARMUP_START_RATIO="${AUX_WARMUP_START_RATIO:-0.30}"
AUX_WARMUP_END_RATIO="${AUX_WARMUP_END_RATIO:-0.70}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-spice_constrained_balanced}"
USE_FEATURE_REWEIGHT="${USE_FEATURE_REWEIGHT:-0}"
DETACH_REWEIGHT_MASK="${DETACH_REWEIGHT_MASK:-1}"
REWEIGHT_ALPHA="${REWEIGHT_ALPHA:-0.2}"
LMASK="${LMASK:-0.0}"
LSEM="${LSEM:-0.0}"
SEMANTIC_DETACH_RATIO="${SEMANTIC_DETACH_RATIO:-0.5}"
SEMANTIC_FUSION_GAMMA_INIT="${SEMANTIC_FUSION_GAMMA_INIT:-}"
SEMANTIC_FUSION_GAMMA_MAX="${SEMANTIC_FUSION_GAMMA_MAX:-}"
SEMANTIC_FUSION_NORM_MODE="${SEMANTIC_FUSION_NORM_MODE:-}"
SEMANTIC_FUSION_SPARSE_CHANGE_TOKENS="${SEMANTIC_FUSION_SPARSE_CHANGE_TOKENS:-}"
SEMANTIC_FUSION_RELIABILITY_GATE="${SEMANTIC_FUSION_RELIABILITY_GATE:-}"
SEMANTIC_FUSION_RELIABILITY_GATE_BIAS="${SEMANTIC_FUSION_RELIABILITY_GATE_BIAS:-}"
SEMANTIC_FUSION_GLOBAL_TOKEN="${SEMANTIC_FUSION_GLOBAL_TOKEN:-}"
SEMANTIC_FUSION_GLOBAL_TOKEN_MODE="${SEMANTIC_FUSION_GLOBAL_TOKEN_MODE:-}"
SEMANTIC_FUSION_GATE_WHOLE_ADAPTER="${SEMANTIC_FUSION_GATE_WHOLE_ADAPTER:-}"
USE_CONTENT_WORD_WEIGHT="${USE_CONTENT_WORD_WEIGHT:-}"
CONTENT_WORD_WEIGHT="${CONTENT_WORD_WEIGHT:-}"
NORMALIZE_CONTENT_WORD_WEIGHTS="${NORMALIZE_CONTENT_WORD_WEIGHTS:-}"
MASK_LOSS_TYPE="${MASK_LOSS_TYPE:-}"
SEMANTIC_LOSS_TYPE="${SEMANTIC_LOSS_TYPE:-}"
ALLOW_MISSING_PSEUDO_MASK="${ALLOW_MISSING_PSEUDO_MASK:-0}"
PAPER_SELECTION_MODE="${PAPER_SELECTION_MODE:-1}"
MAX_ITER="${MAX_ITER:-}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-}"
FINETUNE_STEPS="${FINETUNE_STEPS:-}"
FINETUNE_DECODER_ONLY="${FINETUNE_DECODER_ONLY:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
EVAL_INTERVAL="${EVAL_INTERVAL:-}"
LOG_INTERVAL="${LOG_INTERVAL:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
LR_SCHEDULER="${LR_SCHEDULER:-}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-}"
MIN_LR_RATIO="${MIN_LR_RATIO:-}"
PRETRAINED_LR_SCALE="${PRETRAINED_LR_SCALE:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
SEED="${SEED:-1111}"

bool_word() {
  case "$1" in
    1|true|True|TRUE|yes|Yes|YES|on|ON) echo True ;;
    *) echo False ;;
  esac
}

is_true() {
  [ "$(bool_word "$1")" = "True" ]
}

EXP_PATH="$EXP_DIR/$EXP_NAME"
mkdir -p "$EXP_PATH"
LOG_PATH="$EXP_PATH/train.log"

FEATURE_NOTE_ROOT="$FEATURE_ROOT"
if [ -z "$FEATURE_NOTE_ROOT" ]; then
  FEATURE_NOTE_ROOT="$DATA_ROOT/features"
fi
if [ ! -d "$FEATURE_NOTE_ROOT" ]; then
  echo "Warning: feature root does not exist yet: $FEATURE_NOTE_ROOT" | tee -a "$LOG_PATH"
  echo "If training stops on missing .npy features, generate them with scripts/extract_change_dataset_features.py for $DATASET." | tee -a "$LOG_PATH"
fi

COMMON_OPTS=(
  exp_dir "$EXP_DIR"
  exp_name "$EXP_NAME"
  gpu_id "[$PYTORCH_GPU]"
  train.seed "$SEED"
  train.finetune_decoder_only "$(bool_word "$FINETUNE_DECODER_ONLY")"
  model.enable_aux_mask "$(bool_word "$ENABLE_AUX_MASK")"
  train.use_semantic_aux "$(bool_word "$USE_AUX_SEMANTIC")"
  train.lambda_mask "$LMASK"
  train.lambda_semantic "$LSEM"
  train.use_semantic_partial_detach "$(bool_word "$USE_SEMANTIC_PARTIAL_DETACH")"
  train.semantic_detach_ratio "$SEMANTIC_DETACH_RATIO"
  train.use_aux_warmup "$(bool_word "$USE_AUX_WARMUP")"
  train.aux_warmup_start_ratio "$AUX_WARMUP_START_RATIO"
  train.aux_warmup_end_ratio "$AUX_WARMUP_END_RATIO"
  train.selection_strategy "$SELECTION_STRATEGY"
  train.use_feature_reweight "$(bool_word "$USE_FEATURE_REWEIGHT")"
  train.detach_reweight_mask "$(bool_word "$DETACH_REWEIGHT_MASK")"
  train.reweight_alpha "$REWEIGHT_ALPHA"
  data.allow_missing_pseudo_mask "$(bool_word "$ALLOW_MISSING_PSEUDO_MASK")"
  data.use_change_mask "$(bool_word "$USE_CHANGE_MASK")"
  data.mask_type "$MASK_TYPE"
  data.use_semantic_maps "$(bool_word "$USE_SEMANTIC_MAPS")"
  model.semantic_input_mode "$SEMANTIC_INPUT_MODE"
  data.semantic_diff_only "$(bool_word "$SEMANTIC_DIFF_ONLY")"
  data.semantic_diff_binary "$(bool_word "$SEMANTIC_DIFF_BINARY")"
  data.semantic_unknown_change_class "$SEMANTIC_UNKNOWN_CHANGE_CLASS"
)

if [ -n "${SEMANTIC_DIFF_CONFIDENCE_ROOT:-}" ]; then
  COMMON_OPTS+=(data.semantic_diff_confidence_root "$SEMANTIC_DIFF_CONFIDENCE_ROOT")
fi
if [ -n "${SEMANTIC_DIFF_CONFIDENCE_PHASE:-}" ]; then
  COMMON_OPTS+=(data.semantic_diff_confidence_phase "$SEMANTIC_DIFF_CONFIDENCE_PHASE")
fi

if [ -n "$USE_CONTENT_WORD_WEIGHT" ]; then
  COMMON_OPTS+=(train.use_content_word_weight "$(bool_word "$USE_CONTENT_WORD_WEIGHT")")
fi
if [ -n "$NORMALIZE_CONTENT_WORD_WEIGHTS" ]; then
  COMMON_OPTS+=(train.normalize_content_word_weights "$(bool_word "$NORMALIZE_CONTENT_WORD_WEIGHTS")")
fi

if [ -n "$SEMANTIC_FUSION_GAMMA_INIT" ]; then
  COMMON_OPTS+=(model.semantic_fusion_gamma_init "$SEMANTIC_FUSION_GAMMA_INIT")
fi
if [ -n "$SEMANTIC_FUSION_GAMMA_MAX" ]; then
  COMMON_OPTS+=(model.semantic_fusion_gamma_max "$SEMANTIC_FUSION_GAMMA_MAX")
fi
if [ -n "$SEMANTIC_FUSION_NORM_MODE" ]; then
  COMMON_OPTS+=(model.semantic_fusion_norm_mode "$SEMANTIC_FUSION_NORM_MODE")
fi
if [ -n "$SEMANTIC_FUSION_SPARSE_CHANGE_TOKENS" ]; then
  COMMON_OPTS+=(model.semantic_fusion_sparse_change_tokens "$(bool_word "$SEMANTIC_FUSION_SPARSE_CHANGE_TOKENS")")
fi
if [ -n "$SEMANTIC_FUSION_RELIABILITY_GATE" ]; then
  COMMON_OPTS+=(model.semantic_fusion_reliability_gate "$(bool_word "$SEMANTIC_FUSION_RELIABILITY_GATE")")
fi
if [ -n "$SEMANTIC_FUSION_RELIABILITY_GATE_BIAS" ]; then
  COMMON_OPTS+=(model.semantic_fusion_reliability_gate_bias "$SEMANTIC_FUSION_RELIABILITY_GATE_BIAS")
fi
if [ -n "$SEMANTIC_FUSION_GLOBAL_TOKEN" ]; then
  COMMON_OPTS+=(model.semantic_fusion_global_token "$(bool_word "$SEMANTIC_FUSION_GLOBAL_TOKEN")")
fi
if [ -n "$SEMANTIC_FUSION_GLOBAL_TOKEN_MODE" ]; then
  COMMON_OPTS+=(model.semantic_fusion_global_token_mode "$SEMANTIC_FUSION_GLOBAL_TOKEN_MODE")
fi
if [ -n "$SEMANTIC_FUSION_GATE_WHOLE_ADAPTER" ]; then
  COMMON_OPTS+=(model.semantic_fusion_gate_whole_adapter "$(bool_word "$SEMANTIC_FUSION_GATE_WHOLE_ADAPTER")")
fi
if [ -n "${SEMANTIC_FUSION_VISUAL_CONSISTENCY_GATE:-}" ]; then
  COMMON_OPTS+=(model.semantic_fusion_visual_consistency_gate "$(bool_word "$SEMANTIC_FUSION_VISUAL_CONSISTENCY_GATE")")
fi
if [ -n "${SEMANTIC_FUSION_VISUAL_FALLBACK:-}" ]; then
  COMMON_OPTS+=(model.semantic_fusion_visual_fallback "$(bool_word "$SEMANTIC_FUSION_VISUAL_FALLBACK")")
fi
if [ -n "${SEMANTIC_FUSION_WARMUP_STEPS:-}" ]; then
  COMMON_OPTS+=(model.semantic_fusion_warmup_steps "$SEMANTIC_FUSION_WARMUP_STEPS")
fi
if [ -n "$CONTENT_WORD_WEIGHT" ]; then
  COMMON_OPTS+=(train.content_word_weight "$CONTENT_WORD_WEIGHT")
fi
if [ -n "$NUM_MASK_CLASSES" ]; then
  COMMON_OPTS+=(data.num_mask_classes "$NUM_MASK_CLASSES" model.num_mask_classes "$NUM_MASK_CLASSES")
fi
if [ -n "$NUM_SEMANTIC_CLASSES" ]; then
  COMMON_OPTS+=(data.num_semantic_classes "$NUM_SEMANTIC_CLASSES" model.num_semantic_classes "$NUM_SEMANTIC_CLASSES")
fi
if [ -n "$SEMANTIC_MAP_ROOT" ]; then COMMON_OPTS+=(data.semantic_map_root "$SEMANTIC_MAP_ROOT"); fi
if [ -n "$SEMANTIC_BEFORE_PHASE" ]; then COMMON_OPTS+=(data.semantic_before_phase "$SEMANTIC_BEFORE_PHASE"); fi
if [ -n "$SEMANTIC_AFTER_PHASE" ]; then COMMON_OPTS+=(data.semantic_after_phase "$SEMANTIC_AFTER_PHASE"); fi
if [ -n "$SEMANTIC_DIFF_ROOT" ]; then COMMON_OPTS+=(data.semantic_diff_root "$SEMANTIC_DIFF_ROOT"); fi
if [ -n "$SEMANTIC_DIFF_PHASE" ]; then COMMON_OPTS+=(data.semantic_diff_phase "$SEMANTIC_DIFF_PHASE"); fi
if [ -n "$MASK_LOSS_TYPE" ]; then
  COMMON_OPTS+=(train.mask_loss_type "$MASK_LOSS_TYPE")
fi
if [ -n "$SEMANTIC_LOSS_TYPE" ]; then
  COMMON_OPTS+=(train.semantic_loss_type "$SEMANTIC_LOSS_TYPE")
fi
if [ -n "$MAX_ITER" ]; then
  COMMON_OPTS+=(train.max_iter "$MAX_ITER")
fi
if [ -n "$SNAPSHOT_INTERVAL" ]; then
  COMMON_OPTS+=(train.snapshot_interval "$SNAPSHOT_INTERVAL")
fi
if [ -n "$FINETUNE_STEPS" ]; then
  COMMON_OPTS+=(train.finetune_steps "$FINETUNE_STEPS")
fi
if [ -n "$SAVE_INTERVAL" ]; then
  COMMON_OPTS+=(train.save_interval "$SAVE_INTERVAL")
fi
if [ -n "$EVAL_INTERVAL" ]; then
  COMMON_OPTS+=(train.eval_interval "$EVAL_INTERVAL")
fi
if [ -n "$LOG_INTERVAL" ]; then
  COMMON_OPTS+=(train.log_interval "$LOG_INTERVAL")
fi
if [ -n "$BATCH_SIZE" ]; then
  COMMON_OPTS+=(data.train.batch_size "$BATCH_SIZE")
fi
if [ -n "$LR" ]; then
  COMMON_OPTS+=(train.optim.lr "$LR")
fi
if [ -n "$LR_SCHEDULER" ]; then
  COMMON_OPTS+=(train.optim.scheduler "$LR_SCHEDULER")
fi
if [ -n "$LR_WARMUP_STEPS" ]; then
  COMMON_OPTS+=(train.optim.warmup_steps "$LR_WARMUP_STEPS")
fi
if [ -n "$MIN_LR_RATIO" ]; then
  COMMON_OPTS+=(train.optim.min_lr_ratio "$MIN_LR_RATIO")
fi
if [ -n "$PRETRAINED_LR_SCALE" ]; then
  COMMON_OPTS+=(train.optim.pretrained_lr_scale "$PRETRAINED_LR_SCALE")
fi
if [ -n "$INIT_CHECKPOINT" ]; then
  COMMON_OPTS+=(train.init_checkpoint "$INIT_CHECKPOINT")
fi
if is_true "$PAPER_SELECTION_MODE"; then
  COMMON_OPTS+=(train.paper_selection_mode True)
fi

TRAIN_ARGS=(--cfg "$BASE_CFG" --model "$MODEL_TYPE" --dataset "$DATASET" --data_root "$DATA_ROOT" --output_dir "$EXP_PATH")
if [ -n "$FEATURE_ROOT" ]; then TRAIN_ARGS+=(--feature_root "$FEATURE_ROOT"); fi
if is_true "$USE_CHANGE_MASK"; then TRAIN_ARGS+=(--use_change_mask); fi
if [ -n "$MASK_TYPE" ]; then TRAIN_ARGS+=(--mask_type "$MASK_TYPE"); fi
if [ -n "$NUM_MASK_CLASSES" ]; then TRAIN_ARGS+=(--num_mask_classes "$NUM_MASK_CLASSES"); fi
if is_true "$USE_SEMANTIC_MAPS"; then TRAIN_ARGS+=(--use_semantic_maps); fi
if [ -n "$SEMANTIC_INPUT_MODE" ]; then TRAIN_ARGS+=(--semantic_input_mode "$SEMANTIC_INPUT_MODE"); fi
if [ -n "$NUM_SEMANTIC_CLASSES" ]; then TRAIN_ARGS+=(--num_semantic_classes "$NUM_SEMANTIC_CLASSES"); fi
if [ -n "$SEMANTIC_MAP_ROOT" ]; then TRAIN_ARGS+=(--semantic_map_root "$SEMANTIC_MAP_ROOT"); fi
if [ -n "$SEMANTIC_BEFORE_PHASE" ]; then TRAIN_ARGS+=(--semantic_before_phase "$SEMANTIC_BEFORE_PHASE"); fi
if [ -n "$SEMANTIC_AFTER_PHASE" ]; then TRAIN_ARGS+=(--semantic_after_phase "$SEMANTIC_AFTER_PHASE"); fi
if [ -n "$SEMANTIC_DIFF_ROOT" ]; then TRAIN_ARGS+=(--semantic_diff_root "$SEMANTIC_DIFF_ROOT"); fi
if [ -n "$SEMANTIC_DIFF_PHASE" ]; then TRAIN_ARGS+=(--semantic_diff_phase "$SEMANTIC_DIFF_PHASE"); fi
if is_true "$SEMANTIC_DIFF_ONLY"; then TRAIN_ARGS+=(--semantic_diff_only); fi
if is_true "$SEMANTIC_DIFF_BINARY"; then TRAIN_ARGS+=(--semantic_diff_binary); fi
if is_true "$USE_AUX_SEMANTIC"; then TRAIN_ARGS+=(--use_aux_semantic); fi
if is_true "$ENABLE_AUX_MASK"; then TRAIN_ARGS+=(--use_aux_mask); fi
if is_true "$USE_SEMANTIC_PARTIAL_DETACH"; then TRAIN_ARGS+=(--use_semantic_partial_detach); fi
if [ -n "$USE_CONTENT_WORD_WEIGHT" ] && is_true "$USE_CONTENT_WORD_WEIGHT"; then TRAIN_ARGS+=(--use_content_word_weight); fi
if [ -n "$NORMALIZE_CONTENT_WORD_WEIGHTS" ] && is_true "$NORMALIZE_CONTENT_WORD_WEIGHTS"; then TRAIN_ARGS+=(--normalize_content_word_weights); fi
if is_true "$USE_FEATURE_REWEIGHT"; then TRAIN_ARGS+=(--use_feature_reweight); else TRAIN_ARGS+=(--no_feature_reweight); fi
TRAIN_ARGS+=(--no_semantic_hard_gate)
if is_true "$DETACH_REWEIGHT_MASK"; then TRAIN_ARGS+=(--detach_reweight_mask); fi
if [ -n "$INIT_CHECKPOINT" ]; then TRAIN_ARGS+=(--init_checkpoint "$INIT_CHECKPOINT"); fi
if is_true "$FINETUNE_DECODER_ONLY"; then TRAIN_ARGS+=(--finetune_decoder_only); fi
if is_true "$PAPER_SELECTION_MODE"; then TRAIN_ARGS+=(--paper_selection_mode); fi

{
  echo "========== training $EXP_NAME =========="
  echo "dataset=$DATASET"
  echo "data_root=$DATA_ROOT"
  echo "base_cfg=$BASE_CFG"
  echo "model=$MODEL_TYPE"
  echo "finetune_decoder_only=$(bool_word "$FINETUNE_DECODER_ONLY")"
  "$PYTHON" train_card_spot.py "${TRAIN_ARGS[@]}" "${COMMON_OPTS[@]}"
} 2>&1 | tee "$LOG_PATH"
