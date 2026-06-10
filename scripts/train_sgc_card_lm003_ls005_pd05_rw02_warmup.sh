#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_GPU="${PYTORCH_GPU:-0}"

EXP_NAME="${EXP_NAME:-sgc_card_lm003_ls005_pd05_rw02_warmup}"
EXP_DIR="${EXP_DIR:-./experiments}"
EXP_PATH="$EXP_DIR/$EXP_NAME"
BASE_CFG="${BASE_CFG:-configs/dynamic/transformer_levir_cc_sgc_card.yaml}"
ANNO="${ANNO:-./Levir-CC/levir_cc_captions_reformat.json}"

MAX_ITER="${MAX_ITER:-10000}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1000}"
LMASK="${LMASK:-0.003}"
LSEM="${LSEM:-0.005}"
SEMANTIC_DETACH_RATIO="${SEMANTIC_DETACH_RATIO:-0.5}"
USE_FEATURE_REWEIGHT="${USE_FEATURE_REWEIGHT:-1}"
REWEIGHT_ALPHA="${REWEIGHT_ALPHA:-0.2}"
AUX_WARMUP_START_RATIO="${AUX_WARMUP_START_RATIO:-0.30}"
AUX_WARMUP_END_RATIO="${AUX_WARMUP_END_RATIO:-0.70}"
SEMANTIC_DECAY_START_RATIO="${SEMANTIC_DECAY_START_RATIO:-0.70}"
SEMANTIC_DECAY_FINAL_RATIO="${SEMANTIC_DECAY_FINAL_RATIO:-0.50}"

if [ "$USE_FEATURE_REWEIGHT" = "1" ] || [ "$USE_FEATURE_REWEIGHT" = "true" ] || [ "$USE_FEATURE_REWEIGHT" = "True" ]; then
  FEATURE_REWEIGHT_BOOL=True
else
  FEATURE_REWEIGHT_BOOL=False
fi

mkdir -p "$EXP_PATH"

train_cmd=(
  python train_card_spot.py
  --cfg "$BASE_CFG"
  --exp_name "$EXP_NAME"
  --output_dir "$EXP_PATH"
  --use_aux_mask
  --use_aux_semantic
  --use_semantic_partial_detach
  --semantic_detach_ratio "$SEMANTIC_DETACH_RATIO"
  --lmask "$LMASK"
  --lsem "$LSEM"
  --use_aux_warmup
  --aux_warmup_start_ratio "$AUX_WARMUP_START_RATIO"
  --aux_warmup_end_ratio "$AUX_WARMUP_END_RATIO"
  --semantic_decay_start_ratio "$SEMANTIC_DECAY_START_RATIO"
  --semantic_decay_final_ratio "$SEMANTIC_DECAY_FINAL_RATIO"
)

if [ "$FEATURE_REWEIGHT_BOOL" = "True" ]; then
  train_cmd+=(--use_feature_reweight --reweight_alpha "$REWEIGHT_ALPHA")
fi

train_cmd+=(
  exp_dir "$EXP_DIR"
  exp_name "$EXP_NAME"
  gpu_id "[$PYTORCH_GPU]"
  model.enable_aux_mask True
  train.use_semantic_aux True
  train.lambda_mask "$LMASK"
  train.lambda_semantic "$LSEM"
  train.use_semantic_partial_detach True
  train.semantic_detach_ratio "$SEMANTIC_DETACH_RATIO"
  train.use_feature_reweight "$FEATURE_REWEIGHT_BOOL"
  train.reweight_alpha "$REWEIGHT_ALPHA"
  train.use_aux_warmup True
  train.aux_warmup_start_ratio "$AUX_WARMUP_START_RATIO"
  train.aux_warmup_end_ratio "$AUX_WARMUP_END_RATIO"
  train.semantic_decay_start_ratio "$SEMANTIC_DECAY_START_RATIO"
  train.semantic_decay_final_ratio "$SEMANTIC_DECAY_FINAL_RATIO"
  train.mask_loss_type bce_dice
  train.max_iter "$MAX_ITER"
  train.snapshot_interval "$SNAPSHOT_INTERVAL"
  data.allow_missing_pseudo_mask True
  data.eval_anno_path "$ANNO"
)

printf '%q ' "${train_cmd[@]}" > "$EXP_PATH/train_command.txt"
printf '\n' >> "$EXP_PATH/train_command.txt"

echo "========== train $EXP_NAME =========="
"${train_cmd[@]}" 2>&1 | tee "$EXP_PATH/train.log"
