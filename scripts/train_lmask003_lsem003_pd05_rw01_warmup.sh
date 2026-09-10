#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_GPU="${PYTORCH_GPU:-0}"

EXP_NAME="lmask003_lsem003_pd05_rw01_warmup"
EXP_DIR="${EXP_DIR:-./experiments}"
EXP_PATH="$EXP_DIR/$EXP_NAME"
BASE_CFG="${BASE_CFG:-configs/dynamic/transformer_levir_cc_sgc_card.yaml}"
ANNO="${ANNO:-./Levir-CC/levir_cc_captions_reformat.json}"
MAX_ITER="${MAX_ITER:-10000}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1000}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "$EXP_PATH"

if [ "$OVERWRITE" != "1" ] && [ -d "$EXP_PATH/snapshots" ] && find "$EXP_PATH/snapshots" -type f \( -name '*.pt' -o -name '*.pth' \) | grep -q .; then
  echo "Snapshots already exist under $EXP_PATH/snapshots; refusing to overwrite."
  echo "Use OVERWRITE=1 bash scripts/train_lmask003_lsem003_pd05_rw01_warmup.sh to train again."
  exit 0
fi

echo "========== train $EXP_NAME =========="
echo "lmask=0.003 lsem=0.003 semantic_detach_ratio=0.5 feature_reweight=True reweight_alpha=0.1 aux_warmup=True semantic_decay_final_ratio=0.25"

train_cmd=(
  python train_card_spot.py
  --cfg "$BASE_CFG"
  --exp_name "$EXP_NAME"
  --output_dir "$EXP_PATH"
  --use_aux_mask
  --use_aux_semantic
  --use_semantic_partial_detach
  --semantic_detach_ratio 0.5
  --lmask 0.003
  --lsem 0.003
  --use_feature_reweight
  --reweight_alpha 0.1
  --use_aux_warmup
  --aux_warmup_start_ratio 0.30
  --aux_warmup_end_ratio 0.70
  --semantic_decay_start_ratio 0.70
  --semantic_decay_final_ratio 0.25
  exp_dir "$EXP_DIR"
  exp_name "$EXP_NAME"
  gpu_id "[$PYTORCH_GPU]"
  model.enable_aux_mask True
  train.use_semantic_aux True
  train.lambda_mask 0.003
  train.lambda_semantic 0.003
  train.use_semantic_partial_detach True
  train.semantic_detach_ratio 0.5
  train.use_feature_reweight True
  train.reweight_alpha 0.1
  train.use_aux_warmup True
  train.aux_warmup_start_ratio 0.30
  train.aux_warmup_end_ratio 0.70
  train.semantic_decay_start_ratio 0.70
  train.semantic_decay_final_ratio 0.25
  train.mask_loss_type bce_dice
  train.max_iter "$MAX_ITER"
  train.snapshot_interval "$SNAPSHOT_INTERVAL"
  data.allow_missing_pseudo_mask True
  data.eval_anno_path "$ANNO"
)

printf '%q ' "${train_cmd[@]}" > "$EXP_PATH/train_command.txt"
printf '\n' >> "$EXP_PATH/train_command.txt"
"${train_cmd[@]}" 2>&1 | tee "$EXP_PATH/train.log"
