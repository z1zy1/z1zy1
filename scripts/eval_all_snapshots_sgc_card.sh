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
CSV_PATH="${CSV_PATH:-$EXP_PATH/eval_snapshots.csv}"
FORCE_INFER="${FORCE_INFER:-0}"

LMASK="${LMASK:-0.003}"
LSEM="${LSEM:-0.005}"
SEMANTIC_DETACH_RATIO="${SEMANTIC_DETACH_RATIO:-0.5}"
USE_FEATURE_REWEIGHT="${USE_FEATURE_REWEIGHT:-1}"
REWEIGHT_ALPHA="${REWEIGHT_ALPHA:-0.2}"

if [ "$USE_FEATURE_REWEIGHT" = "1" ] || [ "$USE_FEATURE_REWEIGHT" = "true" ] || [ "$USE_FEATURE_REWEIGHT" = "True" ]; then
  FEATURE_REWEIGHT_BOOL=True
else
  FEATURE_REWEIGHT_BOOL=False
fi

mkdir -p "$EXP_PATH"
rm -f "$CSV_PATH"
LOG_PATH="$EXP_PATH/eval_all_snapshots.log"
rm -f "$LOG_PATH"

SNAPSHOT_DIR="$EXP_PATH/snapshots"
if [ ! -d "$SNAPSHOT_DIR" ]; then
  echo "Snapshot directory does not exist: $SNAPSHOT_DIR" >&2
  exit 1
fi

mapfile -t SNAPSHOTS < <(find "$SNAPSHOT_DIR" -type f \( -name '*.pt' -o -name '*.pth' \) | sort -V)
if [ "${#SNAPSHOTS[@]}" -eq 0 ]; then
  echo "No .pt/.pth snapshots found under $SNAPSHOT_DIR" >&2
  exit 1
fi

COMMON_OPTS=(
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
  data.allow_missing_pseudo_mask True
)

for snapshot in "${SNAPSHOTS[@]}"; do
  snapshot_name="$(basename "$snapshot")"
  snapshot_stem="${snapshot_name%.*}"
  result_dir="$EXP_PATH/eval_sents/$snapshot_stem"
  result_json="$result_dir/sc_results.json"
  mkdir -p "$result_dir"

  echo "========== eval validation snapshot $snapshot ==========" | tee -a "$LOG_PATH"
  if [ "$FORCE_INFER" = "1" ] || [ ! -f "$result_json" ]; then
    python test_card_spot.py \
      --cfg "$BASE_CFG" \
      --snapshot_path "$snapshot" \
      --split val \
      --result_json "$result_json" \
      --gpu "$PYTORCH_GPU" \
      "${COMMON_OPTS[@]}" \
      2>&1 | tee -a "$LOG_PATH"
  else
    echo "Using existing validation captions: $result_json" | tee -a "$LOG_PATH"
  fi

  python scripts/sgc_card_metrics.py \
    --anno "$ANNO" \
    --result_json "$result_json" \
    --snapshot_path "$snapshot" \
    --baseline eval \
    --csv "$CSV_PATH" \
    --append \
    --output_json "$result_dir/metrics.json" \
    2>&1 | tee -a "$LOG_PATH"
done

echo "Wrote validation snapshot CSV: $CSV_PATH"
