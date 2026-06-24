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
CSV_PATH="${CSV_PATH:-}"
FORCE_INFER="${FORCE_INFER:-0}"

DATASET="${DATASET:-}"
DATA_ROOT="${DATA_ROOT:-}"
FEATURE_ROOT="${FEATURE_ROOT:-}"
MODEL_TYPE="${MODEL_TYPE:-sgc_card}"
USE_CHANGE_MASK="${USE_CHANGE_MASK:-1}"
MASK_TYPE="${MASK_TYPE:-binary}"
NUM_MASK_CLASSES="${NUM_MASK_CLASSES:-}"
USE_SEMANTIC_MAPS="${USE_SEMANTIC_MAPS:-0}"
SEMANTIC_INPUT_MODE="${SEMANTIC_INPUT_MODE:-none}"
NUM_SEMANTIC_CLASSES="${NUM_SEMANTIC_CLASSES:-}"
ENABLE_AUX_MASK="${ENABLE_AUX_MASK:-1}"
USE_AUX_SEMANTIC="${USE_AUX_SEMANTIC:-1}"
USE_SEMANTIC_PARTIAL_DETACH="${USE_SEMANTIC_PARTIAL_DETACH:-1}"
ALLOW_MISSING_PSEUDO_MASK="${ALLOW_MISSING_PSEUDO_MASK:-1}"
EVAL_CHANGE_NOCHANGE_SPLIT="${EVAL_CHANGE_NOCHANGE_SPLIT:-0}"
CHANGEFLAG_JSON="${CHANGEFLAG_JSON:-}"
PAPER_SELECTION_MODE="${PAPER_SELECTION_MODE:-0}"
MASK_LOSS_TYPE="${MASK_LOSS_TYPE:-}"
SEMANTIC_LOSS_TYPE="${SEMANTIC_LOSS_TYPE:-}"

usage() {
  echo "Usage: bash scripts/eval_all_snapshots_sgc_card.sh [--exp_dir experiments/<exp_name>] [--force]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --exp_dir)
      EXP_PATH="${2%/}"
      EXP_DIR="$(dirname "$EXP_PATH")"
      EXP_NAME="$(basename "$EXP_PATH")"
      shift 2
      ;;
    --force|--force_infer)
      FORCE_INFER=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

bool_word() {
  case "$1" in
    1|true|True|TRUE|yes|Yes|YES|on|ON) echo True ;;
    *) echo False ;;
  esac
}

is_true() {
  [ "$(bool_word "$1")" = "True" ]
}

CSV_PATH="${CSV_PATH:-$EXP_PATH/eval_snapshots.csv}"

LMASK="${LMASK:-0.003}"
LSEM="${LSEM:-0.005}"
SEMANTIC_DETACH_RATIO="${SEMANTIC_DETACH_RATIO:-0.5}"
USE_FEATURE_REWEIGHT="${USE_FEATURE_REWEIGHT:-1}"
REWEIGHT_ALPHA="${REWEIGHT_ALPHA:-0.2}"

FEATURE_REWEIGHT_BOOL="$(bool_word "$USE_FEATURE_REWEIGHT")"
ENABLE_AUX_MASK_BOOL="$(bool_word "$ENABLE_AUX_MASK")"
USE_AUX_SEMANTIC_BOOL="$(bool_word "$USE_AUX_SEMANTIC")"
USE_SEMANTIC_PARTIAL_DETACH_BOOL="$(bool_word "$USE_SEMANTIC_PARTIAL_DETACH")"
USE_CHANGE_MASK_BOOL="$(bool_word "$USE_CHANGE_MASK")"
USE_SEMANTIC_MAPS_BOOL="$(bool_word "$USE_SEMANTIC_MAPS")"
ALLOW_MISSING_PSEUDO_MASK_BOOL="$(bool_word "$ALLOW_MISSING_PSEUDO_MASK")"

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
  model.enable_aux_mask "$ENABLE_AUX_MASK_BOOL"
  train.use_semantic_aux "$USE_AUX_SEMANTIC_BOOL"
  train.lambda_mask "$LMASK"
  train.lambda_semantic "$LSEM"
  train.use_semantic_partial_detach "$USE_SEMANTIC_PARTIAL_DETACH_BOOL"
  train.semantic_detach_ratio "$SEMANTIC_DETACH_RATIO"
  train.use_feature_reweight "$FEATURE_REWEIGHT_BOOL"
  train.reweight_alpha "$REWEIGHT_ALPHA"
  data.allow_missing_pseudo_mask "$ALLOW_MISSING_PSEUDO_MASK_BOOL"
  data.use_change_mask "$USE_CHANGE_MASK_BOOL"
  data.mask_type "$MASK_TYPE"
  data.use_semantic_maps "$USE_SEMANTIC_MAPS_BOOL"
  model.semantic_input_mode "$SEMANTIC_INPUT_MODE"
)

if [ -n "$NUM_MASK_CLASSES" ]; then
  COMMON_OPTS+=(data.num_mask_classes "$NUM_MASK_CLASSES" model.num_mask_classes "$NUM_MASK_CLASSES")
fi
if [ -n "$NUM_SEMANTIC_CLASSES" ]; then
  COMMON_OPTS+=(data.num_semantic_classes "$NUM_SEMANTIC_CLASSES" model.num_semantic_classes "$NUM_SEMANTIC_CLASSES")
fi
if [ -n "$MASK_LOSS_TYPE" ]; then
  COMMON_OPTS+=(train.mask_loss_type "$MASK_LOSS_TYPE")
fi
if [ -n "$SEMANTIC_LOSS_TYPE" ]; then
  COMMON_OPTS+=(train.semantic_loss_type "$SEMANTIC_LOSS_TYPE")
fi
if is_true "$PAPER_SELECTION_MODE"; then
  COMMON_OPTS+=(train.paper_selection_mode True)
fi

TEST_ARGS=(--cfg "$BASE_CFG" --model "$MODEL_TYPE" --gpu "$PYTORCH_GPU")
if [ -n "$DATASET" ]; then TEST_ARGS+=(--dataset "$DATASET"); fi
if [ -n "$DATA_ROOT" ]; then TEST_ARGS+=(--data_root "$DATA_ROOT"); fi
if [ -n "$FEATURE_ROOT" ]; then TEST_ARGS+=(--feature_root "$FEATURE_ROOT"); fi
if is_true "$USE_CHANGE_MASK"; then TEST_ARGS+=(--use_change_mask); fi
if [ -n "$MASK_TYPE" ]; then TEST_ARGS+=(--mask_type "$MASK_TYPE"); fi
if [ -n "$NUM_MASK_CLASSES" ]; then TEST_ARGS+=(--num_mask_classes "$NUM_MASK_CLASSES"); fi
if is_true "$USE_SEMANTIC_MAPS"; then TEST_ARGS+=(--use_semantic_maps); fi
if [ -n "$SEMANTIC_INPUT_MODE" ]; then TEST_ARGS+=(--semantic_input_mode "$SEMANTIC_INPUT_MODE"); fi
if [ -n "$NUM_SEMANTIC_CLASSES" ]; then TEST_ARGS+=(--num_semantic_classes "$NUM_SEMANTIC_CLASSES"); fi
if is_true "$EVAL_CHANGE_NOCHANGE_SPLIT"; then TEST_ARGS+=(--eval_change_nochange_split); fi
if is_true "$PAPER_SELECTION_MODE"; then TEST_ARGS+=(--paper_selection_mode); fi

for snapshot in "${SNAPSHOTS[@]}"; do
  snapshot_name="$(basename "$snapshot")"
  snapshot_stem="${snapshot_name%.*}"
  result_dir="$EXP_PATH/eval_sents/$snapshot_stem"
  result_json="$result_dir/sc_results.json"
  mkdir -p "$result_dir"

  echo "========== eval validation snapshot $snapshot ==========" | tee -a "$LOG_PATH"
  if [ "$FORCE_INFER" = "1" ] || [ ! -f "$result_json" ]; then
    python test_card_spot.py \
      "${TEST_ARGS[@]}" \
      --snapshot_path "$snapshot" \
      --split val \
      --result_json "$result_json" \
      "${COMMON_OPTS[@]}" \
      2>&1 | tee -a "$LOG_PATH"
  else
    echo "Using existing validation captions: $result_json" | tee -a "$LOG_PATH"
  fi

  METRIC_ARGS=(
    --anno "$ANNO"
    --result_json "$result_json"
    --snapshot_path "$snapshot"
    --baseline eval
    --csv "$CSV_PATH"
    --append
    --output_json "$result_dir/metrics.json"
  )
  if is_true "$EVAL_CHANGE_NOCHANGE_SPLIT"; then
    if [ -z "$CHANGEFLAG_JSON" ]; then
      echo "CHANGEFLAG_JSON is required when EVAL_CHANGE_NOCHANGE_SPLIT=1" >&2
      exit 1
    fi
    METRIC_ARGS+=(--eval_change_nochange_split --changeflag_json "$CHANGEFLAG_JSON" --split val --group_output_dir "$result_dir")
  fi

  python scripts/sgc_card_metrics.py "${METRIC_ARGS[@]}" 2>&1 | tee -a "$LOG_PATH"
done

echo "Wrote validation snapshot CSV: $CSV_PATH"
