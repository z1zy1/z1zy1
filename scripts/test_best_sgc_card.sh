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
BEST_JSON="${BEST_JSON:-$EXP_PATH/best_snapshot.json}"

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

if [ ! -f "$BEST_JSON" ]; then
  echo "Best snapshot JSON does not exist: $BEST_JSON" >&2
  exit 1
fi

BEST_SNAPSHOT="$(
  BEST_JSON_PATH="$BEST_JSON" python -c 'import json, os; print(json.load(open(os.environ["BEST_JSON_PATH"], encoding="utf-8"))["best_snapshot"])'
)"

RESULT_JSON="$EXP_PATH/test_output/captions/test/sc_results.json"
RESULT_METRICS_JSON="$EXP_PATH/test_best_result.json"
RESULT_METRICS_TXT="$EXP_PATH/test_best_result.txt"
mkdir -p "$(dirname "$RESULT_JSON")"

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

echo "========== test best snapshot $BEST_SNAPSHOT =========="
python test_card_spot.py \
  --cfg "$BASE_CFG" \
  --snapshot_path "$BEST_SNAPSHOT" \
  --split test \
  --result_json "$RESULT_JSON" \
  --gpu "$PYTORCH_GPU" \
  "${COMMON_OPTS[@]}" \
  2>&1 | tee "$EXP_PATH/test_best.log"

echo "========== score test output =========="
python scripts/sgc_card_metrics.py \
  --anno "$ANNO" \
  --result_json "$RESULT_JSON" \
  --snapshot_path "$BEST_SNAPSHOT" \
  --baseline test \
  --output_json "$RESULT_METRICS_JSON" \
  --output_txt "$RESULT_METRICS_TXT" \
  2>&1 | tee -a "$EXP_PATH/test_best.log"

echo "Wrote test metrics:"
echo "  $RESULT_METRICS_JSON"
echo "  $RESULT_METRICS_TXT"
