#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
case "${OMP_NUM_THREADS:-}" in
  ''|*[!0-9]*|0) export OMP_NUM_THREADS=1 ;;
esac
PYTORCH_GPU="${PYTORCH_GPU:-0}"

EXP_PATH=""
CHECKPOINT=""
TAG=""
BASE_CFG="${BASE_CFG:-configs/dynamic/transformer_levir_cc_sgc_card.yaml}"
ANNO="${ANNO:-./Levir-CC/levir_cc_captions_reformat.json}"

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

LMASK="${LMASK:-0.003}"
LSEM="${LSEM:-0.005}"
SEMANTIC_DETACH_RATIO="${SEMANTIC_DETACH_RATIO:-0.5}"
USE_FEATURE_REWEIGHT="${USE_FEATURE_REWEIGHT:-0}"
REWEIGHT_ALPHA="${REWEIGHT_ALPHA:-0.2}"

usage() {
  echo "Usage: bash scripts/test_specific_snapshot_sgc_card.sh --exp_dir EXP_DIR --checkpoint CHECKPOINT --tag TAG" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --exp_dir)
      EXP_PATH="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --base_cfg)
      BASE_CFG="$2"
      shift 2
      ;;
    --anno)
      ANNO="$2"
      shift 2
      ;;
    --gpu)
      PYTORCH_GPU="$2"
      shift 2
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

if [ -z "$EXP_PATH" ] || [ -z "$CHECKPOINT" ]; then
  usage
  exit 2
fi

bool_word() {
  case "$1" in
    1|true|True|TRUE|yes|Yes|YES|on|ON) echo True ;;
    *) echo False ;;
  esac
}

is_true() {
  [ "$(bool_word "$1")" = "True" ]
}

EXP_PATH="${EXP_PATH%/}"
if [ -z "$TAG" ]; then
  TAG="$(basename "$CHECKPOINT")"
  TAG="${TAG%.*}"
fi

FEATURE_REWEIGHT_BOOL="$(bool_word "$USE_FEATURE_REWEIGHT")"
ENABLE_AUX_MASK_BOOL="$(bool_word "$ENABLE_AUX_MASK")"
USE_AUX_SEMANTIC_BOOL="$(bool_word "$USE_AUX_SEMANTIC")"
USE_SEMANTIC_PARTIAL_DETACH_BOOL="$(bool_word "$USE_SEMANTIC_PARTIAL_DETACH")"
USE_CHANGE_MASK_BOOL="$(bool_word "$USE_CHANGE_MASK")"
USE_SEMANTIC_MAPS_BOOL="$(bool_word "$USE_SEMANTIC_MAPS")"
ALLOW_MISSING_PSEUDO_MASK_BOOL="$(bool_word "$ALLOW_MISSING_PSEUDO_MASK")"

EXP_DIR="$(dirname "$EXP_PATH")"
EXP_NAME="$(basename "$EXP_PATH")"
mkdir -p "$EXP_PATH"

RESOLVED_CHECKPOINT="$(
  EXP_PATH="$EXP_PATH" CHECKPOINT="$CHECKPOINT" TAG="$TAG" python -c '
import csv
import glob
import os
import re
import sys

exp_path = os.environ["EXP_PATH"]
checkpoint = os.environ["CHECKPOINT"]
tag = os.environ["TAG"]

def checkpoint_number(text):
    matches = re.findall(r"(\d+)", text or "")
    return matches[-1] if matches else None

def existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return os.path.normpath(path)
    return None

paths = [
    checkpoint,
    os.path.abspath(checkpoint),
    os.path.join(exp_path, checkpoint),
    os.path.join(exp_path, os.path.basename(checkpoint)),
    os.path.join(exp_path, "snapshots", os.path.basename(checkpoint)),
    os.path.join(exp_path, "checkpoints", os.path.basename(checkpoint)),
]
root, ext = os.path.splitext(checkpoint)
if ext.lower() == ".pth":
    paths.append(root + ".pt")
elif ext.lower() == ".pt":
    paths.append(root + ".pth")

found = existing(paths)
if found:
    print(found)
    sys.exit(0)

number = checkpoint_number(checkpoint) or checkpoint_number(tag)
patterns = []
basename_no_ext = os.path.splitext(os.path.basename(checkpoint))[0]
if basename_no_ext:
    patterns += [
        os.path.join(exp_path, "snapshots", basename_no_ext + ".*"),
        os.path.join(exp_path, "checkpoints", basename_no_ext + ".*"),
        os.path.join(exp_path, basename_no_ext + ".*"),
    ]
if number:
    patterns += [
        os.path.join(exp_path, "snapshots", "*" + number + "*.pt"),
        os.path.join(exp_path, "snapshots", "*" + number + "*.pth"),
        os.path.join(exp_path, "checkpoints", "*" + number + "*.pt"),
        os.path.join(exp_path, "checkpoints", "*" + number + "*.pth"),
        os.path.join(exp_path, "*" + number + "*.pt"),
        os.path.join(exp_path, "*" + number + "*.pth"),
    ]

csv_path = os.path.join(exp_path, "eval_snapshots.csv")
if os.path.exists(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row.get("snapshot_path", "")
            if (number and number in os.path.basename(raw)) or tag in os.path.basename(raw):
                patterns.append(raw)
                patterns.append(os.path.join(exp_path, raw))
                patterns.append(os.path.join(exp_path, "snapshots", os.path.basename(raw)))

matches = []
for pattern in patterns:
    matches.extend(glob.glob(pattern))
matches = sorted({os.path.normpath(path) for path in matches if os.path.exists(path)})
if matches:
    print(matches[0])
    sys.exit(0)

print("ERROR: checkpoint not found: %s" % checkpoint, file=sys.stderr)
print("Tried experiment directory: %s" % exp_path, file=sys.stderr)
sys.exit(1)
'
)"

RESULT_JSON="$EXP_PATH/test_output/captions/$TAG/sc_results.json"
RESULT_METRICS_JSON="$EXP_PATH/test_${TAG}_result.json"
RESULT_METRICS_TXT="$EXP_PATH/test_${TAG}_result.txt"
LOG_PATH="$EXP_PATH/test_${TAG}.log"
mkdir -p "$(dirname "$RESULT_JSON")"

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

METRIC_ARGS=(
  --anno "$ANNO"
  --result_json "$RESULT_JSON"
  --snapshot_path "$RESOLVED_CHECKPOINT"
  --baseline test
  --output_json "$RESULT_METRICS_JSON"
  --output_txt "$RESULT_METRICS_TXT"
)
if is_true "$EVAL_CHANGE_NOCHANGE_SPLIT"; then
  if [ -z "$CHANGEFLAG_JSON" ]; then
    echo "CHANGEFLAG_JSON is required when EVAL_CHANGE_NOCHANGE_SPLIT=1" >&2
    exit 1
  fi
  METRIC_ARGS+=(--eval_change_nochange_split --changeflag_json "$CHANGEFLAG_JSON" --split test --group_output_dir "$EXP_PATH")
fi

echo "Using checkpoint: $RESOLVED_CHECKPOINT" | tee "$LOG_PATH"
python test_card_spot.py \
  "${TEST_ARGS[@]}" \
  --snapshot_path "$RESOLVED_CHECKPOINT" \
  --split test \
  --result_json "$RESULT_JSON" \
  "${COMMON_OPTS[@]}" \
  2>&1 | tee -a "$LOG_PATH"

python scripts/sgc_card_metrics.py "${METRIC_ARGS[@]}" 2>&1 | tee -a "$LOG_PATH"

echo "Wrote:"
echo "  $RESULT_METRICS_TXT"
echo "  $RESULT_METRICS_JSON"
