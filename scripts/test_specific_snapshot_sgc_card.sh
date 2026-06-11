#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_GPU="${PYTORCH_GPU:-0}"

EXP_PATH=""
CHECKPOINT=""
TAG=""
BASE_CFG="${BASE_CFG:-configs/dynamic/transformer_levir_cc_sgc_card.yaml}"
ANNO="${ANNO:-./Levir-CC/levir_cc_captions_reformat.json}"

LMASK="${LMASK:-0.003}"
LSEM="${LSEM:-0.005}"
SEMANTIC_DETACH_RATIO="${SEMANTIC_DETACH_RATIO:-0.5}"
USE_FEATURE_REWEIGHT="${USE_FEATURE_REWEIGHT:-1}"
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

EXP_PATH="${EXP_PATH%/}"
if [ -z "$TAG" ]; then
  TAG="$(basename "$CHECKPOINT")"
  TAG="${TAG%.*}"
fi

if [ "$USE_FEATURE_REWEIGHT" = "1" ] || [ "$USE_FEATURE_REWEIGHT" = "true" ] || [ "$USE_FEATURE_REWEIGHT" = "True" ]; then
  FEATURE_REWEIGHT_BOOL=True
else
  FEATURE_REWEIGHT_BOOL=False
fi

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

echo "Using checkpoint: $RESOLVED_CHECKPOINT" | tee "$LOG_PATH"
python test_card_spot.py \
  --cfg "$BASE_CFG" \
  --snapshot_path "$RESOLVED_CHECKPOINT" \
  --split test \
  --result_json "$RESULT_JSON" \
  --gpu "$PYTORCH_GPU" \
  "${COMMON_OPTS[@]}" \
  2>&1 | tee -a "$LOG_PATH"

python scripts/sgc_card_metrics.py \
  --anno "$ANNO" \
  --result_json "$RESULT_JSON" \
  --snapshot_path "$RESOLVED_CHECKPOINT" \
  --baseline test \
  --output_json "$RESULT_METRICS_JSON" \
  --output_txt "$RESULT_METRICS_TXT" \
  2>&1 | tee -a "$LOG_PATH"

echo "Wrote:"
echo "  $RESULT_METRICS_TXT"
echo "  $RESULT_METRICS_JSON"
