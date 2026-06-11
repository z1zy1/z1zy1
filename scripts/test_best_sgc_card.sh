#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_PATH="${EXP_PATH:-./experiments/sgc_card_lm003_ls005_pd05_rw02_warmup}"

usage() {
  echo "Usage: bash scripts/test_best_sgc_card.sh [EXP_DIR] [--exp_dir EXP_DIR]" >&2
}

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  EXP_PATH="$1"
  shift
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --exp_dir)
      EXP_PATH="$2"
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

EXP_PATH="${EXP_PATH%/}"
BEST_SNAPSHOT=""
SELECTED_BY=""

read_best_json() {
  local json_path="$1"
  JSON_PATH="$json_path" python -c 'import json, os; print(json.load(open(os.environ["JSON_PATH"], encoding="utf-8"))["best_snapshot"])'
}

if [ -f "$EXP_PATH/best_snapshot_v2.json" ]; then
  BEST_SNAPSHOT="$(read_best_json "$EXP_PATH/best_snapshot_v2.json")"
  SELECTED_BY="v2"
elif [ -f "$EXP_PATH/best_balanced_v2.pth" ]; then
  BEST_SNAPSHOT="$EXP_PATH/best_balanced_v2.pth"
  SELECTED_BY="v2"
elif [ -f "$EXP_PATH/best_snapshot.json" ]; then
  BEST_SNAPSHOT="$(read_best_json "$EXP_PATH/best_snapshot.json")"
  SELECTED_BY="old"
elif [ -f "$EXP_PATH/best_balanced.pth" ]; then
  BEST_SNAPSHOT="$EXP_PATH/best_balanced.pth"
  SELECTED_BY="old"
else
  echo "No best snapshot found under $EXP_PATH." >&2
  echo "Expected one of: best_snapshot_v2.json, best_balanced_v2.pth, best_snapshot.json, best_balanced.pth" >&2
  exit 1
fi

echo "Using snapshot selected by: $SELECTED_BY"
echo "Checkpoint path: $BEST_SNAPSHOT"

bash scripts/test_specific_snapshot_sgc_card.sh \
  --exp_dir "$EXP_PATH" \
  --checkpoint "$BEST_SNAPSHOT" \
  --tag best
