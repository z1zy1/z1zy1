#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

SKIP_TRAIN=0
ONLY_DATASET=""
ONLY_EXP=""
OVERWRITE=0

usage() {
  echo "Usage: bash scripts/run_all_required_paper_experiments.sh [--skip_train] [--only_dataset levir_mci|second_cc] [--only_exp EXP] [--overwrite]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip_train) SKIP_TRAIN=1; shift ;;
    --only_dataset) ONLY_DATASET="$2"; shift 2 ;;
    --only_exp) ONLY_EXP="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

ARGS=()
if [ "$SKIP_TRAIN" -eq 1 ]; then ARGS+=(--skip_train); fi
if [ -n "$ONLY_DATASET" ]; then ARGS+=(--only_dataset "$ONLY_DATASET"); fi
if [ -n "$ONLY_EXP" ]; then ARGS+=(--only_exp "$ONLY_EXP"); fi
if [ "$OVERWRITE" -eq 1 ]; then ARGS+=(--overwrite); fi

if [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "levir_mci" ]; then
  bash scripts/run_levir_mci_required_experiments.sh "${ARGS[@]}"
fi
if [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "second_cc" ]; then
  bash scripts/run_second_cc_required_experiments.sh "${ARGS[@]}"
fi

SUMMARY_ARGS=()
if [ -n "$ONLY_DATASET" ]; then SUMMARY_ARGS+=(--dataset "$ONLY_DATASET"); fi
python scripts/summarize_paper_required_experiments.py "${SUMMARY_ARGS[@]}"
