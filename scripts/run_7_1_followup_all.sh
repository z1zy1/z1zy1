#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
SUMMARY_CSV="${SUMMARY_CSV:-$EXP_ROOT/paper_required_experiments_summary.csv}"
ONLY_EXP=""
OVERWRITE=0
FORCE=0
DRY_RUN=0
SKIP_TRAIN=0
SKIP_TEST=0

usage() {
  echo "Usage: bash scripts/run_7_1_followup_all.sh [--only_exp EXP] [--overwrite] [--force] [--skip_train] [--skip_test] [--dry_run]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --only_exp) ONLY_EXP="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --force) FORCE=1; shift ;;
    --skip_train) SKIP_TRAIN=1; shift ;;
    --skip_test) SKIP_TEST=1; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$EXP_ROOT"

CHECK_ARGS=(--summary_csv "$SUMMARY_CSV" --output "$EXP_ROOT/summary_check_report.txt")
TRAIN_ARGS=()
TEST_ARGS=()
if [ -n "$ONLY_EXP" ]; then
  TRAIN_ARGS+=(--only_exp "$ONLY_EXP")
  TEST_ARGS+=(--only_exp "$ONLY_EXP")
fi
if [ "$OVERWRITE" -eq 1 ]; then TRAIN_ARGS+=(--overwrite); fi
if [ "$FORCE" -eq 1 ]; then TEST_ARGS+=(--force); fi
if [ "$DRY_RUN" -eq 1 ]; then
  TRAIN_ARGS+=(--dry_run)
  TEST_ARGS+=(--dry_run)
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: $PYTHON scripts/check_summary_consistency.py ${CHECK_ARGS[*]}"
else
  "$PYTHON" scripts/check_summary_consistency.py "${CHECK_ARGS[@]}"
fi

if [ "$SKIP_TRAIN" -eq 0 ]; then
  bash scripts/run_7_1_followup_train.sh "${TRAIN_ARGS[@]}"
else
  echo "Skipping follow-up training."
fi

if [ "$SKIP_TEST" -eq 0 ]; then
  bash scripts/run_7_1_followup_test.sh "${TEST_ARGS[@]}"
else
  echo "Skipping follow-up testing and summary update."
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: $PYTHON scripts/check_summary_consistency.py ${CHECK_ARGS[*]}"
else
  "$PYTHON" scripts/check_summary_consistency.py "${CHECK_ARGS[@]}"
fi

echo "7/1 follow-up all flow complete."