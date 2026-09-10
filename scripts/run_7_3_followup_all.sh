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
SKIP_CHECK=0
SKIP_TRAIN=0
SKIP_RESELECT=0
SKIP_TEST=0

usage() {
  echo "Usage: bash scripts/run_7_3_followup_all.sh [--only_exp EXP] [--overwrite] [--force] [--skip_check] [--skip_train] [--skip_reselect] [--skip_test] [--dry_run]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --only_exp) ONLY_EXP="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --force) FORCE=1; shift ;;
    --skip_check) SKIP_CHECK=1; shift ;;
    --skip_train) SKIP_TRAIN=1; shift ;;
    --skip_reselect) SKIP_RESELECT=1; shift ;;
    --skip_test) SKIP_TEST=1; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$EXP_ROOT"

TRAIN_ARGS=()
RESELECT_ARGS=()
TEST_ARGS=()
if [ -n "$ONLY_EXP" ]; then
  TRAIN_ARGS+=(--only_exp "$ONLY_EXP")
  RESELECT_ARGS+=(--only_exp "$ONLY_EXP")
  TEST_ARGS+=(--only_exp "$ONLY_EXP")
fi
if [ "$OVERWRITE" -eq 1 ]; then TRAIN_ARGS+=(--overwrite); fi
if [ "$FORCE" -eq 1 ]; then
  RESELECT_ARGS+=(--force)
  TEST_ARGS+=(--force)
fi
if [ "$DRY_RUN" -eq 1 ]; then
  TRAIN_ARGS+=(--dry_run)
  RESELECT_ARGS+=(--dry_run)
  TEST_ARGS+=(--dry_run)
fi

CHECK_CMD=("$PYTHON" scripts/check_finetune_steps.py \
  "$EXP_ROOT/levir_mci_short_caption_ft_500_lr01" \
  "$EXP_ROOT/levir_mci_short_caption_ft_1000_lr01" \
  --output "$EXP_ROOT/finetune_check/levir_mci_finetune_steps_report.txt")
CONSISTENCY_CMD=("$PYTHON" scripts/check_summary_consistency.py --summary_csv "$SUMMARY_CSV" --output "$EXP_ROOT/summary_check_report.txt")

if [ "$SKIP_CHECK" -eq 0 ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'
    printf ' %q' "${CHECK_CMD[@]}"
    printf '\n'
  else
    "${CHECK_CMD[@]}"
  fi
else
  echo "Skipping finetune step audit."
fi

if [ "$SKIP_TRAIN" -eq 0 ]; then
  bash scripts/run_7_3_followup_train.sh "${TRAIN_ARGS[@]}"
else
  echo "Skipping 7.3 training."
fi

if [ "$SKIP_RESELECT" -eq 0 ]; then
  bash scripts/run_7_3_followup_reselect.sh "${RESELECT_ARGS[@]}"
else
  echo "Skipping 7.3 checkpoint reselect."
fi

if [ "$SKIP_TEST" -eq 0 ]; then
  bash scripts/run_7_3_followup_test_and_summary.sh "${TEST_ARGS[@]}"
else
  echo "Skipping 7.3 testing and summary update."
fi

if [ "$DRY_RUN" -eq 1 ]; then
  printf 'DRY RUN:'
  printf ' %q' "${CONSISTENCY_CMD[@]}"
  printf '\n'
else
  "${CONSISTENCY_CMD[@]}"
fi

echo "7.3 follow-up all flow complete."