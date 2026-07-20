#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

FORCE=0
DRY_RUN=0
SKIP_TRAIN=0
SKIP_RESELECT=0

usage() {
  echo 'Usage: bash scripts/run_7_6_followup_all.sh [--force] [--skip_train] [--skip_reselect] [--dry_run]' >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --overwrite)
      echo '7.6 refuses dirty in-place overwrite; archive old experiment directories first.' >&2
      exit 2 ;;
    --force) FORCE=1; shift ;;
    --skip_train) SKIP_TRAIN=1; shift ;;
    --skip_reselect) SKIP_RESELECT=1; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

train_args=()
reselect_args=()
[ "$FORCE" -eq 0 ] || reselect_args+=(--force)
if [ "$DRY_RUN" -eq 1 ]; then
  train_args+=(--dry_run)
  reselect_args+=(--dry_run)
fi

if [ "$SKIP_TRAIN" -eq 0 ]; then
  bash scripts/run_7_6_followup_train.sh "${train_args[@]}"
fi
if [ "$SKIP_RESELECT" -eq 0 ]; then
  bash scripts/run_7_6_followup_reselect.sh "${reselect_args[@]}"
fi

echo '7.6 stopped after validation lock and full audit.'
echo 'Inspect experiments/7_6_locked_manifest.json, then run test once:'
echo '  bash scripts/run_7_6_followup_test_locked.sh'
