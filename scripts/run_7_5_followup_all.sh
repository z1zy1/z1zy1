#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

OVERWRITE=0
FORCE=0
DRY_RUN=0
SKIP_TRAIN=0
SKIP_RESELECT=0

usage() {
  echo "Usage: bash scripts/run_7_5_followup_all.sh [--overwrite] [--force] [--skip_train] [--skip_reselect] [--dry_run]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --overwrite) OVERWRITE=1; shift ;;
    --force) FORCE=1; shift ;;
    --skip_train) SKIP_TRAIN=1; shift ;;
    --skip_reselect) SKIP_RESELECT=1; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

TRAIN_ARGS=()
RESELECT_ARGS=()
[ "$OVERWRITE" -eq 0 ] || TRAIN_ARGS+=(--overwrite)
[ "$FORCE" -eq 0 ] || RESELECT_ARGS+=(--force)
if [ "$DRY_RUN" -eq 1 ]; then
  TRAIN_ARGS+=(--dry_run)
  RESELECT_ARGS+=(--dry_run)
fi

if [ "$SKIP_TRAIN" -eq 0 ]; then
  bash scripts/run_7_5_followup_train.sh "${TRAIN_ARGS[@]}"
fi
if [ "$SKIP_RESELECT" -eq 0 ]; then
  bash scripts/run_7_5_followup_reselect.sh "${RESELECT_ARGS[@]}"
fi

echo '7.5 flow stopped after validation lock. Test is intentionally separate:'
echo '  bash scripts/run_7_5_followup_test_locked.sh'
