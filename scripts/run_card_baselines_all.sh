#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
DRY_RUN=0
SKIP_TRAIN=0
SKIP_SELECT=0
usage() { echo 'Usage: bash scripts/run_card_baselines_all.sh [--skip_train] [--skip_select] [--dry_run]' >&2; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip_train) SKIP_TRAIN=1; shift ;;
    --skip_select) SKIP_SELECT=1; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
train_args=(); select_args=()
[ "$DRY_RUN" -eq 0 ] || { train_args+=(--dry_run); select_args+=(--dry_run); }
[ "$SKIP_TRAIN" -eq 1 ] || bash scripts/run_card_baselines_train.sh "${train_args[@]}"
[ "$SKIP_SELECT" -eq 1 ] || bash scripts/run_card_baselines_select.sh "${select_args[@]}"
echo 'Baseline flow stopped after validation lock. Review experiments/card_baseline_locked_manifest.json, then run:'
echo '  bash scripts/run_card_baselines_test_locked.sh'
