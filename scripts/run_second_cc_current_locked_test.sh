#!/usr/bin/env bash
set -euo pipefail

# Lock and test the existing three-seed pd=0.5, lr=2e-4 winner before retraining.

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
RUN_ROOT="${CURRENT_RUN_ROOT:-$EXP_ROOT/mci_transfer_crossattn_matrix}"
DATA_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
BASELINE_VAL="${SECOND_CC_BASELINE_VAL_METRICS:-$EXP_ROOT/second_cc_card_rgb_baseline/baseline_best_checkpoint.json}"
LOCK_MANIFEST="${CURRENT_LOCK_MANIFEST:-$EXP_ROOT/second_cc_current_mci_lock.json}"
DRY_RUN=0

if [ "${1:-}" = '--dry-run' ] || [ "${1:-}" = '--dry_run' ]; then
  DRY_RUN=1
  shift
fi
[ "$#" -eq 0 ] || { echo 'Usage: bash scripts/run_second_cc_current_locked_test.sh [--dry-run]' >&2; exit 2; }

lock_command=(
  "$PYTHON" scripts/build_second_cc_optimized_lock.py
  --exp_root "$EXP_ROOT"
  --run_root "$RUN_ROOT"
  --run_prefix second_cc_mci_pd05
  --dataset_root "$DATA_ROOT"
  --baseline "$BASELINE_VAL"
  --specs lr2e4
  --selection_strategy val_baseline_pareto
  --output "$LOCK_MANIFEST"
)

if [ "$DRY_RUN" -eq 1 ]; then
  printf 'DRY RUN:'; printf ' %q' "${lock_command[@]}"; printf '\n'
  LOCK_MANIFEST="$LOCK_MANIFEST" SUMMARY_STEM=second_cc_current_mci \
    bash scripts/run_second_cc_optimized_full_flow.sh --stage test --dry-run
  LOCK_MANIFEST="$LOCK_MANIFEST" SUMMARY_STEM=second_cc_current_mci \
    bash scripts/run_second_cc_optimized_full_flow.sh --stage summary --dry-run
else
  "${lock_command[@]}"
  LOCK_MANIFEST="$LOCK_MANIFEST" SUMMARY_STEM=second_cc_current_mci \
    bash scripts/run_second_cc_optimized_full_flow.sh --stage test
  LOCK_MANIFEST="$LOCK_MANIFEST" SUMMARY_STEM=second_cc_current_mci \
    bash scripts/run_second_cc_optimized_full_flow.sh --stage summary
fi

echo 'Current SECOND-CC MCI winner locked-test flow complete.'
