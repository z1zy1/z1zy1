#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
EXP_ROOT="${EXP_ROOT:-./experiments}"
LEVIR_CC_BASELINE_EXP="${LEVIR_CC_BASELINE_EXP:-card_levir_cc_baseline}"
LEVIR_MCI_BASELINE_EXP="${LEVIR_MCI_BASELINE_EXP:-levir_mci_card_baseline}"
SECOND_CC_BASELINE_EXP="${SECOND_CC_BASELINE_EXP:-second_cc_card_rgb_baseline}"
ONLY_DATASET=""
DRY_RUN=0
FAIL_LOG="${FAIL_LOG:-$EXP_ROOT/card_baseline_select_failures.log}"

usage() {
  echo 'Usage: bash scripts/run_card_baselines_select.sh [--only_dataset DATASET] [--dry_run]' >&2
}
while [ "$#" -gt 0 ]; do
  case "$1" in
    --only_dataset|--only_exp) ONLY_DATASET="$2"; shift 2 ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
case "$ONLY_DATASET" in ''|levir_cc|levir_mci|second_cc) ;; *) usage; exit 2 ;; esac

FAILURES=0
record_failure() {
  echo "FAILED: $1${2:+ - $2}" | tee -a "$FAIL_LOG"
  FAILURES=$((FAILURES + 1))
}
if [ "$DRY_RUN" -eq 0 ]; then
  mkdir -p "$EXP_ROOT"; touch "$FAIL_LOG"
  printf '\n[%s] run_card_baselines_select start\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$FAIL_LOG"
fi

manifest_args=(
  --exp_root "$EXP_ROOT"
  --levir_cc_exp "$LEVIR_CC_BASELINE_EXP"
  --levir_mci_exp "$LEVIR_MCI_BASELINE_EXP"
  --second_cc_exp "$SECOND_CC_BASELINE_EXP"
)
configure_dataset() {
  case "$1" in
    levir_cc) EXP_NAME="$LEVIR_CC_BASELINE_EXP" ;;
    levir_mci) EXP_NAME="$LEVIR_MCI_BASELINE_EXP" ;;
    second_cc) EXP_NAME="$SECOND_CC_BASELINE_EXP" ;;
  esac
}
run_one() {
  local dataset="$1"
  [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "$dataset" ] || return 0
  configure_dataset "$dataset"
  local exp_path="$EXP_ROOT/$EXP_NAME"
  local output_json="$exp_path/baseline_best_checkpoint.json"
  local command=("$PYTHON" scripts/select_best_checkpoint.py --exp_dir "$exp_path" \
    --strategy validation_best_cider --output_json "$output_json" \
    --output_txt "$exp_path/baseline_best_checkpoint.txt")
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
    printf 'DRY RUN: %q scripts/build_card_baseline_manifest.py' "$PYTHON"
    printf ' %q' "${manifest_args[@]}" --only_dataset "$dataset" --validate_only
    printf '\n'
    return
  fi
  # Selection is cheap and must always be recomputed from current val_metrics;
  # never trust a same-named JSON left by an older/incomplete run.
  if ! "${command[@]}"; then record_failure "$dataset selection" 'validation selector failed'; return; fi
  if ! "$PYTHON" scripts/build_card_baseline_manifest.py "${manifest_args[@]}" \
      --only_dataset "$dataset" --validate_only; then
    record_failure "$dataset selection" 'original-CARD/provenance audit failed'
  fi
}

for dataset in levir_cc levir_mci second_cc; do run_one "$dataset"; done
MANIFEST="$EXP_ROOT/card_baseline_locked_manifest.json"
if [ -n "$ONLY_DATASET" ]; then
  echo "Partial selection complete; locked manifest remains unchanged: $MANIFEST"
elif [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: would write validation-locked manifest $MANIFEST"
elif [ "$FAILURES" -eq 0 ]; then
  if ! "$PYTHON" scripts/build_card_baseline_manifest.py "${manifest_args[@]}" --output "$MANIFEST"; then
    record_failure 'manifest' 'failed to audit and lock all three baselines'
  fi
fi
[ "$FAILURES" -eq 0 ] || { echo "$FAILURES baseline selection step(s) failed; see $FAIL_LOG" >&2; exit 1; }
echo "CARD baseline validation-only selection complete: $MANIFEST"
