#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
SUMMARY_CSV="${SUMMARY_CSV:-$EXP_ROOT/paper_required_experiments_summary.csv}"
FORCE=0
ONLY_EXP=""
DRY_RUN=0
FAIL_LOG="$EXP_ROOT/7_3_followup_reselect_failures.log"

EXPERIMENTS=(
  levir_mci_weak_pd08_reselect_strict_spice
  second_cc_crossattn_pd08_lsem0000_reselect_bestcider
  second_cc_crossattn_pd08_lsem0000_reselect_balanced
  second_cc_crossattn_pd08_lsem0000_reselect_spiceconstrained
)

usage() {
  echo "Usage: bash scripts/run_7_3_followup_reselect.sh [--only_exp EXP] [--force] [--dry_run]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --only_exp) ONLY_EXP="$2"; shift 2 ;;
    --force|--overwrite) FORCE=1; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$EXP_ROOT"
: > "$FAIL_LOG"
FAILURES=0

contains_exp() {
  local target="$1"
  [ -z "$ONLY_EXP" ] || [ "$ONLY_EXP" = "$target" ]
}

record_failure() {
  local label="$1"
  local message="${2:-}"
  echo "FAILED: $label" | tee -a "$FAIL_LOG"
  if [ -n "$message" ]; then echo "  $message" | tee -a "$FAIL_LOG"; fi
  FAILURES=$((FAILURES + 1))
}

run_or_log() {
  local label="$1"
  shift
  echo "========== $label =========="
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  if ! "$@"; then
    echo "FAILED: $label" | tee -a "$FAIL_LOG"
    printf '  command:' >> "$FAIL_LOG"
    printf ' %q' "$@" >> "$FAIL_LOG"
    printf '\n' >> "$FAIL_LOG"
    FAILURES=$((FAILURES + 1))
    return 1
  fi
}

configure_reselect() {
  local exp="$1"
  TARGET_EXP="$exp"
  STRICT_ARGS=()
  case "$exp" in
    levir_mci_weak_pd08_reselect_strict_spice)
      SOURCE_EXP=levir_mci_weak_pd08_lm003_ls001_noreweight
      STRATEGY=strict_spice_constrained_balanced
      OUTPUT_JSON_NAME=best_checkpoint_strict_spice.json
      STRICT_ARGS=(--strict_bleu4 0.562 --strict_cider 1.338 --strict_spice 0.336)
      ;;
    second_cc_crossattn_pd08_lsem0000_reselect_bestcider)
      SOURCE_EXP=second_cc_crossattn_pd08_lsem0000
      STRATEGY=best_cider
      OUTPUT_JSON_NAME=best_checkpoint_best_cider.json
      ;;
    second_cc_crossattn_pd08_lsem0000_reselect_balanced)
      SOURCE_EXP=second_cc_crossattn_pd08_lsem0000
      STRATEGY=balanced
      OUTPUT_JSON_NAME=best_checkpoint_balanced.json
      ;;
    second_cc_crossattn_pd08_lsem0000_reselect_spiceconstrained)
      SOURCE_EXP=second_cc_crossattn_pd08_lsem0000
      STRATEGY=spice_constrained_balanced
      OUTPUT_JSON_NAME=best_checkpoint_spice_constrained.json
      ;;
    *) echo "Unknown 7.3 checkpoint reselect experiment: $exp" >&2; return 2 ;;
  esac
}

run_one() {
  local exp="$1"
  configure_reselect "$exp" || { record_failure "configure $exp" "Unknown reselect experiment."; return 1; }
  local source_path="$EXP_ROOT/$SOURCE_EXP"
  local target_path="$EXP_ROOT/$TARGET_EXP"
  local output_json="$target_path/$OUTPUT_JSON_NAME"
  mkdir -p "$target_path"
  if [ "$DRY_RUN" -eq 0 ] && [ ! -d "$source_path" ]; then
    record_failure "reselect $exp" "Source experiment directory missing: $source_path"
    return 1
  fi
  if [ "$FORCE" -eq 0 ] && [ -s "$output_json" ]; then
    echo "Skipping reselect; existing selection file: $output_json"
    return 0
  fi
  run_or_log "select checkpoint $exp from $SOURCE_EXP" \
    "$PYTHON" scripts/select_best_checkpoint.py \
      --exp_dir "$source_path" \
      --summary_csv "$SUMMARY_CSV" \
      --strategy "$STRATEGY" \
      "${STRICT_ARGS[@]}" \
      --output_json "$output_json" \
      --output_txt "$target_path/best_checkpoint.txt" || return 1
  if [ "$DRY_RUN" -eq 0 ]; then
    cp "$output_json" "$target_path/best_checkpoint.json"
    {
      echo "target_exp=$TARGET_EXP"
      echo "source_exp=$SOURCE_EXP"
      echo "strategy=$STRATEGY"
      echo "selection_json=$OUTPUT_JSON_NAME"
    } > "$target_path/reselect_info.txt"
  fi
}

for exp in "${EXPERIMENTS[@]}"; do
  contains_exp "$exp" || continue
  run_one "$exp" || true
done

if [ "$FAILURES" -gt 0 ]; then
  echo "$FAILURES 7.3 checkpoint reselect step(s) failed. See $FAIL_LOG" >&2
  exit 1
fi

echo "7.3 checkpoint reselect flow complete."