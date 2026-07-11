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
FAIL_LOG="$EXP_ROOT/7_4_followup_reselect_failures.log"

EXPERIMENTS=(
  levir_mci_ultrashort_grid_reselect_strict_nearest
  second_cc_crossattn_lsem0000_pdgrid_reselect_balanced
)

usage() {
  echo "Usage: bash scripts/run_7_4_followup_reselect.sh [--only_exp EXP] [--force] [--dry_run]" >&2
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
  SOURCE_EXPS=()
  STRICT_ARGS=()
  case "$exp" in
    levir_mci_ultrashort_grid_reselect_strict_nearest)
      SOURCE_EXPS=(
        levir_mci_ultrashort_caption_ft_100_lr005
        levir_mci_ultrashort_caption_ft_50_lr005
        levir_mci_ultrashort_caption_ft_100_lr002
        levir_mci_ultrashort_caption_ft_100_lr005_dense10
        levir_mci_ultrashort_caption_ft_80_lr004_dense10
        levir_mci_ultrashort_caption_ft_80_lr003_dense10
      )
      STRATEGY=strict_nearest_balanced
      OUTPUT_JSON_NAME=best_checkpoint_strict_nearest.json
      STRICT_ARGS=(--strict_bleu4 0.562 --strict_cider 1.338 --strict_spice 0.336)
      ;;
    second_cc_crossattn_lsem0000_pdgrid_reselect_balanced)
      SOURCE_EXPS=(
        second_cc_crossattn_pd07_lsem0000
        second_cc_crossattn_pd08_lsem0000
        second_cc_crossattn_pd09_lsem0000
      )
      STRATEGY=spice_constrained_balanced
      OUTPUT_JSON_NAME=best_checkpoint_pdgrid_balanced.json
      ;;
    *) echo "Unknown 7.4 checkpoint reselect experiment: $exp" >&2; return 2 ;;
  esac
}

run_one() {
  local exp="$1"
  configure_reselect "$exp" || { record_failure "configure $exp" "Unknown reselect experiment."; return 1; }
  local target_path="$EXP_ROOT/$TARGET_EXP"
  local output_json="$target_path/$OUTPUT_JSON_NAME"
  local primary_path=""
  local source_exp
  local source_path
  local candidate_args=()
  mkdir -p "$target_path"

  for source_exp in "${SOURCE_EXPS[@]}"; do
    source_path="$EXP_ROOT/$source_exp"
    if [ "$DRY_RUN" -eq 0 ] && [ ! -d "$source_path" ]; then
      record_failure "reselect $exp" "Source experiment directory missing: $source_path"
      return 1
    fi
    if [ -z "$primary_path" ]; then
      primary_path="$source_path"
    else
      candidate_args+=(--candidate_exp_dir "$source_path")
    fi
  done

  if [ "$FORCE" -eq 0 ] && [ -s "$output_json" ]; then
    echo "Skipping reselect; existing selection file: $output_json"
    return 0
  fi
  run_or_log "select checkpoint $exp from ${SOURCE_EXPS[*]}" \
    "$PYTHON" scripts/select_best_checkpoint.py \
      --exp_dir "$primary_path" \
      "${candidate_args[@]}" \
      --summary_csv "$SUMMARY_CSV" \
      --strategy "$STRATEGY" \
      "${STRICT_ARGS[@]}" \
      --output_json "$output_json" \
      --output_txt "$target_path/best_checkpoint.txt" || return 1
  if [ "$DRY_RUN" -eq 0 ]; then
    cp "$output_json" "$target_path/best_checkpoint.json"
    {
      echo "target_exp=$TARGET_EXP"
      echo "source_exps=${SOURCE_EXPS[*]}"
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
  echo "$FAILURES 7.4 checkpoint reselect step(s) failed. See $FAIL_LOG" >&2
  exit 1
fi

echo "7.4 checkpoint reselect flow complete."