#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
LEVIR_CC_ROOT="${LEVIR_CC_ROOT:-./Levir-CC}"
LEVIR_MCI_ROOT="${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}"
SECOND_CC_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
LEVIR_CC_BASELINE_VAL_METRICS="${LEVIR_CC_BASELINE_VAL_METRICS:-$EXP_ROOT/card_levir_cc_baseline/baseline_best_checkpoint.json}"
LEVIR_MCI_BASELINE_VAL_METRICS="${LEVIR_MCI_BASELINE_VAL_METRICS:-$EXP_ROOT/levir_mci_card_baseline/baseline_best_checkpoint.json}"
SECOND_CC_BASELINE_VAL_METRICS="${SECOND_CC_BASELINE_VAL_METRICS:-$EXP_ROOT/second_cc_card_rgb_baseline/baseline_best_checkpoint.json}"
PROTECTED_METRICS="${PROTECTED_METRICS:-Bleu_1,Bleu_2,Bleu_3,Bleu_4,METEOR,ROUGE_L,CIDEr}"
ONLY_EXP=""
FORCE=0
DRY_RUN=0
FAIL_LOG="${FAIL_LOG:-$EXP_ROOT/7_6_followup_reselect_failures.log}"

LEVIR_CC_SOURCE_EXPS=(
  sgc_card_lm003_ls005_pd05_rw02_warmup
  levir_cc_decft_cw100_s10_lr5e7 levir_cc_decft_cw100_s10_lr1e6
  levir_cc_decft_cw100_s20_lr5e7 levir_cc_decft_cw100_s20_lr1e6
  levir_cc_decft_cw101_s10_lr5e7 levir_cc_decft_cw101_s10_lr1e6
  levir_cc_decft_cw101_s20_lr5e7 levir_cc_decft_cw101_s20_lr1e6
  levir_cc_decft_cw102_s10_lr5e7 levir_cc_decft_cw102_s10_lr1e6
  levir_cc_decft_cw102_s20_lr5e7 levir_cc_decft_cw102_s20_lr1e6
)
SECOND_CC_SOURCE_EXPS=(
  second_cc_crossattn_pd07_lsem0000
  second_cc_crossattn_pd08_lsem0000
  second_cc_crossattn_pd09_lsem0000
  second_cc_card_semantic_crossattn
)

usage() {
  echo 'Usage: bash scripts/run_7_6_followup_reselect.sh [--only_exp DATASET_OR_LOCK] [--force] [--dry_run]' >&2
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
touch "$FAIL_LOG"
printf '\n[%s] run_7_6_followup_reselect start dry_run=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$DRY_RUN" >> "$FAIL_LOG"

selected() {
  local dataset="$1" lock_id="$2" target="$3"
  [ -z "$ONLY_EXP" ] || [ "$ONLY_EXP" = "$dataset" ] || [ "$ONLY_EXP" = "$lock_id" ] || [ "$ONLY_EXP" = "$target" ]
}

require_baseline() {
  local dataset="$1" path="$2"
  if [ "$DRY_RUN" -eq 0 ] && [ ! -f "$path" ]; then
    echo "Missing audited $dataset CARD validation baseline: $path" | tee -a "$FAIL_LOG" >&2
    return 1
  fi
}

validate_selection() {
  local path="$1" strategy="$2"
  "$PYTHON" - "$path" "$strategy" <<'PY'
import json
import sys

path, strategy = sys.argv[1:]
payload = json.load(open(path, encoding='utf-8-sig'))
assert payload.get('status') == 'done', '%s status=%r' % (path, payload.get('status'))
assert payload.get('selection_strategy') == strategy
assert payload.get('selection_uses_test_metrics') is False
assert payload.get('selection_metric_split') == 'validation'
assert payload.get('selected_checkpoint')
assert float(payload.get('selected_metric_deltas', {}).get('SPICE')) >= -1e-12
PY
}

run_selector() {
  local target="$1" strategy="$2" baseline="$3" expected_gap="$4"
  shift 4
  local sources=("$@")
  local target_path="$EXP_ROOT/$target"
  local output_json="$target_path/best_checkpoint.json"
  mkdir -p "$target_path"
  require_baseline "$target" "$baseline"
  local primary="" source path
  local candidate_args=()
  for source in "${sources[@]}"; do
    path="$EXP_ROOT/$source"
    if [ "$DRY_RUN" -eq 0 ] && [ ! -d "$path" ]; then
      echo "Candidate experiment missing: $path" | tee -a "$FAIL_LOG" >&2
      return 1
    fi
    if [ -z "$primary" ]; then primary="$path"; else candidate_args+=(--candidate_exp_dir "$path"); fi
  done
  if [ "$FORCE" -eq 0 ] && [ -s "$output_json" ]; then
    echo "Skipping existing selection: $output_json"
    [ "$DRY_RUN" -eq 1 ] || validate_selection "$output_json" "$strategy"
    return
  fi
  local command=(
    "$PYTHON" scripts/select_best_checkpoint.py
    --exp_dir "$primary"
    "${candidate_args[@]}"
    --strategy "$strategy"
    --baseline_metrics "$baseline"
    --protected_metrics "$PROTECTED_METRICS"
    --baseline_tolerance 0
    --min_spice_gain 0
    --require_audited_validation_baseline
    --output_json "$output_json"
    --output_txt "$target_path/best_checkpoint.txt"
  )
  if [ -n "$expected_gap" ]; then
    command+=(--stability_window 3 --expected_step_gap "$expected_gap")
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
  else
    "${command[@]}"
    validate_selection "$output_json" "$strategy"
  fi
}

ran_any=0
if selected levir_cc levir_cc levir_cc_7_6_val_pareto_locked; then
  ran_any=1
  run_selector levir_cc_7_6_val_pareto_locked val_baseline_pareto \
    "$LEVIR_CC_BASELINE_VAL_METRICS" '' "${LEVIR_CC_SOURCE_EXPS[@]}"
fi

for seed in 1111 2222 3333; do
  source="levir_mci_masksemantic_repro_seed$seed"
  target="${source}_7_6_val_locked"
  lock_id="levir_mci_seed$seed"
  if selected levir_mci "$lock_id" "$target"; then
    ran_any=1
    run_selector "$target" val_baseline_pareto "$LEVIR_MCI_BASELINE_VAL_METRICS" '' "$source"
  fi
done

if selected second_cc second_cc second_cc_7_6_stable_locked; then
  ran_any=1
  run_selector second_cc_7_6_stable_locked val_baseline_stable_window \
    "$SECOND_CC_BASELINE_VAL_METRICS" 1000 "${SECOND_CC_SOURCE_EXPS[@]}"
fi

[ "$ran_any" -eq 1 ] || { echo "Unknown --only_exp value: $ONLY_EXP" >&2; exit 2; }

MANIFEST="$EXP_ROOT/7_6_locked_manifest.json"
if [ -n "$ONLY_EXP" ]; then
  echo "Partial selection complete; manifest intentionally unchanged: $MANIFEST"
elif [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: would audit and atomically write $MANIFEST"
else
  "$PYTHON" scripts/build_7_6_locked_manifest.py \
    --project_dir "$PROJECT_DIR" \
    --exp_root "$EXP_ROOT" \
    --output "$MANIFEST" \
    --levir_cc_root "$LEVIR_CC_ROOT" \
    --levir_mci_root "$LEVIR_MCI_ROOT" \
    --second_cc_root "$SECOND_CC_ROOT" \
    --levir_cc_baseline "$LEVIR_CC_BASELINE_VAL_METRICS" \
    --levir_mci_baseline "$LEVIR_MCI_BASELINE_VAL_METRICS" \
    --second_cc_baseline "$SECOND_CC_BASELINE_VAL_METRICS"
  "$PYTHON" scripts/build_7_6_locked_manifest.py --verify "$MANIFEST"
fi

echo '7.6 validation-only selection/audit complete. Test remains a separate command.'
