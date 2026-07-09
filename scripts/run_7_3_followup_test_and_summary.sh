#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
LEVIR_ROOT="${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}"
SECOND_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
SUMMARY_CSV="${SUMMARY_CSV:-$EXP_ROOT/paper_required_experiments_summary.csv}"
FORCE=0
ONLY_EXP=""
DRY_RUN=0
FAIL_LOG="$EXP_ROOT/7_3_followup_test_failures.log"

TRAINED_EXPERIMENTS=(
  levir_mci_ultrashort_caption_ft_100_lr005
  levir_mci_ultrashort_caption_ft_50_lr005
  levir_mci_ultrashort_caption_ft_100_lr002
  second_cc_crossattn_pd05_lsem0000
  second_cc_crossattn_pd00_lsem0000
)

RESELECT_EXPERIMENTS=(
  levir_mci_weak_pd08_reselect_strict_spice
  second_cc_crossattn_pd08_lsem0000_reselect_bestcider
  second_cc_crossattn_pd08_lsem0000_reselect_balanced
  second_cc_crossattn_pd08_lsem0000_reselect_spiceconstrained
)

usage() {
  echo "Usage: bash scripts/run_7_3_followup_test_and_summary.sh [--only_exp EXP] [--force] [--dry_run]" >&2
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

is_reselect_exp() {
  local target="$1"
  local item
  for item in "${RESELECT_EXPERIMENTS[@]}"; do
    if [ "$item" = "$target" ]; then return 0; fi
  done
  return 1
}

require_path() {
  local path="$1"
  local label="$2"
  if [ ! -e "$path" ]; then
    echo "$label does not exist: $path" >&2
    return 1
  fi
}

assert_data_ready() {
  require_path "$DATA_ROOT" "DATA_ROOT" || return 1
  require_path "$ANNO" "ANNO" || return 1
  require_path "$CHANGEFLAG_JSON" "CHANGEFLAG_JSON" || return 1
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

json_value() {
  local path="$1"
  local key="$2"
  "$PYTHON" - "$path" "$key" <<'PY'
import json
import sys
path, key = sys.argv[1], sys.argv[2]
try:
    with open(path, encoding='utf-8-sig') as f:
        payload = json.load(f)
except Exception:
    print('')
    raise SystemExit(0)
value = payload
for part in key.split('.'):
    if not isinstance(value, dict):
        value = ''
        break
    value = value.get(part, '')
if value is None:
    value = ''
print(value)
PY
}

clear_exp_env() {
  unset BASE_CFG MODEL_TYPE DATASET DATA_ROOT ANNO CHANGEFLAG_JSON EVAL_CHANGE_NOCHANGE_SPLIT PAPER_SELECTION_MODE
  unset USE_CHANGE_MASK MASK_TYPE NUM_MASK_CLASSES ENABLE_AUX_MASK USE_AUX_SEMANTIC USE_SEMANTIC_MAPS SEMANTIC_INPUT_MODE NUM_SEMANTIC_CLASSES
  unset USE_SEMANTIC_PARTIAL_DETACH SEMANTIC_DETACH_RATIO SEMANTIC_FUSION_GAMMA_INIT USE_FEATURE_REWEIGHT LMASK LSEM MASK_LOSS_TYPE SEMANTIC_LOSS_TYPE
  unset AUX_WARMUP_START_RATIO AUX_WARMUP_END_RATIO USE_AUX_WARMUP SELECTION_STRATEGY LR MAX_ITER FINETUNE_STEPS SNAPSHOT_INTERVAL SAVE_INTERVAL EVAL_INTERVAL LOG_INTERVAL INIT_CHECKPOINT
}

configure_exp() {
  local exp="$1"
  clear_exp_env
  export EXP_DIR="$EXP_ROOT"
  export EXP_NAME="$exp"
  export PAPER_SELECTION_MODE=1
  export USE_AUX_WARMUP=0
  export SELECTION_STRATEGY=spice_constrained_balanced
  export USE_FEATURE_REWEIGHT=0
  case "$exp" in
    levir_mci_ultrashort_caption_ft_100_lr005|levir_mci_ultrashort_caption_ft_50_lr005|levir_mci_ultrashort_caption_ft_100_lr002)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_ROOT" ANNO="$LEVIR_ROOT/levir_mci_captions_reformat.json" CHANGEFLAG_JSON="$LEVIR_ROOT/LevirCCcaptions.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.0 LMASK=0.0 LSEM=0.0 ;;
    second_cc_crossattn_pd05_lsem0000)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5 LMASK=0.0 LSEM=0.0 ;;
    second_cc_crossattn_pd00_lsem0000)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.0 LMASK=0.0 LSEM=0.0 ;;
    levir_mci_weak_pd08_reselect_strict_spice)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_ROOT" ANNO="$LEVIR_ROOT/levir_mci_captions_reformat.json" CHANGEFLAG_JSON="$LEVIR_ROOT/LevirCCcaptions.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=1 MASK_TYPE=multiclass NUM_MASK_CLASSES=3 ENABLE_AUX_MASK=1 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=aux USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.8 LMASK=0.003 LSEM=0.001 MASK_LOSS_TYPE=ce_dice SEMANTIC_LOSS_TYPE=multilabel_bce ;;
    second_cc_crossattn_pd08_lsem0000_reselect_bestcider|second_cc_crossattn_pd08_lsem0000_reselect_balanced|second_cc_crossattn_pd08_lsem0000_reselect_spiceconstrained)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.8 LMASK=0.0 LSEM=0.0 ;;
    *) echo "Unknown 7.3 follow-up test experiment: $exp" >&2; return 2 ;;
  esac
}

reselect_json_name() {
  case "$1" in
    levir_mci_weak_pd08_reselect_strict_spice) echo best_checkpoint_strict_spice.json ;;
    second_cc_crossattn_pd08_lsem0000_reselect_bestcider) echo best_checkpoint_best_cider.json ;;
    second_cc_crossattn_pd08_lsem0000_reselect_balanced) echo best_checkpoint_balanced.json ;;
    second_cc_crossattn_pd08_lsem0000_reselect_spiceconstrained) echo best_checkpoint_spice_constrained.json ;;
    *) echo best_checkpoint.json ;;
  esac
}

mark_summary() {
  local exp="$1"
  local status="$2"
  local note="$3"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: would update summary: $exp status=$status notes=$note"
    return 0
  fi
  "$PYTHON" scripts/update_paper_required_summary.py \
    --summary_csv "$SUMMARY_CSV" \
    --dataset "$DATASET" \
    --exp_name "$exp" \
    --status "$status" \
    --notes "$note"
}

select_trained_checkpoint() {
  local exp_path="$1"
  run_or_log "select best $(basename "$exp_path")" "$PYTHON" scripts/select_best_checkpoint.py \
    --exp_dir "$exp_path" \
    --summary_csv "$SUMMARY_CSV" \
    --strategy spice_constrained_balanced
}

run_test() {
  local exp="$1"
  local exp_path="$2"
  local checkpoint="$3"
  local tag="$4"
  run_or_log "test $exp" bash scripts/test_specific_snapshot_sgc_card.sh \
    --exp_dir "$exp_path" \
    --checkpoint "$checkpoint" \
    --tag "$tag"
}

update_done_summary() {
  local exp="$1"
  local exp_path="$2"
  local best_json="$3"
  run_or_log "update summary $exp" "$PYTHON" scripts/update_paper_required_summary.py \
    --summary_csv "$SUMMARY_CSV" \
    --exp_dir "$exp_path" \
    --exp_name "$exp" \
    --dataset "$DATASET" \
    --best_checkpoint_json "$best_json" \
    --status done
}

run_trained_one() {
  local exp="$1"
  configure_exp "$exp" || return 1
  local exp_path="$EXP_ROOT/$exp"
  if [ "$DRY_RUN" -eq 0 ]; then
    assert_data_ready || { mark_summary "$exp" failed "Dataset or annotation path missing; test not run in this workspace."; return 0; }
    if [ ! -d "$exp_path" ]; then
      mark_summary "$exp" failed "Experiment directory missing; training not available in this workspace."
      return 0
    fi
  fi
  if [ "$DRY_RUN" -eq 1 ] || [ "$FORCE" -eq 1 ] || [ ! -s "$exp_path/best_checkpoint.json" ]; then
    select_trained_checkpoint "$exp_path" || { mark_summary "$exp" failed "Checkpoint selection failed; see $FAIL_LOG."; return 0; }
  fi
  local selected="<selected best checkpoint for $exp>"
  if [ "$DRY_RUN" -eq 0 ]; then
    selected="$(head -n 1 "$exp_path/best_checkpoint.txt")"
  fi
  if [ "$DRY_RUN" -eq 1 ] || [ "$FORCE" -eq 1 ] || [ ! -s "$exp_path/test_metrics.csv" ]; then
    run_test "$exp" "$exp_path" "$selected" paper_best || { mark_summary "$exp" failed "Test failed; see $FAIL_LOG."; return 0; }
  fi
  update_done_summary "$exp" "$exp_path" "$exp_path/best_checkpoint.json"
}

run_reselect_one() {
  local exp="$1"
  configure_exp "$exp" || return 1
  local exp_path="$EXP_ROOT/$exp"
  local best_json="$exp_path/$(reselect_json_name "$exp")"
  if [ "$DRY_RUN" -eq 0 ]; then
    assert_data_ready || { mark_summary "$exp" failed "Dataset or annotation path missing; reselect test not run in this workspace."; return 0; }
  fi
  if [ "$DRY_RUN" -eq 1 ] || [ "$FORCE" -eq 1 ] || [ ! -s "$best_json" ]; then
    local reselect_args=(--only_exp "$exp")
    if [ "$FORCE" -eq 1 ]; then reselect_args+=(--force); fi
    if [ "$DRY_RUN" -eq 1 ]; then reselect_args+=(--dry_run); fi
    run_or_log "reselect $exp" bash scripts/run_7_3_followup_reselect.sh "${reselect_args[@]}" || { mark_summary "$exp" failed "Checkpoint reselect failed; see $FAIL_LOG."; return 0; }
  fi
  local status="done"
  local selected="<selected reselect checkpoint for $exp>"
  if [ "$DRY_RUN" -eq 0 ]; then
    if [ ! -s "$best_json" ]; then
      mark_summary "$exp" failed "Reselect output missing: $best_json"
      return 0
    fi
    status="$(json_value "$best_json" status)"
    selected="$(json_value "$best_json" selected_checkpoint)"
  fi
  if [ "$status" = "no_valid_checkpoint" ] || [ -z "$selected" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "DRY RUN: would mark no_valid_checkpoint for $exp"
    else
      "$PYTHON" scripts/update_paper_required_summary.py \
        --summary_csv "$SUMMARY_CSV" \
        --dataset "$DATASET" \
        --exp_name "$exp" \
        --best_checkpoint_json "$best_json" \
        --status no_valid_checkpoint \
        --notes "no checkpoint satisfies strict SPICE/CIDEr/BLEU constraints"
    fi
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ] || [ "$FORCE" -eq 1 ] || [ ! -s "$exp_path/test_metrics.csv" ]; then
    run_test "$exp" "$exp_path" "$selected" paper_best || { mark_summary "$exp" failed "Test failed; see $FAIL_LOG."; return 0; }
  fi
  update_done_summary "$exp" "$exp_path" "$best_json"
}

for exp in "${TRAINED_EXPERIMENTS[@]}"; do
  contains_exp "$exp" || continue
  run_trained_one "$exp" || true
done

for exp in "${RESELECT_EXPERIMENTS[@]}"; do
  contains_exp "$exp" || continue
  run_reselect_one "$exp" || true
done

if [ "$DRY_RUN" -eq 1 ]; then
  run_or_log "summary consistency" "$PYTHON" scripts/check_summary_consistency.py --summary_csv "$SUMMARY_CSV" --output "$EXP_ROOT/summary_check_report.txt"
else
  "$PYTHON" scripts/check_summary_consistency.py --summary_csv "$SUMMARY_CSV" --output "$EXP_ROOT/summary_check_report.txt"
fi

if [ "$FAILURES" -gt 0 ]; then
  echo "$FAILURES 7.3 follow-up test/update step(s) failed. See $FAIL_LOG" >&2
  exit 1
fi

echo "7.3 follow-up test and summary flow complete."