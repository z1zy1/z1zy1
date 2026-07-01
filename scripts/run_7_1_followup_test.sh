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
FAIL_LOG="$EXP_ROOT/7_1_followup_test_failures.log"

EXPERIMENTS=(
  levir_mci_short_caption_ft_1000_lr01
  levir_mci_short_caption_ft_500_lr01
  levir_mci_short_ft_keep_mask0005
  second_cc_crossattn_pd08_lsem0001
  second_cc_crossattn_pd08_lsem0000
  second_cc_crossattn_pd09_lsem0005
  second_cc_crossattn_pd08_gamma005_lsem0005
)

PROTECTED_EXPERIMENTS=(
  levir_mci_card_mask_semantic
  second_cc_card_semantic_crossattn
)

usage() {
  echo "Usage: bash scripts/run_7_1_followup_test.sh [--only_exp EXP] [--force] [--dry_run]" >&2
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

is_protected_exp() {
  local target="$1"
  local item
  for item in "${PROTECTED_EXPERIMENTS[@]}"; do
    if [ "$item" = "$target" ]; then return 0; fi
  done
  return 1
}

contains_exp() {
  local target="$1"
  [ -z "$ONLY_EXP" ] || [ "$ONLY_EXP" = "$target" ]
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

clear_exp_env() {
  unset BASE_CFG MODEL_TYPE DATASET DATA_ROOT ANNO CHANGEFLAG_JSON EVAL_CHANGE_NOCHANGE_SPLIT PAPER_SELECTION_MODE
  unset USE_CHANGE_MASK MASK_TYPE NUM_MASK_CLASSES ENABLE_AUX_MASK USE_AUX_SEMANTIC USE_SEMANTIC_MAPS SEMANTIC_INPUT_MODE NUM_SEMANTIC_CLASSES
  unset USE_SEMANTIC_PARTIAL_DETACH SEMANTIC_DETACH_RATIO SEMANTIC_FUSION_GAMMA_INIT USE_FEATURE_REWEIGHT LMASK LSEM MASK_LOSS_TYPE SEMANTIC_LOSS_TYPE
  unset AUX_WARMUP_START_RATIO AUX_WARMUP_END_RATIO USE_AUX_WARMUP SELECTION_STRATEGY LR MAX_ITER SNAPSHOT_INTERVAL LOG_INTERVAL INIT_CHECKPOINT
}

configure_common() {
  local exp="$1"
  clear_exp_env
  if is_protected_exp "$exp"; then
    echo "Refusing to run protected existing experiment in follow-up runner: $exp" >&2
    return 2
  fi
  export EXP_DIR="$EXP_ROOT"
  export EXP_NAME="$exp"
  export PAPER_SELECTION_MODE=1
  export USE_AUX_WARMUP=1
  export AUX_WARMUP_START_RATIO=0.30
  export AUX_WARMUP_END_RATIO=0.70
  export SELECTION_STRATEGY=spice_constrained_balanced
  export USE_FEATURE_REWEIGHT=0
}

configure_exp() {
  local exp="$1"
  configure_common "$exp" || return 1
  case "$exp" in
    levir_mci_short_caption_ft_1000_lr01|levir_mci_short_caption_ft_500_lr01)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_ROOT" ANNO="$LEVIR_ROOT/levir_mci_captions_reformat.json" CHANGEFLAG_JSON="$LEVIR_ROOT/LevirCCcaptions.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.0 LMASK=0.0 LSEM=0.0 ;;
    levir_mci_short_ft_keep_mask0005)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_ROOT" ANNO="$LEVIR_ROOT/levir_mci_captions_reformat.json" CHANGEFLAG_JSON="$LEVIR_ROOT/LevirCCcaptions.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=1 MASK_TYPE=multiclass NUM_MASK_CLASSES=3 ENABLE_AUX_MASK=1 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=aux USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.0 LMASK=0.0005 LSEM=0.0 MASK_LOSS_TYPE=ce_dice ;;
    second_cc_crossattn_pd08_lsem0001)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.8 LMASK=0.0 LSEM=0.0001 SEMANTIC_LOSS_TYPE=ce_dice ;;
    second_cc_crossattn_pd08_lsem0000)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.8 LMASK=0.0 LSEM=0.0 USE_AUX_WARMUP=0 ;;
    second_cc_crossattn_pd09_lsem0005)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.9 LMASK=0.0 LSEM=0.0005 SEMANTIC_LOSS_TYPE=ce_dice ;;
    second_cc_crossattn_pd08_gamma005_lsem0005)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.8 SEMANTIC_FUSION_GAMMA_INIT=0.05 LMASK=0.0 LSEM=0.0005 SEMANTIC_LOSS_TYPE=ce_dice ;;
    *) echo "Unknown follow-up experiment: $exp" >&2; return 2 ;;
  esac
}

mark_failed() {
  local exp="$1"
  local note="$2"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: would mark failed: $exp ($note)"
    return 0
  fi
  "$PYTHON" scripts/update_paper_required_summary.py \
    --summary_csv "$SUMMARY_CSV" \
    --dataset "$DATASET" \
    --exp_name "$exp" \
    --status failed \
    --notes "$note"
}

select_checkpoint() {
  local exp_path="$1"
  run_or_log "select best $(basename "$exp_path")" "$PYTHON" scripts/select_best_checkpoint.py \
    --exp_dir "$exp_path" \
    --summary_csv "$SUMMARY_CSV" \
    --strategy spice_constrained_balanced
}

run_one() {
  local exp="$1"
  configure_exp "$exp" || return 1
  local exp_path="$EXP_ROOT/$exp"
  if [ "$DRY_RUN" -eq 0 ]; then
    assert_data_ready || { mark_failed "$exp" "Dataset or annotation path missing; test not run in this workspace."; return 0; }
  fi
  if [ "$DRY_RUN" -eq 0 ] && [ ! -d "$exp_path" ]; then
    mark_failed "$exp" "Experiment directory missing; training not available in this workspace."
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ] || [ "$FORCE" -eq 1 ] || [ ! -s "$exp_path/best_checkpoint.json" ]; then
    select_checkpoint "$exp_path" || { mark_failed "$exp" "Checkpoint selection failed; see $FAIL_LOG."; return 0; }
  fi
  local selected="<selected best checkpoint for $exp>"
  if [ "$DRY_RUN" -eq 0 ]; then
    if [ ! -s "$exp_path/best_checkpoint.txt" ]; then
      mark_failed "$exp" "best_checkpoint.txt missing after checkpoint selection."
      return 0
    fi
    selected="$(head -n 1 "$exp_path/best_checkpoint.txt")"
  fi
  if [ "$DRY_RUN" -eq 1 ] || [ "$FORCE" -eq 1 ] || [ ! -s "$exp_path/test_metrics.csv" ]; then
    run_or_log "test $exp" bash scripts/test_specific_snapshot_sgc_card.sh \
      --exp_dir "$exp_path" \
      --checkpoint "$selected" \
      --tag paper_best || { mark_failed "$exp" "Test failed; see $FAIL_LOG."; return 0; }
  fi
  run_or_log "update summary $exp" "$PYTHON" scripts/update_paper_required_summary.py \
    --summary_csv "$SUMMARY_CSV" \
    --exp_dir "$exp_path" \
    --status done
}

if [ -n "$ONLY_EXP" ] && is_protected_exp "$ONLY_EXP"; then
  echo "Refusing --only_exp $ONLY_EXP: it is protected and not part of the 7 follow-up set." >&2
  exit 2
fi

for exp in "${EXPERIMENTS[@]}"; do
  contains_exp "$exp" || continue
  run_one "$exp" || true
done

if [ "$DRY_RUN" -eq 1 ]; then
  run_or_log "summary consistency" "$PYTHON" scripts/check_summary_consistency.py --summary_csv "$SUMMARY_CSV" --output "$EXP_ROOT/summary_check_report.txt"
else
  "$PYTHON" scripts/check_summary_consistency.py --summary_csv "$SUMMARY_CSV" --output "$EXP_ROOT/summary_check_report.txt"
fi

if [ "$FAILURES" -gt 0 ]; then
  echo "$FAILURES follow-up test/update step(s) failed. See $FAIL_LOG" >&2
  exit 1
fi

echo "7/1 follow-up test and summary flow complete."