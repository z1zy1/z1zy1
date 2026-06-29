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
FORWARD_SANITY=0
FAIL_LOG="$EXP_ROOT/new_required_eval_task_failures.log"
mkdir -p "$EXP_ROOT"
: > "$FAIL_LOG"
FAILURES=0

usage() {
  echo "Usage: bash scripts/run_new_required_eval_tasks.sh [--force] [--forward_sanity]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --force|--overwrite) FORCE=1; shift ;;
    --forward_sanity) FORWARD_SANITY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

run_or_log() {
  local label="$1"
  shift
  echo "========== $label =========="
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
  unset USE_SEMANTIC_PARTIAL_DETACH SEMANTIC_DETACH_RATIO USE_FEATURE_REWEIGHT LMASK LSEM MASK_LOSS_TYPE SEMANTIC_LOSS_TYPE
}

configure_existing() {
  local exp="$1"
  clear_exp_env
  export EXP_DIR="$EXP_ROOT"
  export EXP_NAME="$exp"
  export PAPER_SELECTION_MODE=1
  export USE_FEATURE_REWEIGHT=0
  case "$exp" in
    levir_mci_card_mask_semantic)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_ROOT" ANNO="$LEVIR_ROOT/levir_mci_captions_reformat.json" CHANGEFLAG_JSON="$LEVIR_ROOT/LevirCCcaptions.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=1 MASK_TYPE=multiclass NUM_MASK_CLASSES=3 ENABLE_AUX_MASK=1 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=aux USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.0 LMASK=0.003 LSEM=0.005 MASK_LOSS_TYPE=ce_dice SEMANTIC_LOSS_TYPE=multilabel_bce ;;
    levir_mci_card_mask_semantic_pd05)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_ROOT" ANNO="$LEVIR_ROOT/levir_mci_captions_reformat.json" CHANGEFLAG_JSON="$LEVIR_ROOT/LevirCCcaptions.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=1 MASK_TYPE=multiclass NUM_MASK_CLASSES=3 ENABLE_AUX_MASK=1 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=aux USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5 LMASK=0.003 LSEM=0.005 MASK_LOSS_TYPE=ce_dice SEMANTIC_LOSS_TYPE=multilabel_bce ;;
    levir_mci_card_mask_semantic_pd05_reweight)
      configure_existing levir_mci_card_mask_semantic_pd05
      export EXP_NAME="$exp" USE_FEATURE_REWEIGHT=1 ;;
    second_cc_card_rgb_baseline)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_baseline.yaml" MODEL_TYPE=card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none USE_SEMANTIC_PARTIAL_DETACH=0 LMASK=0.0 LSEM=0.0 ;;
    second_cc_card_semantic_crossattn|second_cc_ours_weak_coupled_final)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5 LMASK=0.0 LSEM=0.005 SEMANTIC_LOSS_TYPE=ce_dice ;;
    *) echo "No config mapping for $exp" >&2; return 2 ;;
  esac
}

select_test_update_existing() {
  local exp="$1"
  local exp_path="$EXP_ROOT/$exp"
  configure_existing "$exp" || return 1
  if [ ! -d "$exp_path" ]; then
    "$PYTHON" scripts/update_paper_required_summary.py --summary_csv "$SUMMARY_CSV" --dataset "$DATASET" --exp_name "$exp" --status failed --notes "Existing run directory missing; checkpoint selection/test not run in this workspace."
    return 0
  fi
  if [ "$FORCE" -eq 1 ] || [ ! -s "$exp_path/best_checkpoint.json" ]; then
    run_or_log "select best existing $exp" "$PYTHON" scripts/select_best_checkpoint.py --exp_dir "$exp_path" --strategy spice_constrained_balanced || return 1
  fi
  if [ -s "$exp_path/best_checkpoint.txt" ]; then
    local selected
    selected="$(head -n 1 "$exp_path/best_checkpoint.txt")"
    if [ "$FORCE" -eq 1 ] || [ ! -s "$exp_path/test_metrics.csv" ]; then
      run_or_log "test existing $exp" bash scripts/test_specific_snapshot_sgc_card.sh --exp_dir "$exp_path" --checkpoint "$selected" --tag paper_best || return 1
    fi
  fi
  run_or_log "update summary existing $exp" "$PYTHON" scripts/update_paper_required_summary.py --summary_csv "$SUMMARY_CSV" --exp_dir "$exp_path" || return 1
}

run_or_log "config check" "$PYTHON" scripts/check_experiment_configs.py --experiments_root "$EXP_ROOT" --output "$EXP_ROOT/config_check/levir_mci_pd05_vs_noreweight_report.txt" || true
"$PYTHON" scripts/update_paper_required_summary.py --summary_csv "$SUMMARY_CSV" --dataset levir_mci --exp_name levir_mci_config_check_pd05_vs_noreweight --method_group config_check --status config_check_done --notes "$EXP_ROOT/config_check/levir_mci_pd05_vs_noreweight_report.txt"

SANITY_ARGS=(scripts/second_cc_semantic_head_sanity_check.py --data_root "$SECOND_ROOT" --output_dir "$EXP_ROOT/sanity_check/second_cc_semantic_head")
if [ "$FORWARD_SANITY" -eq 1 ]; then SANITY_ARGS+=(--forward_batch); fi
run_or_log "second cc semantic sanity" "$PYTHON" "${SANITY_ARGS[@]}" || true
"$PYTHON" scripts/update_paper_required_summary.py --summary_csv "$SUMMARY_CSV" --dataset second_cc --exp_name second_cc_semantic_head_sanity_check --method_group sanity_check --status sanity_check_done --notes "$EXP_ROOT/sanity_check/second_cc_semantic_head/sanity_report.txt"

for exp in \
  levir_mci_card_mask_semantic \
  levir_mci_card_mask_semantic_pd05 \
  levir_mci_card_mask_semantic_pd05_reweight \
  second_cc_card_rgb_baseline \
  second_cc_card_semantic_crossattn \
  second_cc_ours_weak_coupled_final; do
  select_test_update_existing "$exp" || true
done

"$PYTHON" scripts/add_external_result_to_summary.py --summary_csv "$SUMMARY_CSV"

if [ "$FAILURES" -gt 0 ]; then
  echo "$FAILURES eval task step(s) failed. See $FAIL_LOG" >&2
  exit 1
fi

echo "New required eval tasks complete."
