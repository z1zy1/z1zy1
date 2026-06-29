#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
LEVIR_ROOT="${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}"
SECOND_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
FORCE=0
SKIP_TRAIN=0
ONLY_EXP=""
FAIL_LOG="$EXP_ROOT/new_required_train_failures.log"

EXPERIMENTS=(
  levir_mci_weak_pd08_lm003_ls001_noreweight
  levir_mci_weak_pd08_lm001_ls001_noreweight
  levir_mci_weak_pd05_lm003_ls0005_noreweight
  levir_mci_caption_finetune_from_weak_best
  second_cc_crossattn_pd05_lsem0005
  second_cc_crossattn_pd05_lsem001
  second_cc_crossattn_pd08_lsem0005
  second_cc_crossattn_pd08_lsem001
)

usage() {
  echo "Usage: bash scripts/run_new_required_train_experiments.sh [--force] [--skip_train] [--only_exp EXP]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --force|--overwrite) FORCE=1; shift ;;
    --skip_train) SKIP_TRAIN=1; shift ;;
    --only_exp) ONLY_EXP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$EXP_ROOT"
: > "$FAIL_LOG"
FAILURES=0

contains_exp() {
  local exp="$1"
  [ -z "$ONLY_EXP" ] || [ "$ONLY_EXP" = "$exp" ]
}

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

has_complete_test() {
  local exp_path="$1"
  [ -s "$exp_path/test_metrics.csv" ] && [ -s "$exp_path/best_checkpoint.json" ]
}

clear_exp_env() {
  unset BASE_CFG MODEL_TYPE DATASET DATA_ROOT ANNO CHANGEFLAG_JSON EVAL_CHANGE_NOCHANGE_SPLIT PAPER_SELECTION_MODE
  unset USE_CHANGE_MASK MASK_TYPE NUM_MASK_CLASSES ENABLE_AUX_MASK USE_AUX_SEMANTIC USE_SEMANTIC_MAPS SEMANTIC_INPUT_MODE NUM_SEMANTIC_CLASSES
  unset USE_SEMANTIC_PARTIAL_DETACH SEMANTIC_DETACH_RATIO USE_FEATURE_REWEIGHT LMASK LSEM MASK_LOSS_TYPE SEMANTIC_LOSS_TYPE
  unset AUX_WARMUP_START_RATIO AUX_WARMUP_END_RATIO USE_AUX_WARMUP SELECTION_STRATEGY LR MAX_ITER SNAPSHOT_INTERVAL LOG_INTERVAL INIT_CHECKPOINT
}

configure_exp() {
  local exp="$1"
  clear_exp_env
  export EXP_DIR="$EXP_ROOT"
  export EXP_NAME="$exp"
  export PAPER_SELECTION_MODE=1
  export USE_AUX_WARMUP=1
  export AUX_WARMUP_START_RATIO=0.30
  export AUX_WARMUP_END_RATIO=0.70
  export SELECTION_STRATEGY=spice_constrained_balanced
  export USE_FEATURE_REWEIGHT=0
  case "$exp" in
    levir_mci_weak_pd08_lm003_ls001_noreweight)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_ROOT" ANNO="$LEVIR_ROOT/levir_mci_captions_reformat.json" CHANGEFLAG_JSON="$LEVIR_ROOT/LevirCCcaptions.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=1 MASK_TYPE=multiclass NUM_MASK_CLASSES=3 ENABLE_AUX_MASK=1 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=aux USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.8 LMASK=0.003 LSEM=0.001 MASK_LOSS_TYPE=ce_dice SEMANTIC_LOSS_TYPE=multilabel_bce ;;
    levir_mci_weak_pd08_lm001_ls001_noreweight)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_ROOT" ANNO="$LEVIR_ROOT/levir_mci_captions_reformat.json" CHANGEFLAG_JSON="$LEVIR_ROOT/LevirCCcaptions.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=1 MASK_TYPE=multiclass NUM_MASK_CLASSES=3 ENABLE_AUX_MASK=1 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=aux USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.8 LMASK=0.001 LSEM=0.001 MASK_LOSS_TYPE=ce_dice SEMANTIC_LOSS_TYPE=multilabel_bce ;;
    levir_mci_weak_pd05_lm003_ls0005_noreweight)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_ROOT" ANNO="$LEVIR_ROOT/levir_mci_captions_reformat.json" CHANGEFLAG_JSON="$LEVIR_ROOT/LevirCCcaptions.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=1 MASK_TYPE=multiclass NUM_MASK_CLASSES=3 ENABLE_AUX_MASK=1 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=aux USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5 LMASK=0.003 LSEM=0.0005 MASK_LOSS_TYPE=ce_dice SEMANTIC_LOSS_TYPE=multilabel_bce ;;
    levir_mci_caption_finetune_from_weak_best)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_ROOT" ANNO="$LEVIR_ROOT/levir_mci_captions_reformat.json" CHANGEFLAG_JSON="$LEVIR_ROOT/LevirCCcaptions.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.0 LMASK=0.0 LSEM=0.0 LR="${FINETUNE_LR:-0.00005}" MAX_ITER="${FINETUNE_MAX_ITER:-2000}"
      if [ -z "${INIT_CHECKPOINT:-}" ]; then
        run_or_log "find weak init checkpoint" "$PYTHON" scripts/find_best_weak_checkpoint.py --experiments_root "$EXP_ROOT" || return 1
        export INIT_CHECKPOINT="$(head -n 1 "$EXP_ROOT/levir_mci_caption_finetune_from_weak_best_init.txt")"
      fi ;;
    second_cc_crossattn_pd05_lsem0005)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5 LMASK=0.0 LSEM=0.0005 SEMANTIC_LOSS_TYPE=ce_dice ;;
    second_cc_crossattn_pd05_lsem001)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5 LMASK=0.0 LSEM=0.001 SEMANTIC_LOSS_TYPE=ce_dice ;;
    second_cc_crossattn_pd08_lsem0005)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.8 LMASK=0.0 LSEM=0.0005 SEMANTIC_LOSS_TYPE=ce_dice ;;
    second_cc_crossattn_pd08_lsem001)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.8 LMASK=0.0 LSEM=0.001 SEMANTIC_LOSS_TYPE=ce_dice ;;
    *) echo "Unknown experiment: $exp" >&2; return 2 ;;
  esac
}

run_one() {
  local exp="$1"
  configure_exp "$exp" || return 1
  local exp_path="$EXP_ROOT/$exp"
  if [ "$FORCE" -eq 0 ] && has_complete_test "$exp_path"; then
    echo "Skipping existing complete experiment: $exp"
    "$PYTHON" scripts/update_paper_required_summary.py --exp_dir "$exp_path" --status skipped_existing || return 1
    return 0
  fi
  if [ "$SKIP_TRAIN" -eq 0 ]; then
    run_or_log "train $exp" bash scripts/_run_paper_training.sh || return 1
  else
    echo "Skipping training for $exp"
  fi
  if [ "$FORCE" -eq 1 ] || [ ! -s "$exp_path/eval_snapshots.csv" ]; then
    export FORCE_INFER="$FORCE"
    run_or_log "eval snapshots $exp" bash scripts/eval_all_snapshots_sgc_card.sh --exp_dir "$exp_path" || return 1
  fi
  if [ "$FORCE" -eq 1 ] || [ ! -s "$exp_path/best_checkpoint.json" ]; then
    run_or_log "select best $exp" "$PYTHON" scripts/select_best_checkpoint.py --exp_dir "$exp_path" --strategy spice_constrained_balanced || return 1
  fi
  local selected_checkpoint
  selected_checkpoint="$(head -n 1 "$exp_path/best_checkpoint.txt")"
  if [ "$FORCE" -eq 1 ] || [ ! -s "$exp_path/test_metrics.csv" ]; then
    run_or_log "test paper best $exp" bash scripts/test_specific_snapshot_sgc_card.sh --exp_dir "$exp_path" --checkpoint "$selected_checkpoint" --tag paper_best || return 1
  fi
  run_or_log "update summary $exp" "$PYTHON" scripts/update_paper_required_summary.py --exp_dir "$exp_path" --status done || return 1
}

for exp in "${EXPERIMENTS[@]}"; do
  contains_exp "$exp" || continue
  run_one "$exp" || true
done

if [ "$FAILURES" -gt 0 ]; then
  echo "$FAILURES new required experiment step(s) failed. See $FAIL_LOG" >&2
  exit 1
fi

echo "New required training/test flow complete."
