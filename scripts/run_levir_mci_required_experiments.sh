#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
DATA_ROOT="${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}"
SKIP_TRAIN=0
ONLY_DATASET=""
ONLY_EXP=""
OVERWRITE=0
FAIL_LOG="$EXP_ROOT/levir_mci_required_failures.log"

EXPERIMENTS=(
  levir_mci_card_baseline
  levir_mci_card_mask_loss
  levir_mci_card_semantic_loss
  levir_mci_card_mask_semantic
  levir_mci_card_mask_semantic_pd05
  levir_mci_card_mask_semantic_pd05_noreweight
  levir_mci_card_mask_semantic_pd05_reweight
  levir_mci_ours_weak_coupled_final
)

usage() {
  echo "Usage: bash scripts/run_levir_mci_required_experiments.sh [--skip_train] [--only_dataset levir_mci] [--only_exp EXP] [--overwrite]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip_train) SKIP_TRAIN=1; shift ;;
    --only_dataset) ONLY_DATASET="$2"; shift 2 ;;
    --only_exp) ONLY_EXP="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -n "$ONLY_DATASET" ] && [ "$ONLY_DATASET" != "levir_mci" ]; then
  echo "Skipping LEVIR-MCI runner because --only_dataset=$ONLY_DATASET"
  exit 0
fi

mkdir -p "$EXP_ROOT"
: > "$FAIL_LOG"
FAILURES=0
USER_LMASK="${LMASK:-}"
USER_LSEM="${LSEM:-}"
USER_SEMANTIC_DETACH_RATIO="${SEMANTIC_DETACH_RATIO:-}"
USER_REWEIGHT_ALPHA="${REWEIGHT_ALPHA:-}"

contains_exp() {
  local target="$1"
  if [ -z "$ONLY_EXP" ]; then return 0; fi
  [ "$target" = "$ONLY_EXP" ]
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

configure_levir_env() {
  local exp="$1"
  unset NUM_MASK_CLASSES MASK_LOSS_TYPE SEMANTIC_LOSS_TYPE NUM_SEMANTIC_CLASSES REWEIGHT_ALPHA DETACH_REWEIGHT_MASK LMASK LSEM SEMANTIC_DETACH_RATIO
  export EXP_DIR="$EXP_ROOT"
  export EXP_NAME="$exp"
  export DATASET="levir_mci"
  export DATA_ROOT="$DATA_ROOT"
  export ANNO="$DATA_ROOT/levir_mci_captions_reformat.json"
  export CHANGEFLAG_JSON="$DATA_ROOT/LevirCCcaptions.json"
  export EVAL_CHANGE_NOCHANGE_SPLIT=1
  export PAPER_SELECTION_MODE=1
  export ALLOW_MISSING_PSEUDO_MASK=0
  export USE_FEATURE_REWEIGHT=0
  export USE_SEMANTIC_PARTIAL_DETACH=0
  export SEMANTIC_DETACH_RATIO="${USER_SEMANTIC_DETACH_RATIO:-0.5}"
  case "$exp" in
    levir_mci_card_baseline)
      export BASE_CFG="configs/dynamic/transformer_levir_mci_baseline.yaml" MODEL_TYPE=card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none LMASK=0.0 LSEM=0.0 ;;
    levir_mci_card_mask_loss)
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=1 MASK_TYPE=multiclass NUM_MASK_CLASSES=3 ENABLE_AUX_MASK=1 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none LMASK="${USER_LMASK:-0.003}" LSEM=0.0 MASK_LOSS_TYPE=ce_dice ;;
    levir_mci_card_semantic_loss)
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=aux LMASK=0.0 LSEM="${USER_LSEM:-0.005}" SEMANTIC_LOSS_TYPE=multilabel_bce ;;
    levir_mci_card_mask_semantic)
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=1 MASK_TYPE=multiclass NUM_MASK_CLASSES=3 ENABLE_AUX_MASK=1 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=aux LMASK="${USER_LMASK:-0.003}" LSEM="${USER_LSEM:-0.005}" MASK_LOSS_TYPE=ce_dice SEMANTIC_LOSS_TYPE=multilabel_bce ;;
    levir_mci_card_mask_semantic_pd05|levir_mci_card_mask_semantic_pd05_noreweight|levir_mci_ours_weak_coupled_final)
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=1 MASK_TYPE=multiclass NUM_MASK_CLASSES=3 ENABLE_AUX_MASK=1 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=aux USE_SEMANTIC_PARTIAL_DETACH=1 LMASK="${USER_LMASK:-0.003}" LSEM="${USER_LSEM:-0.005}" MASK_LOSS_TYPE=ce_dice SEMANTIC_LOSS_TYPE=multilabel_bce
      if [ "$exp" = "levir_mci_ours_weak_coupled_final" ]; then export SEMANTIC_INPUT_MODE=weak_coupled; fi ;;
    levir_mci_card_mask_semantic_pd05_reweight)
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=1 MASK_TYPE=multiclass NUM_MASK_CLASSES=3 ENABLE_AUX_MASK=1 USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=aux USE_SEMANTIC_PARTIAL_DETACH=1 USE_FEATURE_REWEIGHT=1 REWEIGHT_ALPHA="${USER_REWEIGHT_ALPHA:-0.2}" DETACH_REWEIGHT_MASK=1 LMASK="${USER_LMASK:-0.003}" LSEM="${USER_LSEM:-0.005}" MASK_LOSS_TYPE=ce_dice SEMANTIC_LOSS_TYPE=multilabel_bce ;;
  esac
}

train_script_for() {
  local exp="$1"
  echo "scripts/train_${exp}.sh"
}

for exp in "${EXPERIMENTS[@]}"; do
  contains_exp "$exp" || continue
  configure_levir_env "$exp"
  EXP_PATH="$EXP_ROOT/$exp"
  if [ "$SKIP_TRAIN" -eq 0 ]; then
    run_or_log "train $exp" bash "$(train_script_for "$exp")" || continue
  else
    echo "Skipping training for $exp"
  fi

  if [ "$OVERWRITE" -eq 1 ] || [ ! -f "$EXP_PATH/eval_snapshots.csv" ]; then
    export FORCE_INFER="$OVERWRITE"
    run_or_log "eval snapshots $exp" bash scripts/eval_all_snapshots_sgc_card.sh --exp_dir "$EXP_PATH" || continue
  else
    echo "Using existing eval CSV: $EXP_PATH/eval_snapshots.csv"
  fi

  if [ "$OVERWRITE" -eq 1 ] || [ ! -f "$EXP_PATH/best_snapshot_for_paper.json" ]; then
    run_or_log "select best $exp" python scripts/select_best_snapshot_for_paper.py --exp_dir "$EXP_PATH" || continue
  else
    echo "Using existing paper selection: $EXP_PATH/best_snapshot_for_paper.json"
  fi

  if [ "$OVERWRITE" -eq 1 ] || [ ! -f "$EXP_PATH/test_paper_best_result.json" ]; then
    run_or_log "test paper best $exp" bash scripts/test_specific_snapshot_sgc_card.sh --exp_dir "$EXP_PATH" --checkpoint "$EXP_PATH/best_for_paper.pth" --tag paper_best || continue
  else
    echo "Using existing paper-best test result: $EXP_PATH/test_paper_best_result.json"
  fi
done

python scripts/summarize_paper_required_experiments.py --experiments_root "$EXP_ROOT" --dataset levir_mci

if [ "$FAILURES" -gt 0 ]; then
  echo "$FAILURES LEVIR-MCI step(s) failed. See $FAIL_LOG" >&2
  exit 1
fi

echo "LEVIR-MCI required experiment flow complete."
