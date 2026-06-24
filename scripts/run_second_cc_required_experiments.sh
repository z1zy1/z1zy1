#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
DATA_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
SKIP_TRAIN=0
ONLY_DATASET=""
ONLY_EXP=""
OVERWRITE=0
FAIL_LOG="$EXP_ROOT/second_cc_required_failures.log"

EXPERIMENTS=(
  second_cc_card_rgb_baseline
  second_cc_card_semantic_aux
  second_cc_card_semantic_crossattn
  second_cc_card_semantic_hardgate
  second_cc_ours_weak_coupled_final
  second_cc_mmodalcc_comparison
)

usage() {
  echo "Usage: bash scripts/run_second_cc_required_experiments.sh [--skip_train] [--only_dataset second_cc] [--only_exp EXP] [--overwrite]" >&2
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

if [ -n "$ONLY_DATASET" ] && [ "$ONLY_DATASET" != "second_cc" ]; then
  echo "Skipping SECOND-CC runner because --only_dataset=$ONLY_DATASET"
  exit 0
fi

mkdir -p "$EXP_ROOT"
: > "$FAIL_LOG"
FAILURES=0

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

configure_second_env() {
  local exp="$1"
  unset NUM_MASK_CLASSES MASK_LOSS_TYPE SEMANTIC_LOSS_TYPE NUM_SEMANTIC_CLASSES REWEIGHT_ALPHA DETACH_REWEIGHT_MASK
  export EXP_DIR="$EXP_ROOT"
  export EXP_NAME="$exp"
  export DATASET="second_cc"
  export DATA_ROOT="$DATA_ROOT"
  export ANNO="$DATA_ROOT/second_cc_aug_captions_reformat.json"
  export CHANGEFLAG_JSON="$DATA_ROOT/SECOND-CC-AUG.json"
  export EVAL_CHANGE_NOCHANGE_SPLIT=1
  export PAPER_SELECTION_MODE=1
  export ALLOW_MISSING_PSEUDO_MASK=0
  export USE_CHANGE_MASK=0
  export MASK_TYPE=binary
  export ENABLE_AUX_MASK=0
  export USE_AUX_SEMANTIC=0
  export USE_SEMANTIC_MAPS=0
  export SEMANTIC_INPUT_MODE=none
  export USE_SEMANTIC_PARTIAL_DETACH=0
  export USE_FEATURE_REWEIGHT=0
  export LMASK=0
  export LSEM=0
  case "$exp" in
    second_cc_card_rgb_baseline)
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_baseline.yaml" MODEL_TYPE=card ;;
    second_cc_card_semantic_aux)
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=aux LSEM="${LSEM:-0.005}" SEMANTIC_LOSS_TYPE=ce_dice ;;
    second_cc_card_semantic_crossattn)
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention LSEM="${LSEM:-0.005}" SEMANTIC_LOSS_TYPE=ce_dice ;;
    second_cc_card_semantic_hardgate)
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=hard_gate LSEM="${LSEM:-0.005}" SEMANTIC_LOSS_TYPE=ce_dice ;;
    second_cc_ours_weak_coupled_final)
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=weak_coupled USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO="${SEMANTIC_DETACH_RATIO:-0.5}" LSEM="${LSEM:-0.005}" SEMANTIC_LOSS_TYPE=ce_dice ;;
  esac
}

train_script_for() {
  local exp="$1"
  echo "scripts/train_${exp}.sh"
}

for exp in "${EXPERIMENTS[@]}"; do
  contains_exp "$exp" || continue
  if [ "$exp" = "second_cc_mmodalcc_comparison" ]; then
    run_or_log "external MModalCC $exp" bash scripts/run_second_cc_mmodalcc_comparison.sh || continue
    continue
  fi

  configure_second_env "$exp"
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
    ALLOW_NEG=()
    if [ "$exp" = "second_cc_card_semantic_hardgate" ]; then ALLOW_NEG=(--allow_negative_ablation); fi
    run_or_log "select best $exp" python scripts/select_best_snapshot_for_paper.py --exp_dir "$EXP_PATH" "${ALLOW_NEG[@]}" || continue
  else
    echo "Using existing paper selection: $EXP_PATH/best_snapshot_for_paper.json"
  fi

  if [ "$OVERWRITE" -eq 1 ] || [ ! -f "$EXP_PATH/test_paper_best_result.json" ]; then
    run_or_log "test paper best $exp" bash scripts/test_specific_snapshot_sgc_card.sh --exp_dir "$EXP_PATH" --checkpoint "$EXP_PATH/best_for_paper.pth" --tag paper_best || continue
  else
    echo "Using existing paper-best test result: $EXP_PATH/test_paper_best_result.json"
  fi
done

python scripts/summarize_paper_required_experiments.py --experiments_root "$EXP_ROOT" --dataset second_cc

if [ "$FAILURES" -gt 0 ]; then
  echo "$FAILURES SECOND-CC step(s) failed. See $FAIL_LOG" >&2
  exit 1
fi

echo "SECOND-CC required experiment flow complete."
