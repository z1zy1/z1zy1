#!/usr/bin/env bash
set -euo pipefail

# Fine-tune target-domain models from the LEVIR-MCI seed-1111 step-5000
# checkpoint. Checkpoint selection remains validation-only; testing is
# intentionally a separate, explicit step.

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
MCI_INIT_CHECKPOINT="${MCI_INIT_CHECKPOINT:-$EXP_ROOT/levir_mci_masksemantic_repro_seed1111/snapshots/levir_mci_masksemantic_repro_seed1111_checkpoint_5000.pt}"
LEVIR_CC_ROOT="${LEVIR_CC_ROOT:-./Levir-CC}"
SECOND_CC_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
LEVIR_CC_BASELINE_VAL_METRICS="${LEVIR_CC_BASELINE_VAL_METRICS:-$EXP_ROOT/card_levir_cc_baseline/baseline_best_checkpoint.json}"
SECOND_CC_BASELINE_VAL_METRICS="${SECOND_CC_BASELINE_VAL_METRICS:-$EXP_ROOT/second_cc_card_rgb_baseline/baseline_best_checkpoint.json}"
SECOND_CC_EXP_PREFIX="${SECOND_CC_EXP_PREFIX:-second_cc_mci_xattn_lsem0}"
SECOND_CC_LR="${SECOND_CC_LR:-0.0001}"
SECOND_CC_SEMANTIC_DETACH_RATIO="${SECOND_CC_SEMANTIC_DETACH_RATIO:-0.5}"
SEEDS="${SEEDS:-1111 2222 3333}"
ONLY_DATASET=""
ONLY_SEED=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_SELECT=0

usage() {
  cat <<'EOF' >&2
Usage: bash scripts/run_mci_transfer_finetune.sh [options]

Runs target-domain fine-tuning initialized from the LEVIR-MCI seed-1111,
step-5000 checkpoint. Each selected run trains from scratch in a new output
directory, then selects its checkpoint using only the target validation set.

Options:
  --only_dataset DATASET  Run levir_cc or second_cc only.
  --only_seed SEED        Run one seed from SEEDS only.
  --skip_train            Select existing target runs without training.
  --skip_select           Train only; do not select a checkpoint.
  --dry_run, --dry-run    Print training and selection commands only.

SECOND-CC defaults can be overridden with SECOND_CC_EXP_PREFIX,
SECOND_CC_LR, and SECOND_CC_SEMANTIC_DETACH_RATIO. For the full recommended
matrix, use scripts/run_second_cc_mci_crossattn_matrix.sh.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --only_dataset) ONLY_DATASET="$2"; shift 2 ;;
    --only_seed) ONLY_SEED="$2"; shift 2 ;;
    --skip_train) SKIP_TRAIN=1; shift ;;
    --skip_select) SKIP_SELECT=1; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "$ONLY_DATASET" in
  ''|levir_cc|second_cc) ;;
  *) echo "Unknown --only_dataset value: $ONLY_DATASET" >&2; exit 2 ;;
esac

contains_seed() {
  local wanted="$1" seed
  for seed in $SEEDS; do [ "$seed" = "$wanted" ] && return 0; done
  return 1
}
if [ -n "$ONLY_SEED" ] && ! contains_seed "$ONLY_SEED"; then
  echo "--only_seed must be listed in SEEDS: $ONLY_SEED" >&2
  exit 2
fi

clear_exp_env() {
  unset EXP_NAME DATASET DATA_ROOT FEATURE_ROOT BASE_CFG MODEL_TYPE
  unset INIT_CHECKPOINT SEED LR MAX_ITER FINETUNE_STEPS FINETUNE_DECODER_ONLY
  unset SAVE_INTERVAL EVAL_INTERVAL SNAPSHOT_INTERVAL LOG_INTERVAL BATCH_SIZE
  unset USE_CHANGE_MASK MASK_TYPE NUM_MASK_CLASSES ENABLE_AUX_MASK USE_AUX_SEMANTIC
  unset USE_SEMANTIC_MAPS SEMANTIC_INPUT_MODE NUM_SEMANTIC_CLASSES
  unset USE_SEMANTIC_PARTIAL_DETACH SEMANTIC_DETACH_RATIO USE_FEATURE_REWEIGHT
  unset DETACH_REWEIGHT_MASK REWEIGHT_ALPHA LMASK LSEM USE_AUX_WARMUP
  unset AUX_WARMUP_START_RATIO AUX_WARMUP_END_RATIO ALLOW_MISSING_PSEUDO_MASK
  unset MASK_LOSS_TYPE SEMANTIC_LOSS_TYPE PAPER_SELECTION_MODE SELECTION_STRATEGY
}

configure_levir_cc() {
  local seed="$1"
  clear_exp_env
  export EXP_DIR="$EXP_ROOT" EXP_NAME="levir_cc_mci_init_seed${seed}"
  export DATASET=levir_cc DATA_ROOT="$LEVIR_CC_ROOT" FEATURE_ROOT="$LEVIR_CC_ROOT/features"
  export BASE_CFG=configs/dynamic/transformer_levir_cc_sgc_card.yaml MODEL_TYPE=sgc_card
  # Target-domain LEVIR-CC architecture: source-incompatible mask output is reinitialized.
  export USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=1
  export USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none
  export USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5
  export USE_FEATURE_REWEIGHT=1 DETACH_REWEIGHT_MASK=1 REWEIGHT_ALPHA=0.2
  export ALLOW_MISSING_PSEUDO_MASK=1 LMASK=0.003 LSEM=0.005
  export MASK_LOSS_TYPE=bce_dice SEMANTIC_LOSS_TYPE=multilabel_bce
  export USE_AUX_WARMUP=1 AUX_WARMUP_START_RATIO=0.30 AUX_WARMUP_END_RATIO=0.70
  export INIT_CHECKPOINT="$MCI_INIT_CHECKPOINT" SEED="$seed" LR=0.0002 MAX_ITER=10000
  export SAVE_INTERVAL=1000 EVAL_INTERVAL=1000 SNAPSHOT_INTERVAL=1000 LOG_INTERVAL=100
  export FINETUNE_DECODER_ONLY=0 PAPER_SELECTION_MODE=1 SELECTION_STRATEGY=val_baseline_pareto
  BASELINE_METRICS="$LEVIR_CC_BASELINE_VAL_METRICS"
}

configure_second_cc() {
  local seed="$1"
  clear_exp_env
  export EXP_DIR="$EXP_ROOT" EXP_NAME="${SECOND_CC_EXP_PREFIX}_seed${seed}"
  export DATASET=second_cc DATA_ROOT="$SECOND_CC_ROOT" FEATURE_ROOT="$SECOND_CC_ROOT/features"
  export BASE_CFG=configs/dynamic/transformer_second_cc_aug_sgc_card.yaml MODEL_TYPE=sgc_card
  # SECOND-CC has semantic maps and strong spatial/registration noise. Use them
  # as a cross-attention input, but do not force an incompatible MCI semantic
  # head to optimize an auxiliary loss. The decoder/semantic heads that do not
  # match the MCI checkpoint are deliberately reinitialized by the loader.
  export USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0
  export USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7
  export USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO="$SECOND_CC_SEMANTIC_DETACH_RATIO"
  export USE_FEATURE_REWEIGHT=0 DETACH_REWEIGHT_MASK=1 REWEIGHT_ALPHA=0.2
  export ALLOW_MISSING_PSEUDO_MASK=0 LMASK=0.0 LSEM=0.0
  export USE_AUX_WARMUP=0 AUX_WARMUP_START_RATIO=0.30 AUX_WARMUP_END_RATIO=0.70
  export INIT_CHECKPOINT="$MCI_INIT_CHECKPOINT" SEED="$seed" LR="$SECOND_CC_LR" MAX_ITER=10000
  export SAVE_INTERVAL=1000 EVAL_INTERVAL=1000 SNAPSHOT_INTERVAL=1000 LOG_INTERVAL=100
  export FINETUNE_DECODER_ONLY=0 PAPER_SELECTION_MODE=1 SELECTION_STRATEGY=val_baseline_pareto
  BASELINE_METRICS="$SECOND_CC_BASELINE_VAL_METRICS"
}

has_final_checkpoint() {
  find "$EXP_ROOT/$1/snapshots" -type f \( -name "*checkpoint_10000.pt" -o -name "*checkpoint_10000.pth" \) -print -quit 2>/dev/null | grep -q .
}

has_any_output() {
  [ -d "$EXP_ROOT/$1" ] && find "$EXP_ROOT/$1" -mindepth 1 -print -quit 2>/dev/null | grep -q .
}

train_one() {
  if [ "$DRY_RUN" -eq 0 ]; then
    [ -f "$MCI_INIT_CHECKPOINT" ] || { echo "Missing MCI init checkpoint: $MCI_INIT_CHECKPOINT" >&2; return 1; }
    [ -d "$DATA_ROOT" ] && [ -d "$FEATURE_ROOT" ] || { echo "Missing target dataset or features for $EXP_NAME" >&2; return 1; }
  fi
  if has_final_checkpoint "$EXP_NAME"; then
    echo "Skipping $EXP_NAME; final checkpoint already exists."
    return 0
  fi
  if has_any_output "$EXP_NAME"; then
    echo "Refusing non-empty incomplete experiment directory: $EXP_ROOT/$EXP_NAME" >&2
    return 1
  fi
  printf 'TRANSFER TRAIN: exp=%q dataset=%q seed=%q init=%q\n' "$EXP_NAME" "$DATASET" "$SEED" "$INIT_CHECKPOINT"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo 'DRY RUN: bash scripts/_run_paper_training.sh'
  else
    bash scripts/_run_paper_training.sh
  fi
}

select_one() {
  local output_json="$EXP_ROOT/$EXP_NAME/best_checkpoint.json"
  local command=(
    "$PYTHON" scripts/select_best_checkpoint.py
    --exp_dir "$EXP_ROOT/$EXP_NAME"
    --strategy val_baseline_pareto
    --baseline_metrics "$BASELINE_METRICS"
    --protected_metrics Bleu_1,Bleu_2,Bleu_3,Bleu_4,METEOR,ROUGE_L,CIDEr
    --baseline_tolerance 0
    --min_spice_gain 0
    --require_audited_validation_baseline
    --output_json "$output_json"
    --output_txt "$EXP_ROOT/$EXP_NAME/best_checkpoint.txt"
  )
  if [ "$DRY_RUN" -eq 0 ]; then
    [ -f "$BASELINE_METRICS" ] || { echo "Missing validation baseline: $BASELINE_METRICS" >&2; return 1; }
  fi
  if [ -s "$output_json" ]; then
    echo "Skipping existing validation selection: $output_json"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
  else
    "${command[@]}"
  fi
}

run_dataset() {
  local dataset="$1" seed
  [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "$dataset" ] || return 0
  for seed in $SEEDS; do
    [ -z "$ONLY_SEED" ] || [ "$ONLY_SEED" = "$seed" ] || continue
    case "$dataset" in
      levir_cc) configure_levir_cc "$seed" ;;
      second_cc) configure_second_cc "$seed" ;;
    esac
    [ "$SKIP_TRAIN" -eq 1 ] || train_one
    [ "$SKIP_SELECT" -eq 1 ] || select_one
  done
}

run_dataset levir_cc
run_dataset second_cc

echo 'Transfer training/validation selection complete. Do not use test metrics to choose a run.'
echo 'After reviewing each best_checkpoint.json, test one locked checkpoint per target dataset in a separate directory.'
