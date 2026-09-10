#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
LEVIR_ROOT="${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}"
SECOND_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
OVERWRITE=0
ONLY_EXP=""
DRY_RUN=0
FAIL_LOG="$EXP_ROOT/7_3_followup_train_failures.log"

EXPERIMENTS=(
  levir_mci_ultrashort_caption_ft_100_lr005
  levir_mci_ultrashort_caption_ft_50_lr005
  levir_mci_ultrashort_caption_ft_100_lr002
  second_cc_crossattn_pd05_lsem0000
  second_cc_crossattn_pd00_lsem0000
)

usage() {
  echo "Usage: bash scripts/run_7_3_followup_train.sh [--only_exp EXP] [--overwrite] [--dry_run]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --only_exp) ONLY_EXP="$2"; shift 2 ;;
    --force|--overwrite) OVERWRITE=1; shift ;;
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

has_required_final_checkpoint() {
  local exp_path="$1"
  local expected_step="$2"
  [ -n "$expected_step" ] || return 1
  find "$exp_path/snapshots" -type f \( -name "*checkpoint_${expected_step}.pt" -o -name "*checkpoint_${expected_step}.pth" \) -print -quit 2>/dev/null | grep -q .
}

has_snapshots() {
  local exp_path="$1"
  find "$exp_path/snapshots" -type f \( -name "*.pt" -o -name "*.pth" \) -print -quit 2>/dev/null | grep -q .
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

print_effective_train_env() {
  printf 'TRAIN ENV:'
  for key in EXP_NAME DATASET DATA_ROOT BASE_CFG MODEL_TYPE INIT_CHECKPOINT LR FINETUNE_STEPS SAVE_INTERVAL EVAL_INTERVAL SNAPSHOT_INTERVAL USE_CHANGE_MASK ENABLE_AUX_MASK USE_AUX_SEMANTIC USE_SEMANTIC_MAPS SEMANTIC_INPUT_MODE USE_SEMANTIC_PARTIAL_DETACH SEMANTIC_DETACH_RATIO LMASK LSEM USE_FEATURE_REWEIGHT SELECTION_STRATEGY; do
    local value="${!key-}"
    if [ -n "$value" ]; then printf ' %s=%q' "$key" "$value"; fi
  done
  printf ' bash scripts/_run_paper_training.sh\n'
}

clear_exp_env() {
  unset BASE_CFG MODEL_TYPE DATASET DATA_ROOT ANNO CHANGEFLAG_JSON EVAL_CHANGE_NOCHANGE_SPLIT PAPER_SELECTION_MODE
  unset USE_CHANGE_MASK MASK_TYPE NUM_MASK_CLASSES ENABLE_AUX_MASK USE_AUX_SEMANTIC USE_SEMANTIC_MAPS SEMANTIC_INPUT_MODE NUM_SEMANTIC_CLASSES
  unset USE_SEMANTIC_PARTIAL_DETACH SEMANTIC_DETACH_RATIO SEMANTIC_FUSION_GAMMA_INIT USE_FEATURE_REWEIGHT LMASK LSEM MASK_LOSS_TYPE SEMANTIC_LOSS_TYPE
  unset AUX_WARMUP_START_RATIO AUX_WARMUP_END_RATIO USE_AUX_WARMUP SELECTION_STRATEGY LR MAX_ITER FINETUNE_STEPS SNAPSHOT_INTERVAL SAVE_INTERVAL EVAL_INTERVAL LOG_INTERVAL INIT_CHECKPOINT
}

resolve_levir_init_checkpoint() {
  if [ -n "${LEVIR_WEAK_INIT_CHECKPOINT:-}" ]; then
    INIT_CHECKPOINT="$LEVIR_WEAK_INIT_CHECKPOINT"
    export INIT_CHECKPOINT
    return 0
  fi
  local source_exp="$EXP_ROOT/levir_mci_weak_pd08_lm003_ls001_noreweight"
  local txt="$source_exp/best_checkpoint.txt"
  if [ -s "$txt" ]; then
    INIT_CHECKPOINT="$(head -n 1 "$txt")"
    export INIT_CHECKPOINT
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    INIT_CHECKPOINT="<best checkpoint from levir_mci_weak_pd08_lm003_ls001_noreweight>"
    export INIT_CHECKPOINT
    return 0
  fi
  "$PYTHON" scripts/find_best_weak_checkpoint.py \
    --experiments_root "$EXP_ROOT" \
    --exp_names levir_mci_weak_pd08_lm003_ls001_noreweight \
    --output_json "$EXP_ROOT/levir_mci_short_finetune_init.json" \
    --output_txt "$EXP_ROOT/levir_mci_short_finetune_init.txt"
  INIT_CHECKPOINT="$(head -n 1 "$EXP_ROOT/levir_mci_short_finetune_init.txt")"
  export INIT_CHECKPOINT
}

configure_common() {
  local exp="$1"
  clear_exp_env
  export EXP_DIR="$EXP_ROOT"
  export EXP_NAME="$exp"
  export PAPER_SELECTION_MODE=1
  export USE_AUX_WARMUP=0
  export SELECTION_STRATEGY=spice_constrained_balanced
  export USE_FEATURE_REWEIGHT=0
}

configure_exp() {
  local exp="$1"
  configure_common "$exp"
  case "$exp" in
    levir_mci_ultrashort_caption_ft_100_lr005)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_ROOT" ANNO="$LEVIR_ROOT/levir_mci_captions_reformat.json" CHANGEFLAG_JSON="$LEVIR_ROOT/LevirCCcaptions.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.0 LMASK=0.0 LSEM=0.0 LR="${LEVIR_ULTRASHORT_LR005:-0.00001}" FINETUNE_STEPS=100 SAVE_INTERVAL=50 EVAL_INTERVAL=50 SNAPSHOT_INTERVAL=50 LOG_INTERVAL=10
      resolve_levir_init_checkpoint ;;
    levir_mci_ultrashort_caption_ft_50_lr005)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_ROOT" ANNO="$LEVIR_ROOT/levir_mci_captions_reformat.json" CHANGEFLAG_JSON="$LEVIR_ROOT/LevirCCcaptions.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.0 LMASK=0.0 LSEM=0.0 LR="${LEVIR_ULTRASHORT_LR005:-0.00001}" FINETUNE_STEPS=50 SAVE_INTERVAL=25 EVAL_INTERVAL=25 SNAPSHOT_INTERVAL=25 LOG_INTERVAL=10
      resolve_levir_init_checkpoint ;;
    levir_mci_ultrashort_caption_ft_100_lr002)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_ROOT" ANNO="$LEVIR_ROOT/levir_mci_captions_reformat.json" CHANGEFLAG_JSON="$LEVIR_ROOT/LevirCCcaptions.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_levir_mci_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.0 LMASK=0.0 LSEM=0.0 LR="${LEVIR_ULTRASHORT_LR002:-0.000004}" FINETUNE_STEPS=100 SAVE_INTERVAL=50 EVAL_INTERVAL=50 SNAPSHOT_INTERVAL=50 LOG_INTERVAL=10
      resolve_levir_init_checkpoint ;;
    second_cc_crossattn_pd05_lsem0000)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5 LMASK=0.0 LSEM=0.0 ;;
    second_cc_crossattn_pd00_lsem0000)
      export DATASET=second_cc DATA_ROOT="$SECOND_ROOT" ANNO="$SECOND_ROOT/second_cc_aug_captions_reformat.json" CHANGEFLAG_JSON="$SECOND_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
      export BASE_CFG="configs/dynamic/transformer_second_cc_aug_sgc_card.yaml" MODEL_TYPE=sgc_card USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.0 LMASK=0.0 LSEM=0.0 ;;
    *) echo "Unknown 7.3 follow-up training experiment: $exp" >&2; return 2 ;;
  esac
}

run_one() {
  local exp="$1"
  configure_exp "$exp" || { record_failure "configure $exp" "Unknown experiment."; return 1; }
  if [ "$DRY_RUN" -eq 0 ]; then
    assert_data_ready || { record_failure "data readiness $exp" "Set LEVIR_MCI_ROOT/SECOND_CC_ROOT paths before running on the server."; return 1; }
  fi
  local exp_path="$EXP_ROOT/$exp"
  local expected_step="${FINETUNE_STEPS:-}"
  print_effective_train_env
  if [ "$OVERWRITE" -eq 0 ]; then
    if [ -n "$expected_step" ] && has_required_final_checkpoint "$exp_path" "$expected_step"; then
      echo "Skipping training; final checkpoint for expected step $expected_step already exists under $exp_path/snapshots"
      return 0
    fi
    if [ -z "$expected_step" ] && has_snapshots "$exp_path"; then
      echo "Skipping training; existing snapshots found under $exp_path/snapshots"
      return 0
    fi
  fi
  run_or_log "train $exp" bash scripts/_run_paper_training.sh || return 1
}

for exp in "${EXPERIMENTS[@]}"; do
  contains_exp "$exp" || continue
  run_one "$exp" || true
done

if [ "$FAILURES" -gt 0 ]; then
  echo "$FAILURES 7.3 follow-up training step(s) failed. See $FAIL_LOG" >&2
  exit 1
fi

echo "7.3 follow-up training flow complete."