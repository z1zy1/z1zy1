#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
LEVIR_CC_ROOT="${LEVIR_CC_ROOT:-./Levir-CC}"
LEVIR_MCI_ROOT="${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}"
LEVIR_CC_SOURCE_EXP="${LEVIR_CC_SOURCE_EXP:-sgc_card_lm003_ls005_pd05_rw02_warmup}"
LEVIR_CC_SOURCE_MANIFEST="${LEVIR_CC_SOURCE_MANIFEST:-$EXP_ROOT/7_5_locked_manifest.json}"
ONLY_EXP=""
DRY_RUN=0
FAIL_LOG="${FAIL_LOG:-$EXP_ROOT/7_6_followup_train_failures.log}"

LEVIR_CC_SPECS=(
  'levir_cc_decft_cw100_s10_lr5e7,1.00,10,0.0000005'
  'levir_cc_decft_cw100_s10_lr1e6,1.00,10,0.000001'
  'levir_cc_decft_cw100_s20_lr5e7,1.00,20,0.0000005'
  'levir_cc_decft_cw100_s20_lr1e6,1.00,20,0.000001'
  'levir_cc_decft_cw101_s10_lr5e7,1.01,10,0.0000005'
  'levir_cc_decft_cw101_s10_lr1e6,1.01,10,0.000001'
  'levir_cc_decft_cw101_s20_lr5e7,1.01,20,0.0000005'
  'levir_cc_decft_cw101_s20_lr1e6,1.01,20,0.000001'
  'levir_cc_decft_cw102_s10_lr5e7,1.02,10,0.0000005'
  'levir_cc_decft_cw102_s10_lr1e6,1.02,10,0.000001'
  'levir_cc_decft_cw102_s20_lr5e7,1.02,20,0.0000005'
  'levir_cc_decft_cw102_s20_lr1e6,1.02,20,0.000001'
)
LEVIR_MCI_SPECS=(
  'levir_mci_masksemantic_repro_seed1111,1111'
  'levir_mci_masksemantic_repro_seed2222,2222'
  'levir_mci_masksemantic_repro_seed3333,3333'
)

usage() {
  echo 'Usage: bash scripts/run_7_6_followup_train.sh [--only_exp EXP] [--dry_run]' >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --only_exp) ONLY_EXP="$2"; shift 2 ;;
    --force|--overwrite)
      echo '7.6 refuses in-place overwrite; archive the old experiment directory or use a new EXP_ROOT.' >&2
      exit 2 ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$EXP_ROOT"
touch "$FAIL_LOG"
printf '\n[%s] run_7_6_followup_train start dry_run=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$DRY_RUN" >> "$FAIL_LOG"

record_failure() {
  echo "FAILED: $1${2:+ - $2}" | tee -a "$FAIL_LOG" >&2
}

resolve_levir_cc_source() {
  if [ "$DRY_RUN" -eq 1 ]; then
    LEVIR_CC_LOCKED_CHECKPOINT='<checkpoint from audited LEVIR-CC validation lock>'
    LEVIR_CC_SOURCE_CONFIG='<audited LEVIR-CC source config>'
    export LEVIR_CC_LOCKED_CHECKPOINT LEVIR_CC_SOURCE_CONFIG
    return 0
  fi
  local args=(
    "$PYTHON" scripts/resolve_7_6_levir_cc_source.py
    --expected_source_exp "$LEVIR_CC_SOURCE_EXP"
  )
  if [ -n "${LEVIR_CC_SOURCE_CHECKPOINT:-}" ]; then
    [ -n "${LEVIR_CC_SOURCE_SELECTION_JSON:-}" ] || {
      echo 'LEVIR_CC_SOURCE_CHECKPOINT requires LEVIR_CC_SOURCE_SELECTION_JSON.' >&2
      return 1
    }
    args+=(--selection_json "$LEVIR_CC_SOURCE_SELECTION_JSON" --checkpoint "$LEVIR_CC_SOURCE_CHECKPOINT")
  else
    args+=(--manifest "$LEVIR_CC_SOURCE_MANIFEST")
  fi
  local source_info=()
  mapfile -t source_info < <("${args[@]}")
  [ "${#source_info[@]}" -eq 3 ] || {
    echo 'LEVIR-CC validation-locked source audit failed.' >&2
    return 1
  }
  LEVIR_CC_LOCKED_CHECKPOINT="${source_info[0]}"
  LEVIR_CC_SOURCE_CONFIG="${source_info[1]}"
  LEVIR_CC_SOURCE_DIR="${source_info[2]}"
  export LEVIR_CC_LOCKED_CHECKPOINT LEVIR_CC_SOURCE_CONFIG LEVIR_CC_SOURCE_DIR
}

clear_exp_env() {
  unset EXP_NAME DATASET DATA_ROOT ANNO CHANGEFLAG_JSON FEATURE_ROOT BASE_CFG MODEL_TYPE
  unset INIT_CHECKPOINT SEED LR MAX_ITER FINETUNE_STEPS FINETUNE_DECODER_ONLY
  unset SAVE_INTERVAL EVAL_INTERVAL SNAPSHOT_INTERVAL LOG_INTERVAL BATCH_SIZE
  unset USE_CHANGE_MASK MASK_TYPE NUM_MASK_CLASSES ENABLE_AUX_MASK USE_AUX_SEMANTIC
  unset USE_SEMANTIC_MAPS SEMANTIC_INPUT_MODE NUM_SEMANTIC_CLASSES
  unset USE_SEMANTIC_PARTIAL_DETACH SEMANTIC_DETACH_RATIO SEMANTIC_FUSION_GAMMA_INIT SEMANTIC_FUSION_GAMMA_MAX
  unset USE_FEATURE_REWEIGHT DETACH_REWEIGHT_MASK REWEIGHT_ALPHA LMASK LSEM
  unset USE_AUX_WARMUP AUX_WARMUP_START_RATIO AUX_WARMUP_END_RATIO ALLOW_MISSING_PSEUDO_MASK
  unset USE_CONTENT_WORD_WEIGHT CONTENT_WORD_WEIGHT NORMALIZE_CONTENT_WORD_WEIGHTS
  unset MASK_LOSS_TYPE SEMANTIC_LOSS_TYPE PAPER_SELECTION_MODE SELECTION_STRATEGY
}

configure_common() {
  clear_exp_env
  export EXP_DIR="$EXP_ROOT" EXP_NAME="$1" PAPER_SELECTION_MODE=1
  export USE_FEATURE_REWEIGHT=0 DETACH_REWEIGHT_MASK=1 REWEIGHT_ALPHA=0.2
  export ALLOW_MISSING_PSEUDO_MASK=0
}

configure_levir_cc() {
  local exp="$1" content_weight="$2" steps="$3" lr="$4"
  configure_common "$exp"
  export DATASET=levir_cc DATA_ROOT="$LEVIR_CC_ROOT"
  export ANNO="$LEVIR_CC_ROOT/levir_cc_captions_reformat.json"
  export BASE_CFG=configs/dynamic/transformer_levir_cc_sgc_card.yaml MODEL_TYPE=sgc_card
  # Preserve the validation-locked source model and frozen detector forward exactly.
  # Decoder-only here means the complete speaker is trainable; CARD/change detector is frozen.
  export USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=1 USE_AUX_SEMANTIC=1
  export USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none
  export USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5
  export USE_FEATURE_REWEIGHT=1 DETACH_REWEIGHT_MASK=1 REWEIGHT_ALPHA=0.2
  export USE_AUX_WARMUP=1 AUX_WARMUP_START_RATIO=0.30 AUX_WARMUP_END_RATIO=0.70
  export ALLOW_MISSING_PSEUDO_MASK=1 LMASK=0.003 LSEM=0.005
  export MASK_LOSS_TYPE=bce_dice SEMANTIC_LOSS_TYPE=multilabel_bce
  export USE_CONTENT_WORD_WEIGHT=1 CONTENT_WORD_WEIGHT="$content_weight" NORMALIZE_CONTENT_WORD_WEIGHTS=1
  export FINETUNE_STEPS="$steps" FINETUNE_DECODER_ONLY=1 LR="$lr"
  export SAVE_INTERVAL=5 EVAL_INTERVAL=5 SNAPSHOT_INTERVAL=5 LOG_INTERVAL=5 SEED=1111
  if [ -z "${LEVIR_CC_LOCKED_CHECKPOINT:-}" ]; then
    resolve_levir_cc_source || return 1
  fi
  BASE_CFG="$LEVIR_CC_SOURCE_CONFIG"
  INIT_CHECKPOINT="$LEVIR_CC_LOCKED_CHECKPOINT"
  export BASE_CFG INIT_CHECKPOINT
}

configure_levir_mci() {
  local exp="$1" seed="$2"
  configure_common "$exp"
  export DATASET=levir_mci DATA_ROOT="$LEVIR_MCI_ROOT"
  export ANNO="$LEVIR_MCI_ROOT/levir_mci_captions_reformat.json"
  export CHANGEFLAG_JSON="$LEVIR_MCI_ROOT/LevirCCcaptions.json"
  export BASE_CFG=configs/dynamic/transformer_levir_mci_sgc_card.yaml MODEL_TYPE=sgc_card
  export USE_CHANGE_MASK=1 MASK_TYPE=multiclass NUM_MASK_CLASSES=3 ENABLE_AUX_MASK=1
  export USE_AUX_SEMANTIC=1 USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=aux
  export USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.5
  export USE_AUX_WARMUP=1 AUX_WARMUP_START_RATIO=0.30 AUX_WARMUP_END_RATIO=0.70
  export LMASK=0.003 LSEM=0.005 MASK_LOSS_TYPE=ce_dice SEMANTIC_LOSS_TYPE=multilabel_bce
  export USE_CONTENT_WORD_WEIGHT=0 NORMALIZE_CONTENT_WORD_WEIGHTS=0
  export MAX_ITER=10000 FINETUNE_DECODER_ONLY=0 LR=0.0002
  export SAVE_INTERVAL=1000 EVAL_INTERVAL=1000 SNAPSHOT_INTERVAL=1000 LOG_INTERVAL=100 SEED="$seed"
}

has_final_checkpoint() {
  find "$EXP_ROOT/$1/snapshots" -type f \( -name "*checkpoint_$2.pt" -o -name "*checkpoint_$2.pth" \) -print -quit 2>/dev/null | grep -q .
}

has_any_output() {
  [ -d "$EXP_ROOT/$1" ] && find "$EXP_ROOT/$1" -mindepth 1 -print -quit 2>/dev/null | grep -q .
}

run_configured() {
  local exp="$1" final_step="$2"
  [ -z "$ONLY_EXP" ] || [ "$ONLY_EXP" = "$exp" ] || return 0
  if [ "$DRY_RUN" -eq 0 ]; then
    [ -d "$DATA_ROOT" ] && [ -f "$ANNO" ] || { record_failure "$exp" 'dataset root or annotation missing'; return 1; }
  fi
  if has_final_checkpoint "$exp" "$final_step"; then
    echo "Skipping $exp; final checkpoint $final_step exists."
    return 0
  fi
  if has_any_output "$exp"; then
    record_failure "$exp" 'non-empty incomplete directory; inspect/archive it or use a new EXP_ROOT'
    return 1
  fi
  printf 'TRAIN ENV: EXP_NAME=%q DATASET=%q SEED=%q INIT_CHECKPOINT=%q LR=%q MAX_ITER=%q FINETUNE_STEPS=%q FINETUNE_DECODER_ONLY=%q CONTENT_WORD_WEIGHT=%q SAVE_INTERVAL=%q\n' \
    "$EXP_NAME" "$DATASET" "$SEED" "${INIT_CHECKPOINT:-}" "$LR" "${MAX_ITER:-}" "${FINETUNE_STEPS:-}" "$FINETUNE_DECODER_ONLY" "${CONTENT_WORD_WEIGHT:-}" "$SAVE_INTERVAL"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo 'DRY RUN: bash scripts/_run_paper_training.sh'
  else
    bash scripts/_run_paper_training.sh
  fi
}

known_exp=0
for spec in "${LEVIR_CC_SPECS[@]}"; do
  IFS=',' read -r exp content_weight steps lr <<< "$spec"
  if [ -z "$ONLY_EXP" ] || [ "$ONLY_EXP" = "$exp" ]; then
    known_exp=1
    configure_levir_cc "$exp" "$content_weight" "$steps" "$lr"
    run_configured "$exp" "$steps"
  fi
done

for spec in "${LEVIR_MCI_SPECS[@]}"; do
  IFS=',' read -r exp seed <<< "$spec"
  if [ -z "$ONLY_EXP" ] || [ "$ONLY_EXP" = "$exp" ]; then
    known_exp=1
    configure_levir_mci "$exp" "$seed"
    run_configured "$exp" 10000
  fi
done

if [ -n "$ONLY_EXP" ] && [ "$known_exp" -eq 0 ]; then
  echo "Unknown 7.6 experiment: $ONLY_EXP" >&2
  exit 2
fi

echo '7.6 follow-up training flow complete.'
