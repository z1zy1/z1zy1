#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
LEVIR_CC_ROOT="${LEVIR_CC_ROOT:-./Levir-CC}"
LEVIR_MCI_ROOT="${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}"
SECOND_CC_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
LEVIR_CC_SOURCE_EXP="${LEVIR_CC_SOURCE_EXP:-sgc_card_lm003_ls005_pd05_rw02_warmup}"
LEVIR_MCI_SOURCE_EXP="${LEVIR_MCI_SOURCE_EXP:-levir_mci_card_mask_semantic}"
SECOND_CC_SOURCE_EXP="${SECOND_CC_SOURCE_EXP:-second_cc_crossattn_pd08_lsem0000}"
ONLY_EXP=""
OVERWRITE=0
DRY_RUN=0
FAIL_LOG="${FAIL_LOG:-$EXP_ROOT/7_5_followup_train_failures.log}"
EXPERIMENTS=(
  levir_cc_caption_ft_cw103_norm_30_lr2e6
  levir_cc_caption_ft_cw105_norm_30_lr2e6
  levir_mci_masksemantic_caption_ft_cw103_norm_30_lr2e6
  levir_mci_masksemantic_caption_ft_cw105_norm_30_lr2e6
  second_cc_pd08_gamma005_ft_100_lr2e6
  second_cc_pd08_gamma010_ft_100_lr2e6
)

usage() {
  echo "Usage: bash scripts/run_7_5_followup_train.sh [--only_exp EXP] [--overwrite] [--dry_run]" >&2
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
touch "$FAIL_LOG"
printf '\n[%s] run_7_5_followup_train start dry_run=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$DRY_RUN" >> "$FAIL_LOG"
FAILURES=0

record_failure() {
  echo "FAILED: $1${2:+ - $2}" | tee -a "$FAIL_LOG"
  FAILURES=$((FAILURES + 1))
}

read_locked_checkpoint() {
  local source_path="$1"
  local artifact
  for artifact in best_checkpoint.json best_snapshot.json best_snapshot_v2.json best_snapshot_for_paper.json best_checkpoint.txt best_snapshot.txt; do
    artifact="$source_path/$artifact"
    [ -s "$artifact" ] || continue
    "$PYTHON" - "$artifact" "$source_path" <<'PY'
import json
import os
import sys

artifact, source = sys.argv[1:]
if artifact.lower().endswith('.txt'):
    raw = next((x.strip() for x in open(artifact, encoding='utf-8-sig') if x.strip()), '')
else:
    payload = json.load(open(artifact, encoding='utf-8-sig'))
    raw = ''
    for path in (
        ('selected_checkpoint_path',),
        ('selected_checkpoint',),
        ('best_snapshot',),
        ('snapshot_path',),
        ('best', 'snapshot_path'),
        ('copy_path',),
    ):
        value = payload
        for part in path:
            value = value.get(part, '') if isinstance(value, dict) else ''
        if value:
            raw = str(value)
            break
candidates = [
    raw,
    os.path.abspath(raw),
    os.path.join(source, raw),
    os.path.join(source, 'snapshots', os.path.basename(raw)),
]
print(next((os.path.normpath(p) for p in candidates if p and os.path.isfile(p)), raw))
PY
    return 0
  done
  return 1
}

resolve_init_checkpoint() {
  local explicit="$1"
  local source_exp="$2"
  local label="$3"
  if [ -n "$explicit" ]; then
    INIT_CHECKPOINT="$explicit"
  elif [ "$DRY_RUN" -eq 1 ]; then
    INIT_CHECKPOINT="<validation-locked checkpoint for $source_exp>"
  else
    INIT_CHECKPOINT="$(read_locked_checkpoint "$EXP_ROOT/$source_exp" || true)"
  fi
  if [ "$DRY_RUN" -eq 0 ] && [ ! -f "$INIT_CHECKPOINT" ]; then
    echo "$label requires an explicit checkpoint or validation selection artifact under $EXP_ROOT/$source_exp; resolved: $INIT_CHECKPOINT" >&2
    return 1
  fi
  export INIT_CHECKPOINT
}

clear_exp_env() {
  unset EXP_NAME DATASET DATA_ROOT ANNO BASE_CFG MODEL_TYPE INIT_CHECKPOINT LR FINETUNE_STEPS SAVE_INTERVAL EVAL_INTERVAL SNAPSHOT_INTERVAL LOG_INTERVAL
  unset USE_CHANGE_MASK MASK_TYPE NUM_MASK_CLASSES ENABLE_AUX_MASK USE_AUX_SEMANTIC USE_SEMANTIC_MAPS SEMANTIC_INPUT_MODE NUM_SEMANTIC_CLASSES
  unset USE_SEMANTIC_PARTIAL_DETACH SEMANTIC_DETACH_RATIO SEMANTIC_FUSION_GAMMA_INIT SEMANTIC_FUSION_GAMMA_MAX USE_FEATURE_REWEIGHT LMASK LSEM USE_AUX_WARMUP ALLOW_MISSING_PSEUDO_MASK
  unset USE_CONTENT_WORD_WEIGHT CONTENT_WORD_WEIGHT NORMALIZE_CONTENT_WORD_WEIGHTS
}

configure_common() {
  clear_exp_env
  export EXP_DIR="$EXP_ROOT" EXP_NAME="$1" PAPER_SELECTION_MODE=1 USE_AUX_WARMUP=0
  export USE_FEATURE_REWEIGHT=0 USE_CHANGE_MASK=0 ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0
  export USE_CONTENT_WORD_WEIGHT=0 NORMALIZE_CONTENT_WORD_WEIGHTS=0
  export LMASK=0.0 LSEM=0.0 LR=0.000002 SAVE_INTERVAL=10 EVAL_INTERVAL=10 SNAPSHOT_INTERVAL=10 LOG_INTERVAL=10
}

configure_exp() {
  local exp="$1"
  configure_common "$exp"
  case "$exp" in
    levir_cc_caption_ft_cw103_norm_30_lr2e6|levir_cc_caption_ft_cw105_norm_30_lr2e6)
      export DATASET=levir_cc DATA_ROOT="$LEVIR_CC_ROOT"
      export ANNO="$LEVIR_CC_ROOT/levir_cc_captions_reformat.json"
      export BASE_CFG=configs/dynamic/transformer_levir_cc_sgc_card.yaml MODEL_TYPE=sgc_card
      export USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.0
      export FINETUNE_STEPS=30 USE_CONTENT_WORD_WEIGHT=1 NORMALIZE_CONTENT_WORD_WEIGHTS=1
      if [[ "$exp" == *cw103* ]]; then export CONTENT_WORD_WEIGHT=1.03; else export CONTENT_WORD_WEIGHT=1.05; fi
      resolve_init_checkpoint "${LEVIR_CC_SOURCE_CHECKPOINT:-}" "$LEVIR_CC_SOURCE_EXP" LEVIR-CC
      ;;
    levir_mci_masksemantic_caption_ft_cw103_norm_30_lr2e6|levir_mci_masksemantic_caption_ft_cw105_norm_30_lr2e6)
      export DATASET=levir_mci DATA_ROOT="$LEVIR_MCI_ROOT"
      export ANNO="$LEVIR_MCI_ROOT/levir_mci_captions_reformat.json"
      export BASE_CFG=configs/dynamic/transformer_levir_mci_sgc_card.yaml MODEL_TYPE=sgc_card
      export USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.0
      export FINETUNE_STEPS=30 USE_CONTENT_WORD_WEIGHT=1 NORMALIZE_CONTENT_WORD_WEIGHTS=1
      if [[ "$exp" == *cw103* ]]; then export CONTENT_WORD_WEIGHT=1.03; else export CONTENT_WORD_WEIGHT=1.05; fi
      resolve_init_checkpoint "${LEVIR_MCI_SOURCE_CHECKPOINT:-}" "$LEVIR_MCI_SOURCE_EXP" LEVIR-MCI
      ;;
    second_cc_pd08_gamma005_ft_100_lr2e6|second_cc_pd08_gamma010_ft_100_lr2e6)
      export DATASET=second_cc DATA_ROOT="$SECOND_CC_ROOT"
      export ANNO="$SECOND_CC_ROOT/second_cc_aug_captions_reformat.json"
      export BASE_CFG=configs/dynamic/transformer_second_cc_aug_sgc_card.yaml MODEL_TYPE=sgc_card
      export USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7
      export USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.8 FINETUNE_STEPS=100
      export SAVE_INTERVAL=20 EVAL_INTERVAL=20 SNAPSHOT_INTERVAL=20 LOG_INTERVAL=20
      if [[ "$exp" == *gamma005* ]]; then export SEMANTIC_FUSION_GAMMA_MAX=0.05; else export SEMANTIC_FUSION_GAMMA_MAX=0.10; fi
      resolve_init_checkpoint "${SECOND_CC_SOURCE_CHECKPOINT:-}" "$SECOND_CC_SOURCE_EXP" SECOND-CC
      ;;
    *) echo "Unknown 7.5 experiment: $exp" >&2; return 2 ;;
  esac
}

has_final_checkpoint() {
  find "$EXP_ROOT/$1/snapshots" -type f \( -name "*checkpoint_$2.pt" -o -name "*checkpoint_$2.pth" \) -print -quit 2>/dev/null | grep -q .
}

run_one() {
  local exp="$1"
  [ -z "$ONLY_EXP" ] || [ "$ONLY_EXP" = "$exp" ] || return 0
  configure_exp "$exp" || { record_failure "$exp" "configuration/source checkpoint failed"; return; }
  if [ "$DRY_RUN" -eq 0 ]; then
    [ -d "$DATA_ROOT" ] && [ -f "$ANNO" ] || { record_failure "$exp" "dataset root or annotation missing"; return; }
  fi
  if [ "$OVERWRITE" -eq 0 ] && has_final_checkpoint "$exp" "$FINETUNE_STEPS"; then
    echo "Skipping $exp; checkpoint $FINETUNE_STEPS exists."
    return
  fi
  printf 'TRAIN ENV: EXP_NAME=%q DATASET=%q INIT_CHECKPOINT=%q LR=%q FINETUNE_STEPS=%q CONTENT_WORD_WEIGHT=%q SEMANTIC_FUSION_GAMMA_MAX=%q\n' \
    "$EXP_NAME" "$DATASET" "$INIT_CHECKPOINT" "$LR" "$FINETUNE_STEPS" "${CONTENT_WORD_WEIGHT:-}" "${SEMANTIC_FUSION_GAMMA_MAX:-}"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo 'DRY RUN: bash scripts/_run_paper_training.sh'
  elif ! bash scripts/_run_paper_training.sh; then
    record_failure "$exp" "training command failed"
  fi
}

for exp in "${EXPERIMENTS[@]}"; do
  run_one "$exp"
done

[ "$FAILURES" -eq 0 ] || { echo "$FAILURES 7.5 training step(s) failed; see $FAIL_LOG" >&2; exit 1; }
echo '7.5 follow-up training flow complete.'
