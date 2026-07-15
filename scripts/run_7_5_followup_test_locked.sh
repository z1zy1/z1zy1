#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
MANIFEST="${LOCKED_MANIFEST:-$EXP_ROOT/7_5_locked_manifest.json}"
LEVIR_CC_ROOT="${LEVIR_CC_ROOT:-./Levir-CC}"
LEVIR_MCI_ROOT="${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}"
SECOND_CC_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
ONLY_DATASET=""
FORCE=0
DRY_RUN=0
FAIL_LOG="${FAIL_LOG:-$EXP_ROOT/7_5_followup_locked_test_failures.log}"

usage() {
  echo "Usage: bash scripts/run_7_5_followup_test_locked.sh [--only_dataset DATASET] [--force] [--dry_run]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --only_dataset|--only_exp) ONLY_DATASET="$2"; shift 2 ;;
    --force|--overwrite) FORCE=1; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$EXP_ROOT"
touch "$FAIL_LOG"
printf '\n[%s] run_7_5_followup_test_locked start dry_run=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$DRY_RUN" >> "$FAIL_LOG"
FAILURES=0

record_failure() {
  echo "FAILED: $1${2:+ - $2}" | tee -a "$FAIL_LOG"
  FAILURES=$((FAILURES + 1))
}

json_value() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding='utf-8-sig'))
for part in sys.argv[2].split('.'):
    value = value.get(part, '') if isinstance(value, dict) else ''
if value is None:
    value = ''
print(('1' if value else '0') if isinstance(value, bool) else value)
PY
}

resolved_value() {
  local path="$1"
  local key="$2"
  local fallback="$3"
  local value
  value="$(json_value "$path" "$key")"
  if [ -n "$value" ]; then printf '%s\n' "$value"; else printf '%s\n' "$fallback"; fi
}

same_checkpoint_path() {
  "$PYTHON" - "$1" "$2" <<'PY'
import os
import sys

def canonical(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))

raise SystemExit(0 if canonical(sys.argv[1]) == canonical(sys.argv[2]) else 1)
PY
}

configure_dataset_paths() {
  case "$1" in
    levir_cc)
      DATASET=levir_cc
      DATA_ROOT="$LEVIR_CC_ROOT"
      ANNO="$LEVIR_CC_ROOT/levir_cc_captions_reformat.json"
      EVAL_CHANGE_NOCHANGE_SPLIT=0
      CHANGEFLAG_JSON=""
      ;;
    levir_mci)
      DATASET=levir_mci
      DATA_ROOT="$LEVIR_MCI_ROOT"
      ANNO="$LEVIR_MCI_ROOT/levir_mci_captions_reformat.json"
      EVAL_CHANGE_NOCHANGE_SPLIT=1
      CHANGEFLAG_JSON="$LEVIR_MCI_ROOT/LevirCCcaptions.json"
      ;;
    second_cc)
      DATASET=second_cc
      DATA_ROOT="$SECOND_CC_ROOT"
      ANNO="$SECOND_CC_ROOT/second_cc_aug_captions_reformat.json"
      EVAL_CHANGE_NOCHANGE_SPLIT=1
      CHANGEFLAG_JSON="$SECOND_CC_ROOT/SECOND-CC-AUG.json"
      ;;
    *) return 2 ;;
  esac
  export DATASET DATA_ROOT ANNO EVAL_CHANGE_NOCHANGE_SPLIT CHANGEFLAG_JSON
}

configure_from_locked_source() {
  local dataset="$1"
  local source_cfg="$2"
  local source_name="$3"
  configure_dataset_paths "$dataset"
  BASE_CFG="$source_cfg"
  MODEL_TYPE="$(resolved_value "$source_cfg" model.type '')"
  if [ -z "$MODEL_TYPE" ]; then
    case "$source_name" in *baseline*) MODEL_TYPE=card ;; *) MODEL_TYPE=sgc_card ;; esac
  fi
  FEATURE_ROOT="$(resolved_value "$source_cfg" data.default_feature_dir '')"
  USE_CHANGE_MASK="$(resolved_value "$source_cfg" data.use_change_mask 0)"
  MASK_TYPE="$(resolved_value "$source_cfg" data.mask_type binary)"
  NUM_MASK_CLASSES="$(resolved_value "$source_cfg" model.num_mask_classes 1)"
  USE_SEMANTIC_MAPS="$(resolved_value "$source_cfg" data.use_semantic_maps 0)"
  SEMANTIC_INPUT_MODE="$(resolved_value "$source_cfg" model.semantic_input_mode none)"
  NUM_SEMANTIC_CLASSES="$(resolved_value "$source_cfg" model.num_semantic_classes 0)"
  ENABLE_AUX_MASK="$(resolved_value "$source_cfg" model.enable_aux_mask 0)"
  USE_AUX_SEMANTIC="$(resolved_value "$source_cfg" train.use_semantic_aux 0)"
  USE_SEMANTIC_PARTIAL_DETACH="$(resolved_value "$source_cfg" train.use_semantic_partial_detach 0)"
  SEMANTIC_DETACH_RATIO="$(resolved_value "$source_cfg" train.semantic_detach_ratio 0.0)"
  SEMANTIC_FUSION_GAMMA_INIT="$(resolved_value "$source_cfg" model.semantic_fusion_gamma_init '')"
  SEMANTIC_FUSION_GAMMA_MAX="$(json_value "$MANIFEST" "datasets.$dataset.semantic_fusion_gamma_max")"
  USE_FEATURE_REWEIGHT="$(resolved_value "$source_cfg" train.use_feature_reweight 0)"
  REWEIGHT_ALPHA="$(resolved_value "$source_cfg" train.reweight_alpha 0.2)"
  ALLOW_MISSING_PSEUDO_MASK="$(resolved_value "$source_cfg" data.allow_missing_pseudo_mask 0)"
  LMASK="$(resolved_value "$source_cfg" train.lambda_mask 0.0)"
  LSEM="$(resolved_value "$source_cfg" train.lambda_semantic 0.0)"
  MASK_LOSS_TYPE="$(resolved_value "$source_cfg" train.mask_loss_type '')"
  SEMANTIC_LOSS_TYPE="$(resolved_value "$source_cfg" train.semantic_loss_type '')"
  PAPER_SELECTION_MODE=1
  export BASE_CFG MODEL_TYPE FEATURE_ROOT USE_CHANGE_MASK MASK_TYPE NUM_MASK_CLASSES
  export USE_SEMANTIC_MAPS SEMANTIC_INPUT_MODE NUM_SEMANTIC_CLASSES ENABLE_AUX_MASK USE_AUX_SEMANTIC
  export USE_SEMANTIC_PARTIAL_DETACH SEMANTIC_DETACH_RATIO SEMANTIC_FUSION_GAMMA_INIT SEMANTIC_FUSION_GAMMA_MAX
  export USE_FEATURE_REWEIGHT REWEIGHT_ALPHA ALLOW_MISSING_PSEUDO_MASK LMASK LSEM MASK_LOSS_TYPE SEMANTIC_LOSS_TYPE PAPER_SELECTION_MODE
}

run_one() {
  local dataset="$1"
  [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "$dataset" ] || return 0
  local target checkpoint source_cfg source_name result
  if [ "$DRY_RUN" -eq 1 ] && [ ! -f "$MANIFEST" ]; then
    target="${dataset}_7_5_val_pareto_locked"
    checkpoint="<locked $dataset checkpoint>"
    source_cfg="<locked source resolved_config.json>"
    source_name="<locked source>"
    configure_dataset_paths "$dataset"
    BASE_CFG="$source_cfg"
    MODEL_TYPE=sgc_card
    SEMANTIC_FUSION_GAMMA_MAX="<locked gamma_max>"
    export BASE_CFG MODEL_TYPE SEMANTIC_FUSION_GAMMA_MAX
  else
    [ -f "$MANIFEST" ] || { record_failure "$dataset" "locked manifest missing: $MANIFEST"; return; }
    [ "$(json_value "$MANIFEST" status)" = validation_locked ] || { record_failure "$dataset" 'manifest status is not validation_locked'; return; }
    [ "$(json_value "$MANIFEST" selection_uses_test_metrics)" = 0 ] || { record_failure "$dataset" 'manifest is not validation-only'; return; }
    [ "$(json_value "$MANIFEST" "datasets.$dataset.status")" = done ] || { record_failure "$dataset" 'dataset lock status is not done'; return; }
    target="$(json_value "$MANIFEST" "datasets.$dataset.target_exp")"
    checkpoint="$(json_value "$MANIFEST" "datasets.$dataset.selected_checkpoint")"
    source_cfg="$(json_value "$MANIFEST" "datasets.$dataset.source_resolved_config")"
    source_name="$(json_value "$MANIFEST" "datasets.$dataset.selected_source_exp_name")"
    [ -n "$target" ] || { record_failure "$dataset" 'locked target_exp is empty'; return; }
    [ -n "$checkpoint" ] || { record_failure "$dataset" 'locked checkpoint is empty'; return; }
    [ -n "$source_cfg" ] || { record_failure "$dataset" 'locked source_resolved_config is empty'; return; }
    [ -n "$source_name" ] || { record_failure "$dataset" 'locked selected_source_exp_name is empty'; return; }
    [ -f "$checkpoint" ] || { record_failure "$dataset" "locked checkpoint missing: $checkpoint"; return; }
    [ -f "$source_cfg" ] || { record_failure "$dataset" "source resolved config missing: $source_cfg"; return; }
    configure_from_locked_source "$dataset" "$source_cfg" "$source_name"
  fi
  result="$EXP_ROOT/$target/test_7_5_locked_result.json"
  if [ "$FORCE" -eq 0 ] && [ -s "$result" ]; then
    local existing_snapshot
    existing_snapshot="$(json_value "$result" snapshot_path 2>/dev/null || true)"
    if [ -n "$existing_snapshot" ] && same_checkpoint_path "$existing_snapshot" "$checkpoint"; then
      echo "Skipping $dataset; locked test result matches checkpoint: $result"
      return
    fi
    record_failure "$dataset" "existing result snapshot_path does not match locked checkpoint; inspect it or rerun with --force: $result"
    return
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    [ -d "$DATA_ROOT" ] && [ -f "$ANNO" ] || { record_failure "$dataset" 'dataset root or annotation missing'; return; }
  fi
  local command=(
    bash scripts/test_specific_snapshot_sgc_card.sh
    --exp_dir "$EXP_ROOT/$target"
    --checkpoint "$checkpoint"
    --tag 7_5_locked
  )
  printf 'LOCKED TEST: dataset=%q source=%q checkpoint=%q gamma_max=%q\n' "$dataset" "$source_name" "$checkpoint" "${SEMANTIC_FUSION_GAMMA_MAX:-0}"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'
    printf ' %q' "${command[@]}"
    printf '\n'
  elif ! "${command[@]}"; then
    record_failure "$dataset" 'locked test failed'
  fi
}

for dataset in levir_cc levir_mci second_cc; do
  run_one "$dataset"
done

[ "$FAILURES" -eq 0 ] || { echo "$FAILURES locked test step(s) failed; see $FAIL_LOG" >&2; exit 1; }
echo '7.5 locked checkpoints tested once; no candidate-wide testing was performed.'
