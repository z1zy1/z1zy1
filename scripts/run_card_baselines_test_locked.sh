#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
EXP_ROOT="${EXP_ROOT:-./experiments}"
MANIFEST="${CARD_BASELINE_LOCKED_MANIFEST:-$EXP_ROOT/card_baseline_locked_manifest.json}"
ONLY_DATASET=""
FORCE=0
DRY_RUN=0
FAIL_LOG="${FAIL_LOG:-$EXP_ROOT/card_baseline_test_failures.log}"
usage() { echo 'Usage: bash scripts/run_card_baselines_test_locked.sh [--only_dataset DATASET] [--force] [--dry_run]' >&2; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --only_dataset|--only_exp) ONLY_DATASET="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
case "$ONLY_DATASET" in ''|levir_cc|levir_mci|second_cc) ;; *) usage; exit 2 ;; esac

FAILURES=0
record_failure() { echo "FAILED: $1${2:+ - $2}" | tee -a "$FAIL_LOG"; FAILURES=$((FAILURES + 1)); }
if [ "$DRY_RUN" -eq 0 ]; then
  mkdir -p "$EXP_ROOT"; touch "$FAIL_LOG"
  printf '\n[%s] run_card_baselines_test_locked start\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$FAIL_LOG"
  if ! "$PYTHON" scripts/build_card_baseline_manifest.py --verify_manifest "$MANIFEST"; then
    record_failure manifest 'locked manifest/config/checkpoint verification failed'
  fi
fi

json_value() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding='utf-8-sig'))
for part in sys.argv[2].split('.'):
    value = value.get(part, '') if isinstance(value, dict) else ''
if value is None: value = ''
print(('1' if value else '0') if isinstance(value, bool) else value)
PY
}
same_path() {
  "$PYTHON" - "$1" "$2" <<'PY'
import os, sys
canon = lambda p: os.path.normcase(os.path.abspath(os.path.normpath(p)))
raise SystemExit(0 if canon(sys.argv[1]) == canon(sys.argv[2]) else 1)
PY
}
validate_result() {
  "$PYTHON" - "$1" "$2" <<'PY'
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from summarize_card_baseline_tests import validate_locked_test_result
validate_locked_test_result(sys.argv[1], sys.argv[2])
PY
}

configure_dataset() {
  local dataset="$1" data_root="$2"
  DATASET="$dataset"
  DATA_ROOT="$data_root"
  MODEL_TYPE=card
  USE_CHANGE_MASK=0; MASK_TYPE=binary; NUM_MASK_CLASSES=""
  USE_SEMANTIC_MAPS=0; SEMANTIC_INPUT_MODE=none; NUM_SEMANTIC_CLASSES=""
  ENABLE_AUX_MASK=0; USE_AUX_SEMANTIC=0; USE_SEMANTIC_PARTIAL_DETACH=0
  ALLOW_MISSING_PSEUDO_MASK=0; PAPER_SELECTION_MODE=0
  MASK_LOSS_TYPE=""; SEMANTIC_LOSS_TYPE=""
  LMASK=0.0; LSEM=0.0; SEMANTIC_DETACH_RATIO=0.0
  SEMANTIC_FUSION_GAMMA_INIT=0.0; SEMANTIC_FUSION_GAMMA_MAX=0.0
  USE_FEATURE_REWEIGHT=0; REWEIGHT_ALPHA=0.0
  case "$dataset" in
    levir_cc)
      EVAL_CHANGE_NOCHANGE_SPLIT=0; CHANGEFLAG_JSON="" ;;
    levir_mci)
      EVAL_CHANGE_NOCHANGE_SPLIT=1; CHANGEFLAG_JSON="$data_root/LevirCCcaptions.json" ;;
    second_cc)
      EVAL_CHANGE_NOCHANGE_SPLIT=1; CHANGEFLAG_JSON="$data_root/SECOND-CC-AUG.json" ;;
  esac
  export DATASET DATA_ROOT BASE_CFG ANNO FEATURE_ROOT MODEL_TYPE
  export USE_CHANGE_MASK MASK_TYPE NUM_MASK_CLASSES USE_SEMANTIC_MAPS SEMANTIC_INPUT_MODE NUM_SEMANTIC_CLASSES
  export ENABLE_AUX_MASK USE_AUX_SEMANTIC USE_SEMANTIC_PARTIAL_DETACH ALLOW_MISSING_PSEUDO_MASK
  export EVAL_CHANGE_NOCHANGE_SPLIT CHANGEFLAG_JSON PAPER_SELECTION_MODE MASK_LOSS_TYPE SEMANTIC_LOSS_TYPE
  export LMASK LSEM SEMANTIC_DETACH_RATIO SEMANTIC_FUSION_GAMMA_INIT SEMANTIC_FUSION_GAMMA_MAX
  export USE_FEATURE_REWEIGHT REWEIGHT_ALPHA
}

run_one() {
  local dataset="$1"
  [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "$dataset" ] || return 0
  local exp_dir checkpoint result source_name data_root
  if [ "$DRY_RUN" -eq 1 ]; then
    exp_dir="$EXP_ROOT/card_baseline_locked_tests/$dataset"
    checkpoint="<locked-$dataset-checkpoint>"
    result="$exp_dir/test_card_baseline_locked_result.json"
    source_name="<locked-$dataset-baseline>"
    BASE_CFG="<locked-resolved-config.yaml>"; ANNO="<locked-eval-annotation>"; FEATURE_ROOT="<locked-feature-root>"
    data_root="<locked-data-root>"
  else
    [ "$FAILURES" -eq 0 ] || return
    exp_dir="$(json_value "$MANIFEST" "datasets.$dataset.test_exp_dir")"
    checkpoint="$(json_value "$MANIFEST" "datasets.$dataset.selected_checkpoint")"
    result="$(json_value "$MANIFEST" "datasets.$dataset.test_result")"
    source_name="$(json_value "$MANIFEST" "datasets.$dataset.exp_name")"
    BASE_CFG="$(json_value "$MANIFEST" "datasets.$dataset.resolved_config_yaml")"
    ANNO="$(json_value "$MANIFEST" "datasets.$dataset.eval_anno_path")"
    FEATURE_ROOT="$(json_value "$MANIFEST" "datasets.$dataset.feature_root")"
    data_root="$(json_value "$MANIFEST" "datasets.$dataset.data_root")"
  fi
  configure_dataset "$dataset" "$data_root"
  if [ "$DRY_RUN" -eq 0 ] && { [ ! -d "$DATA_ROOT" ] || [ ! -d "$FEATURE_ROOT" ] || [ ! -f "$ANNO" ]; }; then
    record_failure "$dataset test" 'locked data root/features/annotation missing'; return
  fi
  if [ "$FORCE" -eq 0 ] && [ -s "$result" ]; then
    local tested
    tested="$(json_value "$result" snapshot_path 2>/dev/null || true)"
    if [ -n "$tested" ] && same_path "$tested" "$checkpoint"; then
      if validate_result "$result" "$checkpoint"; then
        echo "Skipping matching, complete locked baseline test: $result"; return
      fi
      record_failure "$dataset test" "matching result is missing valid 8-metric output; inspect or pass --force: $result"; return
    fi
    record_failure "$dataset test" "stale result does not match locked checkpoint; inspect or pass --force: $result"; return
  fi
  local command=(bash scripts/test_specific_snapshot_sgc_card.sh --exp_dir "$exp_dir" --checkpoint "$checkpoint" --tag card_baseline_locked)
  printf 'LOCKED BASELINE TEST: dataset=%q source=%q checkpoint=%q\n' "$dataset" "$source_name" "$checkpoint"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
  elif ! "${command[@]}"; then
    record_failure "$dataset test" 'locked test failed'
  elif ! validate_result "$result" "$checkpoint"; then
    record_failure "$dataset test" 'test command ended without a matching complete 8-metric result'
  fi
}

for dataset in levir_cc levir_mci second_cc; do run_one "$dataset"; done
[ "$FAILURES" -eq 0 ] || { echo "$FAILURES baseline locked-test step(s) failed; see $FAIL_LOG" >&2; exit 1; }
if [ -z "$ONLY_DATASET" ]; then
  summary=("$PYTHON" scripts/summarize_card_baseline_tests.py --manifest "$MANIFEST" \
    --output_json "$EXP_ROOT/card_baseline_test_summary.json" --output_csv "$EXP_ROOT/card_baseline_test_summary.csv")
  if [ "$DRY_RUN" -eq 1 ]; then printf 'DRY RUN:'; printf ' %q' "${summary[@]}"; printf '\n'; else "${summary[@]}"; fi
else
  echo 'Partial locked test complete; run without --only_dataset to verify all three and refresh the summary.'
fi
echo 'Each locked CARD baseline checkpoint was tested at most once; test metrics never affected selection.'
