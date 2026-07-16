#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
LEVIR_CC_SOURCE_EXP="${LEVIR_CC_SOURCE_EXP:-sgc_card_lm003_ls005_pd05_rw02_warmup}"
LEVIR_CC_CARD_BASELINE_EXP="${LEVIR_CC_CARD_BASELINE_EXP:-card_levir_cc_baseline}"
LEVIR_MCI_SOURCE_EXP="${LEVIR_MCI_SOURCE_EXP:-levir_mci_card_mask_semantic}"
LEVIR_MCI_CARD_BASELINE_EXP="${LEVIR_MCI_CARD_BASELINE_EXP:-levir_mci_card_baseline}"
SECOND_CC_SOURCE_EXP="${SECOND_CC_SOURCE_EXP:-second_cc_crossattn_pd08_lsem0000}"
SECOND_CC_TRADITIONAL_EXP="${SECOND_CC_TRADITIONAL_EXP:-second_cc_card_semantic_crossattn}"
SECOND_CC_CARD_BASELINE_EXP="${SECOND_CC_CARD_BASELINE_EXP:-second_cc_card_rgb_baseline}"
PROTECTED_METRICS="${PROTECTED_METRICS:-Bleu_1,Bleu_2,Bleu_3,Bleu_4,METEOR,ROUGE_L,CIDEr}"
ONLY_EXP=""
FORCE=0
DRY_RUN=0
FAIL_LOG="${FAIL_LOG:-$EXP_ROOT/7_5_followup_reselect_failures.log}"

usage() {
  echo "Usage: bash scripts/run_7_5_followup_reselect.sh [--only_exp DATASET_OR_TARGET] [--force] [--dry_run]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --only_exp) ONLY_EXP="$2"; shift 2 ;;
    --force|--overwrite) FORCE=1; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$EXP_ROOT"
touch "$FAIL_LOG"
printf '\n[%s] run_7_5_followup_reselect start dry_run=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$DRY_RUN" >> "$FAIL_LOG"
FAILURES=0

record_failure() {
  echo "FAILED: $1${2:+ - $2}" | tee -a "$FAIL_LOG"
  FAILURES=$((FAILURES + 1))
}

configure_dataset() {
  case "$1" in
    levir_cc)
      TARGET_EXP=levir_cc_7_5_val_pareto_locked
      BASELINE_METRICS="${LEVIR_CC_BASELINE_VAL_METRICS:-}"
      TOLERANCE="${LEVIR_CC_BASELINE_TOLERANCE:-0}"
      SOURCE_EXPS=("$LEVIR_CC_SOURCE_EXP" levir_cc_caption_ft_cw103_norm_30_lr2e6 levir_cc_caption_ft_cw105_norm_30_lr2e6 "$LEVIR_CC_CARD_BASELINE_EXP")
      ;;
    levir_mci)
      TARGET_EXP=levir_mci_7_5_val_pareto_locked
      BASELINE_METRICS="${LEVIR_MCI_BASELINE_VAL_METRICS:-}"
      TOLERANCE="${LEVIR_MCI_BASELINE_TOLERANCE:-0}"
      SOURCE_EXPS=("$LEVIR_MCI_SOURCE_EXP" levir_mci_masksemantic_caption_ft_cw103_norm_30_lr2e6 levir_mci_masksemantic_caption_ft_cw105_norm_30_lr2e6 "$LEVIR_MCI_CARD_BASELINE_EXP")
      ;;
    second_cc)
      TARGET_EXP=second_cc_7_5_val_pareto_locked
      BASELINE_METRICS="${SECOND_CC_BASELINE_VAL_METRICS:-}"
      TOLERANCE="${SECOND_CC_BASELINE_TOLERANCE:-0}"
      SOURCE_EXPS=("$SECOND_CC_SOURCE_EXP" second_cc_pd08_gamma005_ft_100_lr2e6 second_cc_pd08_gamma010_ft_100_lr2e6 "$SECOND_CC_TRADITIONAL_EXP" "$SECOND_CC_CARD_BASELINE_EXP")
      ;;
    *) return 2 ;;
  esac
}

selected_dataset() {
  [ -z "$ONLY_EXP" ] || [ "$ONLY_EXP" = "$1" ] || [ "$ONLY_EXP" = "$TARGET_EXP" ]
}

run_one() {
  local dataset="$1"
  configure_dataset "$dataset"
  selected_dataset "$dataset" || return 0
  local target_path="$EXP_ROOT/$TARGET_EXP"
  local output_json="$target_path/best_checkpoint.json"
  mkdir -p "$target_path"
  if [ "$FORCE" -eq 0 ] && [ -s "$output_json" ]; then
    echo "Skipping $dataset reselect; locked selection exists: $output_json"
    return
  fi
  if [ "$DRY_RUN" -eq 1 ] && [ -z "$BASELINE_METRICS" ]; then
    BASELINE_METRICS="<explicit ${dataset} CARD validation metrics JSON>"
  fi
  if [ "$DRY_RUN" -eq 0 ] && { [ -z "$BASELINE_METRICS" ] || [ ! -f "$BASELINE_METRICS" ]; }; then
    record_failure "$dataset reselect" "set ${dataset^^}_BASELINE_VAL_METRICS to an existing validation-only metrics file"
    return
  fi
  local primary=""
  local source path
  local candidate_args=()
  for source in "${SOURCE_EXPS[@]}"; do
    path="$EXP_ROOT/$source"
    if [ "$DRY_RUN" -eq 0 ] && [ ! -d "$path" ]; then
      record_failure "$dataset reselect" "candidate experiment missing: $path"
      return
    fi
    if [ -z "$primary" ]; then
      primary="$path"
    else
      candidate_args+=(--candidate_exp_dir "$path")
    fi
  done
  local command=(
    "$PYTHON" scripts/select_best_checkpoint.py
    --exp_dir "$primary"
    "${candidate_args[@]}"
    --strategy val_baseline_pareto
    --baseline_metrics "$BASELINE_METRICS"
    --protected_metrics "$PROTECTED_METRICS"
    --baseline_tolerance "$TOLERANCE"
    --output_json "$output_json"
    --output_txt "$target_path/best_checkpoint.txt"
  )
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'
    printf ' %q' "${command[@]}"
    printf '\n'
  elif ! "${command[@]}"; then
    record_failure "$dataset reselect" "selector failed"
  fi
}

for dataset in levir_cc levir_mci second_cc; do
  run_one "$dataset"
done

MANIFEST="$EXP_ROOT/7_5_locked_manifest.json"
if [ -n "$ONLY_EXP" ]; then
  echo "Partial reselect requested; locked manifest remains unchanged: $MANIFEST"
elif [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: would write validation-locked manifest $MANIFEST"
elif [ "$FAILURES" -eq 0 ]; then
  "$PYTHON" - "$EXP_ROOT" "$MANIFEST" "$LEVIR_CC_CARD_BASELINE_EXP" "$LEVIR_MCI_CARD_BASELINE_EXP" "$SECOND_CC_CARD_BASELINE_EXP" <<'PY'
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from resolve_experiment_config import resolve_experiment_config

root, output, levir_cc_base, levir_mci_base, second_cc_base = sys.argv[1:]
targets = {
    'levir_cc': 'levir_cc_7_5_val_pareto_locked',
    'levir_mci': 'levir_mci_7_5_val_pareto_locked',
    'second_cc': 'second_cc_7_5_val_pareto_locked',
}
baseline_names = {
    'levir_cc': levir_cc_base,
    'levir_mci': levir_mci_base,
    'second_cc': second_cc_base,
}
locked = {}
for dataset, target in targets.items():
    selection_path = os.path.join(root, target, 'best_checkpoint.json')
    if not os.path.isfile(selection_path):
        raise SystemExit('Missing locked selection: %s' % selection_path)
    selection = json.load(open(selection_path, encoding='utf-8-sig'))
    if selection.get('selection_uses_test_metrics') is not False:
        raise SystemExit('Selection is not marked validation-only: %s' % selection_path)
    if selection.get('status') != 'done':
        raise SystemExit(
            'Selection is not done: %s status=%s'
            % (selection_path, selection.get('status'))
        )
    checkpoint = (
        selection.get('selected_checkpoint_path')
        or selection.get('selected_checkpoint')
    )
    if not checkpoint:
        raise SystemExit('Selection has no checkpoint: %s' % selection_path)
    if not os.path.isfile(checkpoint):
        raise SystemExit('Locked checkpoint missing: %s' % checkpoint)
    source_dir = selection.get('selected_source_exp_dir', '')
    source_name = selection.get('selected_source_exp_name', '')
    if not source_dir or not os.path.isdir(source_dir):
        raise SystemExit('Selected source experiment is missing: %s' % source_dir)
    try:
        source_config_path, resolved, source_config_artifact = resolve_experiment_config(source_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))
    locked[dataset] = {
        'target_exp': target,
        'selection_json': os.path.normpath(selection_path),
        'status': 'done',
        'selected_checkpoint': checkpoint,
        'selected_source_exp_dir': source_dir,
        'selected_source_exp_name': source_name,
        'selected_is_baseline': source_name == baseline_names[dataset],
        'selected_val_metrics': selection.get('selected_val_metrics', {}),
        'selected_metric_deltas': selection.get('selected_metric_deltas', {}),
        'source_config': os.path.normpath(source_config_path),
        'source_config_artifact': source_config_artifact,
        'source_resolved_config': os.path.normpath(source_config_path),
        'semantic_fusion_gamma_max': resolved.get('model', {}).get(
            'semantic_fusion_gamma_max',
            0.0,
        ),
    }
manifest = {
    'status': 'validation_locked',
    'selection_uses_test_metrics': False,
    'locked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'datasets': locked,
}
with open(output, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(json.dumps(manifest, indent=2, ensure_ascii=False))
PY
fi

[ "$FAILURES" -eq 0 ] || { echo "$FAILURES 7.5 reselect step(s) failed; see $FAIL_LOG" >&2; exit 1; }
if [ -n "$ONLY_EXP" ]; then
  echo "7.5 partial validation-only reselect complete; manifest unchanged: $MANIFEST"
else
  echo "7.5 validation-only reselect complete: $MANIFEST"
fi
