#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
MANIFEST="${LOCKED_MANIFEST:-$EXP_ROOT/7_6_locked_manifest.json}"
LEVIR_CC_ROOT="${LEVIR_CC_ROOT:-./Levir-CC}"
LEVIR_MCI_ROOT="${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}"
SECOND_CC_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
ONLY_DATASET=""
ONLY_LOCK=""
DRY_RUN=0
FAIL_LOG="${FAIL_LOG:-$EXP_ROOT/7_6_followup_locked_test_failures.log}"

usage() {
  echo 'Usage: bash scripts/run_7_6_followup_test_locked.sh [--only_dataset DATASET] [--only_lock LOCK_ID] [--dry_run]' >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --only_dataset) ONLY_DATASET="$2"; shift 2 ;;
    --only_lock|--only_exp) ONLY_LOCK="$2"; shift 2 ;;
    --force|--overwrite)
      echo '7.6 locked tests refuse --force/--overwrite; an existing test result is immutable.' >&2
      exit 2
      ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$EXP_ROOT"
touch "$FAIL_LOG"
[ -f "$MANIFEST" ] || { echo "7.6 locked manifest missing: $MANIFEST" >&2; exit 1; }
"$PYTHON" scripts/build_7_6_locked_manifest.py --verify "$MANIFEST"
"$PYTHON" - "$MANIFEST" "$LEVIR_CC_ROOT" "$LEVIR_MCI_ROOT" "$SECOND_CC_ROOT" <<'PY'
import json
import os
import sys

manifest = json.load(open(sys.argv[1], encoding='utf-8-sig'))
canonical = lambda path: os.path.normcase(os.path.realpath(os.path.abspath(os.path.normpath(path))))
runtime = dict(zip(('levir_cc', 'levir_mci', 'second_cc'), sys.argv[2:]))
for dataset, path in runtime.items():
    locked = manifest['dataset_roots'][dataset]
    if canonical(path) != canonical(locked):
        raise SystemExit('%s runtime root differs from locked root: %s != %s' % (dataset, path, locked))
PY

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
  local path="$1" key="$2" fallback="$3" value
  value="$(json_value "$path" "$key")"
  if [ -n "$value" ]; then printf '%s\n' "$value"; else printf '%s\n' "$fallback"; fi
}

validate_result() {
  local result="$1" checkpoint="$2"
  "$PYTHON" - "$result" "$checkpoint" <<'PY'
import json
import math
import os
import sys

result_path, expected = sys.argv[1:]
payload = json.load(open(result_path, encoding='utf-8-sig'))
actual = payload.get('snapshot_path')
canonical = lambda path: os.path.normcase(os.path.realpath(os.path.abspath(os.path.normpath(path))))
if not actual or canonical(actual) != canonical(expected):
    raise SystemExit('test result checkpoint mismatch: %s != %s' % (actual, expected))
metrics = payload.get('metrics')
required = ('Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE')
if not isinstance(metrics, dict):
    raise SystemExit('test result has no metrics object: %s' % result_path)
for metric in required:
    value = metrics.get(metric)
    if isinstance(value, bool):
        raise SystemExit('invalid boolean test metric %s' % metric)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise SystemExit('missing/invalid test metric %s' % metric)
    if not math.isfinite(value):
        raise SystemExit('non-finite test metric %s' % metric)
PY
}

configure_dataset_paths() {
  local dataset="$1" audited_root="$2" audited_anno="$3" audited_changeflag="$4"
  case "$dataset" in
    levir_cc)
      DATASET=levir_cc DATA_ROOT="$audited_root" ANNO="$audited_anno"
      EVAL_CHANGE_NOCHANGE_SPLIT=0 CHANGEFLAG_JSON=""
      ;;
    levir_mci)
      DATASET=levir_mci DATA_ROOT="$audited_root" ANNO="$audited_anno"
      EVAL_CHANGE_NOCHANGE_SPLIT=1 CHANGEFLAG_JSON="$audited_changeflag"
      ;;
    second_cc)
      DATASET=second_cc DATA_ROOT="$audited_root" ANNO="$audited_anno"
      EVAL_CHANGE_NOCHANGE_SPLIT=1 CHANGEFLAG_JSON="$audited_changeflag"
      ;;
    *) echo "Unknown locked dataset: $1" >&2; return 2 ;;
  esac
  export DATASET DATA_ROOT ANNO EVAL_CHANGE_NOCHANGE_SPLIT CHANGEFLAG_JSON
}

configure_from_source() {
  local dataset="$1" source_cfg="$2" audited_root="$3" audited_anno="$4" audited_changeflag="$5" audited_feature="$6"
  configure_dataset_paths "$dataset" "$audited_root" "$audited_anno" "$audited_changeflag"
  BASE_CFG="$source_cfg"
  MODEL_TYPE="$(resolved_value "$source_cfg" model.type sgc_card)"
  FEATURE_ROOT="$audited_feature"
  USE_CHANGE_MASK="$(resolved_value "$source_cfg" data.use_change_mask 0)"
  MASK_TYPE="$(resolved_value "$source_cfg" data.mask_type binary)"
  NUM_MASK_CLASSES="$(resolved_value "$source_cfg" model.num_mask_classes '')"
  USE_SEMANTIC_MAPS="$(resolved_value "$source_cfg" data.use_semantic_maps 0)"
  SEMANTIC_INPUT_MODE="$(resolved_value "$source_cfg" model.semantic_input_mode none)"
  NUM_SEMANTIC_CLASSES="$(resolved_value "$source_cfg" model.num_semantic_classes '')"
  ENABLE_AUX_MASK="$(resolved_value "$source_cfg" model.enable_aux_mask 0)"
  USE_AUX_SEMANTIC="$(resolved_value "$source_cfg" train.use_semantic_aux 0)"
  USE_SEMANTIC_PARTIAL_DETACH="$(resolved_value "$source_cfg" train.use_semantic_partial_detach 0)"
  SEMANTIC_DETACH_RATIO="$(resolved_value "$source_cfg" train.semantic_detach_ratio 0.5)"
  SEMANTIC_FUSION_GAMMA_INIT="$(resolved_value "$source_cfg" model.semantic_fusion_gamma_init '')"
  SEMANTIC_FUSION_GAMMA_MAX="$(resolved_value "$source_cfg" model.semantic_fusion_gamma_max 0.0)"
  USE_FEATURE_REWEIGHT="$(resolved_value "$source_cfg" train.use_feature_reweight 0)"
  REWEIGHT_ALPHA="$(resolved_value "$source_cfg" train.reweight_alpha 0.2)"
  ALLOW_MISSING_PSEUDO_MASK="$(resolved_value "$source_cfg" data.allow_missing_pseudo_mask 0)"
  LMASK="$(resolved_value "$source_cfg" train.lambda_mask 0.0)"
  LSEM="$(resolved_value "$source_cfg" train.lambda_semantic 0.0)"
  MASK_LOSS_TYPE="$(resolved_value "$source_cfg" train.mask_loss_type '')"
  SEMANTIC_LOSS_TYPE="$(resolved_value "$source_cfg" train.semantic_loss_type '')"
  PAPER_SELECTION_MODE=1 EXP_DIR="$EXP_ROOT"
  export BASE_CFG MODEL_TYPE FEATURE_ROOT USE_CHANGE_MASK MASK_TYPE NUM_MASK_CLASSES
  export USE_SEMANTIC_MAPS SEMANTIC_INPUT_MODE NUM_SEMANTIC_CLASSES ENABLE_AUX_MASK USE_AUX_SEMANTIC
  export USE_SEMANTIC_PARTIAL_DETACH SEMANTIC_DETACH_RATIO SEMANTIC_FUSION_GAMMA_INIT SEMANTIC_FUSION_GAMMA_MAX
  export USE_FEATURE_REWEIGHT REWEIGHT_ALPHA ALLOW_MISSING_PSEUDO_MASK LMASK LSEM MASK_LOSS_TYPE SEMANTIC_LOSS_TYPE
  export PAPER_SELECTION_MODE EXP_DIR
}

validate_runtime_binding() {
  local dataset="$1" source_cfg="$2" audited_root="$3" audited_anno="$4" audited_changeflag="$5" audited_feature="$6"
  "$PYTHON" - "$dataset" "$source_cfg" "$audited_root" "$audited_anno" "$audited_changeflag" "$audited_feature" <<'PY'
import json
import os
import sys

dataset, config_path, root, anno, changeflag, feature = sys.argv[1:]
config = json.load(open(config_path, encoding='utf-8-sig'))
data = config.get('data', {})
canonical = lambda path: os.path.normcase(os.path.realpath(os.path.abspath(os.path.normpath(path))))
root = canonical(root)
expected = {
    'levir_cc': {
        'eval_anno_path': os.path.join(root, 'levir_cc_captions_reformat.json'),
        'vocab_json': os.path.join(root, 'transformer_levir_vocab.json'),
        'h5_label_file': os.path.join(root, 'transformer_levir_labels.h5'),
        'splits_json': os.path.join(root, 'splits.json'),
    },
    'levir_mci': {
        'eval_anno_path': os.path.join(root, 'levir_mci_captions_reformat.json'),
        'vocab_json': os.path.join(root, 'transformer_levir_mci_vocab.json'),
        'h5_label_file': os.path.join(root, 'transformer_levir_mci_labels.h5'),
        'splits_json': os.path.join(root, 'splits.json'),
    },
    'second_cc': {
        'eval_anno_path': os.path.join(root, 'second_cc_aug_captions_reformat.json'),
        'vocab_json': os.path.join(root, 'transformer_second_cc_aug_vocab.json'),
        'h5_label_file': os.path.join(root, 'transformer_second_cc_aug_labels.h5'),
        'splits_json': os.path.join(root, 'splits.json'),
        'semantic_map_root': root,
    },
}[dataset]
expected['default_feature_dir'] = feature
expected['semantic_feature_dir'] = feature
for key, runtime_path in expected.items():
    source_path = data.get(key)
    if not source_path or canonical(source_path) != canonical(runtime_path):
        raise SystemExit('%s runtime/source path mismatch for data.%s: %s != %s' % (
            dataset, key, runtime_path, source_path))
if canonical(anno) != canonical(expected['eval_anno_path']):
    raise SystemExit('%s audited annotation differs from runtime annotation.' % dataset)
expected_changeflag = '' if dataset == 'levir_cc' else os.path.join(
    root, 'LevirCCcaptions.json' if dataset == 'levir_mci' else 'SECOND-CC-AUG.json')
if canonical(changeflag) != canonical(expected_changeflag):
    raise SystemExit('%s audited changeflag differs from runtime changeflag.' % dataset)
if dataset == 'second_cc':
    if data.get('semantic_before_phase') != os.path.join('sem', 'A'):
        raise SystemExit('SECOND semantic_before_phase differs from locked runtime layout.')
    if data.get('semantic_after_phase') != os.path.join('sem', 'B'):
        raise SystemExit('SECOND semantic_after_phase differs from locked runtime layout.')
PY
}

lock_rows() {
  "$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding='utf-8-sig'))
for lock in manifest['locks']:
    artifacts = {item['role']: item['path'] for item in lock['data_audit']['artifacts']}
    resolved_paths = lock['data_audit']['resolved_paths']
    fields = [str(lock.get(key, '')) for key in (
        'lock_id', 'dataset', 'seed', 'target_exp', 'selected_checkpoint',
        'source_config', 'selected_source_exp_name', 'test_result',
    )]
    fields.extend([
        manifest['dataset_roots'][lock['dataset']],
        artifacts['data.eval_anno_path'],
        artifacts.get('data.changeflag_json', ''),
        resolved_paths['default_feature_dir'],
    ])
    print('|'.join(fields))
PY
}

matched=0
while IFS='|' read -r lock_id dataset seed target checkpoint source_cfg source_name result audited_root audited_anno audited_changeflag audited_feature; do
  [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "$dataset" ] || continue
  [ -z "$ONLY_LOCK" ] || [ "$ONLY_LOCK" = "$lock_id" ] || [ "$ONLY_LOCK" = "$target" ] || continue
  matched=1
  configure_from_source "$dataset" "$source_cfg" "$audited_root" "$audited_anno" "$audited_changeflag" "$audited_feature"
  validate_runtime_binding "$dataset" "$source_cfg" "$audited_root" "$audited_anno" "$audited_changeflag" "$audited_feature"
  if [ -s "$result" ]; then
    validate_result "$result" "$checkpoint"
    echo "Skipping $lock_id; existing result matches locked checkpoint: $result"
    continue
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    [ -d "$DATA_ROOT" ] && [ -f "$ANNO" ] || { echo "$lock_id dataset/annotation missing" >&2; exit 1; }
  fi
  command=(
    bash scripts/test_specific_snapshot_sgc_card.sh
    --exp_dir "$EXP_ROOT/$target"
    --checkpoint "$checkpoint"
    --tag 7_6_locked
  )
  printf 'LOCKED TEST: lock=%q dataset=%q seed=%q source=%q checkpoint=%q gamma_max=%q\n' \
    "$lock_id" "$dataset" "$seed" "$source_name" "$checkpoint" "$SEMANTIC_FUSION_GAMMA_MAX"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
  else
    "${command[@]}"
    [ -s "$result" ] || { echo "Locked result not written: $result" >&2; exit 1; }
    validate_result "$result" "$checkpoint"
  fi
done < <(lock_rows)

[ "$matched" -eq 1 ] || { echo 'No lock matched the requested filter.' >&2; exit 2; }

if [ "$DRY_RUN" -eq 0 ] && [ -z "$ONLY_DATASET" ] && [ -z "$ONLY_LOCK" ]; then
  "$PYTHON" scripts/summarize_7_6_locked_tests.py \
    --manifest "$MANIFEST" \
    --baseline_summary "$EXP_ROOT/card_baseline_test_summary.json" \
    --output_json "$EXP_ROOT/7_6_locked_test_summary.json" \
    --output_csv "$EXP_ROOT/7_6_locked_test_summary.csv"
fi

echo '7.6 locked test flow complete; only five validation-locked checkpoints were eligible.'
