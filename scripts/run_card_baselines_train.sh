#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
EXP_ROOT="${EXP_ROOT:-./experiments}"
LEVIR_CC_ROOT="${LEVIR_CC_ROOT:-./Levir-CC}"
LEVIR_MCI_ROOT="${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}"
SECOND_CC_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
LEVIR_CC_BASELINE_EXP="${LEVIR_CC_BASELINE_EXP:-card_levir_cc_baseline}"
LEVIR_MCI_BASELINE_EXP="${LEVIR_MCI_BASELINE_EXP:-levir_mci_card_baseline}"
SECOND_CC_BASELINE_EXP="${SECOND_CC_BASELINE_EXP:-second_cc_card_rgb_baseline}"
export LEVIR_CC_BASELINE_EXP LEVIR_MCI_BASELINE_EXP SECOND_CC_BASELINE_EXP
ONLY_DATASET=""
DRY_RUN=0
FAIL_LOG="${FAIL_LOG:-$EXP_ROOT/card_baseline_train_failures.log}"

usage() {
  echo 'Usage: bash scripts/run_card_baselines_train.sh [--only_dataset levir_cc|levir_mci|second_cc] [--dry_run]' >&2
}
while [ "$#" -gt 0 ]; do
  case "$1" in
    --only_dataset|--only_exp) ONLY_DATASET="$2"; shift 2 ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
case "$ONLY_DATASET" in ''|levir_cc|levir_mci|second_cc) ;; *) usage; exit 2 ;; esac

FAILURES=0
record_failure() {
  echo "FAILED: $1${2:+ - $2}" | tee -a "$FAIL_LOG"
  FAILURES=$((FAILURES + 1))
}
if [ "$DRY_RUN" -eq 0 ]; then
  mkdir -p "$EXP_ROOT"
  touch "$FAIL_LOG"
  printf '\n[%s] run_card_baselines_train start\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$FAIL_LOG"
fi

configure_dataset() {
  case "$1" in
    levir_cc)
      EXP_NAME="$LEVIR_CC_BASELINE_EXP"; DATA_ROOT="$LEVIR_CC_ROOT"
      TRAIN_SCRIPT=scripts/train_levir_cc_card_baseline.sh; ROOT_VAR=LEVIR_CC_ROOT; EXP_VAR=LEVIR_CC_BASELINE_EXP ;;
    levir_mci)
      EXP_NAME="$LEVIR_MCI_BASELINE_EXP"; DATA_ROOT="$LEVIR_MCI_ROOT"
      TRAIN_SCRIPT=scripts/train_levir_mci_card_baseline.sh; ROOT_VAR=LEVIR_MCI_ROOT; EXP_VAR=LEVIR_MCI_BASELINE_EXP ;;
    second_cc)
      EXP_NAME="$SECOND_CC_BASELINE_EXP"; DATA_ROOT="$SECOND_CC_ROOT"
      TRAIN_SCRIPT=scripts/train_second_cc_card_rgb_baseline.sh; ROOT_VAR=SECOND_CC_ROOT; EXP_VAR=SECOND_CC_BASELINE_EXP ;;
  esac
}

training_complete() {
  "$PYTHON" - "$1" "$2" <<'PY'
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from build_card_baseline_manifest import audit_resolved_config

exp_dir, dataset = sys.argv[1:]
cfg_path = os.path.join(exp_dir, 'resolved_config.json')
metrics_path = os.path.join(exp_dir, 'val_metrics.csv')
best_path = os.path.join(exp_dir, 'best_training_checkpoints.json')
if not all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in (cfg_path, metrics_path, best_path)):
    raise SystemExit(1)
try:
    cfg = audit_resolved_config(cfg_path, dataset)
    max_iter = int(cfg['train']['max_iter'])
    with open(metrics_path, newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
except Exception:
    raise SystemExit(1)

def existing_checkpoint(raw):
    raw = str(raw or '').strip()
    candidates = (
        raw,
        os.path.abspath(raw),
        os.path.join(exp_dir, raw),
        os.path.join(exp_dir, os.path.basename(raw)),
        os.path.join(exp_dir, 'snapshots', os.path.basename(raw)),
        os.path.join(exp_dir, 'checkpoints', os.path.basename(raw)),
    )
    return any(path and os.path.isfile(path) for path in candidates)

for row in rows:
    try:
        step = int(float(row.get('iter') or row.get('step') or ''))
    except (TypeError, ValueError):
        continue
    if step == max_iter and existing_checkpoint(row.get('snapshot_path') or row.get('checkpoint_path')):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

run_one() {
  local dataset="$1"
  [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "$dataset" ] || return 0
  configure_dataset "$dataset"
  local exp_path="$EXP_ROOT/$EXP_NAME"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN: %s=%q EXP_DIR=%q %s=%q PYTHON=%q bash %q\n' \
      "$EXP_VAR" "$EXP_NAME" "$EXP_ROOT" "$ROOT_VAR" "$DATA_ROOT" "$PYTHON" "$TRAIN_SCRIPT"
    return
  fi
  if training_complete "$exp_path" "$dataset"; then
    echo "Skipping completed baseline training: $exp_path"
    return
  fi
  if [ -d "$exp_path" ] && [ -n "$(find "$exp_path" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    record_failure "$dataset training" "refusing to overwrite non-empty incomplete directory: $exp_path; archive it or choose another EXP_ROOT"
    return
  fi
  mkdir -p "$exp_path"
  export EXP_DIR="$EXP_ROOT" PYTHON
  case "$dataset" in
    levir_cc) export LEVIR_CC_ROOT="$DATA_ROOT" ;;
    levir_mci) export LEVIR_MCI_ROOT="$DATA_ROOT" ;;
    second_cc) export SECOND_CC_ROOT="$DATA_ROOT" ;;
  esac
  if ! bash "$TRAIN_SCRIPT"; then
    record_failure "$dataset training" "see $exp_path/train.log"
    return
  fi
  if ! training_complete "$exp_path" "$dataset"; then
    record_failure "$dataset training" 'training ended without a canonical max-iter validation checkpoint'
  fi
}

for dataset in levir_cc levir_mci second_cc; do run_one "$dataset"; done
[ "$FAILURES" -eq 0 ] || { echo "$FAILURES baseline training step(s) failed; see $FAIL_LOG" >&2; exit 1; }
echo 'Original CARD baseline training stage complete.'
