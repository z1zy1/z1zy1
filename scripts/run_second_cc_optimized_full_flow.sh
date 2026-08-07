#!/usr/bin/env bash
set -euo pipefail

# Validation-only LR refinement followed by an immutable three-seed locked test.

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
RUN_ROOT="${RUN_ROOT:-$EXP_ROOT/second_cc_optimized_runs}"
DATA_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
FEATURE_ROOT="${FEATURE_ROOT:-$DATA_ROOT/features}"
BASELINE_VAL="${SECOND_CC_BASELINE_VAL_METRICS:-$EXP_ROOT/second_cc_card_rgb_baseline/baseline_best_checkpoint.json}"
BASELINE_TEST_SUMMARY="${SECOND_CC_BASELINE_TEST_SUMMARY:-$EXP_ROOT/card_baseline_test_summary.json}"
MCI_INIT_CHECKPOINT="${MCI_INIT_CHECKPOINT:-$EXP_ROOT/levir_mci_masksemantic_repro_seed1111/snapshots/levir_mci_masksemantic_repro_seed1111_checkpoint_5000.pt}"
LOCK_MANIFEST="${LOCK_MANIFEST:-$EXP_ROOT/second_cc_optimized_lock.json}"
SUMMARY_STEM="${SUMMARY_STEM:-second_cc_optimized}"
SEEDS="${SEEDS:-1111 2222 3333}"
STAGE=all
ONLY_SPEC=""
ONLY_SEED=""
DRY_RUN=0
CONFIRM_PRUNE=0
FAILURES=0

# tag, target-module learning rate. Loaded parameters use PRETRAINED_LR_SCALE.
SPECS=(
  'lr15e5,0.00015'
  'lr20e5,0.00020'
  'lr25e5,0.00025'
)

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/run_second_cc_optimized_full_flow.sh [options]

Options:
  --stage STAGE      all, train, select, lock, test, summary, or prune
  --only_spec SPEC   lr15e5, lr20e5, or lr25e5
  --only_seed SEED   one seed listed in SEEDS
  --dry-run          print commands without training/testing
  --confirm-prune    with --stage prune, delete snapshots outside the lock

The all stage runs train -> select -> lock -> test -> summary. Checkpoints are
selected only from validation metrics. Existing locked test results are immutable.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --only_spec) ONLY_SPEC="$2"; shift 2 ;;
    --only_seed) ONLY_SEED="$2"; shift 2 ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    --confirm-prune) CONFIRM_PRUNE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "$STAGE" in all|train|select|lock|test|summary|prune) ;; *) usage; exit 2 ;; esac
if [ -n "$ONLY_SPEC" ] && ! printf '%s\n' "${SPECS[@]}" | cut -d, -f1 | grep -Fxq "$ONLY_SPEC"; then
  echo "Unknown --only_spec: $ONLY_SPEC" >&2; exit 2
fi
if [ -n "$ONLY_SEED" ] && ! printf '%s\n' $SEEDS | grep -Fxq "$ONLY_SEED"; then
  echo "--only_seed must be listed in SEEDS: $ONLY_SEED" >&2; exit 2
fi
if [ "$STAGE" = all ] && { [ -n "$ONLY_SPEC" ] || [ -n "$ONLY_SEED" ]; }; then
  echo 'Filters cannot be used with --stage all because locking requires the complete matrix.' >&2
  exit 2
fi

run_or_print() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

has_final_checkpoint() {
  find "$RUN_ROOT/$1/snapshots" -type f \( -name '*checkpoint_10000.pt' -o -name '*checkpoint_10000.pth' \) -print -quit 2>/dev/null | grep -q .
}

has_any_output() {
  [ -d "$RUN_ROOT/$1" ] && find "$RUN_ROOT/$1" -mindepth 1 -print -quit 2>/dev/null | grep -q .
}

configure_training() {
  local spec="$1" lr="$2" seed="$3"
  export EXP_DIR="$RUN_ROOT" EXP_NAME="second_cc_opt_${spec}_seed${seed}"
  export DATASET=second_cc DATA_ROOT FEATURE_ROOT
  export BASE_CFG=configs/dynamic/transformer_second_cc_aug_sgc_card.yaml MODEL_TYPE=sgc_card
  export USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0
  export USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7
  export USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5
  export USE_FEATURE_REWEIGHT=0 DETACH_REWEIGHT_MASK=1 REWEIGHT_ALPHA=0.2
  export ALLOW_MISSING_PSEUDO_MASK=0 LMASK=0.0 LSEM=0.0 USE_AUX_WARMUP=0
  export INIT_CHECKPOINT="$MCI_INIT_CHECKPOINT" SEED="$seed" LR="$lr" MAX_ITER=10000
  export SNAPSHOT_INTERVAL=1000 SAVE_INTERVAL=1000 EVAL_INTERVAL=1000 LOG_INTERVAL=100
  export FINETUNE_DECODER_ONLY=0 PAPER_SELECTION_MODE=1 SELECTION_STRATEGY=val_baseline_stable_window
  export LR_SCHEDULER=warmup_cosine LR_WARMUP_STEPS=500 MIN_LR_RATIO=0.05 PRETRAINED_LR_SCALE=0.1
  export SEMANTIC_FUSION_GAMMA_INIT=0.01 SEMANTIC_FUSION_GAMMA_MAX=0.5
}

train_one() {
  if has_final_checkpoint "$EXP_NAME"; then
    echo "Skipping $EXP_NAME; final checkpoint exists."
    return
  fi
  if has_any_output "$EXP_NAME"; then
    echo "Refusing non-empty incomplete run: $RUN_ROOT/$EXP_NAME" >&2
    return 1
  fi
  run_or_print bash scripts/_run_paper_training.sh
}

select_one() {
  local output="$RUN_ROOT/$EXP_NAME/best_checkpoint.json"
  local command=(
    "$PYTHON" scripts/select_best_checkpoint.py
    --exp_dir "$RUN_ROOT/$EXP_NAME"
    --strategy val_baseline_stable_window
    --stability_window 3
    --expected_step_gap 1000
    --baseline_metrics "$BASELINE_VAL"
    --protected_metrics Bleu_1,Bleu_2,Bleu_3,Bleu_4,METEOR,ROUGE_L,CIDEr
    --baseline_tolerance 0
    --min_spice_gain 0
    --require_audited_validation_baseline
    --output_json "$output"
    --output_txt "$RUN_ROOT/$EXP_NAME/best_checkpoint.txt"
  )
  if [ -s "$output" ]; then
    echo "Skipping existing validation selection: $output"
  else
    run_or_print "${command[@]}"
  fi
}

matrix_stage() {
  local action="$1" raw spec lr seed
  mkdir -p "$RUN_ROOT"
  for raw in "${SPECS[@]}"; do
    IFS=',' read -r spec lr <<< "$raw"
    [ -z "$ONLY_SPEC" ] || [ "$ONLY_SPEC" = "$spec" ] || continue
    for seed in $SEEDS; do
      [ -z "$ONLY_SEED" ] || [ "$ONLY_SEED" = "$seed" ] || continue
      configure_training "$spec" "$lr" "$seed"
      if ! "${action}_one"; then
        FAILURES=$((FAILURES + 1))
      fi
    done
  done
}

lock_validation() {
  run_or_print "$PYTHON" scripts/build_second_cc_optimized_lock.py \
    --exp_root "$EXP_ROOT" --run_root "$RUN_ROOT" --dataset_root "$DATA_ROOT" \
    --baseline "$BASELINE_VAL" --seeds "$SEEDS" --output "$LOCK_MANIFEST"
}

json_value() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding='utf-8-sig'))
for part in sys.argv[2].split('.'):
    value = value.get(part, '') if isinstance(value, dict) else ''
print(value if value is not None else '')
PY
}

lock_rows() {
  "$PYTHON" - "$LOCK_MANIFEST" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding='utf-8-sig'))
for lock in manifest['locks']:
    print('|'.join(str(value) for value in (
        lock['lock_id'], lock['seed'], lock['target_exp'], lock['checkpoint']['path'],
        lock['source_config'], lock['test_result'])))
PY
}

validate_test_result() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json, os, sys
payload = json.load(open(sys.argv[1], encoding='utf-8-sig'))
canonical = lambda path: os.path.realpath(os.path.abspath(os.path.normpath(path)))
if canonical(payload.get('snapshot_path', '')) != canonical(sys.argv[2]):
    raise SystemExit('Existing test result is bound to a different checkpoint.')
PY
}

test_locked() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: verify $LOCK_MANIFEST and test its three validation-locked rows"
    return
  fi
  [ -f "$LOCK_MANIFEST" ] || { echo "Missing lock manifest: $LOCK_MANIFEST" >&2; return 1; }
  run_or_print "$PYTHON" scripts/build_second_cc_optimized_lock.py --verify "$LOCK_MANIFEST"
  local lock_id seed target checkpoint source_cfg result
  while IFS='|' read -r lock_id seed target checkpoint source_cfg result; do
    if [ -s "$result" ]; then
      validate_test_result "$result" "$checkpoint"
      echo "Skipping immutable locked test: $lock_id"
      continue
    fi
    export DATASET=second_cc DATA_ROOT FEATURE_ROOT
    export ANNO="$DATA_ROOT/second_cc_aug_captions_reformat.json"
    export CHANGEFLAG_JSON="$DATA_ROOT/SECOND-CC-AUG.json" EVAL_CHANGE_NOCHANGE_SPLIT=1
    export BASE_CFG="$source_cfg" MODEL_TYPE=sgc_card PAPER_SELECTION_MODE=1
    export USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0 USE_AUX_SEMANTIC=0
    export USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7
    export USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5
    export USE_FEATURE_REWEIGHT=0 ALLOW_MISSING_PSEUDO_MASK=0 LMASK=0 LSEM=0
    mkdir -p "$EXP_ROOT/$target"
    bash scripts/test_specific_snapshot_sgc_card.sh \
      --exp_dir "$EXP_ROOT/$target" --checkpoint "$checkpoint" --tag optimized_locked
    [ -s "$result" ] || { echo "Locked result not written: $result" >&2; return 1; }
    validate_test_result "$result" "$checkpoint"
  done < <(lock_rows)
}

summarize_locked() {
  run_or_print "$PYTHON" scripts/summarize_second_cc_optimized_tests.py \
    --manifest "$LOCK_MANIFEST" --baseline_summary "$BASELINE_TEST_SUMMARY" \
    --output_json "$EXP_ROOT/${SUMMARY_STEM}_test_summary.json" \
    --output_csv "$EXP_ROOT/${SUMMARY_STEM}_test_summary.csv"
}

prune_unlocked() {
  local command=(
    "$PYTHON" scripts/prune_second_cc_optimized_snapshots.py
    --manifest "$LOCK_MANIFEST"
    --output "$EXP_ROOT/${SUMMARY_STEM}_prune_plan.json"
  )
  [ "$CONFIRM_PRUNE" -eq 0 ] || command+=(--apply)
  run_or_print "${command[@]}"
}

if [ "$DRY_RUN" -eq 0 ]; then
  case "$STAGE" in
    train|all)
      "$PYTHON" -c 'import torch, yaml'
      [ -d "$DATA_ROOT" ] && [ -d "$FEATURE_ROOT" ] || { echo 'SECOND-CC data/features missing.' >&2; exit 1; }
      [ -f "$MCI_INIT_CHECKPOINT" ] || { echo "MCI init checkpoint missing: $MCI_INIT_CHECKPOINT" >&2; exit 1; }
      ;;
    test)
      "$PYTHON" -c 'import torch, yaml'
      [ -d "$DATA_ROOT" ] && [ -d "$FEATURE_ROOT" ] || { echo 'SECOND-CC data/features missing.' >&2; exit 1; }
      ;;
  esac
fi

case "$STAGE" in
  train) matrix_stage train ;;
  select) matrix_stage select ;;
  lock) lock_validation ;;
  test) test_locked ;;
  summary) summarize_locked ;;
  prune) prune_unlocked ;;
  all)
    matrix_stage train
    [ "$FAILURES" -eq 0 ] || { echo "$FAILURES training runs failed." >&2; exit 1; }
    matrix_stage select
    lock_validation
    test_locked
    summarize_locked
    ;;
esac

[ "$FAILURES" -eq 0 ] || { echo "$FAILURES matrix actions failed." >&2; exit 1; }
echo "SECOND-CC optimized flow stage '$STAGE' complete."
