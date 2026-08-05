#!/usr/bin/env bash
set -euo pipefail

# Train and validation-select the SECOND-CC transfer candidates recommended by
# the MCI transfer analysis. The LEVIR-MCI and LEVIR-CC validation locks are
# audited before training so a SECOND-CC change cannot silently replace them.

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
SECOND_CC_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
MCI_INIT_CHECKPOINT="${MCI_INIT_CHECKPOINT:-$EXP_ROOT/levir_mci_masksemantic_repro_seed1111/snapshots/levir_mci_masksemantic_repro_seed1111_checkpoint_5000.pt}"
SECOND_CC_BASELINE_VAL_METRICS="${SECOND_CC_BASELINE_VAL_METRICS:-$EXP_ROOT/second_cc_card_rgb_baseline/baseline_best_checkpoint.json}"
LEVIR_CC_GUARD_ROOT="${LEVIR_CC_GUARD_ROOT:-$EXP_ROOT}"
LEVIR_MCI_GUARD_ROOT="${LEVIR_MCI_GUARD_ROOT:-$EXP_ROOT}"
SEEDS="${SEEDS:-1111 2222 3333}"
ONLY_SPEC=""
ONLY_SEED=""
DRY_RUN=0
SKIP_TRAIN=0
SKIP_SELECT=0
PASSED=0
FAILED=0

# mode, detach ratio, learning rate, initialization kind
SPECS=(
  'mci_pd05_lr1e4,0.5,0.0001,mci'
  'mci_pd05_lr2e4,0.5,0.0002,mci'
  'mci_pd07_lr1e4,0.7,0.0001,mci'
  'scratch_pd05_lr2e4,0.5,0.0002,scratch'
)

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/run_second_cc_mci_crossattn_matrix.sh [options]

Options:
  --only_spec NAME   Run one matrix spec: mci_pd05_lr1e4, mci_pd05_lr2e4,
                     mci_pd07_lr1e4, or scratch_pd05_lr2e4.
  --only_seed SEED   Run one seed from SEEDS only.
  --skip_train       Select existing runs without training.
  --skip_select      Train only; do not select validation checkpoints.
  --dry_run, --dry-run
                     Print commands without changing experiment outputs.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --only_spec) ONLY_SPEC="$2"; shift 2 ;;
    --only_seed) ONLY_SEED="$2"; shift 2 ;;
    --skip_train) SKIP_TRAIN=1; shift ;;
    --skip_select) SKIP_SELECT=1; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

contains_seed() {
  local wanted="$1" seed
  for seed in $SEEDS; do [ "$seed" = "$wanted" ] && return 0; done
  return 1
}

contains_spec() {
  local wanted="$1" spec name
  for spec in "${SPECS[@]}"; do
    IFS=',' read -r name _ <<< "$spec"
    [ "$name" = "$wanted" ] && return 0
  done
  return 1
}

[ -z "$ONLY_SPEC" ] || contains_spec "$ONLY_SPEC" || { echo "Unknown --only_spec: $ONLY_SPEC" >&2; exit 2; }
[ -z "$ONLY_SEED" ] || contains_seed "$ONLY_SEED" || { echo "--only_seed must be listed in SEEDS: $ONLY_SEED" >&2; exit 2; }

audit_guard_selection() {
  local path="$1" label="$2"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: audit $label at $path"
    return 0
  fi
  "$PYTHON" - "$path" "$label" <<'PY'
import json
import sys

path, label = sys.argv[1:]
with open(path, encoding='utf-8-sig') as handle:
    payload = json.load(handle)
assert payload.get('status') == 'done', '%s is not selected: %s' % (label, path)
assert payload.get('selection_uses_test_metrics') is False, '%s uses test metrics' % label
assert payload.get('selection_metric_split') == 'validation', '%s is not validation-selected' % label
assert payload.get('selected_checkpoint'), '%s has no selected checkpoint' % label
deltas = payload.get('selected_metric_deltas') or {}
required = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
missing = [metric for metric in required if metric not in deltas]
assert not missing, '%s is missing deltas: %s' % (label, ','.join(missing))
bad = {metric: deltas[metric] for metric in required if float(deltas[metric]) <= 0.0}
assert not bad, '%s does not strictly beat baseline: %s' % (label, bad)
PY
}

audit_existing_guards() {
  local seed path
  for seed in $SEEDS; do
    path="$LEVIR_CC_GUARD_ROOT/levir_cc_mci_init_seed${seed}/best_checkpoint.json"
    audit_guard_selection "$path" "levir_cc seed ${seed}"
    path="$LEVIR_MCI_GUARD_ROOT/levir_mci_masksemantic_repro_seed${seed}_7_6_val_locked/best_checkpoint.json"
    audit_guard_selection "$path" "levir_mci seed ${seed}"
  done
}

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

configure() {
  local spec="$1" seed="$2" detach="$3" lr="$4" init_kind="$5"
  clear_exp_env
  export EXP_DIR="$EXP_ROOT" EXP_NAME="second_cc_${spec}_seed${seed}"
  export DATASET=second_cc DATA_ROOT="$SECOND_CC_ROOT" FEATURE_ROOT="$SECOND_CC_ROOT/features"
  export BASE_CFG=configs/dynamic/transformer_second_cc_aug_sgc_card.yaml MODEL_TYPE=sgc_card
  export USE_CHANGE_MASK=0 MASK_TYPE=binary ENABLE_AUX_MASK=0
  export USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7
  export USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO="$detach"
  export USE_FEATURE_REWEIGHT=0 DETACH_REWEIGHT_MASK=1 REWEIGHT_ALPHA=0.2
  export ALLOW_MISSING_PSEUDO_MASK=0 LMASK=0.0 LSEM=0.0 USE_AUX_WARMUP=0
  export INIT_CHECKPOINT=""
  if [ "$init_kind" = mci ]; then INIT_CHECKPOINT="$MCI_INIT_CHECKPOINT"; fi
  export INIT_CHECKPOINT SEED="$seed" LR="$lr" MAX_ITER=10000
  export SAVE_INTERVAL=1000 EVAL_INTERVAL=1000 SNAPSHOT_INTERVAL=1000 LOG_INTERVAL=100
  export FINETUNE_DECODER_ONLY=0 PAPER_SELECTION_MODE=1 SELECTION_STRATEGY=val_baseline_pareto
}

has_final_checkpoint() {
  find "$EXP_ROOT/$1/snapshots" -type f \( -name '*checkpoint_10000.pt' -o -name '*checkpoint_10000.pth' \) -print -quit 2>/dev/null | grep -q .
}

has_any_output() {
  [ -d "$EXP_ROOT/$1" ] && find "$EXP_ROOT/$1" -mindepth 1 -print -quit 2>/dev/null | grep -q .
}

train_one() {
  local exp="$EXP_NAME"
  if [ "$DRY_RUN" -eq 0 ]; then
    [ -d "$DATA_ROOT" ] && [ -d "$FEATURE_ROOT" ] || { echo "Missing SECOND-CC dataset/features for $exp" >&2; return 1; }
    if [ -n "$INIT_CHECKPOINT" ]; then
      [ -f "$INIT_CHECKPOINT" ] || { echo "Missing MCI init checkpoint: $INIT_CHECKPOINT" >&2; return 1; }
    fi
  fi
  if has_final_checkpoint "$exp"; then
    echo "Skipping $exp; final checkpoint already exists."
    return 0
  fi
  if has_any_output "$exp"; then
    echo "Refusing non-empty incomplete experiment directory: $EXP_ROOT/$exp" >&2
    return 1
  fi
  printf 'SECOND-CC TRAIN: exp=%q seed=%q lr=%q detach=%q init=%q\n' "$exp" "$SEED" "$LR" "$SEMANTIC_DETACH_RATIO" "${INIT_CHECKPOINT:-scratch}"
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
    --baseline_metrics "$SECOND_CC_BASELINE_VAL_METRICS"
    --protected_metrics Bleu_1,Bleu_2,Bleu_3,Bleu_4,METEOR,ROUGE_L,CIDEr
    --baseline_tolerance 0
    --min_spice_gain 0
    --require_audited_validation_baseline
    --output_json "$output_json"
    --output_txt "$EXP_ROOT/$EXP_NAME/best_checkpoint.txt"
  )
  if [ "$DRY_RUN" -eq 0 ]; then
    [ -f "$SECOND_CC_BASELINE_VAL_METRICS" ] || { echo "Missing SECOND-CC validation baseline: $SECOND_CC_BASELINE_VAL_METRICS" >&2; return 1; }
  fi
  if [ -s "$output_json" ]; then
    echo "Skipping existing validation selection: $output_json"
  elif [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
  else
    "${command[@]}"
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    if audit_guard_selection "$output_json" "$EXP_NAME"; then
      PASSED=$((PASSED + 1))
    else
      echo "FAILED validation guard: $EXP_NAME" >&2
      FAILED=$((FAILED + 1))
    fi
  fi
}

mkdir -p "$EXP_ROOT"
if [ "$DRY_RUN" -eq 0 ]; then
  audit_existing_guards
fi

for raw_spec in "${SPECS[@]}"; do
  IFS=',' read -r spec detach lr init_kind <<< "$raw_spec"
  [ -z "$ONLY_SPEC" ] || [ "$ONLY_SPEC" = "$spec" ] || continue
  for seed in $SEEDS; do
    [ -z "$ONLY_SEED" ] || [ "$ONLY_SEED" = "$seed" ] || continue
    configure "$spec" "$seed" "$detach" "$lr" "$init_kind"
    [ "$SKIP_TRAIN" -eq 1 ] || train_one || { FAILED=$((FAILED + 1)); continue; }
    [ "$SKIP_SELECT" -eq 1 ] || select_one
  done
done

if [ "$DRY_RUN" -eq 0 ]; then
  audit_existing_guards
  echo "SECOND-CC matrix complete: passed=$PASSED failed=$FAILED"
  if [ "$SKIP_SELECT" -eq 0 ]; then
    [ "$PASSED" -gt 0 ] || { echo 'No SECOND-CC candidate strictly beats the audited validation baseline.' >&2; exit 1; }
    [ "$FAILED" -eq 0 ] || { echo 'Some SECOND-CC candidates failed the strict validation guard; inspect their best_checkpoint.json files.' >&2; exit 1; }
  fi
fi

echo 'Training/validation selection complete. Test only a reviewed, validation-locked checkpoint in a separate directory.'
