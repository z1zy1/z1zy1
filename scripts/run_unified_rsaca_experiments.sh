#!/usr/bin/env bash
set -euo pipefail

# Full CARD + RSACA matrix: three datasets x three seeds. Checkpoint selection
# is validation-only; test outputs are immutable once written.
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-./experiments/unified_rsaca}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-./experiments/card_baseline_test_summary.json}"
SEEDS="${SEEDS:-1111 2222 3333}"
STAGE="all"
ONLY_DATASET=""
ONLY_SEED=""
DRY_RUN=0

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/run_unified_rsaca_experiments.sh [options]

Options:
  --stage all|preflight|train|select|test|summary
  --dataset levir_cc|levir_mci|second_cc
  --seed 1111|2222|3333
  --dry-run

The all stage runs preflight -> train -> validation selection -> locked test -> summary.
Use DATA_ROOT/FEATURE_ROOT overrides only through the dataset-specific *_ROOT variables.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --dataset) ONLY_DATASET="$2"; shift 2 ;;
    --seed) ONLY_SEED="$2"; shift 2 ;;
    --dry-run|--dry_run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "$STAGE" in all|preflight|train|select|test|summary) ;; *) usage; exit 2 ;; esac
DATASETS="levir_cc levir_mci second_cc"
if [ -n "$ONLY_DATASET" ] && ! printf '%s\n' $DATASETS | grep -Fxq "$ONLY_DATASET"; then
  echo "Unknown dataset: $ONLY_DATASET" >&2; exit 2
fi
if [ -n "$ONLY_SEED" ] && ! printf '%s\n' $SEEDS | grep -Fxq "$ONLY_SEED"; then
  echo "Seed is not in SEEDS: $ONLY_SEED" >&2; exit 2
fi

run_or_print() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

configure_case() {
  local dataset="$1" seed="$2"
  export EXP_DIR="$RUN_ROOT" EXP_NAME="unified_rsaca_${dataset}_seed${seed}"
  export DATASET="$dataset" SEED="$seed" MODEL_TYPE=sgc_card
  export MAX_ITER="${MAX_ITER:-10000}" SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1000}"
  export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}" EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
  export LR="${LR:-0.0002}" LR_SCHEDULER="${LR_SCHEDULER:-warmup_cosine}"
  export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}" MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
  export USE_CHANGE_MASK=0 MASK_TYPE=binary NUM_MASK_CLASSES=1 ENABLE_AUX_MASK=0
  export USE_AUX_SEMANTIC=0 USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention
  export NUM_SEMANTIC_CLASSES=7 USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5
  export USE_FEATURE_REWEIGHT=0 DETACH_REWEIGHT_MASK=1 LMASK=0 LSEM=0
  export SEMANTIC_FUSION_GAMMA_INIT="${SEMANTIC_FUSION_GAMMA_INIT:-0.01}"
  export SEMANTIC_FUSION_GAMMA_MAX="${SEMANTIC_FUSION_GAMMA_MAX:-0.5}"
  export SEMANTIC_FUSION_NORM_MODE="${SEMANTIC_FUSION_NORM_MODE:-legacy_post_norm}"
  export SEMANTIC_FUSION_GLOBAL_TOKEN_MODE="${SEMANTIC_FUSION_GLOBAL_TOKEN_MODE:-all_mean}"
  export SEMANTIC_FUSION_VISUAL_CONSISTENCY_GATE="${SEMANTIC_FUSION_VISUAL_CONSISTENCY_GATE:-0}"
  export SEMANTIC_FUSION_VISUAL_FALLBACK="${SEMANTIC_FUSION_VISUAL_FALLBACK:-0}"
  export SEMANTIC_FUSION_WARMUP_STEPS="${SEMANTIC_FUSION_WARMUP_STEPS:-0}"
  export SEMANTIC_DIFF_ONLY=0 SEMANTIC_DIFF_BINARY=0 SEMANTIC_UNKNOWN_CHANGE_CLASS=6
  export SEMANTIC_MAP_ROOT= SEMANTIC_BEFORE_PHASE= SEMANTIC_AFTER_PHASE=
  export SEMANTIC_DIFF_ROOT= SEMANTIC_DIFF_PHASE=
  export SEMANTIC_DIFF_CONFIDENCE_ROOT="${SEMANTIC_DIFF_CONFIDENCE_ROOT:-}"
  export SEMANTIC_DIFF_CONFIDENCE_PHASE="${SEMANTIC_DIFF_CONFIDENCE_PHASE:-}"
  case "$dataset" in
    levir_cc)
      export DATA_ROOT="${LEVIR_CC_ROOT:-./Levir-CC}"
      export FEATURE_ROOT="${LEVIR_CC_FEATURE_ROOT:-$DATA_ROOT/features}"
      export BASE_CFG=configs/dynamic/transformer_levir_cc_sgc_card.yaml
      export SEMANTIC_DIFF_ROOT="$DATA_ROOT/pseudo_masks" SEMANTIC_DIFF_ONLY=1 SEMANTIC_DIFF_BINARY=1
      export ALLOW_MISSING_PSEUDO_MASK=1 ANNO="$DATA_ROOT/levir_cc_captions_reformat.json"
      ;;
    levir_mci)
      export DATA_ROOT="${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}"
      export FEATURE_ROOT="${LEVIR_MCI_FEATURE_ROOT:-$DATA_ROOT/features}"
      export BASE_CFG=configs/dynamic/transformer_levir_mci_sgc_card.yaml
      export SEMANTIC_DIFF_ROOT="$DATA_ROOT/images" SEMANTIC_DIFF_PHASE=label SEMANTIC_DIFF_ONLY=1
      export ALLOW_MISSING_PSEUDO_MASK=0 ANNO="$DATA_ROOT/levir_mci_captions_reformat.json"
      ;;
    second_cc)
      export DATA_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
      export FEATURE_ROOT="${SECOND_CC_FEATURE_ROOT:-$DATA_ROOT/features}"
      export BASE_CFG=configs/dynamic/transformer_second_cc_aug_sgc_card.yaml
      export SEMANTIC_MAP_ROOT="$DATA_ROOT" SEMANTIC_BEFORE_PHASE=sem/A SEMANTIC_AFTER_PHASE=sem/B
      export ALLOW_MISSING_PSEUDO_MASK=0 ANNO="$DATA_ROOT/second_cc_aug_captions_reformat.json"
      ;;
  esac
  # Keep the recorded config and the actual validation selector consistent.
  export PAPER_SELECTION_MODE=1
  export SELECTION_STRATEGY="${SELECTION_STRATEGY:-paper_balanced}"
  export SELECTION_METRIC="${SELECTION_METRIC:-paper_balanced}"
}

selected_checkpoint() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8-sig') as f:
    print(json.load(f)['best_snapshot'])
PY
}

preflight() {
  run_or_print "$PYTHON" scripts/check_unified_semantic_inputs.py \
    --levir_cc_root "${LEVIR_CC_ROOT:-./Levir-CC}" \
    --levir_mci_root "${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}" \
    --second_cc_root "${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
  run_or_print "$PYTHON" scripts/audit_unified_semantic_inputs.py \
    --levir_cc_root "${LEVIR_CC_ROOT:-./Levir-CC}" \
    --levir_mci_root "${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}" \
    --second_cc_root "${SECOND_CC_ROOT:-./SECOND-CC-AUG}" \
    --output "$RUN_ROOT/semantic_input_audit.json"
}

train_one() {
  local exp_path="$RUN_ROOT/$EXP_NAME"
  if [ -f "$exp_path/snapshots/${EXP_NAME}_checkpoint_${MAX_ITER}.pt" ] || [ -f "$exp_path/snapshots/${EXP_NAME}_checkpoint_${MAX_ITER}.pth" ]; then
    echo "Skipping completed run: $EXP_NAME"; return
  fi
  if [ -d "$exp_path" ] && find "$exp_path" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing non-empty incomplete run: $exp_path" >&2; return 1
  fi
  run_or_print bash scripts/_run_paper_training.sh
}

select_one() {
  local exp_path="$RUN_ROOT/$EXP_NAME"
  local output="$exp_path/best_snapshot_for_paper.json"
  [ -s "$output" ] && { echo "Skipping validation selection: $output"; return; }
  run_or_print "$PYTHON" scripts/select_best_snapshot_for_paper.py \
    --exp_dir "$exp_path" --csv "$exp_path/val_metrics.csv" \
    --metric "$SELECTION_METRIC" \
    --output_json "$output" --copy_path "$exp_path/best_for_paper.pth"
}

test_one() {
  local exp_path="$RUN_ROOT/$EXP_NAME"
  local selection="$exp_path/best_snapshot_for_paper.json"
  local result="$exp_path/test_unified_locked_result.json"
  [ -s "$result" ] && { echo "Skipping immutable test: $result"; return; }
  [ -s "$selection" ] || { echo "Missing validation selection: $selection" >&2; return 1; }
  local checkpoint
  checkpoint="$(selected_checkpoint "$selection")"
  run_or_print bash scripts/test_specific_snapshot_sgc_card.sh \
    --exp_dir "$exp_path" --checkpoint "$checkpoint" --tag unified_locked --anno "$ANNO"
}

matrix() {
  local action="$1" dataset seed
  for dataset in $DATASETS; do
    [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "$dataset" ] || continue
    for seed in $SEEDS; do
      [ -z "$ONLY_SEED" ] || [ "$ONLY_SEED" = "$seed" ] || continue
      configure_case "$dataset" "$seed"
      echo "========== RSACA ${STAGE}: dataset=${dataset} seed=${seed} run_root=${RUN_ROOT} norm_mode=${SEMANTIC_FUSION_NORM_MODE} selection=${SELECTION_METRIC} =========="
      "$action"
    done
  done
}

case "$STAGE" in
  preflight) preflight ;;
  train) matrix train_one ;;
  select) matrix select_one ;;
  test) matrix test_one ;;
  summary)
    run_or_print "$PYTHON" scripts/summarize_unified_rsaca.py \
      --run_root "$RUN_ROOT" --baseline "$BASELINE_SUMMARY" \
      --output "$RUN_ROOT/summary.json"
    ;;
  all)
    preflight
    matrix train_one
    matrix select_one
    matrix test_one
    run_or_print "$PYTHON" scripts/summarize_unified_rsaca.py \
      --run_root "$RUN_ROOT" --baseline "$BASELINE_SUMMARY" \
      --output "$RUN_ROOT/summary.json"
    ;;
esac
