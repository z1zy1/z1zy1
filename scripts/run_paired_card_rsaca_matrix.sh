#!/usr/bin/env bash
set -euo pipefail

# Strict 3 datasets x 2 arms x 3 seeds matrix.  CARD and RSACA are launched
# from the same locked protocol; the arm-specific semantic switches are the
# only intended difference.
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
PAIR_ROOT="${PAIR_ROOT:-./experiments/paired_card_rsaca_whole_gate_v1}"
SEEDS="${SEEDS:-1111 2222 3333}"
STAGE="all"
ONLY_DATASET=""
ONLY_SEED=""
DRY_RUN=0
RESET_INCOMPLETE=0
FORCE_SELECT=0
TEST_TAG="${TEST_TAG:-paired_locked}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/run_paired_card_rsaca_matrix.sh [options]

Runs a fixed CARD versus whole-adapter RSACA matrix for LEVIR-CC,
LEVIR-MCI, and SECOND-CC. Each arm selects its checkpoint on validation data
only, then evaluates that selection on the test set once.

Options:
  --stage all|preflight|train|select|test|summary
  --dataset levir_cc|levir_mci|second_cc
  --seed 1111|2222|3333
  --dry-run
  --reset-incomplete
  --force-select       Refresh a pre-test validation selection only.
  --test-tag TAG       Use an independent immutable test result tag (default: paired_locked).

Environment:
  PAIR_ROOT, SEEDS, PYTHON, NUM_WORKERS, OMP_NUM_THREADS, CUDA_VISIBLE_DEVICES,
  PYTORCH_GPU, LEVIR_CC_ROOT, LEVIR_MCI_ROOT, SECOND_CC_ROOT.

The root protocol lock records the Git commit, tracked source digest, Python,
CUDA visibility, worker count, datasets, and seeds. It refuses later stages if
any of those common conditions differ.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --dataset) ONLY_DATASET="$2"; shift 2 ;;
    --seed) ONLY_SEED="$2"; shift 2 ;;
    --dry-run|--dry_run) DRY_RUN=1; shift ;;
    --reset-incomplete|--reset_incomplete) RESET_INCOMPLETE=1; shift ;;
    --force-select|--force_select) FORCE_SELECT=1; shift ;;
    --test-tag) TEST_TAG="$2"; shift 2 ;;
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

source_digest() {
  {
    git ls-files -- models/CARD.py train_card_spot.py utils configs/dynamic scripts/_run_paper_training.sh \
      scripts/test_specific_snapshot_sgc_card.sh scripts/select_best_snapshot_for_paper.py
    printf '%s\n' scripts/run_paired_card_rsaca_matrix.sh scripts/summarize_paired_card_rsaca_matrix.py
  } | sort -u | while IFS= read -r path; do
    [ -f "$path" ] && sha256sum "$path"
  done | sha256sum | awk '{print $1}'
}

ensure_protocol_lock() {
  local git_commit source_hash source_status
  git_commit="$(git rev-parse HEAD)"
  source_hash="$(source_digest)"
  source_status="$(git status --porcelain -- models/CARD.py train_card_spot.py utils configs/dynamic scripts | sha256sum | awk '{print $1}')"
  run_or_print "$PYTHON" scripts/lock_paired_card_rsaca_protocol.py \
    --pair-root "$PAIR_ROOT" --git-commit "$git_commit" --source-digest "$source_hash" \
    --source-status-digest "$source_status" --python "$PYTHON" \
    --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-}" --pytorch-gpu "${PYTORCH_GPU:-0}" \
    --omp-num-threads "${OMP_NUM_THREADS:-}" --num-workers "${NUM_WORKERS:-default}" \
    --datasets "$DATASETS" --seeds "$SEEDS" --rsaca-arm whole_adapter_gate_v1
}

configure_common() {
  local dataset="$1" seed="$2" arm="$3"
  export EXP_DIR="$PAIR_ROOT/$arm" EXP_NAME="${arm}_${dataset}_seed${seed}"
  export DATASET="$dataset" SEED="$seed" MODEL_TYPE=sgc_card REQUIRE_CUDA
  export MAX_ITER="${MAX_ITER:-10000}" SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1000}"
  export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}" EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
  export LR="${LR:-0.0002}" LR_SCHEDULER="${LR_SCHEDULER:-warmup_cosine}"
  export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}" MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
  export USE_CHANGE_MASK=0 MASK_TYPE=binary NUM_MASK_CLASSES=1 ENABLE_AUX_MASK=0
  export USE_AUX_SEMANTIC=0 USE_FEATURE_REWEIGHT=0 DETACH_REWEIGHT_MASK=1 LMASK=0 LSEM=0
  export SEMANTIC_DIFF_ONLY=0 SEMANTIC_DIFF_BINARY=0 SEMANTIC_UNKNOWN_CHANGE_CLASS=6
  export SEMANTIC_MAP_ROOT= SEMANTIC_BEFORE_PHASE= SEMANTIC_AFTER_PHASE=
  export SEMANTIC_DIFF_ROOT= SEMANTIC_DIFF_PHASE= SEMANTIC_DIFF_CONFIDENCE_ROOT= SEMANTIC_DIFF_CONFIDENCE_PHASE=
  export PAPER_SELECTION_MODE=1
  export SELECTION_STRATEGY="${SELECTION_STRATEGY:-paper_balanced_no_spice}"
  export SELECTION_METRIC="${SELECTION_METRIC:-paper_balanced_no_spice}"
  case "$dataset" in
    levir_cc)
      export DATA_ROOT="${LEVIR_CC_ROOT:-./Levir-CC}" FEATURE_ROOT="${LEVIR_CC_FEATURE_ROOT:-${LEVIR_CC_ROOT:-./Levir-CC}/features}"
      export BASE_CFG=configs/dynamic/transformer_levir_cc_sgc_card.yaml ANNO="$DATA_ROOT/levir_cc_captions_reformat.json"
      ;;
    levir_mci)
      export DATA_ROOT="${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}" FEATURE_ROOT="${LEVIR_MCI_FEATURE_ROOT:-${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}/features}"
      export BASE_CFG=configs/dynamic/transformer_levir_mci_sgc_card.yaml ANNO="$DATA_ROOT/levir_mci_captions_reformat.json"
      ;;
    second_cc)
      export DATA_ROOT="${SECOND_CC_ROOT:-./SECOND-CC-AUG}" FEATURE_ROOT="${SECOND_CC_FEATURE_ROOT:-${SECOND_CC_ROOT:-./SECOND-CC-AUG}/features}"
      export BASE_CFG=configs/dynamic/transformer_second_cc_aug_sgc_card.yaml ANNO="$DATA_ROOT/second_cc_aug_captions_reformat.json"
      ;;
  esac
  if [ "$arm" = card ]; then
    export USE_SEMANTIC_MAPS=0 SEMANTIC_INPUT_MODE=none NUM_SEMANTIC_CLASSES=7
    export USE_SEMANTIC_PARTIAL_DETACH=0 SEMANTIC_DETACH_RATIO=0.0 ALLOW_MISSING_PSEUDO_MASK=0
    export SEMANTIC_FUSION_GAMMA_INIT= SEMANTIC_FUSION_GAMMA_MAX= SEMANTIC_FUSION_NORM_MODE=
    export SEMANTIC_FUSION_SPARSE_CHANGE_TOKENS= SEMANTIC_FUSION_RELIABILITY_GATE=
    export SEMANTIC_FUSION_RELIABILITY_GATE_BIAS= SEMANTIC_FUSION_GLOBAL_TOKEN=
    export SEMANTIC_FUSION_GLOBAL_TOKEN_MODE= SEMANTIC_FUSION_GATE_WHOLE_ADAPTER=
    export SEMANTIC_FUSION_DETACH_RELIABILITY_INPUTS=
  else
    export USE_SEMANTIC_MAPS=1 SEMANTIC_INPUT_MODE=cross_attention NUM_SEMANTIC_CLASSES=7
    export USE_SEMANTIC_PARTIAL_DETACH=1 SEMANTIC_DETACH_RATIO=0.5
    export SEMANTIC_FUSION_GAMMA_INIT=0.01 SEMANTIC_FUSION_GAMMA_MAX=0.5
    export SEMANTIC_FUSION_NORM_MODE=legacy_post_norm SEMANTIC_FUSION_SPARSE_CHANGE_TOKENS=1
    export SEMANTIC_FUSION_RELIABILITY_GATE=1 SEMANTIC_FUSION_RELIABILITY_GATE_BIAS=-1.5
    export SEMANTIC_FUSION_GLOBAL_TOKEN=1 SEMANTIC_FUSION_GLOBAL_TOKEN_MODE=all_mean
    export SEMANTIC_FUSION_GATE_WHOLE_ADAPTER=1 SEMANTIC_FUSION_DETACH_RELIABILITY_INPUTS=0
    case "$dataset" in
      levir_cc)
        export SEMANTIC_DIFF_ROOT="$DATA_ROOT/pseudo_masks" SEMANTIC_DIFF_ONLY=1 SEMANTIC_DIFF_BINARY=1
        export ALLOW_MISSING_PSEUDO_MASK=1
        ;;
      levir_mci)
        export SEMANTIC_DIFF_ROOT="$DATA_ROOT/images" SEMANTIC_DIFF_PHASE=label SEMANTIC_DIFF_ONLY=1
        export ALLOW_MISSING_PSEUDO_MASK=0
        ;;
      second_cc)
        export SEMANTIC_MAP_ROOT="$DATA_ROOT" SEMANTIC_BEFORE_PHASE=sem/A SEMANTIC_AFTER_PHASE=sem/B
        export ALLOW_MISSING_PSEUDO_MASK=0
        ;;
    esac
  fi
}

selected_checkpoint() {
  "$PYTHON" - "$1" <<'PY'
import json, os, sys
with open(sys.argv[1], encoding='utf-8-sig') as handle:
    snapshot = json.load(handle).get('best_snapshot', '')
if not snapshot or not os.path.isfile(snapshot):
    raise SystemExit('Selected checkpoint is missing or is not a file: %s' % snapshot)
print(snapshot)
PY
}

preflight() {
  run_or_print "$PYTHON" scripts/check_unified_semantic_inputs.py \
    --levir_cc_root "${LEVIR_CC_ROOT:-./Levir-CC}" --levir_mci_root "${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}" \
    --second_cc_root "${SECOND_CC_ROOT:-./SECOND-CC-AUG}"
  run_or_print "$PYTHON" scripts/audit_unified_semantic_inputs.py \
    --levir_cc_root "${LEVIR_CC_ROOT:-./Levir-CC}" --levir_mci_root "${LEVIR_MCI_ROOT:-./LEVIR-MCI-dataset}" \
    --second_cc_root "${SECOND_CC_ROOT:-./SECOND-CC-AUG}" --output "$PAIR_ROOT/semantic_input_audit.json"
}

train_one() {
  local exp_path="$EXP_DIR/$EXP_NAME"
  if [ -f "$exp_path/snapshots/${EXP_NAME}_checkpoint_${MAX_ITER}.pt" ] || [ -f "$exp_path/snapshots/${EXP_NAME}_checkpoint_${MAX_ITER}.pth" ]; then
    echo "Skipping completed run: $EXP_NAME"; return
  fi
  if [ -d "$exp_path" ] && find "$exp_path" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    if [ "$RESET_INCOMPLETE" -ne 1 ]; then
      echo "Refusing non-empty incomplete run: $exp_path" >&2; return 1
    fi
    local archive="$PAIR_ROOT/aborted/${EXP_NAME}_$(date -u +%Y%m%dT%H%M%SZ)"
    if [ "$DRY_RUN" -eq 1 ]; then echo "DRY RUN: archive $exp_path -> $archive"; else mkdir -p "$PAIR_ROOT/aborted"; mv "$exp_path" "$archive"; fi
  fi
  run_or_print bash scripts/_run_paper_training.sh
}

select_one() {
  local exp_path="$EXP_DIR/$EXP_NAME" output="$EXP_DIR/$EXP_NAME/best_snapshot_for_paper.json"
  if [ -s "$output" ]; then
    if selected_checkpoint "$output" >/dev/null 2>&1; then
      if [ "$FORCE_SELECT" -ne 1 ]; then echo "Skipping validation selection: $output"; return; fi
      if find "$exp_path" -maxdepth 1 -type f -name 'test_*_result.json' -print -quit | grep -q .; then
        echo "Refusing selection refresh after test: $exp_path" >&2
        return 1
      fi
    else
      echo "Ignoring invalid validation selection: $output" >&2
    fi
  fi
  run_or_print "$PYTHON" scripts/select_best_snapshot_for_paper.py --exp_dir "$exp_path" --csv "$exp_path/val_metrics.csv" \
    --metric "$SELECTION_METRIC" --output_json "$output" --copy_path "$exp_path/best_for_paper.pth"
}

test_one() {
  local exp_path="$EXP_DIR/$EXP_NAME" result="$EXP_DIR/$EXP_NAME/test_${TEST_TAG}_result.json" selection="$EXP_DIR/$EXP_NAME/best_snapshot_for_paper.json"
  [ -s "$result" ] && { echo "Skipping immutable test: $result"; return; }
  [ "$DRY_RUN" -eq 1 ] || [ -s "$selection" ] || { echo "Missing validation selection: $selection" >&2; return 1; }
  local checkpoint="$exp_path/best_for_paper.pth"
  if [ "$DRY_RUN" -ne 1 ]; then checkpoint="$(selected_checkpoint "$selection")"; fi
  run_or_print env SEED="$SEED" bash scripts/test_specific_snapshot_sgc_card.sh --exp_dir "$exp_path" --checkpoint "$checkpoint" --tag "$TEST_TAG" --anno "$ANNO"
}

matrix() {
  local action="$1" dataset seed arm
  for dataset in $DATASETS; do
    [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "$dataset" ] || continue
    for seed in $SEEDS; do
      [ -z "$ONLY_SEED" ] || [ "$ONLY_SEED" = "$seed" ] || continue
      for arm in card rsaca; do
        configure_common "$dataset" "$seed" "$arm"
        echo "========== paired ${STAGE}: arm=$arm dataset=$dataset seed=$seed root=$PAIR_ROOT selection=$SELECTION_METRIC =========="
        "$action"
      done
    done
  done
}

summarize() {
  run_or_print "$PYTHON" scripts/summarize_paired_card_rsaca_matrix.py --pair-root "$PAIR_ROOT" \
    --datasets "${ONLY_DATASET:-levir_cc,levir_mci,second_cc}" --seeds "${ONLY_SEED:-1111,2222,3333}" \
    --output "$PAIR_ROOT/summary.json"
}

ensure_protocol_lock
case "$STAGE" in
  preflight) preflight ;;
  train) matrix train_one ;;
  select) matrix select_one ;;
  test) matrix test_one ;;
  summary) summarize ;;
  all) preflight; matrix train_one; matrix select_one; matrix test_one; summarize ;;
esac
