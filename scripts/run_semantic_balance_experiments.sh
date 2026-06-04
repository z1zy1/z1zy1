#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_GPU="${PYTORCH_GPU:-0}"

BASE_CFG="${BASE_CFG:-configs/dynamic/transformer_levir_cc_aux_mask_conf_semantic.yaml}"
ANNO="${ANNO:-./Levir-CC/levir_cc_captions_reformat.json}"
EXP_DIR="${EXP_DIR:-./outputs/semantic_balance}"
RESULTS_DIR="${RESULTS_DIR:-./results/semantic_balance}"
MAX_ITER="${MAX_ITER:-10000}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1000}"
MIN_CKPT="${MIN_CKPT:-1000}"
MAX_CKPT="${MAX_CKPT:-10000}"
STABLE_WINDOW="${STABLE_WINDOW:-1}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

COMMON_OPTS=(
  exp_dir "$EXP_DIR"
  gpu_id "[$PYTORCH_GPU]"
  model.enable_aux_mask True
  train.use_mask_conf_filter True
  train.mask_conf_threshold 0.5
  train.use_mask_warmup False
  train.mask_warmup_steps 0
  train.use_semantic_aux True
  train.lambda_semantic 0.10
  train.semantic_loss_type multilabel_bce
  train.semantic_tag_file configs/semantic_tags/levir_cc_object_action_tags.txt
  train.semantic_aux_dropout 0.1
  train.semantic_normalize_synonyms True
  train.use_relation_aux False
  train.max_iter "$MAX_ITER"
  train.snapshot_interval "$SNAPSHOT_INTERVAL"
  data.dataset rcc_dataset_transformer_levir
  data.default_feature_dir ./Levir-CC/features
  data.semantic_feature_dir ./Levir-CC/features
  data.default_img_dir ./Levir-CC/images
  data.semantic_img_dir ./Levir-CC/images
  data.default_phase A
  data.semantic_phase B
  data.pseudo_mask_root ./Levir-CC/pseudo_masks
  data.allow_missing_pseudo_mask False
  data.splits_json ./Levir-CC/splits.json
  data.vocab_json ./Levir-CC/transformer_levir_vocab.json
  data.h5_label_file ./Levir-CC/transformer_levir_labels.h5
)

prepare_output_dirs() {
  local exp_name="$1"
  local out_dir="$EXP_DIR/$exp_name"
  local result_dir="$RESULTS_DIR/$exp_name"

  if [ "$ALLOW_EXISTING" != "1" ] && { [ -e "$out_dir" ] || [ -e "$result_dir" ]; }; then
    echo "Refusing to overwrite existing output for $exp_name." >&2
    echo "Existing path: $out_dir or $result_dir" >&2
    echo "Set ALLOW_EXISTING=1 only if you intentionally want to reuse the directory." >&2
    exit 1
  fi

  mkdir -p "$out_dir" "$result_dir"
}

run_one() {
  local tag="$1"
  local exp_name="$2"
  local lambda_mask="$3"
  local use_semantic_warmup="$4"
  local semantic_late_start="$5"
  local semantic_warmup_start="$6"
  local semantic_warmup_end="$7"
  local semantic_start_iter="$8"

  prepare_output_dirs "$exp_name"

  local out_dir="$EXP_DIR/$exp_name"
  local result_dir="$RESULTS_DIR/$exp_name"
  local eval_dir="$out_dir/eval_sents"
  local select_dir="$result_dir/snapshot_selection"
  local test_caption_dir="$out_dir/test_output/captions"

  local -a opts=(
    "${COMMON_OPTS[@]}"
    exp_name "$exp_name"
    train.lambda_mask "$lambda_mask"
    train.use_semantic_warmup "$use_semantic_warmup"
    train.semantic_warmup_start "$semantic_warmup_start"
    train.semantic_warmup_end "$semantic_warmup_end"
    train.semantic_warmup_type linear
    train.semantic_late_start "$semantic_late_start"
    train.semantic_start_iter "$semantic_start_iter"
  )

  echo "========== [$tag] train $exp_name =========="
  python train_card_spot.py \
    --cfg "$BASE_CFG" \
    "${opts[@]}" \
    2>&1 | tee "$out_dir/train.log"

  echo "========== [$tag] eval validation snapshots =========="
  python evaluate_spot.py \
    --results_dir "$eval_dir" \
    --anno "$ANNO" \
    2>&1 | tee "$result_dir/eval.log"
  cp "$eval_dir/eval_results.txt" "$result_dir/eval_results.txt"

  echo "========== [$tag] select best snapshot from validation =========="
  python scripts/select_best_snapshot_from_eval_txt.py \
    --input "$eval_dir/eval_results.txt" \
    --output_dir "$select_dir" \
    --config_name "$exp_name" \
    --min_ckpt "$MIN_CKPT" \
    --max_ckpt "$MAX_CKPT" \
    --stable_window "$STABLE_WINDOW" \
    --save_all \
    2>&1 | tee "$result_dir/select_snapshot.log"

  local best_ckpt
  best_ckpt="$(awk -F, 'NR==2 {gsub(/\r/, "", $2); print $2}' "$select_dir/best_checkpoint.csv")"
  if [ -z "$best_ckpt" ]; then
    echo "Failed to read selected checkpoint from $select_dir/best_checkpoint.csv" >&2
    exit 1
  fi

  echo "========== [$tag] test selected snapshot $best_ckpt =========="
  python test_card_spot.py \
    --cfg "$BASE_CFG" \
    --snapshot "$best_ckpt" \
    --gpu "$PYTORCH_GPU" \
    "${opts[@]}" \
    2>&1 | tee "$result_dir/test.log"

  echo "========== [$tag] eval test output =========="
  python evaluate_spot.py \
    --results_dir "$test_caption_dir" \
    --anno "$ANNO" \
    2>&1 | tee "$result_dir/test_eval.log"
  cp "$test_caption_dir/eval_results.txt" "$result_dir/test_results.txt"

  echo "$tag,$exp_name,$best_ckpt,$select_dir/best_checkpoint.md,$result_dir/test_results.txt" >> "$RESULTS_DIR/summary.csv"
}

if [ "$#" -eq 0 ]; then
  TARGETS=(A B C D)
else
  TARGETS=("$@")
fi

mkdir -p "$RESULTS_DIR"
echo "tag,exp_name,best_ckpt,selection_md,test_results" > "$RESULTS_DIR/summary.csv"

for target in "${TARGETS[@]}"; do
  case "$target" in
    A)
      run_one A lmask001_lsem01 0.01 False False 3000 7000 5000
      ;;
    B)
      run_one B lmask0015_lsem01 0.015 False False 3000 7000 5000
      ;;
    C)
      run_one C lmask002_lsem01_semwarmup 0.02 True False 3000 7000 5000
      ;;
    D)
      run_one D lmask002_lsem01_semlatestart 0.02 False True 3000 7000 5000
      ;;
    *)
      echo "Unknown experiment target: $target. Valid targets: A B C D" >&2
      exit 1
      ;;
  esac
done

echo "Finished requested experiments. Summary: $RESULTS_DIR/summary.csv"
