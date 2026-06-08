#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_GPU="${PYTORCH_GPU:-0}"

BASE_CFG="${BASE_CFG:-configs/dynamic/transformer_levir_cc_aux_mask_conf_semantic.yaml}"
ANNO="${ANNO:-./Levir-CC/levir_cc_captions_reformat.json}"
EXP_ROOT="${EXP_ROOT:-./outputs}"
RESULTS_ROOT="${RESULTS_ROOT:-./results}"
MAX_ITER="${MAX_ITER:-10000}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1000}"
MIN_CKPT="${MIN_CKPT:-1000}"
MAX_CKPT="${MAX_CKPT:-10000}"
STABLE_WINDOW="${STABLE_WINDOW:-1}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

prepare_output_dirs() {
  local exp_name="$1"
  local out_dir="$EXP_ROOT/$exp_name"
  local result_dir="$RESULTS_ROOT/$exp_name"

  if [ "$ALLOW_EXISTING" != "1" ] && { [ -e "$out_dir" ] || [ -e "$result_dir" ]; }; then
    echo "Refusing to overwrite existing output for $exp_name." >&2
    echo "Existing path: $out_dir or $result_dir" >&2
    echo "Set ALLOW_EXISTING=1 only if you intentionally want to reuse the directory." >&2
    exit 1
  fi

  mkdir -p "$out_dir" "$result_dir"
}

write_args_file() {
  local file="$1"
  local tag="$2"
  local exp_name="$3"
  local out_dir="$4"
  local lambda_mask="$5"
  local semantic_late_start="$6"
  local semantic_start_iter="$7"
  local use_semantic_detach="$8"
  shift 8

  {
    echo "tag: $tag"
    echo "exp_name: $exp_name"
    echo "output_dir: $out_dir"
    echo "lambda_mask: $lambda_mask"
    echo "lambda_semantic: 0.10"
    echo "use_mask_aux: True"
    echo "use_semantic_aux: True"
    echo "semantic_late_start: $semantic_late_start"
    echo "semantic_start_iter: $semantic_start_iter"
    echo "use_semantic_warmup: False"
    echo "use_semantic_detach: $use_semantic_detach"
    echo ""
    echo "train_command:"
    printf '  %q' "$@"
    echo ""
  } > "$file"
}

write_test_template() {
  local file="$1"
  local exp_name="$2"
  local best_ckpt="$3"
  local use_semantic_detach="$4"

  cat > "$file" <<EOF
# Optional test command for the selected validation snapshot.
# Do not use test scores for checkpoint selection.
python test_card_spot.py \\
  --cfg "$BASE_CFG" \\
  --snapshot "$best_ckpt" \\
  --gpu "$PYTORCH_GPU" \\
  exp_dir "$EXP_ROOT" \\
  exp_name "$exp_name" \\
  model.enable_aux_mask True \\
  train.use_semantic_aux True \\
  train.lambda_mask 0.02 \\
  train.lambda_semantic 0.10 \\
  train.use_semantic_warmup False \\
  train.use_semantic_detach "$use_semantic_detach"

python evaluate_spot.py \\
  --results_dir "$EXP_ROOT/$exp_name/test_output/captions" \\
  --anno "$ANNO"
EOF
}

run_one() {
  local tag="$1"
  local exp_name="$2"
  local semantic_late_start="$3"
  local semantic_start_iter="$4"
  local use_semantic_detach="$5"

  prepare_output_dirs "$exp_name"

  local out_dir="$EXP_ROOT/$exp_name"
  local result_dir="$RESULTS_ROOT/$exp_name"
  local eval_dir="$out_dir/eval_sents"
  local select_dir="$result_dir/snapshot_selection"

  local -a train_cmd=(
    python train_card_spot.py
    --cfg "$BASE_CFG"
    --exp_name "$exp_name"
    --output_dir "$out_dir"
    --use_mask_aux
    --use_semantic_aux
    --lambda_mask 0.02
    --lambda_semantic 0.10
  )

  if [ "$semantic_late_start" = "True" ]; then
    train_cmd+=(--semantic_late_start --semantic_start_iter "$semantic_start_iter")
  fi
  if [ "$use_semantic_detach" = "True" ]; then
    train_cmd+=(--use_semantic_detach)
  fi

  train_cmd+=(
    gpu_id "[$PYTORCH_GPU]"
    train.max_iter "$MAX_ITER"
    train.snapshot_interval "$SNAPSHOT_INTERVAL"
    train.use_mask_warmup False
    train.use_semantic_warmup False
    train.semantic_late_start "$semantic_late_start"
    train.semantic_start_iter "$semantic_start_iter"
    train.use_semantic_detach "$use_semantic_detach"
  )

  write_args_file \
    "$out_dir/args.txt" \
    "$tag" \
    "$exp_name" \
    "$out_dir" \
    "0.02" \
    "$semantic_late_start" \
    "$semantic_start_iter" \
    "$use_semantic_detach" \
    "${train_cmd[@]}"

  echo "========== [$tag] train $exp_name =========="
  "${train_cmd[@]}" 2>&1 | tee "$out_dir/train.log"

  echo "========== [$tag] eval validation snapshots =========="
  python evaluate_spot.py \
    --results_dir "$eval_dir" \
    --anno "$ANNO" \
    2>&1 | tee "$result_dir/eval.log"
  cp -f "$eval_dir/eval_results.txt" "$result_dir/eval_results.txt"

  echo "========== [$tag] select best snapshot from validation =========="
  python scripts/select_best_snapshot_from_eval_txt.py \
    --input "$result_dir/eval_results.txt" \
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

  write_test_template "$result_dir/test_command_template.txt" "$exp_name" "$best_ckpt" "$use_semantic_detach"
  echo "$tag,$exp_name,$best_ckpt,$select_dir/best_checkpoint.md,$result_dir/test_command_template.txt" >> "$RESULTS_ROOT/three_semantic_experiments_summary.csv"
}

if [ "$#" -eq 0 ]; then
  TARGETS=(E1 E2 E3)
else
  TARGETS=("$@")
fi

mkdir -p "$RESULTS_ROOT"
echo "tag,exp_name,best_ckpt,selection_md,test_command_template" > "$RESULTS_ROOT/three_semantic_experiments_summary.csv"

for target in "${TARGETS[@]}"; do
  case "$target" in
    E1|latestart4000)
      run_one E1 lmask002_lsem01_latestart4000 True 4000 False
      ;;
    E2|latestart6000)
      run_one E2 lmask002_lsem01_latestart6000 True 6000 False
      ;;
    E3|semantic_detach)
      run_one E3 lmask002_lsem01_semantic_detach False 5000 True
      ;;
    *)
      echo "Unknown experiment target: $target. Valid targets: E1 E2 E3" >&2
      exit 1
      ;;
  esac
done

echo "Finished requested experiments. Summary: $RESULTS_ROOT/three_semantic_experiments_summary.csv"
