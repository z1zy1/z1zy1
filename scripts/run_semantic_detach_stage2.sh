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
RUN_TEST="${RUN_TEST:-1}"
TARGETS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-test)
      RUN_TEST=0
      shift
      ;;
    --run-test)
      RUN_TEST=1
      shift
      ;;
    *)
      TARGETS+=("$1")
      shift
      ;;
  esac
done

if [ "${#TARGETS[@]}" -eq 0 ]; then
  TARGETS=(A B C D)
fi

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
  local lambda_semantic="$6"
  local seed="$7"
  shift 7

  {
    echo "tag: $tag"
    echo "exp_name: $exp_name"
    echo "output_dir: $out_dir"
    echo "seed: $seed"
    echo "lambda_mask: $lambda_mask"
    echo "lambda_semantic: $lambda_semantic"
    echo "use_mask_aux: True"
    echo "use_semantic_aux: True"
    echo "use_semantic_detach: True"
    echo "use_semantic_warmup: False"
    echo "semantic_late_start: False"
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
  local lambda_mask="$4"
  local lambda_semantic="$5"
  local seed="$6"

  cat > "$file" <<EOF
#!/usr/bin/env bash
set -euo pipefail

python test_card_spot.py \\
  --cfg "$BASE_CFG" \\
  --snapshot "$best_ckpt" \\
  --gpu "$PYTORCH_GPU" \\
  exp_dir "$EXP_ROOT" \\
  exp_name "$exp_name" \\
  model.enable_aux_mask True \\
  train.use_semantic_aux True \\
  train.lambda_mask "$lambda_mask" \\
  train.lambda_semantic "$lambda_semantic" \\
  train.use_semantic_warmup False \\
  train.semantic_late_start False \\
  train.use_semantic_detach True \\
  train.seed "$seed"

python evaluate_spot.py \\
  --results_dir "$EXP_ROOT/$exp_name/test_output/captions" \\
  --anno "$ANNO"
EOF
  chmod +x "$file"
}

run_one() {
  local tag="$1"
  local exp_name="$2"
  local lambda_mask="$3"
  local lambda_semantic="$4"
  local seed="$5"

  prepare_output_dirs "$exp_name"

  local out_dir="$EXP_ROOT/$exp_name"
  local result_dir="$RESULTS_ROOT/$exp_name"
  local eval_dir="$out_dir/eval_sents"
  local select_dir="$result_dir/snapshot_selection"
  local test_template="$result_dir/test_command_template.sh"

  local -a train_cmd=(
    python train_card_spot.py
    --cfg "$BASE_CFG"
    --exp_name "$exp_name"
    --output_dir "$out_dir"
    --use_mask_aux
    --use_semantic_aux
    --use_semantic_detach
    --lambda_mask "$lambda_mask"
    --lambda_semantic "$lambda_semantic"
    --seed "$seed"
    gpu_id "[$PYTORCH_GPU]"
    train.max_iter "$MAX_ITER"
    train.snapshot_interval "$SNAPSHOT_INTERVAL"
    train.use_mask_warmup False
    train.use_semantic_warmup False
    train.semantic_late_start False
    train.use_semantic_detach True
  )

  write_args_file "$out_dir/args.txt" "$tag" "$exp_name" "$out_dir" "$lambda_mask" "$lambda_semantic" "$seed" "${train_cmd[@]}"

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

  write_test_template "$test_template" "$exp_name" "$best_ckpt" "$lambda_mask" "$lambda_semantic" "$seed"

  if [ "$RUN_TEST" = "1" ]; then
    echo "========== [$tag] test selected snapshot $best_ckpt =========="
    "$test_template" 2>&1 | tee "$result_dir/test.log"
    cp -f "$EXP_ROOT/$exp_name/test_output/captions/eval_results.txt" "$result_dir/test_results.txt"
  else
    echo "Skipped test execution. Template: $test_template"
  fi

  echo "$tag,$exp_name,$seed,$best_ckpt,$select_dir/best_checkpoint.md,$test_template" >> "$RESULTS_ROOT/semantic_detach_stage2_summary.csv"
}

run_target() {
  case "$1" in
    A|lmask002_lsem0075)
      run_one A lmask002_lsem0075_semantic_detach 0.02 0.075 1111
      ;;
    B|seed1)
      run_one B lmask002_lsem01_semantic_detach_seed1 0.02 0.10 1
      ;;
    C|seed2)
      run_one C lmask002_lsem01_semantic_detach_seed2 0.02 0.10 2
      ;;
    D|seed3)
      run_one D lmask002_lsem01_semantic_detach_seed3 0.02 0.10 3
      ;;
    *)
      echo "Unknown stage2 target: $1. Valid targets: A B C D" >&2
      exit 1
      ;;
  esac
}

maybe_summarize_seeds() {
  local seed1="$RESULTS_ROOT/lmask002_lsem01_semantic_detach_seed1/test_results.txt"
  local seed2="$RESULTS_ROOT/lmask002_lsem01_semantic_detach_seed2/test_results.txt"
  local seed3="$RESULTS_ROOT/lmask002_lsem01_semantic_detach_seed3/test_results.txt"
  if [ -f "$seed1" ] && [ -f "$seed2" ] && [ -f "$seed3" ]; then
    python scripts/summarize_semantic_detach_seeds.py \
      --results_root "$RESULTS_ROOT" \
      --inputs "$seed1" "$seed2" "$seed3" \
      --output_csv "$RESULTS_ROOT/lmask002_lsem01_semantic_detach_seed_summary.csv" \
      --output_md "$RESULTS_ROOT/lmask002_lsem01_semantic_detach_seed_summary.md"
  else
    echo "Seed summary skipped because one or more test_results.txt files are missing."
  fi
}

mkdir -p "$RESULTS_ROOT"
echo "tag,exp_name,seed,best_ckpt,selection_md,test_command_template" > "$RESULTS_ROOT/semantic_detach_stage2_summary.csv"

for target in "${TARGETS[@]}"; do
  run_target "$target"
done

maybe_summarize_seeds

echo "Finished stage2 experiments. Summary: $RESULTS_ROOT/semantic_detach_stage2_summary.csv"
