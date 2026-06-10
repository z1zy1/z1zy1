#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_DIR="${EXP_DIR:-./experiments}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_TEST="${RUN_TEST:-1}"

EXP_NAMES=()

run_one() {
  local exp_name="$1"
  local lmask="$2"
  local lsem="$3"
  local use_reweight="$4"
  local reweight_alpha="$5"
  local exp_path="$EXP_DIR/$exp_name"

  EXP_NAMES+=("$exp_name")
  mkdir -p "$exp_path"

  echo "========== ablation train/eval/test: $exp_name =========="
  if [ "$RUN_TRAIN" = "1" ]; then
    EXP_NAME="$exp_name" \
    LMASK="$lmask" \
    LSEM="$lsem" \
    USE_FEATURE_REWEIGHT="$use_reweight" \
    REWEIGHT_ALPHA="$reweight_alpha" \
    EXP_DIR="$EXP_DIR" \
    bash scripts/train_sgc_card_lm003_ls005_pd05_rw02_warmup.sh
  fi

  if [ "$RUN_EVAL" = "1" ]; then
    EXP_NAME="$exp_name" \
    LMASK="$lmask" \
    LSEM="$lsem" \
    USE_FEATURE_REWEIGHT="$use_reweight" \
    REWEIGHT_ALPHA="$reweight_alpha" \
    EXP_DIR="$EXP_DIR" \
    bash scripts/eval_all_snapshots_sgc_card.sh

    python scripts/select_best_snapshot_sgc_card.py \
      --csv "$exp_path/eval_snapshots.csv" \
      --output_json "$exp_path/best_snapshot.json" \
      --copy_path "$exp_path/best_balanced.pth"
  fi

  if [ "$RUN_TEST" = "1" ]; then
    EXP_NAME="$exp_name" \
    LMASK="$lmask" \
    LSEM="$lsem" \
    USE_FEATURE_REWEIGHT="$use_reweight" \
    REWEIGHT_ALPHA="$reweight_alpha" \
    EXP_DIR="$EXP_DIR" \
    bash scripts/test_best_sgc_card.sh
  fi
}

run_one lmask003_lsem005_partialdetach05 0.003 0.005 0 0.2
run_one lmask002_lsem005_partialdetach05 0.002 0.005 0 0.2
run_one lmask003_lsem0075_partialdetach05 0.003 0.0075 0 0.2
run_one lmask003_lsem005_partialdetach05_reweight02 0.003 0.005 1 0.2
run_one lmask005_lsem003_partialdetach05_reweight02 0.005 0.003 1 0.2

python scripts/summarize_sgc_card_ablation.py \
  --exp_dir "$EXP_DIR" \
  --exp_names "${EXP_NAMES[@]}" \
  --output "$EXP_DIR/sgc_card_ablation_summary.csv"

echo "Finished ablation. Summary: $EXP_DIR/sgc_card_ablation_summary.csv"
