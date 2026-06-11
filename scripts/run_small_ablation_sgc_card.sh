#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_ROOT="${EXP_ROOT:-./experiments}"
OVERWRITE=0
TARGETS=()

usage() {
  echo "Usage: bash scripts/run_small_ablation_sgc_card.sh [--overwrite] [exp_name ...]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      TARGETS+=("$1")
      shift
      ;;
  esac
done

if [ "${#TARGETS[@]}" -eq 0 ]; then
  TARGETS=(
    lmask003_lsem005_pd05_warmup_no_reweight
    lmask003_lsem005_pd05_rw01_warmup
    lmask003_lsem003_pd05_rw01_warmup
  )
fi

train_script_for() {
  case "$1" in
    lmask003_lsem005_pd05_warmup_no_reweight)
      echo "scripts/train_lmask003_lsem005_pd05_warmup_no_reweight.sh"
      ;;
    lmask003_lsem005_pd05_rw01_warmup)
      echo "scripts/train_lmask003_lsem005_pd05_rw01_warmup.sh"
      ;;
    lmask003_lsem003_pd05_rw01_warmup)
      echo "scripts/train_lmask003_lsem003_pd05_rw01_warmup.sh"
      ;;
    *)
      echo "Unknown small ablation experiment: $1" >&2
      return 1
      ;;
  esac
}

has_snapshots() {
  local exp_path="$1"
  [ -d "$exp_path/snapshots" ] && find "$exp_path/snapshots" -type f \( -name '*.pt' -o -name '*.pth' \) | grep -q .
}

run_step() {
  local exp_name="$1"
  local step="$2"
  shift 2
  echo "========== [$exp_name] $step =========="
  if ! "$@"; then
    echo "ERROR: [$exp_name] failed at step: $step" >&2
    exit 1
  fi
}

for exp_name in "${TARGETS[@]}"; do
  exp_path="$EXP_ROOT/$exp_name"
  train_script="$(train_script_for "$exp_name")"

  if [ "$OVERWRITE" = "0" ] && has_snapshots "$exp_path"; then
    echo "========== [$exp_name] train =========="
    echo "Snapshots already exist under $exp_path/snapshots; skipping training. Use --overwrite to train again."
  else
    run_step "$exp_name" train env EXP_DIR="$EXP_ROOT" OVERWRITE="$OVERWRITE" bash "$train_script"
  fi

  if [ "$OVERWRITE" = "1" ] || [ ! -f "$exp_path/eval_snapshots.csv" ]; then
    if [ "$OVERWRITE" = "1" ]; then
      run_step "$exp_name" "eval all snapshots" bash scripts/eval_all_snapshots_sgc_card.sh --exp_dir "$exp_path" --force
    else
      run_step "$exp_name" "eval all snapshots" bash scripts/eval_all_snapshots_sgc_card.sh --exp_dir "$exp_path"
    fi
  else
    echo "========== [$exp_name] eval all snapshots =========="
    echo "Found $exp_path/eval_snapshots.csv; skipping eval. Use --overwrite to recompute."
  fi

  if [ "$OVERWRITE" = "1" ] || [ ! -f "$exp_path/best_snapshot_v2.json" ]; then
    run_step "$exp_name" "select best snapshot v2" python scripts/select_best_snapshot_sgc_card_v2.py \
      --csv "$exp_path/eval_snapshots.csv" \
      --exp_dir "$exp_path"
  else
    echo "========== [$exp_name] select best snapshot v2 =========="
    echo "Found $exp_path/best_snapshot_v2.json; skipping selection. Use --overwrite to recompute."
  fi

  if [ "$OVERWRITE" = "1" ] || [ ! -f "$exp_path/test_top_snapshots_summary.csv" ]; then
    run_step "$exp_name" "test top snapshots" bash scripts/test_top_snapshots_sgc_card.sh --exp_dir "$exp_path"
  else
    echo "========== [$exp_name] test top snapshots =========="
    echo "Found $exp_path/test_top_snapshots_summary.csv; skipping test. Use --overwrite to recompute."
  fi

  if [ "$OVERWRITE" = "1" ] || [ ! -f "$exp_path/snapshot_compare_report.csv" ]; then
    run_step "$exp_name" "compare snapshot results" python scripts/compare_snapshot_results.py --exp_dir "$exp_path"
  else
    echo "========== [$exp_name] compare snapshot results =========="
    echo "Found $exp_path/snapshot_compare_report.csv; skipping compare. Use --overwrite to recompute."
  fi
done

run_step small_ablation summarize python scripts/summarize_small_ablation_sgc_card.py \
  --exp_root "$EXP_ROOT" \
  --exp_names "${TARGETS[@]}"

echo "Finished small ablation."
echo "Summary CSV: $EXP_ROOT/small_ablation_sgc_card_summary.csv"
echo "Summary report: $EXP_ROOT/small_ablation_sgc_card_report.txt"
