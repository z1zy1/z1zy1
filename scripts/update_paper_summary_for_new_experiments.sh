#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

EXP_ROOT="${EXP_ROOT:-./experiments}"
SUMMARY_CSV="${SUMMARY_CSV:-$EXP_ROOT/paper_required_experiments_summary.csv}"

EXPERIMENTS=(
  levir_mci_weak_pd08_lm003_ls001_noreweight
  levir_mci_weak_pd08_lm001_ls001_noreweight
  levir_mci_weak_pd05_lm003_ls0005_noreweight
  levir_mci_caption_finetune_from_weak_best
  second_cc_crossattn_pd05_lsem0005
  second_cc_crossattn_pd05_lsem001
  second_cc_crossattn_pd08_lsem0005
  second_cc_crossattn_pd08_lsem001
)

for exp in "${EXPERIMENTS[@]}"; do
  exp_path="$EXP_ROOT/$exp"
  if [ -d "$exp_path" ]; then
    "$PYTHON" scripts/update_paper_required_summary.py --summary_csv "$SUMMARY_CSV" --exp_dir "$exp_path"
  else
    dataset=""
    case "$exp" in
      levir_mci*) dataset=levir_mci ;;
      second_cc*) dataset=second_cc ;;
    esac
    "$PYTHON" scripts/update_paper_required_summary.py --summary_csv "$SUMMARY_CSV" --dataset "$dataset" --exp_name "$exp" --status failed --notes "Experiment directory missing; not run in this workspace."
  fi
done

"$PYTHON" scripts/add_external_result_to_summary.py --summary_csv "$SUMMARY_CSV"

echo "Updated $SUMMARY_CSV"
