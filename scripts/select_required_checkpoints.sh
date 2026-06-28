#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
EXP_ROOT="${EXP_ROOT:-./experiments}"
BASELINE_METRICS="${BASELINE_METRICS:-}"
EXPERIMENTS=(
  levir_mci_card_baseline
  levir_mci_card_mask
  levir_mci_card_semantic
  levir_mci_card_mask_semantic
  levir_mci_card_mask_semantic_pd05
  levir_mci_card_mask_semantic_pd05_noreweight
  levir_mci_card_mask_semantic_pd05_reweight
  levir_mci_wcsg_card_final
  second_cc_card_rgb_baseline
  second_cc_semantic_aux
  second_cc_semantic_crossattn
  second_cc_semantic_hard_gate
  second_cc_wcsg_card_final
)
for exp in "${EXPERIMENTS[@]}"; do
  exp_dir="$EXP_ROOT/$exp"
  if [ ! -d "$exp_dir" ]; then
    echo "Skipping missing experiment directory: $exp_dir"
    continue
  fi
  args=(--exp_dir "$exp_dir" --strategy spice_constrained_balanced)
  if [ -n "$BASELINE_METRICS" ]; then args+=(--baseline_metrics "$BASELINE_METRICS"); fi
  python scripts/select_best_checkpoint.py "${args[@]}"
done