#!/usr/bin/env bash
set -euo pipefail

# Generate ChangeFormer + BIT maps in a temporary directory, fuse them, then
# optionally replace the active pseudo_masks directory while preserving a
# timestamped backup. Never delete old masks before complete generation.
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
DATA_ROOT="${DATA_ROOT:-./Levir-CC}"
WORK_ROOT="${WORK_ROOT:-./experiments/levir_cc_change_ensemble}"
CF_REPO="${CF_REPO:-}"
CF_CHECKPOINT="${CF_CHECKPOINT:-}"
BIT_REPO="${BIT_REPO:-}"
BIT_CHECKPOINT="${BIT_CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
REPLACE_EXISTING=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --replace-existing) REPLACE_EXISTING=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: CF_REPO=... CF_CHECKPOINT=... BIT_REPO=... BIT_CHECKPOINT=... \
  bash scripts/generate_levir_ensemble_masks.sh [--replace-existing]

Creates probability maps, fused masks, confidence maps, and uncertainty maps.
Replacement keeps a timestamped backup of the previous pseudo_masks directory.
EOF
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for value in CF_REPO CF_CHECKPOINT BIT_REPO BIT_CHECKPOINT; do
  if [ -z "${!value}" ]; then echo "Missing required variable: $value" >&2; exit 2; fi
done

TMP_ROOT="${WORK_ROOT}.tmp.$$"
CF_PRED="$TMP_ROOT/changeformer_probs"
BIT_PRED="$TMP_ROOT/bit_probs"
MASK_ROOT="$TMP_ROOT/pseudo_masks"
CONF_ROOT="$TMP_ROOT/pseudo_confidence"
UNC_ROOT="$TMP_ROOT/pseudo_uncertainty"
mkdir -p "$TMP_ROOT"
trap 'if [ -d "$TMP_ROOT" ]; then mv "$TMP_ROOT" "${WORK_ROOT}.failed.$$"; fi' EXIT

python scripts/infer_external_change_detector.py \
  --repo "$CF_REPO" --model ChangeFormerV6 --checkpoint "$CF_CHECKPOINT" \
  --dataset_root "$DATA_ROOT" --output_root "$CF_PRED" \
  --batch_size "$BATCH_SIZE" --device "$DEVICE" --overwrite

python scripts/infer_external_change_detector.py \
  --repo "$BIT_REPO" --model base_transformer_pos_s4_dd8_dedim8 --checkpoint "$BIT_CHECKPOINT" \
  --dataset_root "$DATA_ROOT" --output_root "$BIT_PRED" \
  --batch_size "$BATCH_SIZE" --device "$DEVICE" --overwrite

python scripts/fuse_change_detector_predictions.py \
  --changeformer_root "$CF_PRED" --bit_root "$BIT_PRED" \
  --mask_root "$MASK_ROOT" --confidence_root "$CONF_ROOT" \
  --uncertainty_root "$UNC_ROOT" --overwrite

for split in train val test; do
  expected=$(find "$DATA_ROOT/images/$split/A" -maxdepth 1 -type f | wc -l)
  actual=$(find "$MASK_ROOT/$split" -maxdepth 1 -type f -name '*.png' | wc -l)
  [ "$expected" -eq "$actual" ] || { echo "Count mismatch for $split: expected=$expected actual=$actual" >&2; exit 1; }
done

if [ "$REPLACE_EXISTING" -eq 1 ]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  old_mask="$DATA_ROOT/pseudo_masks"
  old_conf="$DATA_ROOT/pseudo_confidence"
  old_unc="$DATA_ROOT/pseudo_uncertainty"
  [ -d "$old_mask" ] && mv "$old_mask" "${old_mask}.backup_${stamp}"
  [ -d "$old_conf" ] && mv "$old_conf" "${old_conf}.backup_${stamp}"
  [ -d "$old_unc" ] && mv "$old_unc" "${old_unc}.backup_${stamp}"
  mv "$MASK_ROOT" "$old_mask"
  mv "$CONF_ROOT" "$old_conf"
  mv "$UNC_ROOT" "$old_unc"
  echo "Replaced pseudo masks; old directories preserved with suffix _backup_${stamp}."
else
  echo "Generated masks under $MASK_ROOT (active pseudo_masks was not changed)."
fi

trap - EXIT
rm -rf "$TMP_ROOT"
