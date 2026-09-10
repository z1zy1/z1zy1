#!/usr/bin/env bash
set -euo pipefail

# Generate ChangeFormer + BIT maps in a temporary directory, fuse them, then
# optionally replace the active pseudo_masks directory. Probability maps use
# 8-bit PNG by default so the complete flow fits on a nearly full volume.
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
DATA_ROOT="${DATA_ROOT:-./Levir-CC}"
WORK_ROOT="${WORK_ROOT:-./experiments/levir_cc_change_ensemble}"
CF_REPO="${CF_REPO:-/root/autodl-tmp/external_models/ChangeFormer}"
CF_CHECKPOINT="${CF_CHECKPOINT:-$CF_REPO/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/best_ckpt.pt}"
BIT_REPO="${BIT_REPO:-/root/autodl-tmp/external_models/BIT_CD}"
BIT_CHECKPOINT="${BIT_CHECKPOINT:-$BIT_REPO/checkpoints/BIT_LEVIR/best_ckpt.pt}"
if [ ! -d "$CF_REPO" ] || [ ! -d "$BIT_REPO" ]; then
  echo "External detector repositories are missing. Set CF_REPO and BIT_REPO." >&2
  exit 2
fi
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DELETE_EXISTING=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --replace-existing) DELETE_EXISTING=1; shift ;;
    --delete-existing) DELETE_EXISTING=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/generate_levir_ensemble_masks.sh [--delete-existing]

Creates new masks, confidence maps, and uncertainty maps using the bundled
ChangeFormer and BIT checkpoints. Existing pseudo-mask directories are deleted
before inference when --delete-existing/--replace-existing is supplied.
Environment variables may override the default repository/checkpoint paths.
EOF
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for value in CF_REPO CF_CHECKPOINT BIT_REPO BIT_CHECKPOINT; do
  if [ ! -e "${!value}" ]; then echo "Missing path for $value: ${!value}" >&2; exit 2; fi
done

python - "$DEVICE" <<'PY'
import importlib.util
import sys
import torch

device = sys.argv[1]
if device.startswith('cuda') and not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable; run this script on a GPU host or set DEVICE=cpu.')
if importlib.util.find_spec('einops') is None:
    raise SystemExit('Missing Python dependency: einops. Install it in the active environment first.')
try:
    import timm
except ImportError:
    raise SystemExit('Missing Python dependency: timm. Install ChangeFormer requirements first.')
if getattr(timm, '__version__', None) != '0.4.12':
    raise SystemExit('Incompatible timm version %s; ChangeFormer with this environment requires timm==0.4.12.' % getattr(timm, '__version__', 'unknown'))
PY

if [ "$DELETE_EXISTING" -eq 1 ]; then
  for old_dir in "$DATA_ROOT/pseudo_masks" "$DATA_ROOT/pseudo_confidence" "$DATA_ROOT/pseudo_uncertainty"; do
    if [ -d "$old_dir" ]; then
      echo "Deleting existing $old_dir"
      find "$old_dir" -depth -delete
    fi
  done
fi

TMP_ROOT="${WORK_ROOT}.tmp.$$"
CF_PRED="$TMP_ROOT/changeformer_probs"
BIT_PRED="$TMP_ROOT/bit_probs"
MASK_ROOT="$TMP_ROOT/pseudo_masks"
CONF_ROOT="$TMP_ROOT/pseudo_confidence"
UNC_ROOT="$TMP_ROOT/pseudo_uncertainty"
mkdir -p "$TMP_ROOT"
cleanup_tmp() {
  if [ -d "$TMP_ROOT" ]; then
    find "$TMP_ROOT" -depth -delete
  fi
}
trap cleanup_tmp EXIT

python scripts/infer_external_change_detector.py \
  --repo "$CF_REPO" --model ChangeFormerV6 --checkpoint "$CF_CHECKPOINT" \
  --dataset_root "$DATA_ROOT" --output_root "$CF_PRED" \
  --batch_size "$BATCH_SIZE" --device "$DEVICE" --output_format png --overwrite

python scripts/infer_external_change_detector.py \
  --repo "$BIT_REPO" --model base_transformer_pos_s4_dd8_dedim8 --checkpoint "$BIT_CHECKPOINT" \
  --dataset_root "$DATA_ROOT" --output_root "$BIT_PRED" \
  --batch_size "$BATCH_SIZE" --device "$DEVICE" --output_format png --overwrite

python scripts/fuse_change_detector_predictions.py \
  --changeformer_root "$CF_PRED" --bit_root "$BIT_PRED" \
  --mask_root "$MASK_ROOT" --confidence_root "$CONF_ROOT" \
  --uncertainty_root "$UNC_ROOT" --overwrite

for split in train val test; do
  expected=$(find "$DATA_ROOT/images/$split/A" -maxdepth 1 -type f | wc -l)
  actual=$(find "$MASK_ROOT/$split" -maxdepth 1 -type f -name '*.png' | wc -l)
  [ "$expected" -eq "$actual" ] || { echo "Count mismatch for $split: expected=$expected actual=$actual" >&2; exit 1; }
done

if [ "$DELETE_EXISTING" -eq 1 ]; then
  mv "$MASK_ROOT" "$DATA_ROOT/pseudo_masks"
  mv "$CONF_ROOT" "$DATA_ROOT/pseudo_confidence"
  mv "$UNC_ROOT" "$DATA_ROOT/pseudo_uncertainty"
  echo "Installed new pseudo masks, confidence, and uncertainty maps."
else
  echo "Generated masks under $MASK_ROOT (active pseudo_masks was not changed)."
fi

trap - EXIT
cleanup_tmp
