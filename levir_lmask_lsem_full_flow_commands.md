# LEVIR lmask/lsem experiments: Linux full-flow commands

> Note: your table contains four concrete experiments, A-D. You mentioned five experiments, so this document includes the four listed experiments and leaves an E template comment in the script for the missing fifth setting.

## Experiment Mapping

| Exp | Setting | Command values |
| --- | --- | --- |
| A | lmask005 + lsem0075 | `train.lambda_mask=0.05`, `train.lambda_semantic=0.075` |
| B | lmask003 + lsem0075 | `train.lambda_mask=0.03`, `train.lambda_semantic=0.075` |
| C | lmask002 + lsem01 | `train.lambda_mask=0.02`, `train.lambda_semantic=0.10` |
| D | lmask003 + lsem01 + semantic warmup | `train.lambda_mask=0.03`, `train.lambda_semantic=0.10`, `train.semantic_warmup_steps=1000` |

Important implementation detail: in the current code, `train.semantic_warmup_steps` only warms up the relation auxiliary weights (`lambda_obj`, `lambda_act`, `lambda_rel`) when `train.use_relation_aux=True`. It does not warm up `train.lambda_semantic` itself. The D command below records `semantic_warmup_steps=1000`, but if you want warmup for `lambda_semantic`, the training code needs a small change.

## 0. Environment Setup

Run this once on the Linux machine:

```bash
cd /path/to/CARD

# If the environment does not exist yet:
# conda create -n card python=3.8 -y
conda activate card
python -m pip install -r requirements.txt

# Set this to your Python-3 compatible coco-caption repo.
export COCO_CAPTION_DIR=/path/to/coco-caption_python3
COCO_CAPTION_DIR="${COCO_CAPTION_DIR%/}"
sed -i "s#^COCO_PATH = .*#COCO_PATH = '${COCO_CAPTION_DIR}/' # i.e. /home/user/code/coco-caption#" utils/eval_utils_spot.py

# If LEVIR caption preprocessing outputs are missing, rebuild them.
if [ ! -f ./Levir-CC/splits.json ] || [ ! -f ./Levir-CC/transformer_levir_vocab.json ] || [ ! -f ./Levir-CC/transformer_levir_labels.h5 ] || [ ! -f ./Levir-CC/levir_cc_captions_reformat.json ]; then
  python scripts/preprocess_captions_levir.py
fi

# If ResNet features are missing, extract them. This preserves train/val/test and A/B subdirectories.
if [ ! -d ./Levir-CC/features ]; then
  python scripts/extract_features.py \
    --input_image_dir ./Levir-CC/images \
    --output_dir ./Levir-CC/features \
    --recursive \
    --batch_size 128
fi

# Optional visdom server for training curves.
python -m visdom.server -port 8097 >/tmp/card_visdom_8097.log 2>&1 &
```

## 1. Full Flow Script

Save the following as `run_levir_lmask_lsem_full_flow.sh` on Linux, then run it with `bash run_levir_lmask_lsem_full_flow.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/CARD}"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_GPU="${PYTORCH_GPU:-0}"

BASE_CFG="${BASE_CFG:-configs/dynamic/transformer_levir_cc_aux_mask_conf_semantic.yaml}"
ANNO="${ANNO:-./Levir-CC/levir_cc_captions_reformat.json}"
EXP_DIR="${EXP_DIR:-./experiments}"
LOG_DIR="${LOG_DIR:-./run_logs/lmask_lsem}"
MAX_ITER="${MAX_ITER:-10000}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1000}"
MIN_CKPT="${MIN_CKPT:-1000}"
MAX_CKPT="${MAX_CKPT:-10000}"
STABLE_WINDOW="${STABLE_WINDOW:-1}"
SEMANTIC_WARMUP_STEPS="${SEMANTIC_WARMUP_STEPS:-1000}"

mkdir -p "$LOG_DIR"

COMMON_OPTS=(
  exp_dir "$EXP_DIR"
  gpu_id "[$PYTORCH_GPU]"
  model.enable_aux_mask True
  train.use_mask_conf_filter True
  train.mask_conf_threshold 0.5
  train.use_mask_warmup False
  train.mask_warmup_steps 0
  train.use_semantic_aux True
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

run_one() {
  local tag="$1"
  local exp_name="$2"
  local lambda_mask="$3"
  local lambda_semantic="$4"
  local semantic_warmup_steps="${5:-0}"

  local -a opts=(
    "${COMMON_OPTS[@]}"
    exp_name "$exp_name"
    train.lambda_mask "$lambda_mask"
    train.lambda_semantic "$lambda_semantic"
  )

  if [ "$semantic_warmup_steps" != "0" ]; then
    opts+=(train.semantic_warmup_steps "$semantic_warmup_steps")
  fi

  local out_dir="$EXP_DIR/$exp_name"
  local eval_dir="$out_dir/eval_sents"
  local select_dir="$out_dir/snapshot_selection"
  local test_caption_dir="$out_dir/test_output/captions"

  echo "========== [$tag] train: $exp_name =========="
  python train_card_spot.py \
    --cfg "$BASE_CFG" \
    "${opts[@]}" \
    2>&1 | tee "$LOG_DIR/${exp_name}_train.log"

  echo "========== [$tag] validation evaluation =========="
  rm -f "$eval_dir/eval_results.txt"
  python evaluate_spot.py \
    --results_dir "$eval_dir" \
    --anno "$ANNO" \
    2>&1 | tee "$LOG_DIR/${exp_name}_val_eval.log"

  echo "========== [$tag] select best snapshot =========="
  python scripts/select_best_snapshot_from_eval_txt.py \
    --input "$eval_dir/eval_results.txt" \
    --output_dir "$select_dir" \
    --config_name "$exp_name" \
    --min_ckpt "$MIN_CKPT" \
    --max_ckpt "$MAX_CKPT" \
    --stable_window "$STABLE_WINDOW" \
    --save_all \
    2>&1 | tee "$LOG_DIR/${exp_name}_select_snapshot.log"

  local best_ckpt
  best_ckpt="$(awk -F, 'NR==2 {gsub(/\r/, "", $2); print $2}' "$select_dir/best_checkpoint.csv")"
  if [ -z "$best_ckpt" ]; then
    echo "Failed to read selected checkpoint from $select_dir/best_checkpoint.csv" >&2
    exit 1
  fi
  echo "[$tag] selected checkpoint: $best_ckpt"

  echo "========== [$tag] test with selected snapshot =========="
  python test_card_spot.py \
    --cfg "$BASE_CFG" \
    --snapshot "$best_ckpt" \
    --gpu "$PYTORCH_GPU" \
    "${opts[@]}" \
    2>&1 | tee "$LOG_DIR/${exp_name}_test.log"

  echo "========== [$tag] test evaluation =========="
  rm -f "$test_caption_dir/eval_results.txt"
  python evaluate_spot.py \
    --results_dir "$test_caption_dir" \
    --anno "$ANNO" \
    2>&1 | tee "$LOG_DIR/${exp_name}_test_eval.log"

  echo "$tag,$exp_name,$best_ckpt,$select_dir/best_checkpoint.md,$test_caption_dir/eval_results.txt" >> "$LOG_DIR/summary.csv"
}

if [ "$#" -eq 0 ]; then
  TARGETS=(A B C D)
else
  TARGETS=("$@")
fi

echo "tag,exp_name,best_ckpt,selection_md,test_eval_txt" > "$LOG_DIR/summary.csv"

for target in "${TARGETS[@]}"; do
  case "$target" in
    A)
      run_one A card_levir_A_lmask005_lsem0075 0.05 0.075 0
      ;;
    B)
      run_one B card_levir_B_lmask003_lsem0075 0.03 0.075 0
      ;;
    C)
      run_one C card_levir_C_lmask002_lsem01 0.02 0.10 0
      ;;
    D)
      run_one D card_levir_D_lmask003_lsem01_semwarm 0.03 0.10 "$SEMANTIC_WARMUP_STEPS"
      ;;
    # E)
    #   Fill in the missing fifth experiment from your table, for example:
    #   run_one E card_levir_E_lmaskXXX_lsemYYY 0.XX 0.YYY 0
    #   ;;
    *)
      echo "Unknown experiment target: $target" >&2
      echo "Valid concrete targets in the provided table: A B C D" >&2
      exit 1
      ;;
  esac
done

echo "All requested experiments finished."
echo "Summary: $LOG_DIR/summary.csv"
```

## 2. Run Commands

Run all concrete experiments from the table:

```bash
bash run_levir_lmask_lsem_full_flow.sh
```

Run only selected experiments:

```bash
bash run_levir_lmask_lsem_full_flow.sh A C
```

Use another physical GPU while keeping PyTorch's visible GPU index as 0:

```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_GPU=0 bash run_levir_lmask_lsem_full_flow.sh A
```

Change training length or selection range:

```bash
MAX_ITER=12000 MAX_CKPT=12000 STABLE_WINDOW=3 bash run_levir_lmask_lsem_full_flow.sh
```

## 3. Outputs To Check

For each experiment, the key files are:

```text
experiments/<exp_name>/snapshots/<exp_name>_checkpoint_<iter>.pt
experiments/<exp_name>/eval_sents/eval_results.txt
experiments/<exp_name>/snapshot_selection/best_checkpoint.md
experiments/<exp_name>/snapshot_selection/best_checkpoint.csv
experiments/<exp_name>/snapshot_selection/all_checkpoint_scores.csv
experiments/<exp_name>/test_output/captions/test/sc_results.json
experiments/<exp_name>/test_output/captions/eval_results.txt
run_logs/lmask_lsem/summary.csv
```

