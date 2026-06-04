# LEVIR lmask/lsem 实验 Linux 全流程命令

说明：你说要跑“五个实验”，但表格里目前只有 A-D 四个具体实验。因此本文档默认生成 A-D 四个实验的完整流程，并在脚本里保留 E 的占位位置。若第五个实验有具体设置，把 E 的 `run_one` 行补上即可。

## 1. 实验设置对应关系

| 实验 | 设置 | 命令中的关键参数 |
| --- | --- | --- |
| A | lmask005 + lsem0075 | `train.lambda_mask=0.05`, `train.lambda_semantic=0.075` |
| B | lmask003 + lsem0075 | `train.lambda_mask=0.03`, `train.lambda_semantic=0.075` |
| C | lmask002 + lsem01 | `train.lambda_mask=0.02`, `train.lambda_semantic=0.10` |
| D | lmask003 + lsem01 + semantic warmup | `train.lambda_mask=0.03`, `train.lambda_semantic=0.10`, `train.semantic_warmup_steps=1000` |

重要提醒：当前 `train_card_spot.py` 里的 `train.semantic_warmup_steps` 只用于 relation auxiliary 的 `lambda_obj/lambda_act/lambda_rel` warmup；如果 `train.use_relation_aux=False`，它不会让 `train.lambda_semantic` 本身 warmup。下面 D 实验会把该参数写入配置，便于记录实验意图；如果你要真正对 semantic loss 做 warmup，需要先改训练代码。

## 2. Linux 环境准备命令

先在 AutoDL 上进入项目目录并激活环境：

```bash
cd /root/autodl-tmp/z1zy1
conda activate card

python -m pip install -r requirements.txt
python -m pip install pandas
```

如果你还没有配置 COCO caption eval，请先确认 `utils/eval_utils_spot.py` 顶部的 `COCO_PATH` 指向 Python 3 可用的 coco-caption 目录。例如：

```bash
grep -n "COCO_PATH" utils/eval_utils_spot.py
```

如果路径不对，可以手动编辑 `utils/eval_utils_spot.py`，把 `COCO_PATH` 改成你的实际路径。

可选：启动 visdom，用于训练曲线展示。

```bash
python -m visdom.server -port 8097 > /tmp/card_visdom_8097.log 2>&1 &
```

## 3. 生成全流程脚本

在 Linux 项目根目录执行下面命令，生成 `run_levir_lmask_lsem_full_flow.sh`：

```bash
cat > run_levir_lmask_lsem_full_flow.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/z1zy1}"
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
STABLE_WINDOW="${STABLE_WINDOW:-3}"
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

  echo "========== [$tag] 训练: $exp_name =========="
  python train_card_spot.py \
    --cfg "$BASE_CFG" \
    "${opts[@]}" \
    2>&1 | tee "$LOG_DIR/${exp_name}_train.log"

  echo "========== [$tag] 验证集评估 =========="
  rm -f "$eval_dir/eval_results.txt"
  python evaluate_spot.py \
    --results_dir "$eval_dir" \
    --anno "$ANNO" \
    2>&1 | tee "$LOG_DIR/${exp_name}_val_eval.log"

  echo "========== [$tag] 选择最佳 snapshot =========="
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
    echo "无法从 $select_dir/best_checkpoint.csv 读取 selected_checkpoint" >&2
    exit 1
  fi
  echo "[$tag] selected checkpoint: $best_ckpt"

  echo "========== [$tag] 使用最佳 snapshot 跑 test =========="
  python test_card_spot.py \
    --cfg "$BASE_CFG" \
    --snapshot "$best_ckpt" \
    --gpu "$PYTORCH_GPU" \
    "${opts[@]}" \
    2>&1 | tee "$LOG_DIR/${exp_name}_test.log"

  echo "========== [$tag] test 结果评估 =========="
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
    #   你表格里的第五个实验尚未给出。补充后可写成：
    #   run_one E card_levir_E_lmaskXXX_lsemYYY 0.XX 0.YYY 0
    #   ;;
    *)
      echo "未知实验编号: $target" >&2
      echo "当前表格中可运行的实验编号: A B C D" >&2
      exit 1
      ;;
  esac
done

echo "全部实验完成。"
echo "汇总文件: $LOG_DIR/summary.csv"
EOF

chmod +x run_levir_lmask_lsem_full_flow.sh
```

## 4. 运行命令

运行表格里的全部四个实验：

```bash
bash run_levir_lmask_lsem_full_flow.sh
```

只运行某几个实验：

```bash
bash run_levir_lmask_lsem_full_flow.sh A C
```

指定物理 GPU 1 运行，脚本内部仍使用 PyTorch 可见 GPU 0：

```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_GPU=0 bash run_levir_lmask_lsem_full_flow.sh A
```

把 snapshot 选择稳定窗口改成 1：

```bash
STABLE_WINDOW=1 bash run_levir_lmask_lsem_full_flow.sh
```

训练到 12000 iter，并在 1000-12000 范围内选 snapshot：

```bash
MAX_ITER=12000 MAX_CKPT=12000 STABLE_WINDOW=3 bash run_levir_lmask_lsem_full_flow.sh
```

## 5. 每个实验实际执行的流程

脚本对每个实验都会按顺序执行：

```bash
# 1. 训练，并在每 1000 iter 保存 snapshot，同时生成验证集 captions
python train_card_spot.py --cfg configs/dynamic/transformer_levir_cc_aux_mask_conf_semantic.yaml ...

# 2. 评估 validation/eval_sents，生成 eval_results.txt
python evaluate_spot.py --results_dir experiments/<exp_name>/eval_sents --anno ./Levir-CC/levir_cc_captions_reformat.json

# 3. 用统一 score 选择最佳 checkpoint
python scripts/select_best_snapshot_from_eval_txt.py \
  --input experiments/<exp_name>/eval_sents/eval_results.txt \
  --output_dir experiments/<exp_name>/snapshot_selection \
  --config_name <exp_name> \
  --min_ckpt 1000 \
  --max_ckpt 10000 \
  --stable_window 3 \
  --save_all

# 4. 读取 best_checkpoint.csv 中的 selected_checkpoint，并跑 test
python test_card_spot.py --cfg configs/dynamic/transformer_levir_cc_aux_mask_conf_semantic.yaml --snapshot <best_ckpt> --gpu 0 ...

# 5. 评估 test captions
python evaluate_spot.py --results_dir experiments/<exp_name>/test_output/captions --anno ./Levir-CC/levir_cc_captions_reformat.json
```

## 6. 关键输出文件

每个实验完成后重点看这些文件：

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

## 7. 常见问题

如果出现 `--input: command not found`，说明你在 Linux bash 里误用了 PowerShell 的反引号。Linux 多行命令换行必须用反斜杠 `\`，且反斜杠后面不要有空格。

如果 `evaluate_spot.py` 报 `Result file already exists!`，删除对应目录下旧的 `eval_results.txt` 后重跑。上面的脚本已经自动执行 `rm -f`。

如果 `select_best_snapshot_from_eval_txt.py` 提示 pandas 缺失，执行：

```bash
python -m pip install pandas
```
