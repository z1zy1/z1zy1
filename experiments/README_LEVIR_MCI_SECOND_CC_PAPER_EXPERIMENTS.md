# LEVIR-MCI / SECOND-CC 论文实验流程

本项目保留现有 CARD 预提取 `.npy` 特征管线。新增的数据集自适应代码会按如下方式映射数据集：

- `levir_mci` 对应 `LEVIR-MCI-dataset`，目录结构为 `images/<split>/A,B,label,label_rgb`。
- `second_cc` 对应 `SECOND-CC-AUG`，目录结构为 `<split>/rgb/A,B`、`<split>/sem/A,B`、`pseudo_masks/<split>`，并读取 JSON 中的 `changeflag` 字段。

运行前请先设置数据集根目录：

```bash
export LEVIR_MCI_ROOT=/path/to/LEVIR-MCI-dataset
export SECOND_CC_ROOT=/path/to/SECOND-CC-AUG
```

如果缺少 `features/` 目录，需要先使用现有特征提取流程生成预提取特征，例如运行 `scripts/extract_change_dataset_features.py`。这些训练脚本会继续使用预提取特征，不会切换为端到端原图训练。

## 必需实验流程

运行全部必需实验：

```bash
bash scripts/run_all_required_paper_experiments.sh
```

只运行某个数据集或某个实验：

```bash
bash scripts/run_all_required_paper_experiments.sh --only_dataset levir_mci
bash scripts/run_all_required_paper_experiments.sh --only_dataset second_cc --only_exp second_cc_ours_weak_coupled_final
```

不重新训练，只检查已有 snapshot：

```bash
bash scripts/run_all_required_paper_experiments.sh --skip_train --only_dataset levir_mci --only_exp levir_mci_ours_weak_coupled_final
```

使用 `--overwrite` 可以重新生成推理、snapshot 选择和测试输出。

每个实验会写入：

- `experiments/<exp_name>/train.log`
- `experiments/<exp_name>/snapshots/`
- `experiments/<exp_name>/eval_snapshots.csv`
- `experiments/<exp_name>/best_snapshot_for_paper.json`
- `experiments/<exp_name>/best_for_paper.pth`
- `experiments/<exp_name>/test_paper_best_result.json`
- 启用 change/no-change 分组评估时，还会写入 `experiments/<exp_name>/test_group_summary.csv`

## 实验脚本

LEVIR-MCI：

- `scripts/train_levir_mci_card_baseline.sh`
- `scripts/train_levir_mci_card_mask_loss.sh`
- `scripts/train_levir_mci_card_semantic_loss.sh`
- `scripts/train_levir_mci_card_mask_semantic.sh`
- `scripts/train_levir_mci_card_mask_semantic_pd05.sh`
- `scripts/train_levir_mci_card_mask_semantic_pd05_noreweight.sh`
- `scripts/train_levir_mci_card_mask_semantic_pd05_reweight.sh`
- `scripts/train_levir_mci_ours_weak_coupled_final.sh`

SECOND-CC：

- `scripts/train_second_cc_card_rgb_baseline.sh`
- `scripts/train_second_cc_card_semantic_aux.sh`
- `scripts/train_second_cc_card_semantic_crossattn.sh`
- `scripts/train_second_cc_card_semantic_hardgate.sh`
- `scripts/train_second_cc_ours_weak_coupled_final.sh`
- `scripts/run_second_cc_mmodalcc_comparison.sh`

Hard gate 和 feature reweight 实验会被视为负向消融。除非给 `scripts/select_best_snapshot_for_paper.py` 显式传入 `--allow_negative_ablation`，否则它们不会参与自动最终模型选择。

## 汇总与可视化

汇总必需实验结果：

```bash
python scripts/summarize_paper_required_experiments.py --experiments_root experiments
```

生成定性案例可视化：

```bash
python scripts/visualize_paper_cases.py \
  --dataset levir_mci \
  --data_root "$LEVIR_MCI_ROOT" \
  --caption_json "$LEVIR_MCI_ROOT/LevirCCcaptions.json" \
  --result_json experiments/levir_mci_ours_weak_coupled_final/test_output/captions/paper_best/sc_results.json
```

对于 SECOND-CC，请使用 `--dataset second_cc`，并将 `--caption_json` 设置为 `"$SECOND_CC_ROOT/SECOND-CC-AUG.json"`。

## MModalCC

MModalCC 结果不会被伪造。汇总脚本只会在用户提供 `external_results/second_cc_mmodalcc_results.csv` 时读取外部结果；如果该文件不存在，对应行会标记为 `source=N/A` 和 `status=missing_external_results`。
