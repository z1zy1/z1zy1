# CARD 实验复现指南

## 1. 项目与实验范围

本项目研究双时相遥感图像的变化描述。当前主干是 CARD：前后时相视觉特征先形成公共/差异表征，再由 DynamicSpeaker Transformer 解码描述。

实际支持 LEVIR-CC、LEVIR-MCI 和 SECOND-CC，以及 CARD baseline、mask auxiliary loss、semantic auxiliary loss、partial detach、mask feature reweight、semantic cross-attention、semantic hard gate、内容词加权 CE 和 speaker-only 保守微调。

utils.utils.RewardCriterion 只是遗留工具，当前训练入口没有 SCST/reward 训练循环。AMP、DDP 和完整 optimizer/scheduler resume 也未接入，不能写成已实现能力。

真实调用链：

~~~text
训练命令
  -> 默认配置与 YAML 合并
  -> CLI/opts 覆盖、别名同步、合法性检查
  -> 数据集与 DataLoader
  -> CARD + DynamicSpeaker
  -> caption/CARD/可选辅助损失
  -> optimizer + StepLR
  -> train loop
  -> validation generation 与指标
  -> checkpoint 和 val_metrics.csv
  -> validation-only checkpoint selector
  -> manifest 锁定
  -> 一次 locked Test
  -> 结果汇总
~~~

Test 指标不得参与 checkpoint 选择。7.5/7.6 流程对此有额外 manifest 审计。

## 2. 主要代码目录

- configs/config_transformer.py：默认值、YAML/CLI 合并及未知键拒绝。
- configs/dynamic、configs/levir_mci、configs/second_cc：已有实验配置。
- datasets/datasets.py：数据集注册映射与公共 DataLoader 构建参数。
- datasets/rcc_dataset_transformer_levir.py：三个当前数据集的加载、预检与 collate。
- models/CARD.py：CARD 主干及 mask/semantic/fusion/gate/reweight。
- models/transformer_decoder.py：DynamicSpeaker 和 caption loss。
- train_card_spot.py：保留的训练权威入口。
- test_card_spot.py：保留的 val/test 推理权威入口。
- utils/experiment_tracking.py：resolved config、哈希和复现实验工件。
- utils/experiment_runtime.py：结构化指标、环境/Git、状态和异常工件。
- utils/metrics.py：标准指标名与 MetricTracker。
- utils/checkpointing.py：旧 checkpoint 字段兼容视图。
- utils/config_validation.py：resolved config 合法性检查。
- utils/seed.py：Python、NumPy、PyTorch 随机种子。
- scripts/select_best_checkpoint.py：论文与 7.5/7.6 已验证的选择器。
- scripts/build_7_6_locked_manifest.py：7.6 五个锁和完整工件审计。
- tools：新增统一入口；旧入口没有删除。
- tests：workflow、manifest、selector、摘要和公共基础设施测试。

## 3. 环境

远程完整实验使用既有 Conda 环境：

~~~bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate card
cd /root/autodl-tmp/z1zy1
~~~

依赖以 requirements.txt 和远程实际环境为准。新训练生成 environment.txt，记录系统、Python、pip freeze、PyTorch、torchvision、CUDA、cuDNN 和 GPU，不应根据本文猜测固定版本。

本地 Windows Python 没有 PyTorch，只用于语法、单元测试、工具 dry-run 和 shell dry-run。完整训练和模型 smoke 在远程 GPU 环境执行。

## 4. 数据集

LEVIR-CC：

~~~text
Levir-CC/
├── features/
├── images/
├── levir_cc_captions_reformat.json
├── transformer_levir_vocab.json
├── transformer_levir_labels.h5
└── splits.json
~~~

LEVIR-MCI：

~~~text
LEVIR-MCI-dataset/
├── features/
├── images/{train,val,test}/{A,B,label}/
├── LevirCCcaptions.json
├── levir_mci_captions_reformat.json
├── transformer_levir_mci_vocab.json
├── transformer_levir_mci_labels.h5
└── splits.json
~~~

其当前 mask 配置是 mask_type=multiclass、num_mask_classes=3。

SECOND-CC：

~~~text
SECOND-CC-AUG/
├── features/
├── rgb/A/
├── rgb/B/
├── sem/A/
├── sem/B/
├── SECOND-CC-AUG.json
├── second_cc_aug_captions_reformat.json
├── transformer_second_cc_aug_vocab.json
├── transformer_second_cc_aug_labels.h5
└── splits.json
~~~

SECOND-CC 语义图按 7 个原始类读取。Dataset 会预检 .npy 特征；缺失错误给出 extract_change_dataset_features.py 命令。不要绕过 splits.json 或重新随机划分。

## 5. 配置系统

加载顺序：

~~~text
config_transformer.py 默认值
  -> --cfg/--config YAML
  -> argparse 显式参数
  -> 末尾 KEY VALUE opts
  -> WCSG 别名同步
  -> train step/save/eval 别名归一化
  -> validate_resolved_config
~~~

未知 YAML/opts 键由原合并器拒绝。关键项包括：

- 实验：exp_dir、exp_name、experiment_group、description、tags、gpu_id；
- 数据：data.dataset、各路径、三段 batch size、num_workers、seed_workers；
- 模型：encoder/decoder 维度、semantic_input_mode、mask/semantic 类数；
- 损失：lambda_mask、lambda_semantic、loss type、warmup、late start；
- 训练：max_iter、finetune_steps、optimizer、StepLR、clip 和保存间隔；
- 复现：train.seed、train.deterministic、train.benchmark；
- 选择：train.selection_strategy。

deterministic 和 benchmark 默认均为 None，即不改变遗留 cuDNN 策略。data.seed_workers
默认 False；打开后才给 DataLoader 注入 worker_init_fn，因此旧实验的采样与 worker
随机行为不变。

常用默认值以 configs/config_transformer.py 为唯一权威；下表用于快速核对：

| 配置 | 类型 | 默认值 | 作用 |
|---|---|---:|---|
| exp_dir | str | ./experiments | 旧版实验根目录 |
| exp_name | str | 空 | 实验目录名，训练时必填 |
| experiment_group | str | 空 | 多 seed 汇总分组，可选 |
| description / tags | str / list | 空 | 不参与计算的实验元信息 |
| gpu_id | list[int] | [0] | 遗留入口使用的 CUDA device |
| data.dataset | str | rcc_dataset | 数据集注册名或项目别名 |
| data.num_workers | int | 8 | DataLoader worker 数 |
| data.seed_workers | bool | False | 是否显式初始化 worker 随机种子 |
| data.train/val/test.batch_size | int | 128 / 64 / 1 | 三阶段 batch size |
| data.*.max_samples | int/null | null | smoke/调试样本上限 |
| model.semantic_input_mode | str | none | none/aux/early_fusion/cross_attention/hard_gate/weak_coupled |
| model.enable_aux_mask | bool | False | mask 分支开关 |
| train.max_iter | int | 10000 | 训练总 iteration |
| train.seed | int | 1111 | Python/NumPy/PyTorch seed |
| train.log_interval | int | 50 | 结构化训练日志间隔 |
| train.snapshot_interval | int | 1000 | 保存并验证的 iteration 间隔 |
| train.optim.type | str | sgdmom | 遗留 optimizer 类型 |
| train.optim.lr | float | 0.01 | 学习率 |
| train.optim.weight_decay | float | 5e-4 | 权重衰减 |
| train.optim.step_size / gamma | int / float | 15 / 0.1 | StepLR 参数 |
| train.grad_clip | float | 1.0 | 梯度裁剪；-1 关闭 |
| train.lambda_mask | float | 0.0 | mask loss 目标权重 |
| train.lambda_semantic | float | 0.05 | semantic loss 目标权重，分支关闭时不生效 |
| train.semantic_detach_ratio | float | 0.5 | partial detach 比例 |
| train.init_checkpoint | str | 空 | 模型权重初始化来源，不是完整 resume |
| train.selection_strategy | str | spice_constrained_balanced | 原验证选择规则 |
| train.deterministic / benchmark | bool/null | null / null | 可选 cuDNN 策略 |

模型维度、词表长度、mask/semantic 类数、warmup、relation auxiliary 和内容词权重
的全部实际值都保存在 config_resolved.yaml；不要只依赖表中默认值判断某个 run。

每次训练保留旧工件并新增：

~~~text
cfg.json
resolved_config.json
resolved_config.yaml
config_hash.txt
args.txt
config_original.yaml
config_resolved.yaml
command.txt
environment.txt
git_info.txt
~~~

测试阶段写 command_val.txt 或 command_test.txt，不覆盖训练 command.txt。

## 6. 模型与损失

总损失保持原实现：

~~~text
main_loss =
    caption_loss
  + 0.001 * CARD_common_constraint
  + 0.001 * CARD_difference_constraint
  + effective_lambda_mask * mask_loss
  + lambda_obj(t) * object_loss
  + lambda_act(t) * action_loss
  + lambda_rel(t) * relation_loss

total_loss = main_loss + effective_lambda_semantic * semantic_loss
~~~

decoder-only 微调冻结 CARD，辅助损失只监控，反向传播使用含 speaker 路径的 main_loss。partial detach 的梯度隔离仍由原训练代码控制。

实现状态：

- CARD baseline、mask head、semantic tag head、dense semantic head、cross-attention：已实现并可用；
- hard gate：已实现但属实验性分支；
- weak_coupled：不是额外网络层，实际由 semantic auxiliary、partial detach 等组合定义；
- confidence filtering：已实现，只在相应 mask 配置生效；
- SCST、AMP、DDP、完整 resume：未实现。

## 7. 输出与日志

为兼容历史脚本，默认仍是 experiments/{exp_name}：

~~~text
experiments/{exp_name}/
├── snapshots/
├── eval_sents/
├── train.log
├── val.log
├── test.log
├── train_log.txt
├── eval_log.txt
├── train_log.jsonl
├── val_metrics.csv
├── metrics.jsonl
├── metrics.csv
├── run_summary.json
├── error.log
├── model_summary.txt
├── config_original.yaml
├── config_resolved.yaml
├── command.txt
├── environment.txt
└── git_info.txt
~~~

新 tools/train.py 默认拒绝复用非空目录，并把相对输出路径解析到项目根目录。
tools/evaluate.py 和 tools/test.py 默认拒绝覆盖已有预测文件；只有显式
--allow-existing 才允许覆盖。旧入口为兼容保留原行为；论文脚本另有覆盖保护。
推荐唯一 run name，例如 20260726_193000_seed1111。日志字段见
docs/LOG_FORMAT.md。

## 8. 训练

统一入口：

~~~bash
python tools/train.py \
  --config configs/levir_mci/wcsg_card_final.yaml \
  --output-root experiments \
  --run-name levir_mci_wcsg_20260726_193000_seed1111 \
  --seed 1111
~~~

旧入口继续可用：

~~~bash
python train_card_spot.py \
  --cfg configs/levir_mci/wcsg_card_final.yaml \
  --dataset levir_mci \
  --data_root /path/to/LEVIR-MCI-dataset
~~~

模块差异应写 YAML 或末尾 opts，而不是复制训练代码：

~~~bash
python tools/train.py \
  --config configs/levir_mci/wcsg_card_final.yaml \
  --run-name mci_no_reweight_seed1111 \
  train.use_feature_reweight False train.seed 1111
~~~

已有 baseline、mask、semantic、partial detach、reweight、SECOND cross-attention/hard-gate 脚本位于 scripts/train_*.sh。7.6 继续使用 scripts/run_7_6_followup_all.sh。

典型既有配置可直接使用：

~~~bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_levir_mci_card_baseline.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/train_levir_mci_card_mask_loss.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/train_levir_mci_card_semantic_loss.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/train_levir_mci_card_mask_semantic_pd05_reweight.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/train_second_cc_card_semantic_crossattn.sh
~~~

使用统一入口跑不同 seed：

~~~bash
for seed in 1111 2222 3333; do
  python tools/train.py \
    --config configs/levir_mci/wcsg_card_final.yaml \
    --run-name "levir_mci_wcsg_seed${seed}" \
    --seed "${seed}"
done
~~~

当前没有完整 resume。若只需要从已有 CARD/speaker 权重做新的初始化，可显式传：

~~~bash
python tools/train.py \
  --config configs/levir_mci/wcsg_card_final.yaml \
  --run-name levir_mci_model_init_seed1111 \
  --init_checkpoint /path/source_checkpoint.pt
~~~

这不会恢复 optimizer、scheduler、epoch 或随机状态，必须视为一个新 run。

## 9. 验证、选择和 Test

验证预测与现有 caption 指标：

~~~bash
python tools/evaluate.py \
  --config configs/levir_mci/wcsg_card_final.yaml \
  --checkpoint /path/checkpoint.pt \
  --result-json /path/run/val_predictions.json \
  --annotation /path/LEVIR-MCI-dataset/levir_mci_captions_reformat.json \
  --run-dir /path/run
~~~

validation-only 选择：

~~~bash
python tools/select_checkpoint.py \
  --run-dir /path/run \
  --primary-metric CIDEr \
  --secondary-metric SPICE \
  --mode max \
  --top-k 5
~~~

加权选择可用 --weights CIDEr=0.6,SPICE=0.4。通用规则在主、次指标相同
时优先较晚的 validation step，并把 tie-breaking 写入 checkpoint_selection.json。
不传 primary/weights 时委托原 scripts/select_best_checkpoint.py，默认
spice_constrained_balanced 不变。

锁定后 Test：

~~~bash
python tools/test.py \
  --config configs/levir_mci/wcsg_card_final.yaml \
  --checkpoint /path/locked_checkpoint.pt \
  --result-json /path/run/test_predictions.json \
  --annotation /path/LEVIR-MCI-dataset/levir_mci_captions_reformat.json \
  --run-dir /path/run
~~~

7.6 必须先审计 7_6_locked_manifest.json，再运行一次 run_7_6_followup_test_locked.sh，不得用通用命令绕过锁。

## 10. checkpoint

当前训练入口实际保存：

~~~python
{
    "change_detector_state": ...,
    "speaker_state": ...,
    "model_cfg": ...
}
~~~

因此不支持 optimizer/scheduler/scaler/random-state 的完整恢复。utils/checkpointing.py 能识别旧字段和标准别名、处理 module. 前缀，但不会隐瞒加载不匹配。decoder-only 初始化仍要求架构完全一致。

best_cider、best_spice、best_balanced 是训练时便利 alias；论文结论以 validation selector 和 manifest 为准。
当前没有 latest.pth；迭代 checkpoint 是可加载的模型权重快照，但不是可完整恢复训练
状态的 checkpoint。文档中的 “best” 指 validation 规则选定或训练期便利 alias，
不能用 Test 指标重选。

## 11. 汇总

~~~bash
python tools/summarize_experiments.py \
  --input-root experiments \
  --output-csv experiments/paper_required_experiments_summary.csv
~~~

汇总器递归读取配置、模型名、状态、选择时的 validation 指标与最终 Test
指标，保留失败/不完整实验；不同 seed 的重复实验增加 mean 与 sample std 行，
相同 seed 的重复 run 不会被误当成多 seed。它不替代 7.6 locked summary 的
provenance 审计。

## 12. smoke test

本地命令检查：

~~~bash
python tools/smoke_test.py \
  --config configs/levir_mci/wcsg_card_final.yaml \
  --output-dir C:/tmp/card_smoke \
  --dry-run
~~~

远程一条样本真实 smoke：

~~~bash
python tools/smoke_test.py \
  --config configs/levir_mci/wcsg_card_final.yaml \
  --output-dir /tmp/card_smoke_mci
~~~

它执行一批训练和 validation、保存 checkpoint、从验证入口重新加载，并校验结构化工件。输出目录必须为空。

## 13. 可复现性

- 固定 seed，多 seed 至少报告 3 次 mean ± sample std；
- 保留原始/解析配置、命令、环境和 Git dirty 状态；
- 不默认更改 deterministic/benchmark；
- validation 选 checkpoint，Test 只在锁定后执行；
- 历史 CSV 必须同时核对 checkpoint step 与 config hash。

## 14. 常见问题

1. 数据路径不存在：核对 root、vocab、H5、splits 和 feature。
2. .npy 缺失：按 Dataset preflight 生成特征。
3. CUDA OOM：用新 run 调整 batch size，不覆盖原实验。
4. checkpoint 无法加载：确认 CARD/speaker state 和 vocab shape。
5. missing/unexpected keys：不得静默忽略；decoder-only 必须零 mismatch。
6. 没有日志：检查输出权限和 error.log。
7. 指标全零：检查 prediction image id 与 annotation。
8. CIDEr/SPICE 失败：检查评价依赖；异常不会伪造成零。
9. warmup：查看 metrics.jsonl 的 effective_lambda 字段（extras 内）。
10. 恢复后学习率异常：当前没有完整 optimizer/scheduler resume。
11. validation 最优而 Test 较差：不能回看 Test 重选。
12. 输出冲突：换 run name，不要滥用 allow-existing。
13. JSON/CSV 损坏：不要并行写同一 run。
14. DataLoader 卡住：smoke 中先设 num_workers=0。
15. 中断：保留 checkpoint 和 run_summary；重新启动时清楚记录初始化来源。
