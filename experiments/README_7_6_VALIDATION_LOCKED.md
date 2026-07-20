# 7.6：保守微调、三随机种子与稳定窗口锁定流程

本流程只在验证集上选择 checkpoint。候选模型选择完成并通过配置/数据/哈希审计后，才允许对五个锁定 checkpoint 各执行一次 Test；Test 指标不参与选择。

## 实验设置

### LEVIR-CC

- 固定初始化来源：7.5 validation-locked manifest 中指向 `sgc_card_lm003_ls005_pd05_rw02_warmup` 的 checkpoint；禁止从任意 `best*.json` 猜测初始化。
- 训练整个 `DynamicSpeaker`（包括词嵌入、Transformer decoder 和输出层），冻结整个 CARD/change detector。
- content weight：`1.00 / 1.01 / 1.02`。
- learning rate：`5e-7 / 1e-6`。
- steps：`10 / 20`。
- 每 `5` 步保存并验证，共 12 个实验。
- 选择规则：七项非 SPICE 指标均不低于新 CARD 验证 baseline，并要求 SPICE 不低于 baseline，再在验证 Pareto 前沿最大化 SPICE。

### LEVIR-MCI

- 固定 `levir_mci_card_mask_semantic` 的完整方法配置，不再搜索超参数。
- `seed=1111 / 2222 / 3333` 各自从头训练 10000 步，禁止 init checkpoint。
- 每个 seed 只从自己的验证 checkpoint 中选择；CARD baseline 仅作阈值，不能成为候选。
- 每个 seed 锁定后各 Test 一次，最终报告八项指标的 mean 和 sample std（`n-1`）。

### SECOND-CC

- 不新增盲目训练网格，审计已有 `pd07 / pd08 / pd09 / traditional semantic-crossattn` 完整训练。
- 选择规则：同一个 source 实验内连续三个验证 checkpoint，固定间隔 1000 步。
- 三个成员都必须存在、具有八项 finite 验证指标，并且七项保护指标和 SPICE 均不低于新 CARD 验证 baseline。
- 按窗口 mean SPICE、最小保护余量、worst SPICE 等依次排序，只锁定窗口中心 checkpoint。

CARD baseline 不会加入任何改进模型候选池；若没有候选满足约束，流程会失败，而不会回退到 baseline 冒充改进结果。

## 远程运行

先确保远端代码已同步，并已经完成三数据集 CARD baseline 的训练、验证选择、锁定 Test 和汇总。验证 baseline 文件默认为：

```text
experiments/card_levir_cc_baseline/baseline_best_checkpoint.json
experiments/levir_mci_card_baseline/baseline_best_checkpoint.json
experiments/second_cc_card_rgb_baseline/baseline_best_checkpoint.json
experiments/card_baseline_locked_manifest.json
experiments/card_baseline_test_summary.json
```

在服务器设置环境：

```bash
cd /root/autodl-tmp/z1zy1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate card

export PYTHON=/root/miniconda3/envs/card/bin/python
export PROJECT_DIR=/root/autodl-tmp/z1zy1
export EXP_ROOT=/root/autodl-tmp/z1zy1/experiments
export LEVIR_CC_ROOT=/root/autodl-tmp/z1zy1/Levir-CC
export LEVIR_MCI_ROOT=/root/autodl-tmp/z1zy1/LEVIR-MCI-dataset
export SECOND_CC_ROOT=/root/autodl-tmp/z1zy1/SECOND-CC-AUG
```

先展开命令，不启动训练：

```bash
bash scripts/run_7_6_followup_all.sh --dry_run
```

正式训练和验证锁定：

```bash
bash scripts/run_7_6_followup_all.sh
```

该命令在验证锁定后主动停止。主要审计文件为：

```text
experiments/7_6_locked_manifest.json
```

检查五个 lock 后，先 dry-run Test：

```bash
$PYTHON scripts/build_7_6_locked_manifest.py --verify "$EXP_ROOT/7_6_locked_manifest.json"
bash scripts/run_7_6_followup_test_locked.sh --dry_run
```

确认无误后，仅执行一次正式 Test：

```bash
bash scripts/run_7_6_followup_test_locked.sh
```

## 输出文件

五个单次 Test 结果分别写到各 target experiment 的：

```text
test_7_6_locked_result.json
test_7_6_locked_result.txt
test_7_6_locked.log
```

总汇总：

```text
experiments/7_6_locked_test_summary.json
experiments/7_6_locked_test_summary.csv
```

汇总中的 LEVIR-MCI 包含三个 seed 的原始结果、八项均值和 sample standard deviation。比较 CARD Test baseline 时使用 `card_baseline_test_summary.json`，不得将 Test baseline 传回 checkpoint selector。

## 安全行为

- 不覆盖 7.5 的 target 目录或结果。
- 已有 Test 结果只有在 checkpoint 路径完全相同且八项指标均为 finite 时才会跳过。
- checkpoint、source config、validation CSV、annotation、vocab、H5 均记录 SHA256；特征目录记录完整文件清单摘要。
- Test 前会重新验证 manifest 和所有已锁定工件。
- `semantic_fusion_gamma_max=0` 仅表示不做正上限裁剪，不代表关闭语义融合。
