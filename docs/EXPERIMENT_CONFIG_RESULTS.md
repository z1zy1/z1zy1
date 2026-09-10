# 实验配置与结果记录

本文件记录各候选配置的可追溯结果。当前新增条目只覆盖新 LEVIR-CC 共识掩码候选；完整逐指标原始结果在 `experiments/reliability_sparse_rsaca_v1_new_masks_20260821/summary.json` 与 `summary.csv`。所有 checkpoint 均由验证集选择，锁定测试不参与选点。

## 新掩码 V1 候选

结果目录：`experiments/reliability_sparse_rsaca_v1_new_masks_20260821`。

| 项目 | 记录值 |
|---|---|
| 模型 | CARD + reliability-gated sparse RSACA |
| 残差归一化 | `context_pre_norm` |
| 语义 token | changed-only K/V，`changed_mean` global token，空变化样本使用 learned fallback token |
| 门控 | 连续 reliability gate 作用于 whole adapter；visual-consistency gate 与 visual fallback 启用 |
| 残差尺度 | `gamma_init=0.01`，`gamma_max=0.1`，gate bias `-2.5`，fusion warmup `2000` steps |
| 训练 | scratch，`10000` steps，seeds `1111/2222/3333`，partial detach `0.5`，无 auxiliary mask/caption loss、hard gate 或 feature reweight |
| checkpoint 选择 | validation-only `paper_balanced_no_spice`，按 CIDEr、BLEU-4、METEOR、ROUGE-L；SPICE 不用于选择但仍报告 |
| 测试协议 | 每个 seed 对验证集选出的 checkpoint 执行一次 immutable locked test；汇总确认 `selection_uses_test_metrics=false` |
| 对照 | `experiments/card_baseline_test_summary.json` 的每数据集单次锁定 CARD 审计基线；尚不是配对的 CARD 三 seed 对照 |

### 数据集输入配置

| 数据集 | 语义输入 | confidence 输入 | 预检/审计 |
|---|---|---|---|
| LEVIR-CC | ChangeFormerV6 + BIT LEVIR checkpoint 共识 `pseudo_masks`，diff-only binary map | `Levir-CC/pseudo_confidence`，仅此数据集启用 | 11163 文件；抽样平均变化覆盖 `0.0454`；空变化比例 `0.5430` |
| LEVIR-MCI | `images/<split>/label`，diff-only multiclass map | 无 | 10077 文件；抽样平均变化覆盖 `0.0699`；空变化比例 `0.5078` |
| SECOND-CC | 成对 `sem/A`、`sem/B` | 无 | A/B 各 10855，paired 10855，缺失 0；抽样平均变化覆盖 `0.2258` |

### 锁定测试均值与波动

数值为 `mean +/- sample std`，delta 相对于上述审计 CARD baseline。`*` 表示均值高于该 baseline。

| 数据集 | Bleu-1 | Bleu-2 | Bleu-3 | Bleu-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LEVIR-CC | 0.8019 +/- 0.0053 | 0.6994 +/- 0.0090 | 0.6156 +/- 0.0138 | 0.5484 +/- 0.0196 | 0.3823 +/- 0.0059 | 0.7224 +/- 0.0034 | 1.2879 +/- 0.0097 | 0.3013 +/- 0.0206* |
| LEVIR-MCI | 0.8086 +/- 0.0079 | 0.7126 +/- 0.0097 | 0.6339 +/- 0.0125 | 0.5687 +/- 0.0170* | 0.3786 +/- 0.0049 | 0.7256 +/- 0.0041 | 1.2963 +/- 0.0158 | 0.3094 +/- 0.0155 |
| SECOND-CC | 0.4948 +/- 0.0101 | 0.3690 +/- 0.0098 | 0.2934 +/- 0.0087 | 0.2421 +/- 0.0084 | 0.1892 +/- 0.0036 | 0.4623 +/- 0.0033 | 0.6633 +/- 0.0047 | 0.2150 +/- 0.0066 |

| 数据集 | B4 delta | CIDEr delta | SPICE delta | 均值提升数 | 忽略 SPICE 的提升数 | 8/8 seed pass |
|---|---:|---:|---:|---:|---:|---:|
| LEVIR-CC | -0.0642 | -0.0556 | +0.0196 | 1/8 | 0/7 | 0/3 |
| LEVIR-MCI | +0.0066 | -0.0416 | -0.0204 | 1/8 | 1/7 | 0/3 |
| SECOND-CC | -0.0532 | -0.1155 | -0.0342 | 0/8 | 0/7 | 0/3 |

### 每 Seed 锁定结果

| 数据集 | seed | selected checkpoint step | Bleu-4 | CIDEr | SPICE | 全部 8 项高于 baseline |
|---|---:|---:|---:|---:|---:|---|
| LEVIR-CC | 1111 | 10000 | 0.5629 | 1.2768 | 0.2829 | 否 |
| LEVIR-CC | 2222 | 9000 | 0.5562 | 1.2922 | 0.2974 | 否 |
| LEVIR-CC | 3333 | 10000 | 0.5261 | 1.2949 | 0.3236 | 否 |
| LEVIR-MCI | 1111 | 9000 | 0.5647 | 1.2781 | 0.3055 | 否 |
| LEVIR-MCI | 2222 | 10000 | 0.5541 | 1.3051 | 0.3264 | 否 |
| LEVIR-MCI | 3333 | 9000 | 0.5873 | 1.3058 | 0.2962 | 否 |
| SECOND-CC | 1111 | 10000 | 0.2461 | 0.6579 | 0.2089 | 否 |
| SECOND-CC | 2222 | 8000 | 0.2324 | 0.6657 | 0.2220 | 否 |
| SECOND-CC | 3333 | 10000 | 0.2477 | 0.6664 | 0.2139 | 否 |

## 结论边界

该候选完成了三数据集、三随机 seed、验证集选点和锁定测试，但 `acceptance_passed=false`。因此它不支持“统一模型在三个数据集上稳定超过 CARD”的主张，也不支持在忽略 SPICE 后的同一主张：LEVIR-CC 为 0/7 非 SPICE 指标提升，LEVIR-MCI 为 1/7，SECOND-CC 为 0/7。

唯一的均值正向信号是 LEVIR-CC 的 SPICE 与 LEVIR-MCI 的 BLEU-4；两者均不足以抵消其余主要指标下降。新掩码及 confidence 输入可以作为已完成的输入质量改进尝试记录，但在重新设计并通过新的验证驱动候选前，不能写作性能增益证据或主结果。

## LEVIR-CC 新掩码匹配对照

结果目录：`experiments/reliability_sparse_rsaca_v1_levir_cc_new_masks_matched_control_retry_20260826`。三个 seed 均由验证集 `paper_balanced` 选点并各执行一次锁定测试，`selection_uses_test_metrics=false`。

| 项目 | 旧 LEVIR-CC pre-norm changed-global | 新掩码对照 |
|---|---|---|
| 语义图 | 旧版外部 `pseudo_masks` | 当前 ChangeFormerV6 + BIT 共识 `pseudo_masks` |
| confidence | 无 | 无 |
| visual gate / fallback / warmup | 关闭 / 关闭 / 0 | 关闭 / 关闭 / 0 |
| 验证选点 | `paper_balanced` | `paper_balanced` |
| 核心 RSACA | `context_pre_norm`、changed-only K/V、`changed_mean`、whole-adapter reliability gate | 相同 |
| DataLoader workers | 8 | 0 |
| 记录提交 | `7e6d1f9` | `3d6f297` |

| 指标 | 旧均值 | 新掩码均值 +/- sample std | 新减旧 | 新减 CARD baseline |
|---|---:|---:|---:|---:|
| Bleu-1 | 0.8323 | 0.8282 +/- 0.0102 | -0.0042 | -0.0122 |
| Bleu-2 | 0.7382 | 0.7290 +/- 0.0132 | -0.0092 | -0.0209 |
| Bleu-3 | 0.6576 | 0.6429 +/- 0.0170 | -0.0147 | -0.0309 |
| Bleu-4 | 0.5905 | 0.5695 +/- 0.0222 | -0.0210 | -0.0430 |
| METEOR | 0.3990 | 0.3911 +/- 0.0062 | -0.0080 | -0.0115 |
| ROUGE-L | 0.7438 | 0.7434 +/- 0.0031 | -0.0004 | +0.0016 |
| CIDEr | 1.3464 | 1.3374 +/- 0.0110 | -0.0090 | -0.0061 |
| SPICE | 0.3070 | 0.3274 +/- 0.0122 | +0.0205 | +0.0457 |

三个新掩码 seed 的 B4/CIDEr 分别为 `0.5784/1.3350`、`0.5442/1.3278`、`0.5860/1.3495`，均未实现 8/8 超过 CARD；均值仅 ROUGE-L 与 SPICE 高于 CARD，忽略 SPICE 后为 `1/7` 指标提升。

结论：该对照不支持“新生成掩码效果更好”。相对旧结果，7/8 均值指标下降，仅 SPICE 上升；但 `NUM_WORKERS` 从 8 变为 0，且两次实验提交不同，因此不能把全部退化严格归因于掩码。它足以表明在当前代码和训练条件下，新掩码没有带来可复现的整体性能增益。
