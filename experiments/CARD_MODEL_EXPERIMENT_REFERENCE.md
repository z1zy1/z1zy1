# CARD 三数据集模型改动与实验进展参考

> 更新日期：2026-07-26  
> 适用项目：`CARD` 远程感知变化描述实验  
> 本地代码：`D:\实验\CARD`  
> 远程代码：`/root/autodl-tmp/z1zy1`  
> 用途：汇总历史对话、当前代码、已完成结果、失败经验与后续建议，作为继续修改模型和设计实验的统一依据。

## 1. 当前结论摘要

1. 项目已经从原始 CARD 的“双时相视觉差异编码 + DynamicSpeaker”扩展为可选的多任务 WCSG-CARD：支持变化掩膜、语义辅助监督、语义图 cross-attention、partial detach、特征重加权、hard gate、关系辅助和内容词加权。
2. LEVIR-MCI 当前最可靠的全指标模型是 `levir_mci_card_mask_semantic`：在现有单次 Test 中，八项描述指标全部高于重新实现的 CARD baseline。7.6 不再继续盲目搜索超参数，而是用相同配置独立训练三个随机种子。
3. SECOND-CC 的历史 `pd08` 和 `semantic_crossattn checkpoint 9000` 都曾在八项指标上超过 CARD baseline，但 7.5 锁定到 `semantic_crossattn checkpoint 8000` 后出现全面退化。该冲突说明不能只看实验名或单个 checkpoint，必须审计源配置，并以验证集稳定窗口锁定模型。
4. LEVIR-CC 的 7.5 模型把 SPICE 提高到 `0.353128`，但 BLEU-1/2/3/4、METEOR 和 CIDEr 相对旧参考 baseline 下滑。因此 7.6 采用极低学习率、10/20 步、只更新语言生成器的保守微调，目标是保留其他指标同时提高 SPICE。
5. 7.6 的代码与流程已经完成，但截至本文更新时没有新的 7.6 Test 结果。首次远程启动在训练前因 7.5 旧配置未显式记录 `model.semantic_input_mode` 而被审计器拒绝；该问题已由提交 `75315f2` 修复，需重新运行。
6. 后续实验必须坚持：验证集选择 checkpoint，Test 只在锁定后执行一次；同一配置至少 3 个随机种子并报告 mean/std；不能再把 Test 阈值直接套到不同尺度的验证指标上。

## 2. 数据集说明

### 2.1 LEVIR-CC

LEVIR-CC 是面向遥感图像变化描述的双时相数据集。公开资料给出的标准规模为 10,077 对 `256×256` 图像、50,385 条人工描述，每对图像有 5 条描述；标准划分为 6,815/1,333/1,929 对训练、验证、测试样本，其中变化与无变化样本数量近似均衡。

官方资料：

- 数据集仓库：https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset
- 论文：https://levir.buaa.edu.cn/publications/ChangeCaptioning.pdf

本项目中的主要输入：

```text
Levir-CC/
├── features/
├── images/
├── levir_cc_captions_reformat.json
├── transformer_levir_vocab.json
├── transformer_levir_labels.h5
└── splits.json
```

在当前方案中，LEVIR-CC 主要承担纯变化描述任务；其像素级掩膜并不完整，因此增强模型允许缺失伪掩膜。研究目标不是单独最大化 SPICE，而是在 BLEU、METEOR、ROUGE-L、CIDEr 基本不下降的前提下改善语义内容。

### 2.2 LEVIR-MCI

LEVIR-MCI 是 LEVIR-CC 的多层次变化理解扩展，保留双时相图像和描述，同时增加变化检测掩膜，用于联合变化检测与变化描述。公开资料列出 10,077 对图像、50,385 条描述和 44,380 张掩膜；官方目录中 `A`、`B` 分别是变化前后图像，`label` 是变化掩膜。

官方资料：

- Change-Agent/MCI 官方仓库：https://github.com/Chen-Yang-Liu/Change-Agent
- 论文：https://arxiv.org/abs/2403.19646
- 汇总资料：https://github.com/Chen-Yang-Liu/Awesome-Change-Captioning

本项目中的主要输入：

```text
LEVIR-MCI-dataset/
├── LevirCCcaptions.json
├── features/
├── images/{train,val,test}/{A,B,label}/
├── levir_mci_captions_reformat.json
├── transformer_levir_mci_vocab.json
├── transformer_levir_mci_labels.h5
└── splits.json
```

当前代码把变化掩膜作为三类多分类目标，配置为 `mask_type=multiclass`、`num_mask_classes=3`，并使用 `ce_dice`。评价除八项描述指标外，还包含 Mask Precision/Recall/F1/IoU、road/building IoU 等像素级指标。

### 2.3 SECOND-CC

SECOND-CC 面向更复杂真实场景下的遥感变化描述，公开论文给出 6,041 对双时相 RGB 图像和 30,205 条描述，并同时提供前后时相语义分割图。它强调光照、视角、配准、模糊和分辨率差异等干扰，对仅依赖 RGB 差异的模型更有挑战。

官方资料：

- 论文：https://arxiv.org/abs/2501.10075
- 官方项目：https://github.com/ChangeCapsInRS/SecondCC

本项目使用的目录结构：

```text
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
```

代码将语义图配置为 7 类，并通过 `semantic_input_mode=cross_attention` 注入 CARD 差异特征。当前本地 Test 文件包含 1,227 个测试样本。

### 2.4 三个数据集的角色差异

| 数据集 | 核心监督 | 当前增强重点 | 主要风险 |
|---|---|---|---|
| LEVIR-CC | 图像对 + 描述 | 语言生成器保守微调、内容词权重 | SPICE 提升时 BLEU/CIDEr 容易下降 |
| LEVIR-MCI | 图像对 + 描述 + 变化掩膜 | mask 与 semantic 辅助多任务 | 多任务权重、梯度干扰、随机种子波动 |
| SECOND-CC | RGB 图像对 + 描述 + 前后语义图 | 语义 cross-attention、partial detach | checkpoint 波动大、配置/路径错配会导致整体崩溃 |

## 3. 原始 CARD 与当前模型结构

### 3.1 原始 CARD 主干

当前 `models/CARD.py` 仍保留原始 CARD 的核心：

1. 对双时相视觉特征分别进行投影和 Transformer 编码。
2. 使用公共特征一致性损失 `loss_con` 对齐变化前后的共同语义。
3. 使用差异特征独立性损失 `loss_ind` 分离两个时相的变化表征。
4. 通过跨时相注意力计算公共上下文，再从原特征中减去公共部分，形成双向差异特征。
5. 拼接并映射双向差异特征，交给 `DynamicSpeaker` Transformer 解码器生成描述。

原始 CARD baseline 在本项目中应理解为：关闭 mask、semantic、reweight、hard gate、relation 等全部增强开关，只保留 CARD 主干和语言解码器。baseline 流程会检查这些开关，防止增强模型被误标为 CARD baseline。

### 3.2 掩膜辅助分支

新增 `AuxMaskHead`，把 CARD 差异 token 恢复为空间特征并输出掩膜 logits：

- 二分类可使用 BCE/Dice；
- LEVIR-MCI 三分类使用 CE/Dice；
- 可计算 Mask Precision、Recall、F1、IoU、mIoU 和分类别 IoU；
- 可作为辅助损失，也可作为差异特征重加权先验。

实践结论：单独增加 mask loss 并不保证描述指标提升；mask 与 semantic 联合配置更稳定，但权重和梯度流向必须控制。

### 3.3 语义辅助分支

代码提供两类语义监督：

- `SemanticAuxHead`：对差异特征池化后预测对象/动作语义标签，适合 LEVIR-CC/LEVIR-MCI 的多标签语义辅助；
- `DenseSemanticHead`：输出逐像素语义类别，适合具备语义分割图的 SECOND-CC。

语义损失支持 multilabel BCE、CE 和 CE+Dice，并可配置 warmup、late start、decay 和 ignore index。

### 3.4 语义 cross-attention

`SemanticCrossAttentionFusion` 对前后时相语义图分别编码，拼接：

```text
[semantic_before, semantic_after, abs(semantic_after - semantic_before)]
```

映射后作为 key/value，CARD 差异特征作为 query，通过多头注意力得到语义上下文，再以可学习 `gamma` 残差融合：

```text
fused = LayerNorm(diff + gamma * semantic_context)
```

必要说明：`semantic_fusion_gamma_max=0` 表示不做正上限裁剪，并不表示关闭语义融合。此前已修复 SECOND-CC 评估显存问题：类别 ID 语义图会先缩放到特征空间尺寸，再执行 embedding，避免在原始分辨率上产生巨量张量。

### 3.5 partial detach 与梯度控制

`partial_detach_feature(x, r)` 实现：

```text
x_partial = r * stop_gradient(x) + (1-r) * x
```

`r=0` 为完全反向传播，`r=1` 为完全截断。实验中的 `pd05/pd07/pd08/pd09` 分别对应 0.5/0.7/0.8/0.9。其目的在于利用语义信息但减少语义辅助任务对视觉变化特征的破坏。

已观察到的规律：

- SECOND-CC 的 `pd08` 在当前历史网格中综合最好；
- `pd09` 更偏向 BLEU，CIDEr 和 SPICE 明显下降；
- `pd07` 与 `pd08` 的 SPICE 差异很小，单次差异不足以证明显著性。

### 3.6 特征重加权与 hard gate

- Feature reweight：用 mask 概率对 caption 输入做软重加权，强度由 `reweight_alpha` 控制；通常会 detach mask，避免描述损失反向污染 mask 分支。
- Hard gate：根据语义差异图对空间 token 做硬门控。历史 SECOND-CC 结果表明 hard gate 虽可提高部分 BLEU，但 SPICE 明显受损，不是当前主路线。

### 3.7 内容词加权

`DynamicSpeaker` 的 caption loss 支持对对象、动作等内容词加权。7.5 使用 `content_word_weight=1.03/1.05`，7.6 收缩为 `1.00/1.01/1.02`，并启用归一化，目的是微调语义表达而不明显改变句法与 n-gram 分布。

### 3.8 7.6 的语言生成器独立微调

新增 `train.finetune_decoder_only`，但代码中的“decoder-only”实际含义是：

- 冻结整个 `CARD/change_detector`；
- 将其固定在 `eval` 模式；
- 训练整个 `DynamicSpeaker`，包括词嵌入、Transformer decoder 和输出层；
- 优化器只接收 `requires_grad=True` 的参数；
- 必须提供结构完全匹配的初始化 checkpoint，missing/unexpected key 会直接失败；
- 冻结 CARD 后，mask/semantic 辅助损失只做监控，反向传播只使用包含 speaker 路径的主损失。

该设计专门用于 LEVIR-CC：保持已经学到的视觉变化表示，只做极小幅度语言端修正。

## 4. 实验演进

### 4.1 初始多任务与消融实验

已完成的主要配置包括：

- LEVIR-MCI：CARD baseline、mask only、semantic only、mask+semantic、partial detach、feature reweight、weak coupled；
- SECOND-CC：RGB CARD baseline、semantic aux、semantic cross-attention、hard gate、weak coupled；
- 增加描述、掩膜、语义和 change/no-change 分组指标；
- 增加配置哈希、resolved config、checkpoint 路径与实验汇总。

这一阶段确认了“增加辅助任务不等于稳定提高描述质量”：语义约束过强会提高某一类指标，却破坏 CIDEr/SPICE 或整体句子质量。

### 4.2 7.1–7.3：短程 caption fine-tune

LEVIR-MCI 从 weak 模型出发做 caption-only 微调：

- 1000/500 步版本实际都选中 step 250，SPICE 达到 `0.350027`，但 BLEU-4 仅 `0.519557`；
- 保留 mask loss 的短微调将 BLEU-4 提到 `0.594504`，但 SPICE 降到 `0.296271`；
- 说明 BLEU/CIDEr 与 SPICE 存在明显权衡，不能单指标选 checkpoint。

随后修复 `finetune_steps`，确保超短实验真实在 50/100 步结束并保存最终 checkpoint。

### 4.3 7.4：LEVIR-MCI 密集网格与 SECOND partial detach

LEVIR-MCI 的关键结果：

| 实验 | 选中 step | BLEU-4 | CIDEr | SPICE | 结论 |
|---|---:|---:|---:|---:|---|
| weak source | 10000 | 0.573020 | 1.352636 | 0.322770 | 传统指标强，SPICE 偏低 |
| 50_lr005 | 50 | 0.562437 | 1.346001 | 0.332140 | 接近旧严格阈值，仅 SPICE 仍不足 |
| 100_lr005_dense10 | 90 | 0.538200 | 1.336385 | 0.348278 | SPICE 高，但 BLEU 损失过大 |
| 80_lr004_dense10 | 70 | 0.551500 | 1.346699 | 0.343762 | 最佳语义/传统指标折中 |
| 80_lr003_dense10 | 10 | 0.574437 | 1.351915 | 0.319743 | 几乎保持源模型，语义无改善 |

旧严格测试目标为 `BLEU-4>=0.562`、`CIDEr>=1.338`、`SPICE>=0.336`，该轮没有模型三项同时满足。

SECOND-CC 的关键结果：

| partial detach | BLEU-4 | CIDEr | SPICE | 结论 |
|---|---:|---:|---:|---|
| pd07 | 0.280804 | 0.816742 | 0.271879 | SPICE 略高，但其余略低 |
| pd08 | 0.284531 | 0.821306 | 0.271452 | 综合最佳，作为主要候选 |
| pd09 | 0.294403 | 0.800944 | 0.254799 | BLEU 提升但语义指标下降 |

### 4.4 CARD baseline 完整流程

新增三数据集原始 CARD 流程，包含训练、验证集 checkpoint 选择、manifest 锁定和 Test 汇总。baseline 仅作为比较和验证阈值，不进入增强模型候选池。

当前已有结果来源并不完全统一：

- LEVIR-MCI、SECOND-CC baseline 数值来自用户提供的 `初始实验结果.csv`；
- LEVIR-CC 旧参考值来自早期 `cc/test_paper_best_result.txt`，需要与远程新 baseline 锁定汇总再次核对；
- 7.6 正式结论应以远程 `card_baseline_locked_manifest.json` 和 `card_baseline_test_summary.json` 为唯一 baseline 来源。

### 4.5 7.5：验证集锁定的三数据集流程

训练设置：

- LEVIR-CC：从 `sgc_card_lm003_ls005_pd05_rw02_warmup` 初始化，content weight 1.03/1.05，30 步，lr `2e-6`；
- LEVIR-MCI：从 `levir_mci_card_mask_semantic` 初始化，同样做 30 步内容词微调；
- SECOND-CC：从 pd08 初始化，100 步，测试 gamma 上限 0.05/0.10；
- checkpoint 只在验证集按 baseline 约束/Pareto 规则锁定，随后 Test。

最终 7.5 Test 实际锁定到：

```text
LEVIR-CC  : sgc_card_lm003_ls005_pd05_rw02_warmup checkpoint 5000
LEVIR-MCI : levir_mci_card_mask_semantic checkpoint 5000
SECOND-CC : second_cc_card_semantic_crossattn checkpoint 8000
```

这说明新增短微调候选并没有在验证约束下胜过已有源模型。

### 4.6 7.6：保守微调、三种子和稳定窗口

LEVIR-CC：

- 精确绑定 7.5 validation-locked checkpoint 与源配置；
- 冻结 CARD，只训练完整 `DynamicSpeaker`；
- content weight：1.00/1.01/1.02；
- lr：`5e-7/1e-6`；
- steps：10/20；
- 每 5 步保存和验证，共 12 组；
- 七项非 SPICE 指标不低于验证 baseline，SPICE 也不得下降，再从 Pareto 前沿选择。

LEVIR-MCI：

- 不再搜索超参数；
- 固定 mask+semantic 完整配置；
- seed 1111/2222/3333 各自从头训练 10,000 步；
- 每个 seed 只能从自己的验证 checkpoint 中选择；
- 最终报告 8 项指标 mean 和 sample std。

SECOND-CC：

- 不新增盲目训练网格；
- 审计 pd07/pd08/pd09 和传统 semantic-crossattn；
- 只接受同一源实验中间隔 1000 步的连续 3-checkpoint 窗口；
- 三个成员都必须具有八项 finite 指标，并逐项不低于验证 baseline；
- 按窗口 mean SPICE、最小保护余量、worst SPICE 等排序，锁定中心 checkpoint。

7.6 还增加了严格 manifest：绑定 baseline、selection、checkpoint、resolved config、annotation、vocab、H5、split、changeflag、semantic tags、特征目录和 SECOND `sem/A`、`sem/B` 的哈希/库存。锁定 Test 拒绝 `--force/--overwrite`。

## 5. 结果与 baseline 对比

### 5.1 当前使用的 baseline

| 数据集 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE | 来源说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LEVIR-CC | 0.828000 | 0.728000 | 0.641000 | 0.565000 | 0.391000 | 0.746000 | 1.348000 | 0.336000 | 旧参考阈值，须用新锁定 baseline 复核 |
| LEVIR-MCI | 0.821720 | 0.721904 | 0.635996 | 0.562134 | 0.388939 | 0.741979 | 1.337970 | 0.329708 | 重新实现 CARD baseline |
| SECOND-CC | 0.623698 | 0.466348 | 0.360868 | 0.283700 | 0.233607 | 0.526760 | 0.779330 | 0.254104 | 重新实现 CARD RGB baseline |

不同文件中的 baseline 可能来自论文值、旧手工阈值或重新实现流程，不能混用。正式论文表应只引用同一代码版本、同一数据划分、同一 Test 脚本生成的锁定 baseline。

### 5.2 7.5 LEVIR-CC 与旧参考 baseline

| 指标 | baseline | 7.5 | 差值 |
|---|---:|---:|---:|
| BLEU-1 | 0.828000 | 0.819841 | -0.008159 |
| BLEU-2 | 0.728000 | 0.717266 | -0.010734 |
| BLEU-3 | 0.641000 | 0.622571 | -0.018429 |
| BLEU-4 | 0.565000 | 0.537259 | -0.027741 |
| METEOR | 0.391000 | 0.387732 | -0.003268 |
| ROUGE-L | 0.746000 | 0.747085 | +0.001085 |
| CIDEr | 1.348000 | 1.341413 | -0.006587 |
| SPICE | 0.336000 | 0.353128 | +0.017128 |

结论：SPICE 明显提升，ROUGE-L 基本持平并略升，但其余六项下降；它是“语义优先”模型，不满足“其他指标不下降”的最终要求。7.6 的超保守 speaker 微调正是针对这个问题。

### 5.3 7.5 LEVIR-MCI 与 CARD baseline

| 指标 | baseline | 7.5 mask+semantic | 差值 |
|---|---:|---:|---:|
| BLEU-1 | 0.821720 | 0.832480 | +0.010760 |
| BLEU-2 | 0.721904 | 0.735057 | +0.013152 |
| BLEU-3 | 0.635996 | 0.645353 | +0.009357 |
| BLEU-4 | 0.562134 | 0.564402 | +0.002268 |
| METEOR | 0.388939 | 0.391825 | +0.002886 |
| ROUGE-L | 0.741979 | 0.751507 | +0.009528 |
| CIDEr | 1.337970 | 1.353239 | +0.015268 |
| SPICE | 0.329708 | 0.341736 | +0.012028 |

结论：这是目前唯一在一次 Test 中八项描述指标全部超过 CARD baseline 的配置，优先级高于只追求更高 SPICE 的超短微调。当前问题不是继续调参，而是用多个随机种子验证提升是否稳定。

`lr004 step70` 的 SPICE 更高（0.343762），但 BLEU-1/2/3/4、METEOR 仍略低于 baseline，因此只适合作为语义折中对照，不应取代 mask+semantic 作为主模型。

### 5.4 SECOND-CC 历史最佳与 CARD baseline

`second_cc_crossattn_pd08_lsem0000 checkpoint 9000`：

| 指标 | baseline | pd08 | 差值 |
|---|---:|---:|---:|
| BLEU-1 | 0.623698 | 0.639224 | +0.015526 |
| BLEU-2 | 0.466348 | 0.479126 | +0.012777 |
| BLEU-3 | 0.360868 | 0.366617 | +0.005749 |
| BLEU-4 | 0.283700 | 0.284531 | +0.000831 |
| METEOR | 0.233607 | 0.246266 | +0.012659 |
| ROUGE-L | 0.526760 | 0.545675 | +0.018915 |
| CIDEr | 0.779330 | 0.821306 | +0.041976 |
| SPICE | 0.254104 | 0.271452 | +0.017347 |

此外，`second_cc_card_semantic_crossattn checkpoint 9000` 的 BLEU-4/CIDEr/SPICE 为 `0.312630/0.831379/0.260486`：它在传统描述指标上更强，pd08 的 SPICE 更高。两者都应进入稳定选择候选池。

### 5.5 SECOND-CC 7.5 锁定 Test 的异常退化

7.5 锁定的 `second_cc_card_semantic_crossattn checkpoint 8000` 得到：

| 指标 | baseline | 7.5 locked | 差值 |
|---|---:|---:|---:|
| BLEU-1 | 0.623698 | 0.520636 | -0.103061 |
| BLEU-2 | 0.466348 | 0.350814 | -0.115535 |
| BLEU-3 | 0.360868 | 0.243124 | -0.117744 |
| BLEU-4 | 0.283700 | 0.174873 | -0.108828 |
| METEOR | 0.233607 | 0.189343 | -0.044264 |
| ROUGE-L | 0.526760 | 0.382861 | -0.143899 |
| CIDEr | 0.779330 | 0.374278 | -0.405052 |
| SPICE | 0.254104 | 0.148793 | -0.105312 |

该结果与同名实验 checkpoint 9000 的历史结果差异极大。合理解释包括：checkpoint 8000 本身处于不稳定区、验证选择没有稳定性约束、旧实验配置解析/运行时路径绑定不完整。不能把它解释为“semantic cross-attention 方法本身失败”。7.6 因此改用三点稳定窗口，并强制 Test 运行时与 source resolved config、数据路径和语义图路径一致。

## 6. 方法学结论与建议

### 6.1 checkpoint 选择

必须遵循：

```text
训练候选 -> 验证集指标 -> 约束/Pareto或稳定窗口 -> 锁定manifest -> Test一次
```

禁止：

- 反复测试多个 checkpoint 后选最好 Test；
- 用 Test 指标阈值直接筛选验证指标；
- 从多个 `best*.json` 文件猜测初始化 checkpoint；
- 在 enhanced candidate 不满足约束时自动回退到 CARD baseline 并将其标为改进模型。

7.4 的 global strict-nearest 曾直接把测试尺度阈值用于更低尺度验证集，造成阈值缺口巨大且选择结果不可信。这是 7.5/7.6 改为 validation-baseline 相对约束的直接原因。

### 6.2 指标解释

- BLEU-1/2/3/4：局部 n-gram 一致性，BLEU-4 对句式和长片段更敏感；
- METEOR：考虑词形与一定程度语义匹配；
- ROUGE-L：最长公共子序列；
- CIDEr：对多参考描述的一致性加权，是变化描述常用主指标；
- SPICE：基于场景图语义，能够反映对象、属性和关系，但与 BLEU/CIDEr 可能存在权衡；
- Mask/Semantic 指标：用于检查视觉辅助任务是否真正学到像素/类别信息，不能替代描述指标。

最终“超过 baseline”应定义为同一锁定 Test 上八项描述指标逐项比较，而不是平均分或只看 BLEU-4/CIDEr/SPICE。

### 6.3 随机种子与显著性

单次实验的微小差异不能作为显著性结论。建议：

- 最低标准：3 个独立随机种子，报告 mean ± sample std；
- 更强标准：参考 LEVIR-MCI 官方建议跑 5 次；
- 同一多种子组除 seed、实验名和输出路径外，完整配置必须一致；
- 如果均值提升小于标准差，应表述为“趋势”而不是“稳定提升”。

### 6.4 后续优先级

1. 首先完成 7.6 训练、验证锁定和一次性 Test，不新增其他网格。
2. LEVIR-CC：若没有候选在验证集同时保护七项指标和 SPICE，则保留 7.5 源模型，不强行 Test 失败候选；下一轮只在 `lr<=5e-7`、`steps<=10` 范围做更小扰动，或尝试只解冻输出层。
3. LEVIR-MCI：以三种子 mask+semantic 为主结果；`lr004 step70` 只作为语义折中对照。
4. SECOND-CC：优先稳定窗口而不是单点最优；分别保留 pd08（SPICE 优先）和 semantic-crossattn（BLEU/CIDEr 优先）的消融解释。
5. 若 7.6 Test 仍出现与历史结果量级不一致，先审计输入、词表、checkpoint、resolved config、语义图和测试样本数，禁止立即修改模型。

### 6.5 推荐消融表

论文或报告建议至少保留以下对照：

| 组别 | mask aux | semantic aux/map | partial detach | reweight/cross-attn | 用途 |
|---|---:|---:|---:|---:|---|
| CARD baseline | × | × | × | × | 原始基线 |
| mask only | √ | × | × | × | 掩膜监督贡献 |
| semantic only | × | √ | 可选 | × | 语义监督贡献 |
| mask+semantic | √ | √ | × | 可选 | LEVIR-MCI 主模型 |
| pd grid | 视数据集 | √ | √ | cross-attn | 梯度耦合敏感性 |
| speaker conservative FT | 源模型保持 | 源模型保持 | 源模型保持 | 源模型保持 | LEVIR-CC 语言端修正 |

## 7. 代码与流程索引

| 文件 | 作用 |
|---|---|
| `models/CARD.py` | CARD 主干、mask/semantic/relation heads、partial detach、reweight、semantic cross-attention |
| `models/transformer_decoder.py` | DynamicSpeaker 与内容词加权 caption loss |
| `datasets/rcc_dataset_transformer_levir.py` | 三数据集图像、掩膜、语义图和描述加载 |
| `utils/dataset_config.py` | LEVIR-CC、LEVIR-MCI、SECOND-CC 数据目录适配 |
| `train_card_spot.py` | 多任务损失、warmup、梯度控制、decoder-only fine-tune、checkpoint 保存 |
| `test_card_spot.py` | 锁定 checkpoint 测试与辅助指标输出 |
| `scripts/select_best_checkpoint.py` | validation baseline Pareto、严格约束、稳定窗口选择 |
| `scripts/build_card_baseline_manifest.py` | 原始 CARD baseline 配置与工件审计 |
| `scripts/build_7_6_locked_manifest.py` | 7.6 五个 lock 的完整工件/路径/指标审计 |
| `scripts/run_7_6_followup_train.sh` | 7.6 的 12 个 LEVIR-CC 微调和 3 个 MCI seed 训练 |
| `scripts/run_7_6_followup_reselect.sh` | 三数据集验证选择与 manifest 生成 |
| `scripts/run_7_6_followup_test_locked.sh` | 五个锁定 checkpoint 的一次性 Test |
| `scripts/summarize_7_6_locked_tests.py` | Test 对 baseline 的差值、均值和样本标准差 |
| `experiments/README_7_5_VALIDATION_LOCKED.md` | 7.5 使用说明 |
| `experiments/README_7_6_VALIDATION_LOCKED.md` | 7.6 使用说明 |

关键提交：

```text
40b3ae4  模型适配 LEVIR-MCI 和 SECOND-CC
a4ddd8d  WCSG-CARD 模型、配置、指标和追踪流程
256f41f  SECOND-CC semantic cross-attention 显存修复
e77625c  7.3 超短微调流程
06e55c8  7.4 密集网格与 reselect
23eb48b  三数据集 CARD baseline 完整流程
d426096  7.5 validation-locked 流程
e6d3d97  7.6 保守微调、三种子与稳定窗口
75315f2  兼容 7.5 LEVIR-CC 旧配置缺失字段
```

## 8. 当前运行状态与下一步操作

截至 2026-07-26：

- 7.1–7.5 历史结果已获得并完成分析；
- 三数据集原始 CARD baseline 流程已实现；
- 7.6 代码已提交；
- 7.6 首次远程运行在训练前被旧配置审计阻止；修复已提交；
- 尚未收到 7.6 训练、锁定 manifest 或 Test 汇总，因此不能声称 7.6 已超过 baseline。

远程重新运行前先确认：

```bash
cd /root/autodl-tmp/z1zy1
git pull
source /root/miniconda3/etc/profile.d/conda.sh
conda activate card

export PYTHON=/root/miniconda3/envs/card/bin/python
export PROJECT_DIR=/root/autodl-tmp/z1zy1
export EXP_ROOT=/root/autodl-tmp/z1zy1/experiments
export LEVIR_CC_ROOT=/root/autodl-tmp/z1zy1/Levir-CC
export LEVIR_MCI_ROOT=/root/autodl-tmp/z1zy1/LEVIR-MCI-dataset
export SECOND_CC_ROOT=/root/autodl-tmp/z1zy1/SECOND-CC-AUG

$PYTHON scripts/resolve_7_6_levir_cc_source.py \
  --manifest "$EXP_ROOT/7_5_locked_manifest.json" \
  --expected_source_exp sgc_card_lm003_ls005_pd05_rw02_warmup
```

审计成功后，在 tmux 中运行：

```bash
bash scripts/run_7_6_followup_all.sh
rc=$?
echo CARD_7_6_EXIT_CODE=$rc
```

训练与验证选择完成后检查：

```text
experiments/7_6_locked_manifest.json
```

确认五个 lock 后再执行一次：

```bash
$PYTHON scripts/build_7_6_locked_manifest.py \
  --verify "$EXP_ROOT/7_6_locked_manifest.json"
bash scripts/run_7_6_followup_test_locked.sh --dry_run
bash scripts/run_7_6_followup_test_locked.sh
```

最终结果：

```text
experiments/7_6_locked_test_summary.json
experiments/7_6_locked_test_summary.csv
```

## 9. 运行与维护注意事项

1. 本地只做语法、单元测试、dry-run 和小规模代码验证；完整训练只在远程 GPU 服务器执行。
2. 非交互 tmux 中不要依赖 `.bashrc` 激活 Conda，使用 `/root/miniconda3/etc/profile.d/conda.sh` 或直接指定环境 Python。
3. 不要把整条 `bash -lc '...'` 再用一层引号作为命令执行，否则 Shell 会把它当作文件名。
4. 在交互式 Shell 中使用 `rc=$?`；只有把命令嵌入 tmux 外层双引号时才写 `rc=\$?`。
5. tmux 会话立即消失通常是启动阶段失败；保留退出码并在失败后 `exec bash` 便于检查。
6. `experiments/7_1_followup_train_failures.log` 等运行日志属于用户工件，不得擅自删除；若阻碍 `git pull`，先移动备份。
7. 不覆盖已有未完成实验目录。7.6 会拒绝脏目录和锁定 Test 覆盖；需要重跑时应归档旧目录或使用新 `EXP_ROOT`。
8. 测试结果文件缺失不一定表示训练失败：训练阶段通常只生成 `val_metrics.csv` 和 checkpoint，Test JSON 只有在执行测试脚本后才出现。
9. `resolved_config.json` 是新流程首选；旧实验可能只有 `cfg.json`。兼容逻辑只能针对注册的旧实验，不能全局放宽。
10. 远程磁盘容量长期接近上限。避免训练过程中复制多个 60+ MB best checkpoint，优先保存路径引用和必要 snapshot。

## 10. 数据来源与解释边界

本文结果来自：

```text
C:\Users\zhangyi\Desktop\2024级专硕企业实习项目\7.1.csv
C:\Users\zhangyi\Desktop\2024级专硕企业实习项目\7.3.csv
C:\Users\zhangyi\Desktop\2024级专硕企业实习项目\7.9.csv
C:\Users\zhangyi\Desktop\2024级专硕企业实习项目\7.12.csv
C:\Users\zhangyi\Desktop\2024级专硕企业实习项目\mci\初始实验结果.csv
C:\Users\zhangyi\Desktop\2024级专硕企业实习项目\levir_cc_test_metrics.csv
C:\Users\zhangyi\Desktop\2024级专硕企业实习项目\levir_mci_test_metrics.csv
C:\Users\zhangyi\Desktop\2024级专硕企业实习项目\second_cc_test_metrics.csv
```

必要边界：

- 历史 CSV 中的同名实验可能对应不同 checkpoint，比较时必须同时记录 checkpoint step；
- 早期 `cc/test_paper_best_result.txt` 的 baseline 属于旧参考值，不应替代新 baseline manifest；
- 7.5 的三份 `*_test_metrics.csv` 是单次锁定 Test，不代表均值或统计显著性；
- 本文不包含尚未完成的 7.6 指标，也不假设新修改一定超过 baseline；
- 当后续生成 `7_6_locked_test_summary.csv` 后，应以它更新第 5 节，并保留本版作为实验设计与问题追踪记录。
