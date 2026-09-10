# 当前最佳统一 RSACA 配置总结

本文档总结当前在 LEVIR-CC、LEVIR-MCI 和 SECOND-CC 上综合表现最好的统一配置。结果来自
`experiments/reliability_sparse_rsaca_v1_whole_gate_retry/summary.json`，三个数据集均使用
scratch 训练、三个随机种子、验证集选点和一次锁定测试。

## 1. 配置结论

当前最佳统一配置为：

`CARD + sparse RSACA + input-conditioned global token + whole-adapter reliability gate`

对应入口为 `scripts/run_reliability_sparse_rsaca_v1_whole_gate.sh`，关键设置如下：

| 组件 | 设置 |
|---|---|
| 主干 | `sgc_card` / CARD |
| 语义融合 | residual cross-attention |
| 语义输入 | `cross_attention`，7 个语义类别 |
| 稀疏策略 | 变化位置 token 用作 cross-attention 的 K/V |
| 全局 token | 开启，使用 `all_mean` 输入相关全局 token |
| 可靠性门控 | 开启，连续 gate，作用于完整 RSACA adapter 输出 |
| gate bias | `-1.5`，使初始语义残差较保守 |
| 残差缩放 | `gamma_init=0.01`，`gamma_max=0.5` |
| 归一化 | `legacy_post_norm`，保持当前锁定结果可复现 |
| partial detach | 开启，`semantic_detach_ratio=0.5` |
| auxiliary loss | mask loss 与 semantic caption loss 均关闭 |
| hard gate / feature reweight | 均关闭 |
| 训练 | scratch，10000 steps，seed `1111/2222/3333` |
| 选点 | `paper_balanced`，只使用验证集 |

## 2. 锁定测试结果

数值为三 seed 均值，括号内为样本标准差。delta 相对于当前 CARD 审计基线。

完整均值如下：

| 数据集 | B1 | B2 | B3 | B4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LEVIR-CC | .8322 | .7379 | .6587 | .5926 | .3982 | .7427 | 1.3431 | .3033 |
| LEVIR-MCI | .8360 | .7349 | .6446 | .5661 | .3999 | .7567 | 1.3742 | .3386 |
| SECOND-CC | .6589 | .5017 | .3934 | .3168 | .2504 | .5499 | .8324 | .2540 |

| 数据集 | B4 | CIDEr | SPICE | 主要结论 |
|---|---:|---:|---:|---|
| LEVIR-CC | `0.5926 +/- 0.0271` (`-0.0200`) | `1.3431 +/- 0.0050` (`-0.0004`) | `0.3033 +/- 0.0269` (`+0.0215`) | 主要 BLEU 指标仍低于 CARD |
| LEVIR-MCI | `0.5661 +/- 0.0339` (`+0.0039`) | `1.3742 +/- 0.0082` (`+0.0362`) | `0.3386 +/- 0.0356` (`+0.0088`) | 八项均值均高于 CARD |
| SECOND-CC | `0.3168 +/- 0.0038` (`+0.0215`) | `0.8324 +/- 0.0093` (`+0.0536`) | `0.2540 +/- 0.0069` (`+0.0048`) | 八项均值均高于 CARD |

综合判断：这是目前跨三个数据集最均衡的已完成统一配置。它在 LEVIR-MCI 和 SECOND-CC
表现明确，但 LEVIR-CC 仍未超过 CARD，因此不能声称“统一模型在三个数据集稳定优于 CARD”。
当前严格配对的 `3 datasets x 2 arms x 3 seeds` 主矩阵仍需使用
`scripts/run_paired_card_rsaca_matrix.sh` 完成后，才能替代单次 CARD 审计基线。

## 3. 模块作用与联系

### 3.1 CARD 视觉主干

CARD 从双时相图像特征中提取视觉变化表示，并将其送入 caption decoder。它是整个系统的
稳定基础路径，也是 RSACA 的 query 来源。RSACA 的目标不是替换 CARD，而是在视觉表示上
增加一个小幅、可回退的语义残差。

### 3.2 语义输入编码

语义图先被映射到与视觉特征相同的 embedding 空间：

- LEVIR-CC：pseudo change mask，使用 diff-only binary 编码；
- LEVIR-MCI：`images/<split>/label`，使用 diff-only multiclass 编码；
- SECOND-CC：成对 `sem/A` 与 `sem/B`，按 `before != after` 得到变化位置。

三个数据集共享融合结构，但语义来源和数据路径允许按数据集变化。

### 3.3 Sparse changed-token K/V

只把检测为变化的位置提供给 cross-attention 的 key/value。这样可以减少无变化背景对
语义融合的干扰，使 query 主要读取与变化相关的地物语义。它依赖前面的语义差分规则；
差分图错误会直接造成 K/V 缺失或错误语义。

### 3.4 Input-conditioned global token

除稀疏变化 token 外，加入一个由输入语义特征汇聚得到的全局 token（当前最佳配置为
`all_mean`）。它为局部变化 token 提供场景上下文，降低稀疏化后上下文不足的风险。它与
changed-token K/V 一起构成 cross-attention 的 context 序列。

### 3.5 Cross-attention adapter 与 residual gamma

视觉 query 读取语义 context 后得到 adapter 输出。该输出不直接覆盖 CARD 特征，而是形成
残差：

`visual_features + gamma * adapter_output`

`gamma` 从很小的 `0.01` 开始，并限制在 `0.5` 以内，使训练初期接近 CARD，同时允许模型
逐步学习有用的语义修正。

### 3.6 Reliability gate

gate 根据视觉变化摘要、语义摘要和变化覆盖率估计当前语义输入是否可靠。gate 不是二值
开关，而是连续系数。当前配置把该系数乘到完整 adapter 输出上，因此低可靠性样本可以
整体减弱 RSACA，保留 CARD 原始视觉表示。

模块关系可以概括为：

```text
双时相图像 -> CARD visual features --------------------+
                                                      |
语义图 -> 差分/变化位置 -> sparse K/V + global token -> cross-attention adapter
                                                      |
视觉摘要 + 语义摘要 + 覆盖率 -> reliability gate ------+
                                                      |
                         visual + gate * gamma * adapter -> caption decoder
```

### 3.7 Partial detach

`semantic_detach_ratio=0.5` 部分阻断语义分支对视觉特征学习的反向影响，减少语义输入噪声
重塑 CARD 主干的风险。它与 reliability gate 互补：partial detach 控制梯度路径，gate
控制前向残差幅度。

## 4. 当前证据边界

1. LEVIR-CC 的 B4 和 CIDEr 均值仍低于 CARD；该配置不是三数据集全指标胜出方案。
2. 当前 delta 主要相对于每个数据集一次 CARD 锁定审计结果；严格配对 CARD/RSACA 矩阵
   尚未完成，不能把差值解释为完全由模型结构造成。
3. 新生成 LEVIR-CC pseudo mask、confidence、visual fallback 和 warmup 的已完成尝试
   没有带来可复现的整体增益，不能作为默认模块。
4. 当前 `context_pre_norm`、detached gate 等方案属于待验证候选，不应覆盖本配置的锁定证据。

## 5. 后续优化方向

### 优先级 A：先完成配对实验与可归因性

运行 `scripts/run_paired_card_rsaca_matrix.sh` 完成 CARD 与 whole-adapter RSACA 的 18 个
scratch 任务。必须保持相同 commit、源码状态、workers、Python/CUDA 环境、seed、训练
日程和验证选点策略。先完成该矩阵，再决定哪些差异值得优化。

### 优先级 B：只针对 LEVIR-CC 做验证集筛选

以本配置为固定对照，逐项验证以下候选，禁止同时改变多个因素：

1. `context_pre_norm`，测试 gamma 接近零时的严格恒等性质；
2. `changed_mean` 替代 `all_mean`，减少无变化背景对全局 token 的污染；
3. gate bias、gamma 上限和 gate 温度的保守小范围搜索；
4. 变化覆盖率校准和无变化样本的 gate 分布；
5. 旧 mask 与新 mask 的严格同提交、同 workers 配对消融。

每个候选先用多个 seed 的验证集筛选，只有在 B4、CIDEr 和非 SPICE 指标均不劣于配对
对照时，才进入一次锁定测试。

### 优先级 C：补充论文级实验分析

- 报告每个 seed、均值、样本标准差和配对 delta；
- 对 sparse token、global token、whole gate、partial detach、norm 分别做消融；
- 增加无变化样本、伪 mask 错误样本和不同变化覆盖率区间的分层结果；
- 提供语义正确、语义错误和失败案例，说明 gate 是否确实抑制了不可靠语义；
- 固定测试集只评估一次，避免根据 test 结果反复改选 checkpoint。

任何模型、语义输入、实验协议、结果或论文结论变化后，都必须运行：

```bash
python scripts/update_experiment_claim_reference.py
```
