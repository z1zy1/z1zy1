# P1 语义对照实验结果（2026-10-03 GPU 批次）

## 1. 批次身份与完成状态

本文件记录当前 AutoDL 批次 `p1_semantic_controls_gpu_20261003T142855Z` 的实际结果，不覆盖历史 `p1_semantic_controls_20260921` 总结，也不把历史 R2 测试结果并入本批统计。

| 项目 | 当前状态 |
|---|---|
| 运行矩阵 | `card`、`plain_fusion`、`rsaca`、`fixed_gate` × `levir_cc`、`levir_mci`、`second_cc` × seed `1111/2222/3333` |
| 运行数量 | 36/36 completed |
| 训练步数 | 每次 10000 steps |
| 验证选点 | 每次从 1000 至 10000、间隔 1000 的验证网格选择 |
| 协议审计 | `protocol_audit_passed=true`，`errors=[]` |
| 验证矩阵 | 完整 36 运行 |
| RSACA−CARD 正向均值 | 11/15 |
| RSACA−CARD 正向 seed×指标 | 34/45 |
| 15/15 准入 | 未达到 |
| 冻结/正式 test | 未执行 |

权威机器证据位于 [`results/p1_semantic_controls_gpu_20261003T142855Z`](../results/p1_semantic_controls_gpu_20261003T142855Z/)。AutoDL 原始目录仍保留 checkpoint、预测和完整训练日志；本仓库只提交轻量证据，避免把大文件或数据集提交到 Git。

## 2. 证据边界

本批可以复核验证 CSV、历史选点记录、运行身份和协议审计。离线验证工具的证据级别为 `CSV plus historical selection; checkpoint identity not independently checked`：它没有重新加载每个 checkpoint 的完整权重并复算全部预测。因此下文的数值结论是验证数值和配对比较结论，不是 checkpoint 本体已完全独立复核的结论。

审计记录的 `statistical_significance` 为 `not inferred from three seeds`。三个 seed 足以报告配对差和样本标准差，但不能据此声称统计显著性；45 个 seed×指标差也不是 45 个独立重复实验。

## 3. RSACA−CARD 主比较

五项指标的验证均值差如下。正向准入单元是 3 个数据集 × 5 个指标，共 15 项。

| 数据集 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE | 正向项 |
|---|---:|---:|---:|---:|---:|---:|
| LEVIR-CC | -0.021124 | -0.004032 | +0.001597 | -0.000291 | +0.022447 | 2/5 |
| LEVIR-MCI | +0.031997 | +0.009666 | +0.013209 | +0.029444 | -0.006392 | 4/5 |
| SECOND-CC | +0.005458 | +0.005658 | +0.018561 | +0.041097 | +0.021142 | 5/5 |

未通过的四个均值单元是 `LEVIR-CC/BLEU-4`、`LEVIR-CC/METEOR`、`LEVIR-CC/CIDEr` 和 `LEVIR-MCI/SPICE`。当前最强的描述是：RSACA 在 SECOND-CC 五项指标均值均为正，并在 LEVIR-MCI 的 BLEU-4、METEOR、ROUGE-L、CIDEr 上为正；它没有在所有数据集和指标上统一优于 CARD。

### 逐 seed 方向

RSACA−CARD 的 45 个配对差中，34 个正、0 个零、11 个负。LEVIR-CC 的 BLEU-4 为 3/3 seed 负，是最稳定的负向边界；LEVIR-MCI 的 BLEU-4、METEOR、ROUGE-L、CIDEr 为 3/3 正；SECOND-CC 的 ROUGE-L 为 3/3 正。逐 seed 方向仍有明显异质性，不能称为全面稳定提升。

## 4. 四组比较

下表汇总 9 个配对 seed 的总体均值、样本标准差和正向数量。样本标准差使用 `ddof=1`；缺失值不会填零。

| 比较 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---|---|---|---|---|
| D−B (`plain_fusion−card`) | +0.006081 ± 0.031906，6/9 | +0.004887 ± 0.009857，8/9 | +0.006343 ± 0.009072，6/9 | +0.013829 ± 0.021783，7/9 | +0.004921 ± 0.021796，5/9 |
| C0−B (`rsaca−card`) | +0.005444 ± 0.029783，5/9 | +0.003764 ± 0.009023，7/9 | +0.011123 ± 0.009594，7/9 | +0.023417 ± 0.026872，8/9 | +0.012399 ± 0.025183，7/9 |
| C0−D (`rsaca−plain_fusion`) | -0.000637 ± 0.019356，5/9 | -0.001123 ± 0.005303，3/9 | +0.004780 ± 0.006515，7/9 | +0.009587 ± 0.022635，6/9 | +0.007478 ± 0.016199，5/9 |
| C0−G (`rsaca−fixed_gate`) | +0.005422 ± 0.023542，4/9 | -0.001656 ± 0.006903，4/9 | +0.005330 ± 0.005100，9/9 | -0.000548 ± 0.016778，5/9 | +0.003338 ± 0.020148，5/9 |

这四组比较支持三个具体观察：

1. `plain_fusion` 已经产生相当部分收益，因此不能把 C0−B 的全部差异归因于 RSACA 结构；C0−D 在 BLEU-4 和 METEOR 上总体略负。
2. RSACA 相对 fixed_gate 最稳定的优势是 ROUGE-L，9/9 个配对 seed 为正；METEOR 总体为负，CIDEr 接近零且略负。
3. CIDEr 是 RSACA 相对 CARD 最稳定的改善指标之一（8/9 个 seed 为正），但这种优势相对 fixed_gate 并未保持。

## 5. LEVIR-CC SPICE 诊断

当前批次 RSACA−CARD 的 LEVIR-CC 选点 SPICE 差异为：

| seed | 差值 |
|---:|---:|
| 1111 | +0.048586 |
| 2222 | -0.001619 |
| 3333 | +0.020373 |

下降集中在 seed 2222，而不是三个 seed 普遍下降。选中 step 也不同：CARD 为 `7000/10000/6000`，RSACA 为 `9000/10000/3000`（对应 seed `1111/2222/3333`）。这说明需要同时查看曲线和固定样本对照，不能从最终选点差值直接推断机制。

本批 CC 诊断生成了 16 个固定样本的匿名人工核查表。对象、动作、属性和关系错误字段仍为 `unlabelled`；文本长度或重复率不能替代人工事实标签。变化状态、变化面积和语义空图分层在没有有效标注来源时保持不可用。

## 6. 与历史总结的关系

历史总结记录的 LEVIR-CC RSACA−CARD 均值为 `(+0.01869, +0.00992, +0.00671, +0.02241, -0.01456)`，而本批为 `(-0.021124, -0.004032, +0.001597, -0.000291, +0.022447)`。这不是舍入误差，多个指标方向不同。两批结果必须分开保存和引用；当前文档只解释本批实际输出，不尝试把两批平均或用旧总结覆盖新证据。

## 7. 结论与后续

本次实验确认了 36-run 训练、验证选点、矩阵完整性和协议审计流程，显示 RSACA 有数据集和指标依赖的局部改善，但没有达到 15/15 准入门槛。因此当前结果适合作为后续模型诊断依据，不支持冻结、正式 test 或“统一模型全面优于 CARD”的论文主张。

后续改进应优先针对 LEVIR-CC 的 BLEU-4/METEOR/CIDEr 负向结果、LEVIR-MCI 的 SPICE 负向结果，以及 RSACA 相对 fixed_gate 在 METEOR/CIDEr 上的不稳定性。任何结构变化都应使用新的独立输出目录、相同 seed 和验证选点规则，并保留本批结果不变。

## 8. 可复核命令

```bash
cd /root/autodl-tmp/z1zy1_v1_accept_20260920
PYTHON=/root/miniconda3/envs/card/bin/python
EXP=/root/autodl-tmp/z1zy1_v1_accept_20260920/experiments/p1_semantic_controls_gpu_20261003T142855Z

$PYTHON scripts/semantic_controls_evidence.py inventory \
  --experiment-root "$EXP" \
  --output-dir /root/autodl-tmp/p1_evidence_inventory

$PYTHON scripts/semantic_controls_evidence.py validation \
  --experiment-root "$EXP" \
  --output-dir /root/autodl-tmp/p1_evidence_validation

$PYTHON scripts/semantic_controls_evidence.py cc-diagnostic \
  --experiment-root "$EXP" \
  --output-dir /root/autodl-tmp/p1_evidence_cc_diagnostic \
  --sample-count 16
```

这些命令要求输出目录不存在或为空；工具拒绝覆盖非空目录、软链接和实验源目录内路径。
