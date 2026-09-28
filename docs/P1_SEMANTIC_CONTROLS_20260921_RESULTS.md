# P1 Semantic Controls 结果总结

结果日期：`2026-09-24`  
协议：`p1_semantic_controls_20260915`  
实验根目录：`experiments/p1_semantic_controls_20260921`

## 执行状态

- 实验矩阵为 4 个方案（`card`、`plain_fusion`、`rsaca`、`fixed_gate`）× 3 个数据集（`levir_cc`、`levir_mci`、`second_cc`）× 3 个 seed（`1111/2222/3333`），共 36 个运行。
- 36/36 个 `run_summary.json` 均为 `status=completed`，且 `final_global_step=10000`；无缺失、无无效 JSON、无运行失败残留。
- 锁定运行时为 Python `3.8.20`、PyTorch `1.10.0+cu113`、NumPy `1.21.4`、CUDA build `11.3`，使用绝对解释器 `/root/miniconda3/bin/python`。
- 源码身份与协议锁一致：`534f22fcd69294890152e1ae69a56bdf4838405a`。本批没有修改模型、语义输入、训练协议或 `protocol.json`。

## 恢复记录

初始失败运行为 `plain_fusion/plain_fusion_second_cc_seed2222`，原因是 checkpoint 写入时磁盘空间不足（`OSError: [Errno 28] No space left on device`）。原目录被完整归档而未删除，证据保存在：

`experiments/p1_semantic_controls_20260921/evidence_failure_20260922T090911Z/`

归档目录为：

`experiments/p1_semantic_controls_20260921/archive_incomplete_plain_fusion_second_cc_seed2222_20260922T092408Z/`

该运行随后在清空的新路径上按原协议从头重训。补跑/补齐的运行是：

`plain_fusion_second_cc_seed2222`、`rsaca_second_cc_seed2222`、`fixed_gate_second_cc_seed2222`、`card_second_cc_seed3333`、`plain_fusion_second_cc_seed3333`、`rsaca_second_cc_seed3333`、`fixed_gate_second_cc_seed3333`。

## 选点与审计

- `select` 退出码：`0`。
- 36 个 checkpoint 均由 validation split 按 `five_metric_equal_weight_log` 选择；没有使用 test 指标选点。
- `audit` 退出码：`0`；`protocol_audit_passed=true`，`errors=[]`。源码、runtime、输入清单、初始化证据、checkpoint 完整性、validation prediction identity 和选择记录均通过审计。
- 审计统计明确标记 `statistical_significance="not inferred from three seeds"`；本批没有进行显著性检验。

## Validation 结果

下面是 `rsaca - card` 的 validation 均值差异。每个数据集有 3 个 seed，指标顺序为 Bleu-4、METEOR、ROUGE-L、CIDEr、SPICE。

| 数据集 | Bleu-4 | METEOR | ROUGE-L | CIDEr | SPICE | 均值正向指标 |
|---|---:|---:|---:|---:|---:|---:|
| LEVIR-CC | +0.01869 | +0.00992 | +0.00671 | +0.02241 | -0.01456 | 4/5 |
| LEVIR-MCI | +0.00296 | +0.00829 | +0.00985 | +0.02294 | +0.00043 | 5/5 |
| SECOND-CC-AUG | +0.00511 | +0.00801 | +0.01812 | +0.04400 | +0.01543 | 5/5 |

合计为 14/15 个 validation 均值指标高于 CARD；45 个配对 seed-指标中有 37 个为正。`plain_fusion` 和 `fixed_gate` 也表现出明显的指标权衡，不能据此选出一个在三个数据集和全部指标上统一优于 CARD 的方案。

## 结论边界

这批实验证明训练、恢复、validation 选点和协议审计流程已经完成，但不支持以下主张：

- 不能写成统一模型在三个数据集上全面优于 CARD。
- 不能把 14/15 validation 均值提升写成协议要求的 15/15 通过。
- 不能把 validation 结果称为正式 locked test 结果。
- 不能从三个 seed 推断统计显著性。

由于协议要求验证均值达到 15/15，`audit.validation.mean_goal=false`，因此没有执行 `freeze` 或正式 `test`。当前不存在 `frozen.json`、正式 test prediction 或 test score 产物。Java、METEOR、SPICE 在 validation 评估中均有输出，但这不等同于完成正式 test 评分链。

## 证据文件

- [协议锁](../experiments/p1_semantic_controls_20260921/protocol.json)
- [审计报告](../experiments/p1_semantic_controls_20260921/audit.json)
- [select 日志](../experiments/p1_semantic_controls_20260921/select_20260924T051736Z.log)
- [audit 日志](../experiments/p1_semantic_controls_20260921/audit_20260924T051855Z.log)
- [训练重试日志](../experiments/p1_semantic_controls_20260921/resume_train_20260922_retry_20260922T092420Z.log)
- [失败证据目录](../experiments/p1_semantic_controls_20260921/evidence_failure_20260922T090911Z/)

