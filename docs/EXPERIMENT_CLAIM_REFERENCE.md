# 实验主张与结果分析参考文档

> 本文档是模型实现、实验设计和论文结论的统一语义来源。修改 CARD/RSACA、数据输入、训练协议、选点策略、测试结果或论文主张后，必须运行 `python scripts/update_experiment_claim_reference.py` 并复核结论。

结果快照时间：`2026-08-12 11:09:44 UTC`

## 1. 原始论文主张

论文希望提出一个基于 CARD 改进的统一模型，并证明该模型在 LEVIR-CC、LEVIR-MCI、SECOND-CC 三个数据集上均优于原始 CARD，从而证明改进方法具有跨数据集有效性。这里的“统一模型”指主体结构、语义融合机制、损失项和核心超参数一致；允许数据路径、词表、序列长度和语义标签来源随数据集变化，并为每个数据集分别训练 checkpoint。

该主张只有在三个数据集都完成同协议、多随机种子、验证集选点和锁定测试后才能成立。单 seed、测试集选点、不同基线 checkpoint 或数据集专用结构都不能单独支撑原主张。

## 2. 当前结论

- **统一 RSACA scratch 三 seed 锁定测试已完成，但尚不能声称三数据集统一有效。**
- 当前 `experiments/unified_rsaca/summary.json` 包含 9 次 RSACA 运行；CARD 对照仍是每数据集 1 次锁定运行，而不是第 3 节要求的 CARD 三 seed 配对矩阵。因此现有 delta 是对单次审计基线的比较，尚不能作为完整 `3 x 2 x 3 = 18` 主实验的最终统计结论。
- LEVIR-CC 均值仅 ROUGE-L、SPICE 提升；BLEU-1 至 BLEU-4、METEOR、CIDEr 均低于 CARD（B4 -0.0585，CIDEr -0.0026）。
- LEVIR-MCI 三 seed 均值 8/8 指标高于 CARD（B4 +0.0141，CIDEr +0.0453，SPICE +0.0091），支持该数据集上的有效性。
- SECOND-CC 三 seed 均值 7/8 指标提升，但 SPICE 低于 CARD -0.0005，未达到全指标统一验收标准。
- 因此统一验收为未通过；MCI-transfer 结果仍只能作为独立迁移证据，不能补足 LEVIR-CC 或 SECOND-CC 的主实验缺口。

## 3. 实现原主张的建议

统一候选采用 **CARD + Residual Semantic Cross-Attention Adapter（RSACA）**：残差 cross-attention、`gamma_init=0.01`、`gamma_max=0.5`、partial detach `0.5`；关闭 auxiliary mask loss、semantic caption loss、hard gate 和 feature reweight。SECOND-CC 使用成对语义图；LEVIR-CC 与 LEVIR-MCI 使用显式标记的 diff-only 语义图接口。

当前锁定结果使用 `semantic_fusion_norm_mode=legacy_post_norm`。审计发现该实现即使在 `gamma=0` 时仍执行 `LayerNorm(query)`，因此不是严格恒等残差；而锁定 checkpoint 的实际 gamma 仅约 0.016-0.025。`context_pre_norm` 将归一化移到语义 context 分支，使 `gamma=0` 时输出严格等于原 CARD 特征。它目前只是针对 LEVIR-CC 退化的待验证结构假设，必须先做验证集驱动的小规模消融，再按 scratch 三 seed 锁定协议复验，不能据此改写现有结果。

主实验必须从 scratch 分别训练 CARD 与 RSACA，避免用 MCI 初始化混淆结构增益。固定 3 个 seed（1111、2222、3333），形成 `3 datasets x 2 models x 3 seeds = 18` 次主实验。MCI-transfer 结果只能作为独立迁移实验。

验收标准：每个数据集的主要指标均值不低于 CARD；至少 BLEU-4、CIDEr、SPICE 的方向一致；报告每 seed、均值、样本标准差；checkpoint 只能根据验证集选择，测试集仅运行一次锁定评估。

完整统一实验入口：`bash scripts/run_unified_rsaca_experiments.sh --stage all`。

## 4. 审计 CARD 基线

| 数据集 | Bleu_1 | Bleu_2 | Bleu_3 | Bleu_4 | METEOR | ROUGE_L | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| levir_cc | 0.8404 | 0.7499 | 0.6738 | 0.6126 | 0.4025 | 0.7417 | 1.3435 | 0.2818 |
| levir_mci | 0.8217 | 0.7219 | 0.6360 | 0.5621 | 0.3889 | 0.7420 | 1.3380 | 0.3297 |
| second_cc | 0.6307 | 0.4748 | 0.3703 | 0.2953 | 0.2366 | 0.5266 | 0.7789 | 0.2491 |

这些数值来自 `experiments/card_baseline_test_summary.json`，是本文档所有 delta 的默认比较基线。它们各自仅对应 1 次 CARD 锁定运行；在补齐 CARD seeds 1111、2222、3333 前，只能作为审计基线，不能替代多 seed 对照均值和样本标准差。SECOND-CC 旧论文汇总中的 checkpoint 7000 与这里的锁定 checkpoint 9000 不一致，最终论文必须使用后者。

## 5. 当前锁定结果总览

| 数据集/方案 | seeds | B4 delta | CIDEr delta | SPICE delta | 提升指标数 | 分析 |
|---|---:|---:|---:|---:|---:|---|
| levir_cc / 7.6 locked | 1 | -0.0753 | -0.0021 | +0.0714 | 2/8 | SPICE 增益不能抵消 BLEU 系列下降，当前候选失败。 |
| levir_mci / 7.6 locked | 3 | -0.0047 | +0.0039 | +0.0067 | 6/8 | 大多数指标改善，但 BLEU-4 未复现单 seed 优势，结论仍为混合。 |
| second_cc / current MCI-transfer cross-attn | 3 | +0.0107 | +0.0351 | +0.0052 | 8/8 | 三 seed 均值全面提升；需用 scratch 对照分离迁移收益。 |
| second_cc / 7.6 legacy locked | 1 | -0.1265 | -0.4190 | -0.1010 | 0/8 | 与最新复现实验冲突，属于旧配置/评估链路失败结果，不作为最终模型证据。 |

## 6. 每项论文实验结果分析

| 数据集 | 实验 | B4 | CIDEr | SPICE | 相对审计基线 | 结论说明 |
|---|---|---:|---:|---:|---|---|
| levir_mci | `levir_mci_card_baseline` | 0.5621 | 1.3380 | 0.3297 | 参考基线，不判定增益 | LEVIR-MCI 的审计基线；所有该数据集改进实验应与它比较。 |
| levir_mci | `levir_mci_card_mask_loss` | 0.5565 | 1.3288 | 0.3302 | 4/8 提升，不支持整体优于基线 | 仅增加 mask loss；BLEU-4、CIDEr 下降，只有 SPICE 极小提升，不支持整体有效。 |
| levir_mci | `levir_mci_card_semantic_loss` | 0.5834 | 1.3507 | 0.3168 | 7/8 提升，但证据混合 | BLEU-4 和 CIDEr 提升，但 SPICE 明显下降；说明语义辅助有潜力，但目标不平衡。 |
| levir_mci | `levir_mci_card_mask_semantic` | 0.5644 | 1.3532 | 0.3417 | 8/8 提升，当前结果支持 | 单 seed 的 8 项指标均高于 CARD，是最强单次结果；三 seed 复现后优势未完全保持。 |
| levir_mci | `levir_mci_card_mask_semantic_pd05` | 0.5590 | 1.3390 | 0.3404 | 6/8 提升，但证据混合 | partial detach 改善 SPICE，但 BLEU-4 低于基线；属于混合结果。 |
| levir_mci | `levir_mci_card_mask_semantic_pd05_noreweight` | 0.5590 | 1.3390 | 0.3404 | 6/8 提升，但证据混合 | 与 pd05 数值完全相同，不能作为独立增益证据；需检查配置/权重是否实际生效。 |
| levir_mci | `levir_mci_card_mask_semantic_pd05_reweight` | 0.5853 | 1.3506 | 0.3189 | 7/8 提升，但证据混合 | BLEU-4 较强，但 SPICE 下降；feature reweight 不适合作为统一模型的固定组件。 |
| levir_mci | `levir_mci_ours_weak_coupled_final` | 0.5590 | 1.3390 | 0.3404 | 6/8 提升，但证据混合 | 与 pd05/noreweight 数值完全相同，且 BLEU-4 下降；不能作为最终统一模型证据。 |
| second_cc | `second_cc_card_rgb_baseline` | 0.2837 | 0.7793 | 0.2541 | 旧参考基线，不用于最终比较 | 旧论文汇总使用 checkpoint 7000；最终比较应改用锁定 CARD checkpoint 9000。 |
| second_cc | `second_cc_card_semantic_aux` | 0.2804 | 0.7915 | 0.2601 | 4/8 提升，不支持整体优于基线 | CIDEr、SPICE 略升但 BLEU-4 下降；单独语义辅助不足以证明全面改进。 |
| second_cc | `second_cc_card_semantic_crossattn` | 0.3126 | 0.8314 | 0.2605 | 8/8 提升，当前结果支持 | 单次结果在主要指标上最均衡，是 RSACA 统一方案的直接结构依据。 |
| second_cc | `second_cc_card_semantic_hardgate` | 0.3200 | 0.7819 | 0.2243 | 7/8 提升，但证据混合 | BLEU-4 提升但 SPICE 大幅下降；硬门控破坏语义质量，不纳入统一方案。 |
| second_cc | `second_cc_ours_weak_coupled_final` | 0.3278 | 0.8032 | 0.2259 | 7/8 提升，但证据混合 | BLEU-4 较高，但 SPICE 明显下降；不能支持整体优于 CARD 的主张。 |
| second_cc | `second_cc_mmodalcc_comparison` | - | - | - | 无可分析结果 | 缺少外部结果，不能进入定量主表或结论。 |

### LEVIR-MCI 三 seed 复现

单 seed `levir_mci_card_mask_semantic` 的 8/8 提升不能直接作为最终结论。三 seed 锁定均值为：

| 方案 | Bleu_1 | Bleu_2 | Bleu_3 | Bleu_4 | METEOR | ROUGE_L | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mask+semantic 3-seed mean | 0.8259 | 0.7254 | 0.6360 | 0.5575 | 0.3918 | 0.7467 | 1.3419 | 0.3365 |
| delta vs CARD | 0.0042 | 0.0035 | -0.0000 | -0.0047 | 0.0029 | 0.0047 | 0.0039 | 0.0067 |

结论：BLEU-3 delta 约为 0，BLEU-4 为负；只能称为多数指标改善，不能称为所有指标稳定优于 CARD。

### SECOND-CC 三 seed 复现

| seed | B4 | CIDEr | SPICE | 8/8 超过基线 |
|---:|---:|---:|---:|---|
| 1111 | 0.3104 | 0.8067 | 0.2547 | 是 |
| 2222 | 0.3093 | 0.8354 | 0.2612 | 是 |
| 3333 | 0.2982 | 0.7998 | 0.2470 | 否 |

结论：均值 8/8 提升，2/3 seeds 全指标通过；seed 3333 的 SPICE 低于基线。该结果支持方案潜力，但因为使用 MCI 初始化，不能替代 scratch 统一主实验。

## 7. 统一 RSACA 实验状态

| 数据集 | mean 8/8 above CARD | B4 delta | CIDEr delta | SPICE delta |
|---|---|---:|---:|---:|
| levir_cc | 否 | -0.0585 | -0.0026 | +0.0612 |
| levir_mci | 是 | +0.0141 | +0.0453 | +0.0091 |
| second_cc | 否 | +0.0113 | +0.0329 | -0.0005 |

总体验收：**未通过**。

## 8. 论文写作边界

当前可以写：提出残差语义交叉注意力改进；统一 RSACA 在 LEVIR-MCI 的 scratch 三 seed 锁定均值 8/8 指标高于 CARD；其在 LEVIR-CC 和 SECOND-CC 呈现明确的指标权衡；消融表明 hard gate 和 feature reweight 会造成指标权衡。

当前不能写：统一模型已在三个数据集全部优于 CARD；所有指标均显著提升；改进完全来自模型结构；LEVIR-CC 或 SECOND-CC 已验证成功；已完成 `3 x 2 x 3 = 18` 次完整主实验。

只有统一 RSACA scratch 矩阵达到第 3 节验收标准后，才可以把主结论升级为“三数据集均有效”。

## 9. 同步规则

以下修改必须同步本文档：模型结构或默认开关；语义标签来源和类别映射；训练步数、seed、初始化、学习率；验证选点与测试锁定策略；任一实验指标或有效性判断；论文主张和对外表述。

```bash
python scripts/update_experiment_claim_reference.py
```

更新后检查 `git diff -- docs/EXPERIMENT_CLAIM_REFERENCE.md`，确认自动表格和人工结论一致，再提交代码。

## 10. 权威数据源

- `experiments/card_baseline_test_summary.json`：三数据集 CARD 锁定基线。
- `experiments/paper_required_experiments_summary.json`：论文要求的单项实验结果。
- `experiments/7_6_locked_test_summary.json`：LEVIR-CC、LEVIR-MCI 和旧 SECOND-CC 锁定结果。
- `experiments/second_cc_current_mci_test_summary.json`：SECOND-CC 的 MCI-transfer 三 seed 结果，只能作为迁移证据。
- `experiments/unified_rsaca/summary.json`：统一 RSACA scratch 三 seed 矩阵结果；存在时作为当前统一结论的直接依据。
