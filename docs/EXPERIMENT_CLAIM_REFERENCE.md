# 实验主张与结果分析参考文档

## 当前依据：P1 R2 配对实验

来源：`experiments/p1_rsaca_20260914/paired_matrix_r2/*/*/test_paired_locked_result.json`；逐运行重新计算，不使用文件内相对旧单 seed 基线的 deltas。

五项指标均值严格提高：**15/15**；逐 seed 指标提高：**41/45**。均值目标：**达到（历史 R2 数值证据）**。

| 数据集 | 指标 | CARD 均值 ± 样本标准差 | RSACA 均值 ± 样本标准差 | 配对差均值 ± 样本标准差 |
|---|---|---:|---:|---:|
| levir_cc | Bleu_4 | 0.555117 ± 0.005960 | 0.558059 ± 0.014903 | +0.002942 ± 0.010977 |
| levir_cc | METEOR | 0.383312 ± 0.004907 | 0.387524 ± 0.003379 | +0.004212 ± 0.008052 |
| levir_cc | ROUGE_L | 0.736866 ± 0.001640 | 0.746052 ± 0.004809 | +0.009185 ± 0.003248 |
| levir_cc | CIDEr | 1.318836 ± 0.014647 | 1.340869 ± 0.012001 | +0.022033 ± 0.012154 |
| levir_cc | SPICE | 0.331107 ± 0.010559 | 0.346477 ± 0.005556 | +0.015371 ± 0.010677 |
| levir_mci | Bleu_4 | 0.554434 ± 0.012383 | 0.567794 ± 0.020836 | +0.013360 ± 0.027516 |
| levir_mci | METEOR | 0.384363 ± 0.000571 | 0.397949 ± 0.007492 | +0.013586 ± 0.007939 |
| levir_mci | ROUGE_L | 0.739663 ± 0.000843 | 0.756220 ± 0.007010 | +0.016558 ± 0.006188 |
| levir_mci | CIDEr | 1.321042 ± 0.008950 | 1.367429 ± 0.019086 | +0.046387 ± 0.010655 |
| levir_mci | SPICE | 0.337520 ± 0.007512 | 0.346526 ± 0.006593 | +0.009006 ± 0.014083 |
| SECOND-CC-AUG | Bleu_4 | 0.308805 ± 0.007321 | 0.319540 ± 0.010910 | +0.010735 ± 0.008100 |
| SECOND-CC-AUG | METEOR | 0.240614 ± 0.001916 | 0.250547 ± 0.001644 | +0.009934 ± 0.000462 |
| SECOND-CC-AUG | ROUGE_L | 0.528250 ± 0.002179 | 0.549251 ± 0.001277 | +0.021001 ± 0.002445 |
| SECOND-CC-AUG | CIDEr | 0.770552 ± 0.004072 | 0.828853 ± 0.002297 | +0.058301 ± 0.003215 |
| SECOND-CC-AUG | SPICE | 0.231390 ± 0.008529 | 0.248154 ± 0.007928 | +0.016765 ± 0.006279 |

评分工件已完成：18/18。旧 run_summary 的 test_completed 标记不覆盖实际评分状态；不能据此重复训练或推理。原始工件与旧验收字段保持不变。

R2 支持在其既有实验条件下三数据集五项均值改善，不证明每个 seed 全部改善、统计显著或全部收益来自融合结构。历史源码、初始化和输入来源的可追溯缺口需要独立核查；dirty 标记本身不是重训理由。

当前后续实验：`p1_semantic_controls_20260915`，详见 [AutoDL 执行协议](SEMANTIC_CONTROLS_AUTODL.md)。B/D/C0/G 分别分离普通语义使用收益、定制融合额外收益和学习门控贡献。R2 不自动并入新协议。新配置冻结前要求验证均值 15/15 改善；不能用已有 R2 测试集开发记录声称测试集重新未见。

均值改善、实际意义与统计显著是不同结论；不能保证投稿录用。当前没有新四组训练结果。

## 历史材料（以下为旧方案和旧验收定义，不代表当前 R2 结论）

旧结果、旧失败结论、旧 8 项及 3 项验收口径仅用于追溯；不得覆盖上文五项指标复算。


> 本文档是模型实现、实验设计和论文结论的统一语义来源。修改 CARD/RSACA、数据输入、训练协议、选点策略、测试结果或论文主张后，必须运行 `python scripts/update_experiment_claim_reference.py` 并复核结论。

结果快照时间：`2026-09-15 11:53:01 UTC`

## 1. 原始论文主张

论文希望提出一个基于 CARD 改进的统一模型，并证明该模型在 LEVIR-CC、LEVIR-MCI、SECOND-CC 三个数据集上均优于原始 CARD，从而证明改进方法具有跨数据集有效性。这里的“统一模型”指主体结构、语义融合机制、损失项和核心超参数一致；允许数据路径、词表、序列长度和语义标签来源随数据集变化，并为每个数据集分别训练 checkpoint。

该主张只有在三个数据集都完成同协议、多随机种子、验证集选点和锁定测试后才能成立。单 seed、测试集选点、不同基线 checkpoint 或数据集专用结构都不能单独支撑原主张。

## 2. 历史阶段结论

- **统一 RSACA scratch 三 seed 锁定测试已完成，但尚不能声称三数据集统一有效。**
- 历史阶段 `experiments/unified_rsaca/summary.json` 包含 9 次 RSACA 运行；CARD 对照仍是每数据集 1 次锁定运行，而不是第 3 节要求的 CARD 三 seed 配对矩阵。因此现有 delta 是对单次审计基线的比较，尚不能作为完整 `3 x 2 x 3 = 18` 主实验的最终统计结论。
- LEVIR-CC 均值仅 ROUGE-L、SPICE 提升；BLEU-1 至 BLEU-4、METEOR、CIDEr 均低于 CARD（B4 -0.0585，CIDEr -0.0026）。
- LEVIR-MCI 三 seed 均值 8/8 指标高于 CARD（B4 +0.0141，CIDEr +0.0453，SPICE +0.0091），支持该数据集上的有效性。
- SECOND-CC 三 seed 均值 7/8 指标提升，但 SPICE 低于 CARD -0.0005，未达到全指标统一验收标准。
- 因此统一验收为未通过；MCI-transfer 结果仍只能作为独立迁移证据，不能补足 LEVIR-CC 或 SECOND-CC 的主实验缺口。

## 3. 实现原主张的建议

统一候选采用 **CARD + Residual Semantic Cross-Attention Adapter（RSACA）**：残差 cross-attention、`gamma_init=0.01`、`gamma_max=0.5`、partial detach `0.5`；关闭 auxiliary mask loss、semantic caption loss、hard gate 和 feature reweight。SECOND-CC 使用成对语义图；LEVIR-CC 与 LEVIR-MCI 使用显式标记的 diff-only 语义图接口。

历史阶段锁定结果使用 `semantic_fusion_norm_mode=legacy_post_norm`。审计发现该实现即使在 `gamma=0` 时仍执行 `LayerNorm(query)`，因此不是严格恒等残差；而锁定 checkpoint 的实际 gamma 仅约 0.016-0.025。`context_pre_norm` 将归一化移到语义 context 分支，使 `gamma=0` 时输出严格等于原 CARD 特征。它目前只是针对 LEVIR-CC 退化的待验证结构假设，必须先做验证集驱动的小规模消融，再按 scratch 三 seed 锁定协议复验，不能据此改写现有结果。

新增的 V1 候选为 **reliability-gated sparse RSACA**：只把变化位置提供给语义 cross-attention 的 K/V，并为无变化样本加入一个可学习回退 token；连续可靠性门控以视觉变化摘要、稀疏语义摘要和变化覆盖率缩放语义残差。成对语义图按 `before != after` 判定变化，diff-only 输入按非零类别判定变化。该规则在三数据集共享，保留锁定 RSACA 的其余训练设置。完整入口：`bash scripts/run_reliability_sparse_rsaca_v1.sh --stage all`；候选筛选阶段仅可运行 `preflight`、`train` 和 `select`。

V1 reliability-gated sparse RSACA 已完成三数据集三 seed 锁定测试，但总体验收为 `False`；LEVIR-CC B4/CIDEr 均值变化为 -0.0506/-0.0038，不能作为三数据集统一有效的证据。

V1 whole-adapter 候选已完成独立三数据集三 seed 锁定测试，但总体验收仍为 `acceptance_passed=false`：LEVIR-CC 的 B4/CIDEr 均值仍低于 CARD（-0.0200、-0.0004），而 LEVIR-MCI 与 SECOND-CC 的均值 8/8 指标提升。该结果是候选证据，不能改写统一主张。
新增待验证候选 `run_reliability_sparse_rsaca_v1_prenorm_changed_global.sh` 使用 `context_pre_norm`、`gamma_max=0.1`、gate bias `-2.5` 和 changed-only global token；其输出目录独立，锁定测试前必须先完成验证集筛选。
新增待验证 LEVIR-CC V2 候选 `run_reliability_sparse_rsaca_v2_detached_gate.sh` 保留 pre-norm changed-global 核心，仅在 reliability MLP 的视觉/语义摘要输入上使用 detach，以阻断门控对 CARD 特征分支的反向塑形；它显式关闭 confidence、visual gate、fallback 和 warmup。入口在训练前要求所选 Python 的 CUDA 可用；中断目录必须通过 `--reset-incomplete` 归档后才能重训，空验证行或非文件 checkpoint 不得进入选择或测试。历史阶段仅完成 seed 3333/1111 的验证筛选（seed 3333: B4=0.4445, CIDEr=1.2615；seed 1111: B4=0.4411, CIDEr=1.2434），尚未运行测试。历史 V1 运行与该筛选不构成严格配对，不能将任何差异归因于 detach；必须先与 `run_reliability_sparse_rsaca_v2_paired_control.sh` 在相同环境下比较。该结果不构成性能增益或论文证据。
本次 V1 实现还支持可选的 `data.semantic_diff_confidence_root`：置信度会对变化位置的 K/V 和 changed-mean global token 加权；视觉一致性门控、低置信度 visual fallback 与 fusion warmup 默认关闭，仅由 V1 候选显式开启。历史阶段 `pseudo_masks` 是二值外部模型输出，尚未提供可验证的逐像素概率，因此不能把该置信度路径或 fallback 设计宣称为已验证的性能增益。
V1 的验证选点新增 `paper_balanced_no_spice`，只在验证集上按 CIDEr、BLEU-4、METEOR、ROUGE-L 加权，暂时忽略 SPICE；这只是选择协议调整，不能替代三数据集多 seed 锁定测试。
LEVIR-CC 掩码重生成入口为 `scripts/generate_levir_ensemble_masks_direct.sh`：ChangeFormerV6 与 BIT 的概率图以 8-bit PNG 流程处理，通过一致性规则融合，并同时输出 confidence/uncertainty；该入口会在推理前删除旧 `pseudo_*` 目录以控制磁盘占用，成功后安装新结果。新掩码三 seed 锁定矩阵已经完成但未达到 CARD 验收，因此该输入替换不能写成已验证的性能增益。
为隔离新 LEVIR-CC 掩码本身的影响，可运行 `scripts/run_levir_cc_new_masks_matched_control.sh`：保留数据集作用域隔离，但关闭 confidence、visual gate、visual fallback 和 fusion warmup，并恢复 `paper_balanced` 验证选点。该入口要求 CUDA，且在受限容器中默认 `NUM_WORKERS=0`；该对照尚无结果，不能据此预判掩码优劣。

新掩码候选 `reliability_sparse_rsaca_v1_new_masks_20260821` 已完成三数据集三 seed 验证选点和锁定测试，且 `selection_uses_test_metrics=false`；但 `acceptance_passed=false`，不能支持“稳定超过 CARD”的主张。
levir_cc：均值 1/8 指标高于 CARD，seed 全指标通过 0/3；忽略 SPICE 后仍仅 0/7 指标提升；B4/CIDEr/SPICE delta=-0.0642/-0.0556/+0.0196。
levir_mci：均值 1/8 指标高于 CARD，seed 全指标通过 0/3；忽略 SPICE 后仍仅 1/7 指标提升；B4/CIDEr/SPICE delta=+0.0066/-0.0416/-0.0204。
second_cc：均值 0/8 指标高于 CARD，seed 全指标通过 0/3；忽略 SPICE 后仍仅 0/7 指标提升；B4/CIDEr/SPICE delta=-0.0532/-0.1155/-0.0342。
该候选使用 LEVIR-CC ChangeFormer+BIT 共识二值掩码及 agreement-aware confidence；LEVIR-MCI/SECOND-CC 语义输入保持原协议。新掩码输入变化尚未带来主张支持证据。

新 LEVIR-CC 掩码匹配对照 `reliability_sparse_rsaca_v1_levir_cc_new_masks_matched_control_retry_20260826` 已完成三 seed 验证选点和锁定测试，且 `selection_uses_test_metrics=false`；均值仅 2/8 指标高于 CARD，忽略 SPICE 后仅 1/7，seed 全指标通过 0/3，不能证明新掩码优于 CARD。
相对旧 LEVIR-CC pre-norm changed-global 结果，B4/CIDEr/SPICE 均值变化为 -0.0210/-0.0090/+0.0205；新掩码在该条件下未带来整体提升。
该对照关闭 confidence、visual gate、fallback 和 warmup，并恢复 `paper_balanced`；但新运行 `num_workers=0`、旧运行 `num_workers=8`，且两次提交不同，故它是反对“新掩码更好”的直接证据，但不能把全部差异严格归因于掩码本身。

严格配对的 `3 datasets x 2 models x 3 seeds` 矩阵已完成；总体验收：**未通过**。
levir_cc：配对 B4/CIDEr/SPICE delta=+0.0029/+0.0220/+0.0154；主要指标逐 seed 非劣=否；8 项均值严格提升=是。
levir_mci：配对 B4/CIDEr/SPICE delta=+0.0134/+0.0464/+0.0090；主要指标逐 seed 非劣=否；8 项均值严格提升=是。
second_cc：配对 B4/CIDEr/SPICE delta=+0.0107/+0.0583/+0.0168；主要指标逐 seed 非劣=是；8 项均值严格提升=是。

主实验必须从 scratch 分别训练 CARD 与 RSACA，避免用 MCI 初始化混淆结构增益。固定 3 个 seed（1111、2222、3333），形成 `3 datasets x 2 models x 3 seeds = 18` 次主实验。MCI-transfer 结果只能作为独立迁移实验。

验收标准：每个数据集的主要指标均值不低于 CARD；至少 BLEU-4、CIDEr、SPICE 的方向一致；报告每 seed、均值、样本标准差；checkpoint 只能根据验证集选择，测试集仅运行一次锁定评估。

历史 RSACA-only 入口：`bash scripts/run_unified_rsaca_experiments.sh --stage all`。严格主实验入口：`bash scripts/run_paired_card_rsaca_matrix.sh --stage all`。

## 4. 审计 CARD 基线

| 数据集 | Bleu_1 | Bleu_2 | Bleu_3 | Bleu_4 | METEOR | ROUGE_L | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| levir_cc | 0.8404 | 0.7499 | 0.6738 | 0.6126 | 0.4025 | 0.7417 | 1.3435 | 0.2818 |
| levir_mci | 0.8217 | 0.7219 | 0.6360 | 0.5621 | 0.3889 | 0.7420 | 1.3380 | 0.3297 |
| second_cc | 0.6307 | 0.4748 | 0.3703 | 0.2953 | 0.2366 | 0.5266 | 0.7789 | 0.2491 |

这些数值来自 `experiments/card_baseline_test_summary.json`，是本文档所有 delta 的默认比较基线。它们各自仅对应 1 次 CARD 锁定运行；在补齐 CARD seeds 1111、2222、3333 前，只能作为审计基线，不能替代多 seed 对照均值和样本标准差。SECOND-CC 旧论文汇总中的 checkpoint 7000 与这里的锁定 checkpoint 9000 不一致，最终论文必须使用后者。

## 5. 历史阶段锁定结果总览

| 数据集/方案 | seeds | B4 delta | CIDEr delta | SPICE delta | 提升指标数 | 分析 |
|---|---:|---:|---:|---:|---:|---|
| levir_cc / 7.6 locked | 1 | -0.0753 | -0.0021 | +0.0714 | 2/8 | SPICE 增益不能抵消 BLEU 系列下降，历史阶段候选失败。 |
| levir_mci / 7.6 locked | 3 | -0.0047 | +0.0039 | +0.0067 | 6/8 | 大多数指标改善，但 BLEU-4 未复现单 seed 优势，结论仍为混合。 |
| second_cc / current MCI-transfer cross-attn | 3 | +0.0107 | +0.0351 | +0.0052 | 8/8 | 三 seed 均值全面提升；需用 scratch 对照分离迁移收益。 |
| second_cc / 7.6 legacy locked | 1 | -0.1265 | -0.4190 | -0.1010 | 0/8 | 与最新复现实验冲突，属于旧配置/评估链路失败结果，不作为最终模型证据。 |

## 6. 每项论文实验结果分析

| 数据集 | 实验 | B4 | CIDEr | SPICE | 相对审计基线 | 结论说明 |
|---|---|---:|---:|---:|---|---|
| levir_mci | `levir_mci_card_baseline` | 0.5621 | 1.3380 | 0.3297 | 参考基线，不判定增益 | LEVIR-MCI 的审计基线；所有该数据集改进实验应与它比较。 |
| levir_mci | `levir_mci_card_mask_loss` | 0.5565 | 1.3288 | 0.3302 | 4/8 提升，不支持整体优于基线 | 仅增加 mask loss；BLEU-4、CIDEr 下降，只有 SPICE 极小提升，不支持整体有效。 |
| levir_mci | `levir_mci_card_semantic_loss` | 0.5834 | 1.3507 | 0.3168 | 7/8 提升，但证据混合 | BLEU-4 和 CIDEr 提升，但 SPICE 明显下降；说明语义辅助有潜力，但目标不平衡。 |
| levir_mci | `levir_mci_card_mask_semantic` | 0.5644 | 1.3532 | 0.3417 | 8/8 提升，历史阶段结果支持 | 单 seed 的 8 项指标均高于 CARD，是最强单次结果；三 seed 复现后优势未完全保持。 |
| levir_mci | `levir_mci_card_mask_semantic_pd05` | 0.5590 | 1.3390 | 0.3404 | 6/8 提升，但证据混合 | partial detach 改善 SPICE，但 BLEU-4 低于基线；属于混合结果。 |
| levir_mci | `levir_mci_card_mask_semantic_pd05_noreweight` | 0.5590 | 1.3390 | 0.3404 | 6/8 提升，但证据混合 | 与 pd05 数值完全相同，不能作为独立增益证据；需检查配置/权重是否实际生效。 |
| levir_mci | `levir_mci_card_mask_semantic_pd05_reweight` | 0.5853 | 1.3506 | 0.3189 | 7/8 提升，但证据混合 | BLEU-4 较强，但 SPICE 下降；feature reweight 不适合作为统一模型的固定组件。 |
| levir_mci | `levir_mci_ours_weak_coupled_final` | 0.5590 | 1.3390 | 0.3404 | 6/8 提升，但证据混合 | 与 pd05/noreweight 数值完全相同，且 BLEU-4 下降；不能作为最终统一模型证据。 |
| second_cc | `second_cc_card_rgb_baseline` | 0.2837 | 0.7793 | 0.2541 | 旧参考基线，不用于最终比较 | 旧论文汇总使用 checkpoint 7000；最终比较应改用锁定 CARD checkpoint 9000。 |
| second_cc | `second_cc_card_semantic_aux` | 0.2804 | 0.7915 | 0.2601 | 4/8 提升，不支持整体优于基线 | CIDEr、SPICE 略升但 BLEU-4 下降；单独语义辅助不足以证明全面改进。 |
| second_cc | `second_cc_card_semantic_crossattn` | 0.3126 | 0.8314 | 0.2605 | 8/8 提升，历史阶段结果支持 | 单次结果在主要指标上最均衡，是 RSACA 统一方案的直接结构依据。 |
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

历史阶段可以写：提出残差语义交叉注意力改进；统一 RSACA 在 LEVIR-MCI 的 scratch 三 seed 锁定均值 8/8 指标高于 CARD；其在 LEVIR-CC 和 SECOND-CC 呈现明确的指标权衡；消融表明 hard gate 和 feature reweight 会造成指标权衡。

历史阶段不能写：统一模型已在三个数据集全部优于 CARD；所有指标均显著提升；改进完全来自模型结构；LEVIR-CC 或 SECOND-CC 已验证成功；已完成 `3 x 2 x 3 = 18` 次完整主实验。

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
- `experiments/unified_rsaca/summary.json`：统一 RSACA scratch 三 seed 矩阵结果；存在时作为历史阶段统一结论的直接依据。
- `experiments/p1_rsaca_20260914/paired_matrix_r2/summary.json`：P1 严格 CARD/whole-adapter RSACA 配对 `3 x 2 x 3` 矩阵；这是历史阶段该阶段主实验结果。
- `experiments/reliability_sparse_rsaca_v1_new_masks_20260821/summary.json`：新 LEVIR-CC 共识掩码候选的三数据集三 seed 锁定汇总，仅作为候选负结果证据。
- `experiments/reliability_sparse_rsaca_v1_levir_cc_new_masks_matched_control_retry_20260826/summary.json`：新 LEVIR-CC 掩码匹配对照的三 seed 锁定汇总，仅作为掩码效果的受限负结果证据。
