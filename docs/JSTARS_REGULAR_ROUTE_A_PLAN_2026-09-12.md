# CARD＋RSACA：J-STARS Regular 后续实验方案

分析日期：2026-09-12。设备：RTX 4090D × 1。目标：9 月完成初稿，尽量在 9 月 30 日前完成必要实验。

**建议继续方案 A。优先处理验证与测试解码不一致，再确认统一协议下的 CARD／RSACA 对照；结构候选以单因素归一化调整为首选，空间语义编码只在诊断支持时进入。** 当前并未达到“三个数据集五项主要指标均提升”：按 recovery v2 结果文件复算，15 个数据集—指标组合中 12 个均值为正，负项是 LEVIR-CC 的 BLEU-4、METEOR，以及 LEVIR-MCI 的 SPICE。

本报告是分析与待实施计划，没有修改模型或训练代码，没有训练、生成预测或重新运行正式评分器。已完成的是远端核实、快照阅读、结果文件统计复算、配置与验证轨迹核对、原始论文及官方代码检索。报告中的候选不代表已经实现或有效。


运行说明：本报告与 [scripts/analyze_evidence.py](../scripts/analyze_evidence.py) 一起发布。从仓库根目录运行 `python scripts/analyze_evidence.py` 可重建本文统计；默认输出到 `experiments/analysis/route_a_20260912/`，也可用 `--output-dir` 指定独立目录。脚本仅依赖Python标准库，读取已归档结果，不执行训练、推理或正式重评分。文中统计输出文件在运行后生成，不需要原先Windows临时目录。完整训练仍按本文阶段与准入条件实施。

## 1. 最新代码、结果状态与可复用证据

### 1.1 本次依据

- 仓库：[z1zy1/z1zy1](https://github.com/z1zy1/z1zy1)。远端实际分支是 **V1**。
- 本次获取并读取的提交：**84b6f36332fb617393ad4116026e68470a787ad4**，提交时间 2026-09-10 18:43:06 +0800。
- 当前工作区仍为 recode；本次没有切换或覆盖它。本次依据的代码快照：[V1 源码与证据目录](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/README.md)。后文源码链接固定到这一审计提交；在服务器可按对应仓库路径读取并比较新版本。
- 相比上次的 `64be6e134fc3ecf6c251333cecfa26712b9af84d`，本次已包含 recovery v2 工件。`models/`、`datasets/`、`train_card_spot.py`、`test_card_spot.py` 在两个已提交版本间没有差异；部分配置、启动包装和评分导入路径发生变化。这不证明历史 dirty 工作区与任一提交完全相同。
- 统计来源为 18 个 `test_paired_recovery_seeded_result.json` 的 `metrics` 字段。**没有使用包装文件中的 `baseline`、`deltas` 或 `ALL_TEST_METRICS_ABOVE_BASELINE`，因为它们不是本轮同 seed CARD 对照。**

来源：[最新 recovery 台账](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/experiments/audit/recovery_v2/paired_matrix_recovery.json)、[示例测试结果](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/experiments/paired_card_rsaca_recovery_v2/card/card_levir_cc_seed1111/test_paired_recovery_seeded_result.json)、[复算程序](../scripts/analyze_evidence.py)。

### 1.2 哪些状态已更新，哪些不能混算

| 证据集合 | 本次读取到的状态 | 用途与限制 |
|---|---|---|
| 9 月 8 日历史注册表 | 165 个运行：完成 103、部分完成 29、失败 1、未知 32；243 条评估中 135 条标为重复 | 历史清单，不是最新 recovery 完成率；不能将评估次数当独立训练次数 |
| recovery v2 | 18/18 arm 为 `VERIFIED_COMPLETE`，9/9 配对已观察到，`qualified_pairs=0` | 训练/验证选择/测试记录完成；源码与历史内容证据仍有限 |
| recovery v2 预测完整性报告 | 18 条均为 `PASS_IDS_AND_CAPTIONS` | 远程审计记录称 ID、覆盖和非空检查通过；本次没有逐预测重新检查 |
| `paired_locked` 与 `paired_recovery_seeded` | 同一批训练可能对应不同评估记录 | 复评不是新增 seed；本报告只使用后者 |
| 旧 unified、whole-gate retry | 常见比较为三 seed 候选对单次 CARD | 可解释历史探索，不能拼入本轮三 seed 配对主表 |
| MCI-transfer 历史组 | 原审计登记 18 个迁移运行 | 不与 scratch 结构消融合并 |
| 已选择 checkpoint 的路径 | 一部分仍指向旧 whole-gate/v1 目录，部分指向 recovery 目录 | 路径迁移不等于新训练；必须结合 checkpoint 内容和运行来源去重 |

来源：[历史注册表](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/experiments/audit/experiment_registry.json)、[评估注册表](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/experiments/audit/evaluation_registry.json)、[完整性报告](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/experiments/audit/recovery_v2/prediction_integrity_report.json)。

`docs/CURRENT_VERIFIED_RESULTS.md` 同时保留早期“只有 1/9 配对”和后续“9/9 配对”的内容；`docs/recovery_v2/CURRENT_VERIFIED_RESULTS.md` 也不是最终台账的完整替代。本次采用最终 recovery JSON 与实际结果文件，不再安排“恢复缺失的 16 个 arm”。仓库外 BEST_UNIFIED_RSACA_CONFIG.md 中的单次 CARD 比较和 `paper_balanced` 描述也不代表本轮实际选点规则。

### 1.3 dirty 的具体影响与复用决定

18 个训练 `git_info.txt` 均记录 commit `64be6e…`、dirty=True。修改列表确实包括三个动态配置、`scripts/_run_paper_training.sh`、`scripts/run_unified_rsaca_experiments.sh`、`utils/eval_utils_spot.py` 等执行相关文件，不能再假设只有文档或输出变化。另一方面，模型与训练主体没有出现在这些已跟踪修改项中，这是部分有利证据，不能扩大成完整源码已还原。

当前协议的源码摘要只列出部分执行依赖，遗漏 decoder、部分数据加载代码及测试入口等，见 [source_digest](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/scripts/run_paired_card_rsaca_matrix.sh#L80)。本次快照没有找到历史完整 patch、完整训练源码快照、checkpoint、原始图像/语义图、`sc_results.json` 或逐样本 SPICE 输出。SHA 与路径清单可以帮助追溯，不能代替缺失文件本身。

**复用策略：**

1. 现有结果和验证 CSV 直接用于描述性统计、问题定位、参数和耗时估计。
2. 远程仍存在的 checkpoint 可用于统一解码后的验证诊断，不必为这一步重新训练。
3. 9 月 13 日前给源码追溯一个有限窗口：若找到运行时 patch/快照，按完整依赖链复核；不能仅更改 dirty 标记。
4. 若仍无法建立兼容性，最终主表使用新的冻结协议训练 B 与 C0/最终候选。这个决定同时解决来源、初始化、解码与选择规则问题，**不是仅因 dirty 就重跑 18 次**。
5. 即使找回历史源码，若更改训练期间验证方式影响随机数流，也不能将历史训练与新训练混成一个严格受控的结构比较。

## 2. 三数据集五项指标的现状与目标差距

以下均为原始 scorer 尺度，未乘 100；CIDEr 不限于 0–1。`±` 是三次训练的样本标准差，差值为 RSACA−CARD。配对仅指相同 seed 标签，历史公共初始化与源码匹配仍有上述限制。

| 数据集 | 指标 | CARD 均值±SD | RSACA 均值±SD | 配对差值均值±SD | 提升seed数 |
|---|---|---:|---:|---:|---:|
| LEVIR-CC | BLEU-4 | 0.572697 ± 0.012601 | 0.560247 ± 0.014113 | -0.012450 ± 0.026610 | 1/3 |
| LEVIR-CC | METEOR | 0.392443 ± 0.003056 | 0.390553 ± 0.004490 | -0.001891 ± 0.007347 | 2/3 |
| LEVIR-CC | ROUGE-L | 0.739745 ± 0.003505 | 0.742681 ± 0.001471 | +0.002936 ± 0.004216 | 3/3 |
| LEVIR-CC | CIDEr | 1.330769 ± 0.010226 | 1.344532 ± 0.006087 | +0.013763 ± 0.013086 | 2/3 |
| LEVIR-CC | SPICE | 0.313952 ± 0.019842 | 0.337276 ± 0.010420 | +0.023324 ± 0.030215 | 2/3 |
| LEVIR-MCI | BLEU-4 | 0.584544 ± 0.018534 | 0.610764 ± 0.005447 | +0.026219 ± 0.013414 | 3/3 |
| LEVIR-MCI | METEOR | 0.396046 ± 0.004580 | 0.412265 ± 0.001475 | +0.016219 ± 0.003362 | 3/3 |
| LEVIR-MCI | ROUGE-L | 0.740188 ± 0.003543 | 0.758551 ± 0.000419 | +0.018363 ± 0.003367 | 3/3 |
| LEVIR-MCI | CIDEr | 1.337229 ± 0.011113 | 1.387842 ± 0.004664 | +0.050613 ± 0.014631 | 3/3 |
| LEVIR-MCI | SPICE | 0.304757 ± 0.011018 | 0.296095 ± 0.002829 | -0.008662 ± 0.008801 | 0/3 |
| SECOND-CC-AUG | BLEU-4 | 0.292357 ± 0.009340 | 0.307620 ± 0.005681 | +0.015264 ± 0.014251 | 2/3 |
| SECOND-CC-AUG | METEOR | 0.237414 ± 0.002662 | 0.250742 ± 0.000124 | +0.013327 ± 0.002656 | 3/3 |
| SECOND-CC-AUG | ROUGE-L | 0.529501 ± 0.003336 | 0.549884 ± 0.003376 | +0.020382 ± 0.005616 | 3/3 |
| SECOND-CC-AUG | CIDEr | 0.782647 ± 0.016431 | 0.825328 ± 0.012742 | +0.042680 ± 0.015673 | 3/3 |
| SECOND-CC-AUG | SPICE | 0.247038 ± 0.011374 | 0.257843 ± 0.010515 | +0.010805 ± 0.017042 | 2/3 |

逐 seed 配对差值：

| 数据集 | seed | BLEU-4 Δ | METEOR Δ | ROUGE-L Δ | CIDEr Δ | SPICE Δ |
|---|---:|---:|---:|---:|---:|---:|
| LEVIR-CC | 1111 | -0.017711 | +0.000815 | +0.007798 | +0.024598 | +0.039038 |
| LEVIR-CC | 2222 | +0.016397 | +0.003721 | +0.000719 | +0.017465 | -0.011510 |
| LEVIR-CC | 3333 | -0.036037 | -0.010207 | +0.000290 | -0.000775 | +0.042444 |
| LEVIR-MCI | 1111 | +0.040602 | +0.019336 | +0.020186 | +0.061749 | -0.017608 |
| LEVIR-MCI | 2222 | +0.014051 | +0.012656 | +0.014477 | +0.034042 | -0.000013 |
| LEVIR-MCI | 3333 | +0.024005 | +0.016665 | +0.020426 | +0.056049 | -0.008365 |
| SECOND-CC-AUG | 1111 | +0.027973 | +0.016289 | +0.018294 | +0.036284 | -0.003275 |
| SECOND-CC-AUG | 2222 | -0.000144 | +0.011158 | +0.026743 | +0.060539 | +0.029751 |
| SECOND-CC-AUG | 3333 | +0.017962 | +0.012535 | +0.016110 | +0.031218 | +0.005939 |

完整逐运行数值及来源：`experiments/analysis/route_a_20260912/per_seed_metrics.csv`（运行复算脚本后生成）；结构化统计：`experiments/analysis/route_a_20260912/paired_statistics.json`（运行复算脚本后生成）。

**当前结论：**

- LEVIR-CC 有两项均值低于 CARD。CIDEr、SPICE 平均收益不能掩盖 BLEU-4 和 METEOR 代价；BLEU-4 配对标准差还大于其平均损失幅度。
- LEVIR-MCI 四项均值、逐 seed 均为正；SPICE 三个 seed 均为负，是最一致的负向现象。
- SECOND-CC-AUG 五项均值为正；BLEU-4、SPICE 各有一个 seed 为负，因此只达到了当前描述性均值目标，不能称每次训练均改善。
- 目前没有做置信区间或显著性检验。这 15 个均值不能证明“全面、显著、跨独立地理域优越”。

最终达标必须在冻结配置下重新评估：每个数据集五项平均差值均大于零。当前绝对 CARD 数值只作诊断参考，不能作为新协议训练时追逐的固定测试阈值。

## 3. 已确认的问题与低成本诊断

### 3.1 先解决验证与测试解码差异

**代码事实：** [训练验证入口](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/train_card_spot.py#L1632) 调用 `speaker.sample(encoder_output)`；[decoder 默认参数](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/models/transformer_decoder.py#L227) 是 `sample_max=0`，走 `torch.multinomial`。而 [测试入口](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/test_card_spot.py#L474) 明确传 `sample_max=1`，采用贪心解码。配置中的 `sample_max`、`beam_size` 字段不能代替对实际调用路径的核实。

**记录事实：** 18 个选点记录均使用 `paper_balanced_no_spice`。从各自 `best.metrics` 复算的验证均值差如下：

| 数据集 | BLEU-4 Δ | METEOR Δ | ROUGE-L Δ | CIDEr Δ | SPICE Δ |
|---|---:|---:|---:|---:|---:|
| LEVIR-CC | +0.011532 | +0.005686 | +0.003798 | +0.008138 | +0.001782 |
| LEVIR-MCI | +0.009234 | +0.000784 | +0.007239 | +0.023105 | +0.001244 |
| SECOND-CC-AUG | +0.007233 | +0.008745 | +0.017321 | +0.037929 | +0.008728 |

来源：[示例选点记录](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/experiments/paired_card_rsaca_recovery_v2/rsaca/rsaca_levir_cc_seed1111/best_snapshot_for_paper.json)，其余按相同目录结构读取。所有 15 项验证均值为正，而测试仅 12 项为正。**这支持优先排查协议，但不能把转负全部归因于随机解码；训练源码来源、验证/测试分布、单次生成噪声等仍需区分。**

当前 scorer 使用固定参考值归一化，no-SPICE 选点权重为 CIDEr 0.35、BLEU-4 0.25、METEOR/ROUGE-L 各 0.20，SPICE 仅可能在并列排序中出现。这与五项主要目标没有完全对应。见 [选择函数](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/scripts/select_best_snapshot_for_paper.py#L136)。检查全部有效历史验证行后，未发现选中点被另一点在五项上同时支配，因此不是简单换一个“已全面更好”的旧点即可解决。

**D0 诊断设计，零新增训练：**

- 第一批读取 18 个已选 checkpoint，仅在验证集以相同贪心解码重新生成和评分，保留旧记录。
- 在 CC、MCI 的 seed1111 上，各固定 B/C0 同一 checkpoint，对同一批验证样本比较三次独立随机采样与一次贪心输出；记录重复生成方差、描述长度、对象/动作遗漏。生成随机 seed 单独命名，不能计为训练 seed。
- 若排序变化或重点负项的诊断仍不清楚，再验证这些运行的相邻 checkpoint；对最终要复用的训练轨迹，按统一预注册规则补全相同的 1000…10000 步验证网格，不能只给候选挑有利点。
- 不在这一阶段复跑或比较新的测试结果。后续真正采用新选点协议时，B/C0/候选同时适用，标记为新协议。
- 首个数据集记录验证生成时间、评分时间、SPICE 冷/热缓存时间，更新后文预算。

### 3.2 其他诊断，按证据触发

| 现象或风险 | 代码/记录证据 | 尚不能确定的部分 | 低成本验证与决策 |
|---|---|---|---|
| CC 的语义融合可能主要反映类别计数与覆盖率 | 二值变化被映射为 class 6；语义 K/V 无位置编码；local token 对同一类别相同 | 这种信息限制是否造成描述错误 | 固定 query、类别及变化数，在 14×14 网格置乱语义位置；测融合与输出差。再看人工错误是否集中于位置/小目标；二者均支持时才进入 P |
| 小变化可能消失 | 类别图在 embedding 前用 nearest 下采样，change mask 也独立 nearest | 原始非空、下采样全空的真实比例 | 按原始变化面积统计非空变空、前后覆盖率；固定样本检查。未统计前不归因于“小目标丢失” |
| 极小 gamma 不意味着当前融合接近恒等 | whole gate + legacy post-norm 对完整 query 做 LayerNorm | 实际残差幅度是否过大、是否与错误相关 | 固定 checkpoint 测 r=‖fused−query‖F/(‖query‖F+ε)，并记录 gamma、gate、coverage；gamma=0 与 gate=0 作路径诊断 |
| partial detach 的含义被旧文档扩大 | `partial_detach_feature` 用于语义编码后的 `sem_diff_feat` | 语义编码梯度缩放是否有益 | 核查梯度路径；不能称其直接阻断 query/视觉骨干梯度。不要将 detach_gate 与此混为一项 |
| SPICE 的对象/属性/关系代价 | MCI 三 seed 测试 SPICE 都下降 | 是内容错误、表达变化还是解析问题 | 获取逐样本/子类 SPICE 与解析日志，对同参考文本核对；无法获取时明确缺失 |
| 相同 seed 未保证共享初始化及样本次序 | CARD 构造后才构造 decoder；RSACA 增加随机初始化参数；DataLoader 无独立 generator，seed_workers=False | 历史公共参数是否相同 | 检查运行初始权重；新协议使用独立公共随机初始化和采样 RNG。它们是实验控制，不是方法贡献 |
| 数据与语义输入对应待验证 | 路径配置齐备，原始数组和数据清单未上传 | 颜色映射、unknown 类、增强来源组、特征抽取权重是否一致 | 在远程按 split＋ID 抽样核对图像、语义图、caption、特征；训练/验证来源组去重 |

代码依据：[类别编码与变化选择](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/models/CARD.py#L275)、[稀疏 attention](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/models/CARD.py#L336)、[门控与归一化](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/models/CARD.py#L449)、[partial detach](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/models/CARD.py#L407)、[数据映射](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/datasets/rcc_dataset_transformer_levir.py#L430)、[构造顺序](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/train_card_spot.py#L972)、[DataLoader](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/datasets/datasets.py#L41)。

补充两项实现边界：`allow_missing_pseudo_mask=True` 不意味着当前 semantic diff 输入可以缺失，diff-only 路径缺图会抛异常；sparse 分支使用全长度 `key_padding_mask`，未物理压缩 token，因此目前不能宣称获得稀疏计算复杂度收益。空 coverage 时 gate=0 是确定规则，不是学会识别可靠性的证据。

人工诊断先做 CC、MCI 各 60 个验证图像对，按固定 seed、变化/无变化与原始面积分层抽样，匿名比较 B/C0。记录对象、动作、属性/关系、位置、遗漏、虚构、长度与重复；分层若有过采样，不能把样本错误率直接写作总体错误率。至少保留双人复核的一部分样本。SPICE 是场景图语义度量，不能独立取代遥感事实核查。[SPICE 原论文](https://arxiv.org/abs/1607.08822)

## 4. 优先结构候选：一个首选、一个有条件备选

### 4.1 N：仅调整融合归一化位置

**启动条件：** D0 后仍有验证上的实质权衡，且残差诊断表明归一化路径值得检验。若统一协议的 C0 已满足五项验证目标，先确认和消融 C0，不为增加创新点强行运行 N。

当前 C0 在 eval 模式下可写成：

`y = x + g · [LN(x + γA) − x]`

当 γ=0、g≠0 时，仍有 `g·[LN(x)−x]`。这与旧说明里的 `x + gγA` 不是同一公式。候选 N 为：

`y = x + g · γ · LN(A)`

**可证伪假设：** 将缩放与归一化限制在语义残差上，能减轻对原有视觉表示的无关改动，从而改善 CC 的 BLEU-4/METEOR 或 MCI 的 SPICE，同时保留其他目标项的收益。恒等性质本身不保证性能提高。

只修改 `model.semantic_fusion_norm_mode: legacy_post_norm → context_pre_norm`。这个功能已在 [CARD.py](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/models/CARD.py#L483) 实现；统一单因素入口、实际配置差分和新协议锁仍需后续实现。global=all_mean、gamma_init=0.01、gamma_max=0.5、gate bias=-1.5、detach_ratio=0.5、所有损失与训练配置保持不变。

不能直接运行旧 `run_reliability_sparse_rsaca_v1_prenorm_changed_global.sh` 作为 N：它还改变 global、gamma 上限、gate bias、visual gate、fallback 和 warmup。历史组合的 CC 三 seed B4 均值约 0.5905，另一次 rerun 约 0.5194，来源/条件未完成严格匹配，不能从前者断言 pre-norm 有效，也不能从后者单独否定它。见 [组合脚本](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/scripts/run_reliability_sparse_rsaca_v1_prenorm_changed_global.sh#L10)。

先在 CC/MCI 上 seeds1111、2222 各运行一次 N，共 4 次；对照为同协议 B/C0。若验证准入通过，再补 CC/MCI seed3333 与 SECOND-CC-AUG 三 seed，共增加 5 次。最大 9 次，不额外搜索 gamma/bias。若有效残差变得过小而融合几乎无作用，记录为假设未获支持，先停止，不立即用多参数搜索“补救”。

### 4.2 P：仅给局部语义 K/V 加二维位置编码

**启动条件：** N 被否定或 N 后仍有明确缺口；位置置乱诊断符合预期，且人工案例确实指出空间对应问题。单纯发现置乱不变不能证明它是性能瓶颈。

候选形式 `z_i = s_i + p_i`，其中 p_i 是固定二维 sin/cos 编码，在与视觉 query 对应的 14×14 网格上生成。只改变局部语义 K/V，global token 与 gate 的语义摘要仍由原 s_i 计算；只对有效变化 token 生效，保持 fallback 和 padding 语义。第一轮不同时加入视觉特征投影、覆盖率池化或额外损失。

改动位置为 [语义融合及 K/V 构造](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/models/CARD.py#L336)。目前没有独立位置编码开关；`semantic_fusion_local_pos_encoding` 可作为后续新增参数名，**它目前不可直接运行**。视觉主干已有位置参数，但不意味着无位置标签的语义 K/V 获得了空间对应。固定 query 时，同类语义位置置换的不变性仍成立。

P 相对于预先选定的统一父配置（C0 或已确认 N）仅增加此项。准入、4＋5 次训练流程与 N 相同。必须同时改善描述事实或主要指标；只有位置敏感性增强、attention 更好看，不算成功。若无证据，不启动 P。九月最多两种结构假设，不展开 N×P×global×loss 的组合网格。

不优先安排 detached gate：已有 CC 两 seed 验证配对中，候选 BLEU-4 均下降，CIDEr/SPICE 方向也不一致，缺乏比 N 更强的启动依据。见 [历史配对文件](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/experiments/reliability_sparse_rsaca_v2_paired_comparison.json)。

## 5. 统一配置与实验矩阵

### 5.1 当前实配及下一阶段固定项

| 类别 | 统一配置/处理 |
|---|---|
| 基础结构 | 本项目适配的 CARD＋DynamicSpeaker；差异表示维度512、8 heads、2层；外部视觉特征保持同来源 |
| C0 融合 | cross_attention；7类编码接口；sparse changed-token、all_mean global、whole-adapter gate |
| C0 数值 | dropout=0.1；gamma_init=0.01；gamma_max=0.5；gate bias=-1.5；legacy_post_norm |
| 反向路径 | semantic partial detach=0.5；detach_reliability_inputs=False |
| 主损失 | caption CE＋0.001 consistency＋0.001 independence；保留 CARD 原有约束 |
| 关闭项 | mask/semantic/relation auxiliary、content-word weighting、feature reweight、hard gate、visual consistency/fallback、fusion warmup |
| 优化 | Adam，lr=2e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0；warmup_cosine，warmup500，min_lr_ratio=0.05 |
| 训练 | scratch；10000 optimization steps；batch32；每图训练caption数1；每1000步保存/验证；seed1111/2222/3333 |
| 验证/测试 | 使用相同贪心解码、同参考集定义、相同 tokenizer/scorer；验证batch64、测试batch1先沿用，需做批大小一致性检查 |
| 随机控制 | 公共模块使用同 seed 的独立随机初始化、保存初始摘要；训练采样与验证 RNG 分离；worker seed显式设置，环境与配置固定 |
| 候选差分 | N只改norm_mode；P只加局部位置编码；其他核心项不得随数据集变动 |

来源：[CC resolved config](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/experiments/paired_card_rsaca_recovery_v2/rsaca/rsaca_levir_cc_seed1111/resolved_config.json)、[主损失](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/train_card_spot.py#L1381)、`experiments/analysis/route_a_20260912/config_variations.json`（运行复算脚本后生成）。若 feature extractor 的权重/预处理记录无法找回，需补充核对；不能仅因目录同名认定相同。预提取视觉特征的 ImageNet 预训练不等于 caption 模型从已有 checkpoint 微调。

公共初始化是为每个 seed 新生成的随机权重，不能从已训练 CARD 向 RSACA 迁移后仍称 scratch。保存公共参数 key/shape/hash，新增模块使用独立 RNG。现有 seed_workers 开关只初始化 worker，尚未提供完整的独立 DataLoader generator；这一点属于后续协议实现工作。

### 5.2 仅允许的数据适配

| 数据集 | 归档 train/val/test 数 | 实际语义输入 | 允许适配 |
|---|---:|---|---|
| LEVIR-CC | 6815/1333/1929 | pseudo_masks，二值变化转 class6 | 路径、词表、序列最大长度、二值映射 |
| LEVIR-MCI | 6815/1333/1929 | images/{split}/label，非binary diff-only语义加载 | 标签映射及路径；当前 mask_type=binary 不代表语义图被二值化 |
| SECOND-CC-AUG | 8438/1190/1227 | sem/A、sem/B 前后语义图 | 配对语义接口、颜色类别映射、词表、序列长度 |

这些计数来自运行与审计记录，本次没有重新读取原始划分。SECOND 官方 AUG 对训练与验证均做增强；需要按来源图像组核对 train/val/test 不重叠，统计区间也按来源组处理。[SECOND 原论文的数据增强说明](https://arxiv.org/html/2501.10075v1)

CC 与 MCI 不是完全独立地理域，MCI 是 CC 的扩展；应称“三种数据/标注条件验证”，不能据此宣称三个独立区域泛化。[Change-Agent 官方数据说明](https://github.com/Chen-Yang-Liu/Change-Agent)

### 5.3 完整矩阵

全部训练行默认沿用 5.1 的 10000步、统一优化器/损失、3 seed；早筛例外单列。`T_B(d)`、`T_R(d)` 是包含训练内验证的运行时间，数值见第7节。F表示验证后冻结的统一最终候选，可为C0、N或条件性P。

| ID/优先级 | 研究问题；对照→候选/唯一变化 | 数据集与seed；新增训练 | 配置、选点与评价 | 成本与复用 | 继续/停止；论文证据 |
|---|---|---|---|---|---|
| D0/P0 | 解码/选择协议是否影响观察；固定checkpoint采样→greedy | 三数据集18个已选点；重点CC/MCI seed1111重复生成 | 只用val，五项＋文本诊断；不改变旧test | 0训练；复用旧checkpoint；先测速 | 解释协议影响，决定是否需结构实验；方法评估协议 |
| D1/P0 | 输入对应、coverage、残差、位置及事实错误 | 三数据集固定诊断样本；CC/MCI各60对重点复核 | 固定eval/query；同类位置置乱、gamma/gate干预 | 0训练；依赖远程原始图、预测与权重 | 无证据则不启动对应结构候选；诊断/失败图 |
| B/P0 | 建立新协议RGB基线 | 三数据集×3seed＝9 | 关闭全部RSACA和aux；统一五项选点 | Σ3T_B；证据兼容时可减少重训 | 来源/训练失败先修复；主表对照 |
| C0/P0 | 当前设计整体是否有效 | 三数据集×3seed＝9 | 5.1当前whole-gate配置 | Σ3T_R；旧结果作pilot | 若达标先补机制证据；与B构成18个arm |
| N/P1 | 去除query归一化是否改善权衡 | CC/MCI×2seed＝4；通过后补5，总计9 | 仅norm_mode；同B/C0协议；五项/r/gamma/gate | 最多Σ3T_R量级；不得用旧组合替代 | 按第6节；N−C0归一化单因素证据 |
| D/P1 | 收益是否来自特定融合而非仅额外输入 | 最低AUG×3＝3；推荐三数据集×3＝9 | 与F同语义来源/编码/norm/detach，普通dense attention，无sparse/global/learned gate组合；保留一致空图规则 | 最低3T_R(AUG)，推荐Σ3T_R；需新训练 | 若D与F等效，收紧结构贡献；同输入主对照 |
| G/P1 | learned gate是否必要 | 最低AUG×3＝3；推荐三数据集×3＝9 | 相对F，非空图gate固定1、空图仍0，其他全同；不直接用现有gate=False代替 | 同D；需明确新开关/固定gate路径 | 无收益则删减gate主张；gate消融 |
| S/P2 | 变化位置选择是否有贡献 | AUG×3＝3 | 相对F只取消局部padding选择；global、gate输入、coverage及空图规则固定 | 约3T_R(AUG)；需实现受控路径 | 不能直接把sparse=False当纯消融，因为当前dense分支还改coverage等；sparse消融 |
| Glo/P2 | global token是否必要 | AUG×3＝3 | 相对F仅移除global，保留fallback/local/gate | 约3T_R(AUG) | 无收益则简化方法；global消融 |
| P/P2条件性 | 语义局部位置是否改善描述 | 与N相同4＋5设计，最多9 | 只加local K/V位置编码；统一父配置冻结 | 最多Σ3T_R量级；后续新增代码 | 位置敏感＋实际指标/事实同时支持才保留 |
| E/P1–P2 | 方法相对领域代表作是否有价值 | 推荐MModalCC在AUG×3；后续Chg2Cap在CC×3 | 官方协议复现与统一评分分开列；不强迫外部方法使用我方优化器 | 3T_ext；权重可先用于接口/评分核对，不能替代训练重复 | 至少一个可核对的外部比较；无兼容条件只作引用表 |

D 是整个融合组合的同输入对照，不是单个组件消融；G/S/Glo 才分别检验单个设计。D 与 F 应尽量控制容量并报告差异。当前 sparse attention 保留完整网格，不能假设 dense 更慢或省去成本测量。

如果最终加入 P，D 与其他对照也应共享相同语义 token 表达；否则无法把差异归因于 sparse/gate。已训练的旧表达D/G不能自动充当新表达的对照。最低预算中的单数据集消融仅支持该数据条件的机制结论。

## 6. 选点、统计与继续/停止规则

### 6.1 将协议修正与结构贡献分开

新协议记为 P1：统一 greedy；显式五项评分；训练/验证随机流隔离；源码/初始参数/输入清单固定。先登记规则，再运行新训练。旧 recovery 保留 P0 身份；P1−P0 不能称为 RSACA 新结构收益。配对锁目前写死 `validation_only: paper_balanced_no_spice`，即使环境覆盖选择规则也不会自动匹配，必须在后续实现中使锁文件记录实际函数和参数。[锁文件代码](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/scripts/lock_paired_card_rsaca_protocol.py#L47)

建议 P1 对所有方法采用统一五项等权对数得分：

`Q(d,k) = (1/5) Σ_m log(max(metric_m(d,k), 1e-12) / r(d,m))`

r 是在候选训练前冻结的正数参考，可用 P1 CARD 验证参考；对同一数据集，它只是常数，不改变checkpoint按五项乘积排序，也不是数据集特定超参数。不得根据测试结果改变权重。某指标缺失/解析失败应判该评分记录无效，不用0代替。并列取较早步数。所有方法用相同保存/验证网格；每个模型每seed只选一个checkpoint，不能分别为五项指标各选一个。

这是新建议，当前 selector 尚未实现。也可以采用已存在的五项 `paper_balanced`，但必须在第一轮前确定且全组统一；本报告预算按新增等权规则作为唯一推荐方案，不同时搜索多套选点函数。综合分数用于选点，**不代替最终15项逐项达标**。

### 6.2 早筛与冻结条件

1. 两seed早筛只控制资源，不作为最终稳定性结论。若候选在CC/MCI十个“数据集×指标”均值上均不低于父配置且至少一项更好，可进入补seed；或相对B的负向项数量减少、没有新增负项、两数据集Q均提高，也可进入补seed。
2. 以上均只使用P1验证。若C0在P1已无验证负项，候选必须按第一条改善，不能仅靠换取单项收益进入。
3. 完成三数据集三seed后，只有所有15项验证平均差值相对B均为正，且没有未解释的重大事实错误，才按当前完整目标冻结为最终候选。若不满足，报告缺口；可触发唯一有依据的备选P，不能开放无限搜索。
4. 同配置seed失败属于运行问题，补同seed；不要以“失败”为名丢弃低分的正常运行。禁止把落选候选的高分seed并入胜者。
5. 9月22日前结束结构搜索。若仍未满足完整目标，继续完成已冻结方案的必要证据和初稿，明确研究目标尚未达成，而不是降低目标或承诺补到全胜。
6. 最终测试后若出现任一负项，论文准确报告。历史测试已经用于开发决策，应披露这一事实；新seed、目录或P1名称不会生成真正未见的测试集。

### 6.3 不确定性与去重

每个seed保存预测、参考ID、checkpoint哈希、实际resolved config和选点轨迹。训练随机性用三seed分布描述；固定checkpoint的样本不确定性可用按图像对的paired bootstrap，AUG按原始来源组抽样。每次重算corpus指标，不能将BLEU等当作逐句得分的简单均值。

三seed可以展示方差，但不足以支撑强烈的训练总体显著性主张。需要显著性时，预先说明检验、单位、假设及多重比较处理；样本级区间不能冒充跨训练随机性的区间。平均差值>0不是正式非劣效检验，也不代表有实用价值；同时报告绝对/相对幅度及人工事实结果。

本次已发现 AUG/RSACA seed2222 的val CSV有20行，其中每个1000…10000步记录恰好重复两遍且内容相同。诊断统计按同一run/step/内容去重，不能当成20次独立验证或更多seed。

## 7. 单张4090D的三档预算与九月底排期

### 7.1 真实计时依据

18个run_summary中的`total_training_seconds`合计 **8.6613小时**。当前代码从训练初始化阶段计到训练结束，包含训练期间验证生成和同步scorer调用；不能再为同一轮内置验证重复计费。环境文件均记录4090D、Python3.8.20、PyTorch1.8.0+cu111、pycocoevalcap1.2。源码/环境仍需按实际运行复核，但这比凭经验猜“每个实验数小时”更有依据。

| 数据集 | CARD 中位/范围（小时） | RSACA 中位/范围（小时） |
|---|---:|---:|
| LEVIR-CC | 0.383 / 0.334–0.546 | 0.374 / 0.367–0.380 |
| LEVIR-MCI | 0.340 / 0.325–0.613 | 0.390 / 0.372–1.007 |
| SECOND-CC-AUG | 0.318 / 0.310–0.319 | 0.756 / 0.715–0.814 |

来源：`experiments/analysis/route_a_20260912/timing_and_selection.json`（运行复算脚本后生成）、[训练计时起点](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/train_card_spot.py#L847)、[计时终点](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/train_card_spot.py#L1705)、[4090D环境记录](https://github.com/z1zy1/z1zy1/blob/84b6f36332fb617393ad4116026e68470a787ad4/experiments/paired_card_rsaca_recovery_v2/rsaca/rsaca_second_cc_seed3333/environment.txt)。

这些是GPU占用期间的**墙钟运行时间代理**，不是GPU kernel累计时间；不包含独立最终测试、D0离线验证、原始特征提取、语义先验生成、外部模型训练或人工诊断。复评时间戳与训练结束时间相隔数天，不能拿差值当测试耗时。AUG/RSACA更慢，不能用CC的0.37小时替代全部模型。

### 7.2 三档预算

以下以历史主表来源暂不能直接复用、B9/C0九次需新训练为保守基准；若证据兼容可扣除对应运行。N不需要则不训练；N中途被否定时按实际4次等计费。F若为条件性P，额外训练单列。

| 档位 | 内部新增完整训练 | 历史中位数外推的训练＋内置验证 | 建议内部GPU占用预留 | 独立外部方法/日历安排 |
|---|---|---|---|---|
| 最低必要 | B9＋C0九次＋D3＋G3＝24次；N通过并完整确认则33次 | 12.2–16.8小时 | **35–60小时**，含约2倍运行余量和D0/测试/评分预留 | 外部先核对官方预测/权重与协议；没有匹配复现时明确缺口。约5–8个工作日，需每天6–10小时GPU窗口 |
| 推荐完整 | B9＋C0九次＋D9＋G9＋S3＋Glo3＝42次；加N则51次 | 21.3–25.9小时 | **55–90小时** | 优先额外MModalCC在AUG三seed，即另加3次、`3T_ext＋评估`。约10–15个工作日，满足下述资源不等式才排入 |
| 可选扩展 | 推荐档＋条件P最多9次＋normalization-only等有诊断依据对照3次 | 比推荐档约再加6.8小时；是同规模估计 | 比推荐档再留15–25小时 | 可再加Chg2Cap/不同seed扩展，按实测另计；不得挤占9月底冻结与写作 |

35–60、55–90小时是**规划预留而非已测结果**。可用公式为：`总占用 = Σ训练运行计时 + D0额外验证 + 最终推理/评分 + 外部复现 + 必要特征/先验生成`。内部候选只有轻量改动时才可借用上述计时；第一个完整P1训练完成后更新每数据集估计。若源码/输入核实要求重新提取全部特征，预算须单独重算。

外部MModalCC官方训练示例包含可训练的ResNet101、30 epochs等设置，与本项目预提取特征的10k步不等价，不能用0.4–0.8小时推算它。[官方训练和评估入口](https://github.com/ChangeCapsInRS/SecondCC)

9月14日前做外部方法安装/数据接口和短段测速，之后用同卡实测估算T_ext。若为9月27日前完成最终结果，则必须满足：`剩余可用GPU小时 ≥ 内部剩余预算 + 3T_ext + 外部评估 + 故障余量`。没有T_ext时不承诺推荐档全部完成；单卡任务顺序执行，CPU整理与写作可以并行。

### 7.3 日历与交付节点

| 日期 | 实验工作 | 同步写作/决策 |
|---|---|---|
| 9/12–9/13 | 取得远程原始工件、完成有限源码追溯、D0首批与数据检查；计时 | 写研究问题、数据条件、现有12/15结果及证据边界 |
| 9/14–9/16 | 实现P1协议与诊断记录，冻结新源码；B/C0对照先CC/MCI、再AUG | 写方法公式和公平比较协议；外部方法接口/计时 |
| 9/17–9/19 | 有需要时N四次早筛；同时完成同输入D的AUG三seed | 若C0已达验证目标则跳过N；整理早筛理由与失败记录 |
| 9/20–9/22 | 胜者补足三数据集三seed；只有诊断支持才进入P，9/22结束结构搜索 | 冻结F；记录15项验证是否达标；不能达标则明确缺口 |
| 9/23–9/25 | D/G扩展、S/Glo必要消融；外部方法按单卡预算排队 | 完成消融表、开销表、方法差异分析 |
| 9/26–9/27 | 冻结后最终测试、统计和案例复核；补运行失败而非追逐分数 | 锁定主表；尚未完成的结果留空，不使用预计值 |
| 9/28–9/30 | 仅补必要核对和报告一致性 | 完成初稿、图表、限制与复现材料 |

若只能每天投入2小时GPU，剩余约两周的可用时间不足以承诺推荐档；最低档也需扣除外部复现并重新核算。这是时间投入约束，不能靠同时开启多个完整训练解决。九月底能否实现15/15性能目标仍由数据决定，算力可行不等于科学目标必然成功。

## 8. 文献定位、外部比较和初稿组织

### 8.1 与现有工作的关系

本次检索截至2026-09-12，侧重原论文和官方代码，不据此声称穷尽所有工作或获得当前SOTA排行榜。

| 工作 | 已核实内容及与本项目关系 | 本轮使用方式 |
|---|---|---|
| CARD，ACL2024 | 原题为Context-aware Difference Distilling for Multi-change Captioning，研究上下文差异蒸馏；原官方仓库示例采用Spot-the-Diff | 明确是本项目适配到遥感数据的直接基线，不把它误写为原论文已在三个遥感数据集报告相同数值。[论文](https://aclanthology.org/2024.acl-long.430/)／[代码](https://github.com/tuyunbin/CARD) |
| MModalCC/SECOND-CC，J-STARS2025 | RGB与语义信息融合，并包含跨模态注意力和门控；官方代码/数据与AUG条件相关 | **最近邻、优先外部比较**。不能把“加入语义cross-attention和gate”本身作为未经比较的独创贡献。[论文](https://arxiv.org/abs/2501.10075)／[代码](https://github.com/ChangeCapsInRS/SecondCC) |
| Change-Agent/MCI，TGRS2024 | LEVIR-MCI与变化检测/描述的多任务设定相关 | 比较其中变化感知模型，注明mask是训练监督还是测试输入，不能把agent能力与caption分数混算。[官方代码与数据](https://github.com/Chen-Yang-Liu/Change-Agent) |
| Chg2Cap，TIP2023 | 有官方训练、测试和预训练权重，提供遥感RGB描述基线 | 次优先外部复现，先CC；核对特征提取、词频阈值、解码和参考集。[官方代码](https://github.com/ShizhenChang/Chg2Cap) |
| RSCaMa，GRSL2024 | 使用state-space模型研究遥感变化描述，有官方代码 | 相关工作或预算允许时的RGB代表，不替换本项目骨干。[论文](https://arxiv.org/abs/2404.18895)／[代码](https://github.com/Chen-Yang-Liu/RSCaMa) |
| HIMEC，2026-08预印本 | 研究方向性变化表示与固定解码接口，报告三seed比较；页面注明投稿TGRS、代码拟在发表时公开 | 说明最新研究已关注表示/接口一致性；列为近期相关工作，不能称已接收或默认可复现，也不能用其SECOND分数直接排名AUG。[原文页面](https://arxiv.org/abs/2608.12502) |

MModalCC的AUG采用与原始SECOND不同的训练/验证条件，官方示例beam=4，而当前本项目测试为greedy。正式比较最好同时列“论文报告值”和“本地同数据/同scorer复现值”，并清楚标记预训练、输入、训练预算和解码差异；只在核对相同设定后进行排名。

### 8.2 当前可以主张什么

当前已有证据支持：一个统一语义融合结构已在三种数据条件运行；recovery v2中CIDEr均值都提高、AUG五项均值为正；同时存在CC与MCI的特定指标代价。

尚没有充分证据支持：收益由sparse/global/gate各自造成、融合利用了准确的语义空间位置、学得可靠性、计算更高效、三数据集五指标最终全部胜出。约2,377,730个新增可训练参数使CC/MCI参数从14,878,509增至17,256,239，增加约16%；AUG从14,956,557增至17,334,287。必须把外部特征提取器和先验生成器成本另列，不能只以adapter参数量宣称端到端轻量。

建议论文问题表述为：**在CARD差异表示上，受控地引入变化语义信息，能否提升遥感变化描述，并通过同输入对照解释融合设计的必要性？** 只有N/P或已有组件消融给出明确证据后，再把相应机制写成贡献。协议修正是可信实验的基础，不是新模型的创新点。

J-STARS官方作者说明强调技术内容的新颖性、重要性，以及完整实验和实验条件描述；它并未给出“五项全升即接收”的规则。本报告的15/15来自用户研究目标，不是期刊硬性录用门槛。[官方作者说明](https://www.grss-ieee.org/publications/jstars-information-for-authors/)

### 8.3 初稿需要的证据与图表

| 初稿部分/图表 | 内容 | 截至本次状态 |
|---|---|---|
| 引言与相关工作 | CARD具体不足的假设、遥感语义输入价值、与MModalCC差异 | 可写，但不足必须写为待验证问题 |
| 方法图与公式 | RGB差异表示、语义输入、local/global/fallback、whole gate、真实norm公式 | 可按当前源码绘制；F冻结后更新 |
| 数据与协议表 | 三数据条件、输入是否含标注、AUG来源组、训练/解码/选点、随机控制 | 路径/运行配置可写，原始数据核对待补 |
| 主表 | B/C0/F每seed、均值±SD、五项配对差值 | recovery表可标探索性；P1最终表待实验 |
| 同输入与消融表 | D、G、S、Glo，必要时N或P | 待实验；负结果同样记录 |
| 协议诊断图 | 同checkpoint随机/greedy、验证轨迹与最终选点 | 待远程checkpoint诊断 |
| 机制/失败案例图 | CC对象/位置/小变化，MCI属性关系，AUG复杂变化 | 待固定样本人工核对；不能只挑成功案例 |
| 成本表 | 参数、显存、训练/推理、语义生成与特征提取 | 当前参数/训练计时可填，其他待测 |
| 讨论与限制 | 标注先验条件、CC/MCI非独立、测试开发使用、剩余负项 | 必须保留，按最终证据更新 |

## 9. 现在最先做的三件事

1. **在远程用现有checkpoint完成D0。** 优先核对验证随机采样、测试greedy的差异；取回预测、参考、逐样本SPICE和真实时间记录。保持现有结果归档独立。
2. **冻结P1公平比较协议。** 完成实际选择规则、统一解码、公共随机初始化、采样随机流和完整来源记录；先确认B/C0，避免把协议收益误判为结构收益。
3. **只启动诊断支持的一个候选。** 有归一化证据则运行N四次早筛；若C0已经满足验证目标，直接优先同输入D和gate消融。P只作有条件备选，九月不进行多模块组合搜索。

本报告保留“三数据集五项指标均提升”的完整目标。当前证据说明该目标尚未达成，但短训练计时使有限、受控的确认实验具有现实可行性。最终论文结论以新协议结果为准，任何未完成或未获支持的部分都应明确保留。
