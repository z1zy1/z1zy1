# WHU-CDC 正式实验总结

实验标识：`whu_cdc_whole_adapter_gate_v1_20261114`
结果目录：`/root/autodl-tmp/z1zy1/experiments/whu_cdc_whole_adapter_gate_v1_20261114_formal`
总结依据：六个正式 run 的预测绑定和评分结果，以及 `summary.json`。

## 1. 实验目的

本实验比较从 scratch 训练的 CARD 与 whole-adapter gate v1 RSACA 在 WHU-CDC 测试集上的变化描述生成结果。比较使用相同的三个随机种子（1111、2222、3333），并分别报告 overall、change 和 no-change 子集上的 BLEU-1、BLEU-2、BLEU-3、BLEU-4、METEOR、ROUGE-L、CIDEr、SPICE。

这里的比较单位是同一 seed 下的 CARD/RSACA 配对 run。RSACA 相对 CARD 的 paired delta 定义为：

\[
\Delta = \mathrm{RSACA} - \mathrm{CARD}.
\]

因此，正号表示 RSACA 数值高于 CARD，负号表示 RSACA 数值低于 CARD，零表示两者相同。本文只报告描述性差异，不将差异解释为统计显著性。

## 2. 数据集与实验设置

- 数据集：WHU-CDC。
- 训练样本数：5,947；验证样本数：743；测试样本数：744。
- 测试子集：change 236 个样本，no-change 508 个样本；两者合计 744。
- 训练步数：10,000；每 1,000 步进行验证、保存和候选评分。
- batch size：32；优化器：Adam；学习率：`2e-4`；scheduler：`warmup_cosine`；warmup：500 步；最低学习率比例：0.05；workers：0。
- 验证解码：greedy。
- 验证选点：`five_metric_equal_weight_log`，平分时选择较早的 step。测试集只使用验证集选出的 checkpoint。
- 评分资源：结果锁定的 Python、Java 11 和评估资源哈希记录在 `protocol_lock.json` 的 `scoring_resources` 中。
- 统计规则：每个方法每个子集使用三个 seed 的算术均值；标准差为三个 seed 的样本标准差（`ddof=1`）。paired delta 的标准差同样在三个同 seed 差值上计算。

本实验是 **whole-system comparison**。RSACA 使用冻结的外部 detector 输入，因此结果不能解释为纯粹的 gate ablation，也不能把差异单独归因于 gate 结构。

## 3. 方法与随机种子

CARD 使用 `model_identity=card`，不启用语义地图和语义 cross-attention。RSACA 使用 `model_identity=whole_adapter_gate_v1_rsaca`，其正式测试配置包括：

- `semantic_input_mode=cross_attention`；`use_semantic_maps=true`；`semantic_diff_only=true`；`semantic_diff_binary=true`；
- sparse change tokens、`global_token_mode=all_mean`、reliability gate、`reliability_gate_bias=-1.5`；
- whole-adapter gate；`semantic_fusion_norm_mode=legacy_post_norm`；
- `gamma_init=0.01`，`gamma_max=0.5`；partial detach ratio 为 0.5；
- auxiliary mask/semantic loss、hard gate、feature reweight、visual consistency gate、visual fallback 和 fusion warmup 均关闭；
- 两个方法均从空的 `init_checkpoint` 开始训练。

正式 seed 为：`1111`、`2222`、`3333`。因此共有 2 个方法 × 3 个 seed = 6 个正式 run。

## 4. 选点策略与 checkpoint

所有 checkpoint 均依据验证集的 `five_metric_equal_weight_log` 选择，未依据测试指标选择。下表中的 selection score 是 `best_snapshot_for_paper.json` 中保存的验证选点分数；checkpoint SHA256 来自 `prediction_record.json` 与 `best_snapshot_for_paper.json` 的绑定记录。

| 方法 | seed | selected step | validation selection score | checkpoint SHA256（前 16 位） |
|---|---:|---:|---:|---|
| CARD | 1111 | 7000 | 0.251831 | `50716970e04d92e7` |
| CARD | 2222 | 4000 | 0.267504 | `0a8dcfc10b3dae01` |
| CARD | 3333 | 9000 | 0.272997 | `8b855b3a3dcae3c6` |
| RSACA | 1111 | 7000 | 0.274637 | `38b45eb3b74456f3` |
| RSACA | 2222 | 9000 | 0.263702 | `d8b529602451b86a` |
| RSACA | 3333 | 9000 | 0.274126 | `efb26e3e648f204b` |

六个 `prediction_record.json` 均为 `prediction_complete`，每个包含 744 条测试预测，并且其 selected step、checkpoint SHA256 与对应的 `best_snapshot_for_paper.json` 一致。六个 run 的 `test_metrics.csv` 是空占位行；正式测试指标来自各 run 的评分日志和最终 `summary.json`。在实验目录中没有找到 `test_metrics.json`，因此本文没有从该缺失文件推测或补造任何指标。

## 5. test 与 summary 完成状态

- `stage_test.json`：`status=completed`。
- `stage_summary.json`：`status=completed`。
- `summary.json`：`status=completed`，`completed_runs=6`。
- 六个评分日志均写出 overall、changed、no_change 三组以及八个指标；每个评分日志均记录测试样本数 744、change 样本数 236、no-change 样本数 508。
- 六个 `prediction_record.json` 均完成，测试预测数量均为 744。

需要保留一个元数据说明：六个 `run_summary.json` 的 `test_inference_completed` 为 `true`，但旧字段 `test_completed` 仍为 `false`。结合已完成的 `stage_test.json`、完整 prediction record、评分日志和 `summary.json`，本文把 test 阶段判定为已完成，同时将这个字段不一致作为结果包的可复现性注意事项，而不是忽略它。

训练日志中可见向本地 Visdom 地址 `localhost:8097` 发送事件时的连接失败 traceback；这些日志同时保存了训练过程的 checkpoint，阶段状态和后续预测/评分均完成。该连接错误不改变本文使用的评分数值，但复现实验时应检查是否需要关闭或配置 Visdom 上报。

## 6. Overall 结果

下表为三个 seed 的 mean ± std。表中所有数值统一保留 6 位小数；`std` 是跨 seed 的样本标准差。

| 指标 | CARD mean ± std | RSACA mean ± std |
|---|---:|---:|
| BLEU-1 | 0.827915 ± 0.012257 | 0.833561 ± 0.012525 |
| BLEU-2 | 0.760869 ± 0.013414 | 0.768232 ± 0.017603 |
| BLEU-3 | 0.701463 ± 0.015465 | 0.713820 ± 0.023606 |
| BLEU-4 | 0.647904 ± 0.019990 | 0.666819 ± 0.032700 |
| METEOR | 0.458112 ± 0.006463 | 0.461148 ± 0.009082 |
| ROUGE-L | 0.816597 ± 0.003638 | 0.813958 ± 0.002056 |
| CIDEr | 1.562965 ± 0.009977 | 1.547152 ± 0.015316 |
| SPICE | 0.383723 ± 0.007519 | 0.362354 ± 0.027343 |

样本数为 744，seed 数为 3。

## 7. Change 与 no-change 结果

### Change 子集

| 指标 | CARD mean ± std | RSACA mean ± std |
|---|---:|---:|
| BLEU-1 | 0.633368 ± 0.017494 | 0.626750 ± 0.011261 |
| BLEU-2 | 0.485022 ± 0.015161 | 0.471181 ± 0.010730 |
| BLEU-3 | 0.358873 ± 0.009712 | 0.344947 ± 0.013477 |
| BLEU-4 | 0.267310 ± 0.003349 | 0.252486 ± 0.017231 |
| METEOR | 0.236218 ± 0.008122 | 0.231400 ± 0.006041 |
| ROUGE-L | 0.466086 ± 0.010936 | 0.462453 ± 0.012464 |
| CIDEr | 0.577773 ± 0.031575 | 0.533664 ± 0.035726 |
| SPICE | 0.188211 ± 0.010196 | 0.183059 ± 0.003889 |

样本数为 236，seed 数为 3。

### No-change 子集

| 指标 | CARD mean ± std | RSACA mean ± std |
|---|---:|---:|
| BLEU-1 | 0.968490 ± 0.006609 | 0.968360 ± 0.008717 |
| BLEU-2 | 0.962453 ± 0.007872 | 0.962562 ± 0.010108 |
| BLEU-3 | 0.958052 ± 0.008702 | 0.958241 ± 0.011450 |
| BLEU-4 | 0.953086 ± 0.009973 | 0.953960 ± 0.013056 |
| METEOR | 0.737756 ± 0.014540 | 0.738083 ± 0.018425 |
| ROUGE-L | 0.979433 ± 0.003692 | 0.977255 ± 0.006004 |
| CIDEr | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| SPICE | 0.474552 ± 0.012823 | 0.445649 ± 0.041844 |

样本数为 508，seed 数为 3。该子集在三个评分日志和 `summary.json` 中均标记为 `degenerate_reference_corpus=true`，且 `reference_set_count=1`。因此 no-change 的 CIDEr 为 0，以及 CIDEr、SPICE 等指标的相对差异，都需要谨慎解释。

## 8. Paired delta

以下为同 seed 配对的 `RSACA - CARD`。正号表示 RSACA 更高，负号表示 RSACA 更低；括号内为三个 paired delta 的样本标准差。

| 子集 | 指标 | paired delta mean ± std | 均值方向 |
|---|---|---:|---|
| overall | BLEU-1 | +0.005647 ± 0.020796 | RSACA 更高 |
| overall | BLEU-2 | +0.007363 ± 0.026185 | RSACA 更高 |
| overall | BLEU-3 | +0.012356 ± 0.033763 | RSACA 更高 |
| overall | BLEU-4 | +0.018915 ± 0.046625 | RSACA 更高 |
| overall | METEOR | +0.003036 ± 0.011790 | RSACA 更高 |
| overall | ROUGE-L | -0.002639 ± 0.004801 | RSACA 更低 |
| overall | CIDEr | -0.015814 ± 0.016284 | RSACA 更低 |
| overall | SPICE | -0.021369 ± 0.030997 | RSACA 更低 |
| change | BLEU-1 | -0.006618 ± 0.011172 | RSACA 更低 |
| change | BLEU-2 | -0.013841 ± 0.005231 | RSACA 更低 |
| change | BLEU-3 | -0.013926 ± 0.005877 | RSACA 更低 |
| change | BLEU-4 | -0.014824 ± 0.014216 | RSACA 更低 |
| change | METEOR | -0.004818 ± 0.002341 | RSACA 更低 |
| change | ROUGE-L | -0.003633 ± 0.002338 | RSACA 更低 |
| change | CIDEr | -0.044110 ± 0.012872 | RSACA 更低 |
| change | SPICE | -0.005152 ± 0.007002 | RSACA 更低 |
| no-change | BLEU-1 | -0.000130 ± 0.014878 | RSACA 更低 |
| no-change | BLEU-2 | +0.000110 ± 0.017521 | RSACA 更高 |
| no-change | BLEU-3 | +0.000190 ± 0.019770 | RSACA 更高 |
| no-change | BLEU-4 | +0.000873 ± 0.022818 | RSACA 更高 |
| no-change | METEOR | +0.000327 ± 0.031021 | RSACA 更高 |
| no-change | ROUGE-L | -0.002177 ± 0.008099 | RSACA 更低 |
| no-change | CIDEr | +0.000000 ± 0.000000 | 相同（均为 0） |
| no-change | SPICE | -0.028903 ± 0.043958 | RSACA 更低 |

按三个 seed 的均值方向看：

- overall：RSACA 在 BLEU-1/2/3/4、METEOR 上高于 CARD，在 ROUGE-L、CIDEr、SPICE 上低于 CARD。
- change：RSACA 在八个指标上均低于 CARD。
- no-change：RSACA 在 BLEU-2/3/4、METEOR 上高于 CARD，在 BLEU-1、ROUGE-L、SPICE 上低于 CARD；CIDEr 两者均为 0。

这些是均值层面的描述性方向。三个 seed 的方向并不完全一致，尤其 overall 和 no-change 的若干指标存在 seed 间反转；本文不把均值差异写成显著性结论。

## 9. 结果分析

**事实。** 六个正式 run 均完成了测试预测和评分，`summary.json` 完成了三个子集、八个指标以及三 seed 聚合。overall 的均值比较表现为 BLEU 系列和 METEOR 较高、ROUGE-L/CIDEr/SPICE 较低。change 子集的八项均值全部低于 CARD。no-change 子集的 BLEU-2/3/4 和 METEOR 略高，但 ROUGE-L 和 SPICE 较低，CIDEr 两边均为零。

**观察。** RSACA 的整体方向不能简化为“全面提升”。整体 BLEU-4 的正 delta 与 CIDEr、SPICE 的负 delta 同时存在；change 子集的 CIDEr delta 为 `-0.044110 ± 0.012872`，是该子集中绝对值最大的均值差之一。no-change 子集的若干 BLEU 差异很小，且其参考语料退化，不能据此形成强结论。

**推断边界。** 本实验只支持在固定 WHU-CDC 数据、固定外部 detector 输入、固定训练和评分协议下的观察性比较。由于 RSACA 同时改变了语义输入和 whole-adapter 机制，结果不能单独证明 gate 的因果贡献，也不能据此声称跨数据集泛化或统计显著性。

## 10. 主要结论

1. WHU-CDC 正式实验的 test 阶段和 summary 阶段均标记为完成，覆盖 CARD/RSACA 两个方法、三个 seed 和 744 个测试样本。
2. 在 overall 的三个 seed 均值上，RSACA 提高了 BLEU-1 至 BLEU-4 和 METEOR，但降低了 ROUGE-L、CIDEr 和 SPICE。
3. 在 change 子集上，RSACA 的八项均值均低于 CARD；因此当前结果不支持 RSACA 在变化样本上的全面优势。
4. no-change 子集的参考语料退化，相关 CIDEr 等指标应谨慎解释；该子集不应被用来夸大 RSACA 的效果。
5. 这是一项 whole-system comparison，冻结外部 detector 输入使其不能被解释为纯 gate ablation。当前结果支持报告受协议约束的描述性差异，不支持无条件的结构因果或显著性结论。

## 11. 局限性与可复现性说明

- **外部 detector provenance。** 当前正式结果包的 `protocol_lock.json`、各 run JSON 和汇总文件记录了 detector 生成后的 mask manifest 哈希，但没有完整呈现 detector 权重来源、训练数据/划分、验证数据/划分及独立的 no-WHU-val/test provenance 审核记录。因此本文不暗示 detector provenance 已被完整验证。
- **逐 checkpoint 历史。** 当前文件记录了每个 run 的验证选点结果、selected step、checkpoint 路径和 SHA256；但没有完整的逐 checkpoint WHU-CDC 历史结果包可供独立复核。选点依据以 `best_snapshot_for_paper.json`、`val_metrics.csv`、selection 日志和 prediction binding 为准，不从缺失历史记录推断额外结论。
- **测试指标文件形态。** 六个 run 目录都存在 `test_metrics.csv`，但内容只有表头和空占位行；未找到 `test_metrics.json`。测试数值来自 `score_*.log` 与 `summary.json`，不能把空 CSV 当作有效指标来源。
- **元数据一致性。** `run_summary.json` 中 `test_completed=false` 与 `test_inference_completed=true` 并存；阶段级状态、prediction record、评分日志和 summary 均显示测试流程完成，因此该字段不一致应在后续运行器中修正或解释。
- **no-change 退化参考语料。** no-change 子集标记为 `degenerate_reference_corpus` 且只有一个 reference set；CIDEr 恒为 0，CIDEr、SPICE 等指标的排序和差异需要谨慎解释。
- **统计范围。** 只有三个 seed，本文报告样本标准差，未进行显著性检验，也未报告置信区间。差异大小不能直接作为显著性证据。
- **训练日志。** 训练日志含有 Visdom 本地连接失败信息；虽然阶段结果已完成，复现时应确认日志上报服务配置不会改变训练流程。

## 12. 运行命令与结果路径

以下是仓库中对应的正式阶段命令形式；实际 run 的完整命令保存在每个 run 目录的 `command.txt` 和 `command_test.txt`。正式实验使用 `/root/miniconda3/envs/card/bin/python`，并以已锁定的 detector/mask、feature 和 protocol manifest 为输入。

```bash
cd /root/autodl-tmp/z1zy1
export CARD_PY=/root/miniconda3/envs/card/bin/python

bash scripts/run_whu_cdc_whole_adapter_gate_v1.sh \
  --stage preflight --python "$CARD_PY"
bash scripts/run_whu_cdc_whole_adapter_gate_v1.sh \
  --stage smoke --execute-real-smoke --python "$CARD_PY"
bash scripts/run_whu_cdc_whole_adapter_gate_v1.sh \
  --stage train --execute-real-training --python "$CARD_PY"
bash scripts/run_whu_cdc_whole_adapter_gate_v1.sh \
  --stage select --python "$CARD_PY"
bash scripts/run_whu_cdc_whole_adapter_gate_v1.sh \
  --stage test --python "$CARD_PY"
bash scripts/run_whu_cdc_whole_adapter_gate_v1.sh \
  --stage summary --python "$CARD_PY"
```

主要结果文件：

- 阶段状态：`experiments/whu_cdc_whole_adapter_gate_v1_20261114_formal/stage_test.json`、`stage_summary.json`。
- 协议锁：`experiments/whu_cdc_whole_adapter_gate_v1_20261114_formal/protocol_lock.json`。
- 最终汇总：`experiments/whu_cdc_whole_adapter_gate_v1_20261114_formal/summary.json`、`summary.csv`。
- 汇总版本：`experiments/whu_cdc_whole_adapter_gate_v1_20261114_formal/summary_versions/47f34ec8dc6fde92873661944036a6dfa3dba8345af1260c9e0b2c14e0cf40bd/summary.json`。
- CARD run：`experiments/whu_cdc_whole_adapter_gate_v1_20261114_formal/card/card_whu_cdc_seed{1111,2222,3333}/`。
- RSACA run：`experiments/whu_cdc_whole_adapter_gate_v1_20261114_formal/rsaca/rsaca_whu_cdc_seed{1111,2222,3333}/`。
- 每个 run 的关键文件：`prediction_record.json`、`best_snapshot_for_paper.json`、`run_summary.json`、`command.txt`、`command_test.txt`、`test.log`、`eval_log.txt`、`test_metrics.csv`。
- 训练、预测、选点和评分日志：`experiments/whu_cdc_whole_adapter_gate_v1_20261114_formal/logs/`，其中 `train_*.log`、`predict_*.log`、`select_*.log`、`score_*.log` 分别记录对应阶段。
- 数据与资产清单：`dataset_manifest.json`、`preprocessing_manifest.json`、`feature_manifest.json`、`data_audit.json`。

本文中的聚合数字逐项对应正式目录的 `summary.json`；`summary.json` 的 `protocol_lock_sha256` 为 `79f9644ee149e8852832450fc0eae54d3ca7419f09e3a4ea29c1f3c7390a10b6`，`scoring_version` 为 `47f34ec8dc6fde92873661944036a6dfa3dba8345af1260c9e0b2c14e0cf40bd`。
