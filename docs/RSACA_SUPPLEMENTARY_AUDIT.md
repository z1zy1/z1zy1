# CARD + RSACA 本地审查与预算

## 来源与技术状态

核验日期为 2026-11-25。原主工作树位于 recode，HEAD 为 d48cc5d24d0a3c84175e80a704929d6772da9e39，含作者的未跟踪材料。V2 工作树 HEAD 为 28dfce3c601076e21e170f5fab4d11fc92a54e6e。开发分支 codex/rsaca-paper-supplement 从 V1 的 995471be45517d9e6894b429f3a6cd48981b4bc3 建独立工作树，origin 为 z1zy1/z1zy1。论文基准读取自 2026-11-17 修订版 source，核对 experiments、methodology、appendices、results_discussion 及关联表。打开 11 月 2 日原始运行包与 11 月 3 日 final3 的运行映射、预测审计和选中预测。O 的轻量档案另见 b7c14537 的对应 results 子目录。

| 系列 | 训练源码 | 历史工件 | 本次统计 |
|---|---|---|---|
| R2 | fc9e7096d473bc9cb4e7682f7287a9597c45f87c | 历史 test 18 个评分 | 历史 test 15/15，保持独立 |
| O | 443d9ca11aed443f8a66d8cf72088c349af2c673 | 四模型 val 36 个验证选点 | 11/15 |
| S | 534f22fcd69294890152e1ae69a56bdf4838405a | 四模型 val 36 个验证选点 | 14/15 |

S/O 独立复算选点、验证指标和验证预测的 ID 及内容身份。两批各 36 个验证选点和选中预测吻合。2026-11-03 的服务器审计记录两批各 36 个选中权重当时存在且与记录匹配。此轮本机使用的是档案中的候选 sidecar 与审计记录，72 个权重本体严格加载、官方评估前重放与正式冻结均待服务器核验。未把本地证据包缺少大文件计为服务器权重缺失。测试输入以原清单的实际路径和内容为准。

S→O 的 Git 差异为四个诊断、编排或证据收集文件，模型、数据集、训练主入口相同。36 对同名 S/O 运行的存档初始参数摘要相符。两组 LEVIR 参考文件的字节哈希不同，排序后 50,385 条 (ID, caption) 相同，各自评估继续使用训练锁绑定的原参考文件。两套视觉缓存须分别对原清单核验，O 的 10,077 对样本仅 159 对缓存的两期哈希均相同。S/O 的数值差异成因待实测历史环境重放。

历史 GPU 环境存档为 GeForce RTX 4090 D。全文只读审查与本地 CPU 验证已实施。正式训练、官方测试、真实 Java 评分链及 GPU 测量安排在服务器阶段。

## 矩阵与预算

| 比较系列 | 计划训练键 | 有原训练记录 | 可直接复用已核权重 | 当前待服务器核验 | 需安排新训练 | 已正式测试 |
|---|---:|---:|---:|---:|---:|---:|
| R2 历史 | 18 | 18 | 独立历史证据 | 本轮不复测 | 0 | 18 历史 |
| O 补充四模型 | 36 | 36 | 0 | 36 | 0 已确认 | 0 |
| S 补充四模型 | 36 | 36 | 0 | 36 | 0 已确认 | 0 |
| Dense Local | 9 | 0 | 0 | 0 | 9 | 0 |
| No Global | 9 | 0 | 0 | 0 | 9 | 0 |
| 新系列共享完整 RSACA | 9 | 0 | 0 | 0 | 9 | 0 |
| new-full 备选 | 54 | 0 | 0 | 0 | 54 | 0 |

O 的 36 次训练总计 16.47 GPU 小时，S 为 18.26 小时，均含周期验证。O 按 LEVIR-CC、LEVIR-MCI、SECOND-CC-AUG 的历史中位每运行 0.34、0.35、0.62 小时。S 对应 0.34、0.36、0.91 小时。估算复用同类卡、缓存布局和加载速度，变体以完整 RSACA 的历史每条件用时为代理。

| 方案 | 训练预估 GPU 小时 | 十点权重峰值估算 | 正式解码样本对次 |
|---|---:|---:|---:|
| 复用 O 或 S 整组 | 0 | 现有选中 2.22 GiB | 每系列 61,020 |
| 两项消融，具备 9 个匹配的历史对照 | 9.77–10.31 | 新增 11.60 GiB | 消融新增 30,510 |
| 两项消融加同系列完整对照 | 14.66–15.46 | 新增 17.40 GiB | 45,765 |
| 同系列重建四模型与两项消融 | 26.24–28.57 | 新增约 33.83 GiB | 91,530 |

十点权重峰值按历史 checkpoint 规模推算，另加输入缓存、预测、评分器缓存、日志、哈希临时 I/O 及失败尝试。单套历史 36 次如保留全十点为 22.23 GiB。当前缺目标机型的存储吞吐、GPU 单价及正式端到端小样计时。额外预算可按每系列样本对次除以实测解码吞吐，加 36 次评分 wall time、一次验证重放，以及审计和准入实际读取字节除以目标存储吞吐计算。SECOND 的历史训练时间波动较大。总账按租用时长乘目标卡单价，单价与上游地图生成成本待核实。

## 参数复核和本地验证

用本地真实词表/H5 绑定接口重新构造六模型。LEVIR 两条件词表 860，SECOND 词表 956。下列总数也等于 requires_grad 参数数，固定门控例外见运行文档。

| 条件 | CARD | Plain Fusion | RSACA、Fixed Gate、Dense Local、No Global |
|---|---:|---:|---:|
| LEVIR-CC | 14,878,509 | 16,729,902 | 17,256,239 |
| LEVIR-MCI | 14,878,509 | 16,729,902 | 17,256,239 |
| SECOND-CC-AUG | 14,956,557 | 16,807,950 | 17,334,287 |

RSACA 相对 CARD 的下游增量为 2,377,730，三个条件均匹配论文附录。已生成 O 验证矢量统计图，并实际对齐和渲染 O LEVIR-MCI val seed 1111 四模型样例 val_000752.png。样例图的实际输入图核过档案哈希，包含五条参考及四条描述。

CPU 审核由 scripts/run_semantic_tests.py 启动。Windows 下用同一 D 盘上的 basetemp 覆盖历史相对路径测试。fixture 的正式解码与 COCO 评分使用模拟实现。CPU 比较使用固定输入和共享权重，覆盖原模型及两项单因素消融，真实历史 S/O checkpoint 前向和全验证重放待服务器执行。

## 交付文件

- scripts/run_rsaca_supplement.py、utils/rsaca_supplement.py 接入历史导入、训练注册、冻结、评估及恢复。
- scripts/run_semantic_controls.py、scripts/select_best_snapshot_p1.py、utils/semantic_controls.py、utils/semantic_control_audit.py、train_card_spot.py、test_card_spot.py 更新既有训练、选点和底层准入。
- utils/rsaca_supplement_report.py、scripts/plot_rsaca_supplement.py 输出配对统计和矢量图。
- scripts/measure_rsaca_supplement.py、models/transformer_decoder.py 提供参数审计及可控推理计时。
- scripts/visualize_paper_cases.py、scripts/rsaca_supplement_cases.py 输出同系列四模型案例。
- tests/test_rsaca_supplement.py、tests/test_semantic_controls.py、scripts/run_semantic_tests.py 覆盖 CPU 回归。
- docs/SEMANTIC_CONTROLS_AUTODL.md、docs/EXPERIMENT_CLAIM_REFERENCE.md、docs/semantic_followup.md、scripts/update_experiment_claim_reference.py、utils/semantic_diagnostics.py 更新准入引用和生成文档。
- [服务器运行说明](RSACA_SUPPLEMENTARY_EXPERIMENTS.md) 与 [待核事项](../review/open_issues.md)。

绘图开发参考 Kassis 等人的 Scientific Agent Skills 中的 Matplotlib 流程。本地开发由 Codex 协助。
