# RSACA 局部位置实验：完整训练、选点与测试结果

报告依据：2026 年 11 月 9–11 日服务器运行记录；原始训练源码为 V2 提交 `28dfce3c601076e21e170f5fab4d11fc92a54e6e`。本文档是在实验结束后补写，不改变该源码身份。

## 结论与边界

- 2 个数据集 × 3 个实验臂 × 3 个种子，共 18 次训练全部完成 10,000 步；每次有 1,000 至 10,000 步的 10 个唯一验证点，18 个验证选点和 18 项锁定测试评分均完成。
- 技术审计对两个数据集均为 complete；冻结记录存在且 validation_admitted=true。验证集预注册候选晋级规则：SECOND-CC 通过，LEVIR-MCI 未通过。
- 测试集 E2−E1：LEVIR-MCI 的五指标均值中仅 SPICE 上升，其余四项下降；SECOND-CC 的五项均值全部下降。SECOND-CC 的验证晋级没有转化为测试均值改善，不能用测试结果改选 checkpoint。
- 每个数据集只有 3 个配对种子；报告均值与样本 SD，不声称统计显著性，也不将本轮结果与历史协议混合。语义图来源仍记为 unknown，不能宣称官方标注来源已核实。

## 协议与运行环境

| 项目 | 记录 |
|---|---|
| 协议 ID | `rsaca_local_position_20261109` |
| 训练源码 | 分支 V2；`28dfce3c601076e21e170f5fab4d11fc92a54e6e`；V1 基线 `995471be45517d9e6894b429f3a6cd48981b4bc3` 为祖先提交 |
| 项目目录 | `/root/autodl-tmp/z1zy1_rsaca_v2_20261109` |
| 本地完整产物根目录 | `/root/autodl-tmp/z1zy1_rsaca_v2_20261109/experiments/server_20261109_1848/rsaca_local_position_20261109` |
| Python / PyTorch / CUDA | `/root/miniconda3/bin/python`；3.8.20 / 1.10.0+cu113 / 11.3 |
| GPU / Java | NVIDIA GeForce RTX 4090 D；OpenJDK 11.0.27 |
| 运行映射 | `CUDA_VISIBLE_DEVICES=0`；`PYTORCH_GPU=0`（进程内逻辑 GPU 0）；`NUM_WORKERS=8` |
| 单价与费用 | 服务器单价未获可核实记录，费用为 unknown；未购买额外资源 |

源码入口：[run_rsaca_position.py](../scripts/run_rsaca_position.py)、[rsaca_position_protocol.py](../utils/rsaca_position_protocol.py)、[semantic_controls.py](../utils/semantic_controls.py)。本 worktree 与其上级未找到 AGENTS.md；本 V2 checkout 未找到 docs/RSACA_MODEL_IMPROVEMENT_PLAN.md，因此没有将这两份缺失文件当作已读依据。

## 数据与输入预检

本报告中的 second_cc 实际对应服务器现有的 SECOND-CC-AUG 数据条件；不得将它解读为未经增强的 SECOND-CC。

| 数据集 | 数据根目录 | 特征根目录 | train / val / test | 预检缺失 | 测试 ID 数 | 来源状态 |
|---|---|---|---:|---:|---:|---|
| levir_mci | `/root/autodl-tmp/z1zy1/LEVIR-MCI-dataset` | `/root/autodl-tmp/z1zy1/LEVIR-MCI-dataset/features` | 6815 / 1333 / 1929 | 0 | 1929 | unknown |
| second_cc | `/root/autodl-tmp/z1zy1/SECOND-CC-AUG` | `/root/autodl-tmp/z1zy1/SECOND-CC-AUG/features` | 8438 / 1190 / 1227 | 0 | 1227 | unknown |

| 数据集 | 原图空变更 | 有效非空邻居 | 下采样后空邻居 | 重复 split/ID |
|---|---:|---:|---:|---:|
| levir_mci | 4978 | 4925 | 174 | 0 |
| second_cc | 2974 | 7750 | 131 | 0 |

预检绑定了每个 split/ID 的双时相视觉特征、对应源 RGB 的 A/B 时相与 256×256 尺寸，以及词表、caption/参考和标签文件。LEVIR-MCI 使用同 split/ID 的 label 差异图；SECOND-CC 使用同 split/ID 的 sem/A 与 sem/B。合法空语义图未被填零替代或删样本。输入清单保存在 `experiments/server_20261109_1848/rsaca_local_position_20261109/inputs_levir_mci.json` 与 `experiments/server_20261109_1848/rsaca_local_position_20261109/inputs_second_cc.json`。

两份清单的 source_kind 均为 unknown，source_evidence 为“user declaration; verify source-generation provenance”。SECOND-CC 原图到增强样本的分组关系未核实；几何预检确认 split/ID/phase/size，但增强几何来源未核实。以上是结果解释的边界，目录名称本身不构成官方标注来源证据。

## 设计、训练与复核

| 实验臂 | 语义输入 | local_position | 作用 |
|---|---|---|---|
| E0 | 关闭 | none | 纯视觉 CARD 基线 |
| E1 | 当前 RSACA cross-attention | none | 无显式局部位置 |
| E2 | 与 E1 相同 | sincos2d_kv | 仅打开局部 KV 位置编码 |

E1/E2 的注册配置差别仅为位置开关；同种子初始化与其余模型配置经过冻结审计。全部从新初始化训练，init_checkpoint 为空，没有使用历史权重暖启动。固定 Adam（初始学习率 0.0002，weight_decay=0）、500 步 warmup + cosine 调度；训练/验证/测试 batch 分别 32/64/1，8 workers，单 GPU 顺序执行。语义辅助与 mask loss 权重均为 0；沿用锁定门控、legacy_post_norm、稀疏邻居和解码器配置，不在测试后更改。

GPU smoke 在独立临时目录执行：E1/E2 初始参数一致；两种位置模式的 70 个中间量均有限；空行 Z 等于同次前向的 Q；混合批的非空邻居有效；视觉输入、视觉分支参数和解码器参数梯度均非零。原始记录：`/tmp/rsaca_position_gpu_smoke_20261109.json`。

2026-11-11 的轻量测试复核：单进程整组调用退出码 137（系统终止，具体原因未从系统日志证实；该容器 memory.max 为 2 GiB）；按文件以及将 semantic_controls 分为五项一批后，113 项均通过、0 项失败或跳过。未修改 CPU 位置阈值。

18 次训练的 total_training_seconds 之和为 18.208 小时；单 GPU 顺序运行，约对应同口径 GPU 时。首轮开始 `2026-11-09T20:02:12+08:00`，末轮结束 `2026-11-10T14:16:03+08:00`。测试评分记录从 `2026-11-10T16:01:50+08:00` 至 `2026-11-11T00:14:28+08:00`，整个测试进程约 8 小时 15 分钟；测试耗时包含 CPU 预处理和重复输入核验，不等同于 GPU 利用时数。

| 数据集 | E0 训练时长（h） | E1 训练时长（h） | E2 训练时长（h） |
|---|---:|---:|---:|
| levir_mci | 0.894 | 2.526 | 2.633 |
| second_cc | 0.992 | 5.624 | 5.539 |

### 逐项完成状态与训练耗时

以下产物目录均相对于上文的本地完整产物根目录；训练耗时取各 run_summary.json 的 total_training_seconds。

| 数据集 | 实验臂 | Seed | 最终步 | 训练耗时（分钟） | 测试评分 | 产物目录 |
|---|---|---:|---:|---:|---|---|
| levir_mci | E0 | 1111 | 10000 | 18.3 | 完成 | E0/E0_levir_mci_seed1111 |
| levir_mci | E0 | 2222 | 10000 | 17.1 | 完成 | E0/E0_levir_mci_seed2222 |
| levir_mci | E0 | 3333 | 10000 | 18.2 | 完成 | E0/E0_levir_mci_seed3333 |
| levir_mci | E1 | 1111 | 10000 | 47.5 | 完成 | E1/E1_levir_mci_seed1111 |
| levir_mci | E1 | 2222 | 10000 | 51.1 | 完成 | E1/E1_levir_mci_seed2222 |
| levir_mci | E1 | 3333 | 10000 | 52.9 | 完成 | E1/E1_levir_mci_seed3333 |
| levir_mci | E2 | 1111 | 10000 | 49.3 | 完成 | E2/E2_levir_mci_seed1111 |
| levir_mci | E2 | 2222 | 10000 | 55.4 | 完成 | E2/E2_levir_mci_seed2222 |
| levir_mci | E2 | 3333 | 10000 | 53.2 | 完成 | E2/E2_levir_mci_seed3333 |
| second_cc | E0 | 1111 | 10000 | 19.6 | 完成 | E0/E0_second_cc_seed1111 |
| second_cc | E0 | 2222 | 10000 | 19.5 | 完成 | E0/E0_second_cc_seed2222 |
| second_cc | E0 | 3333 | 10000 | 20.4 | 完成 | E0/E0_second_cc_seed3333 |
| second_cc | E1 | 1111 | 10000 | 114.6 | 完成 | E1/E1_second_cc_seed1111 |
| second_cc | E1 | 2222 | 10000 | 118.8 | 完成 | E1/E1_second_cc_seed2222 |
| second_cc | E1 | 3333 | 10000 | 104.0 | 完成 | E1/E1_second_cc_seed3333 |
| second_cc | E2 | 1111 | 10000 | 117.4 | 完成 | E2/E2_second_cc_seed1111 |
| second_cc | E2 | 2222 | 10000 | 110.1 | 完成 | E2/E2_second_cc_seed2222 |
| second_cc | E2 | 3333 | 10000 | 104.8 | 完成 | E2/E2_second_cc_seed3333 |

## 验证选点与权重身份

仅在验证集上，以 BLEU-4、METEOR、ROUGE-L、CIDEr、SPICE 五指标计算等权对数分数：score = 五项 log(max(metric, 1e-12) / reference) 的均值；参考值依次为 0.4375、0.3377、0.6942、1.2299、0.2607。分数越大越好；同分取更早步。每行 SHA-256 是实际选中 checkpoint 文件的摘要，并与选点、预测和评分身份记录核对；未拼接不同 checkpoint 的“最佳指标”。

| 数据集 | 实验臂 | Seed | 选中步 | 权重 SHA-256 |
|---|---|---:|---:|---|
| levir_mci | E0 | 1111 | 8000 | `fe5fca0e69c90b0e319de6923b84d21a9d80a646ed807c36506df6e68293fa65` |
| levir_mci | E0 | 2222 | 9000 | `015189a602bf99febed4f6604301a3124f905b8a14caf4d87d644a3aff87d67f` |
| levir_mci | E0 | 3333 | 6000 | `9d5cdcdce868a9b69faeaf544e140471424c2afec666f338d71d7c34063e82a0` |
| levir_mci | E1 | 1111 | 8000 | `87d1254c38c9f46ed3c270aea8f6c3ac251bd5d3f54a01100d7f69ed74c5c521` |
| levir_mci | E1 | 2222 | 9000 | `901201e1353d23ca0e1802840a4e447e8e48612ff98b8e6b739c4247b4dc6379` |
| levir_mci | E1 | 3333 | 8000 | `c4dd36cc7e71afa616378e5c180f2937daddef7bf504185b6a9272916b7dc13f` |
| levir_mci | E2 | 1111 | 8000 | `299d20411447ebcf86ebd27504857e14e02cdf4814fa9526771474a45ec0e711` |
| levir_mci | E2 | 2222 | 7000 | `1a433968f2c7b17ba3f3086695ecac4b7a97c79fce08c78e6ef25808afaac27a` |
| levir_mci | E2 | 3333 | 9000 | `f4820dd3c348e8f416194f4f9eb9cbcc49f24e68ccf65ca742f12324b3f5fec2` |
| second_cc | E0 | 1111 | 10000 | `86a4d862bb162ecaf1d1b29c742d5f76810d5ea6a9150de3e49d744e3ab33610` |
| second_cc | E0 | 2222 | 9000 | `4e64a632cc116767a979e0644625ace0c085b72853950e02112f1ce065eebbd5` |
| second_cc | E0 | 3333 | 9000 | `c3a9f114c6e20c7df517fbe6659d384cf21dc203bf9cba6f99d3cb74665b129a` |
| second_cc | E1 | 1111 | 8000 | `21f528087195d01412d5b187ef0eb11ea86a90549f6161d87c62c07b1ecea4fa` |
| second_cc | E1 | 2222 | 10000 | `3fff3e1a411cb2c1cfcf584026b9787a8f8b4f1091ffab0123d110b85ed5cd59` |
| second_cc | E1 | 3333 | 8000 | `31d280c716dad4daaa196de5b157a3b3f2eb36a2d10a0feaf1e95bbee70fd634` |
| second_cc | E2 | 1111 | 8000 | `77fb20da42453254d0ef44f8b1da7dc45aa8dd98c0fa2ea3888f8cd4e130a968` |
| second_cc | E2 | 2222 | 10000 | `8f73cc75633aa5785b97689c1d84a9a6015a21c3a5161e91917b23b12a332d0f` |
| second_cc | E2 | 3333 | 8000 | `2b3e2990a941705c525f20fdf72218751c8162ab6615b0cc6b90a560c0f5cf42` |

### 所选 checkpoint 的五项验证指标

| 数据集 | 实验臂 | Seed | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---|---:|---:|---:|---:|---:|---:|
| levir_mci | E0 | 1111 | 0.565376 | 0.386477 | 0.745523 | 1.365650 | 0.346430 |
| levir_mci | E0 | 2222 | 0.567984 | 0.392215 | 0.743994 | 1.365423 | 0.339192 |
| levir_mci | E0 | 3333 | 0.578454 | 0.386518 | 0.737931 | 1.340314 | 0.329405 |
| levir_mci | E1 | 1111 | 0.569647 | 0.396631 | 0.755493 | 1.376856 | 0.348723 |
| levir_mci | E1 | 2222 | 0.548448 | 0.385197 | 0.741033 | 1.358948 | 0.337607 |
| levir_mci | E1 | 3333 | 0.602585 | 0.408248 | 0.760469 | 1.404416 | 0.329978 |
| levir_mci | E2 | 1111 | 0.576293 | 0.394703 | 0.753788 | 1.378312 | 0.347303 |
| levir_mci | E2 | 2222 | 0.547944 | 0.385673 | 0.740891 | 1.355989 | 0.347719 |
| levir_mci | E2 | 3333 | 0.570600 | 0.398046 | 0.753019 | 1.389467 | 0.353044 |
| second_cc | E0 | 1111 | 0.306546 | 0.242951 | 0.537699 | 0.797536 | 0.246047 |
| second_cc | E0 | 2222 | 0.327839 | 0.249499 | 0.544563 | 0.817077 | 0.235327 |
| second_cc | E0 | 3333 | 0.317098 | 0.245161 | 0.538804 | 0.789772 | 0.234505 |
| second_cc | E1 | 1111 | 0.300046 | 0.252854 | 0.563647 | 0.846448 | 0.273551 |
| second_cc | E1 | 2222 | 0.338183 | 0.258877 | 0.557985 | 0.860641 | 0.245496 |
| second_cc | E1 | 3333 | 0.328595 | 0.249925 | 0.553804 | 0.829301 | 0.243112 |
| second_cc | E2 | 1111 | 0.301455 | 0.253737 | 0.563126 | 0.861970 | 0.278693 |
| second_cc | E2 | 2222 | 0.338832 | 0.256420 | 0.552646 | 0.840221 | 0.243465 |
| second_cc | E2 | 3333 | 0.333433 | 0.256551 | 0.563432 | 0.858959 | 0.256815 |

## 锁定测试集结果

测试在全部技术冻结的 18 项上运行，包括验证退步种子；每项仅使用本行对应的所选权重。LEVIR-MCI 与 SECOND-CC 的每个测试结果分别覆盖 1,929 和 1,227 个 ID；同数据集各臂/种子的参考 SHA 与样本 ID SHA 一致，测试评分身份的 checkpoint SHA 与上表逐项一致。正式测试集评分是在后续用户授权后执行的。

| 数据集 | 实验臂 | Seed | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---|---:|---:|---:|---:|---:|---:|
| levir_mci | E0 | 1111 | 0.552281 | 0.384099 | 0.740618 | 1.330604 | 0.344452 |
| levir_mci | E0 | 2222 | 0.543269 | 0.383972 | 0.739025 | 1.319656 | 0.338569 |
| levir_mci | E0 | 3333 | 0.567752 | 0.385019 | 0.739344 | 1.312865 | 0.329538 |
| levir_mci | E1 | 1111 | 0.569497 | 0.398541 | 0.762661 | 1.385565 | 0.355730 |
| levir_mci | E1 | 2222 | 0.554451 | 0.394234 | 0.752512 | 1.371125 | 0.345025 |
| levir_mci | E1 | 3333 | 0.577574 | 0.400249 | 0.753774 | 1.366643 | 0.331565 |
| levir_mci | E2 | 1111 | 0.563220 | 0.393796 | 0.757034 | 1.367857 | 0.353227 |
| levir_mci | E2 | 2222 | 0.543796 | 0.389098 | 0.750759 | 1.357282 | 0.354256 |
| levir_mci | E2 | 3333 | 0.555846 | 0.395587 | 0.752663 | 1.375717 | 0.354843 |
| second_cc | E0 | 1111 | 0.300658 | 0.240129 | 0.529387 | 0.775253 | 0.241167 |
| second_cc | E0 | 2222 | 0.314834 | 0.242725 | 0.529625 | 0.768237 | 0.225474 |
| second_cc | E0 | 3333 | 0.310922 | 0.238987 | 0.525738 | 0.768165 | 0.227528 |
| second_cc | E1 | 1111 | 0.297009 | 0.248226 | 0.552981 | 0.841158 | 0.273377 |
| second_cc | E1 | 2222 | 0.328865 | 0.254738 | 0.551990 | 0.829556 | 0.240718 |
| second_cc | E1 | 3333 | 0.327309 | 0.250604 | 0.548992 | 0.831087 | 0.237586 |
| second_cc | E2 | 1111 | 0.288100 | 0.242439 | 0.545376 | 0.825694 | 0.265333 |
| second_cc | E2 | 2222 | 0.327268 | 0.251050 | 0.548599 | 0.819460 | 0.238441 |
| second_cc | E2 | 3333 | 0.321806 | 0.250149 | 0.545600 | 0.820527 | 0.244866 |

### 三种实验臂的测试均值（三种子）

| 数据集 | 实验臂 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---|---:|---:|---:|---:|---:|
| levir_mci | E0 | 0.554434 | 0.384363 | 0.739663 | 1.321042 | 0.337520 |
| levir_mci | E1 | 0.567174 | 0.397675 | 0.756316 | 1.374444 | 0.344107 |
| levir_mci | E2 | 0.554287 | 0.392827 | 0.753485 | 1.366952 | 0.354109 |
| second_cc | E0 | 0.308805 | 0.240614 | 0.528250 | 0.770552 | 0.231390 |
| second_cc | E1 | 0.317728 | 0.251189 | 0.551321 | 0.833934 | 0.250560 |
| second_cc | E2 | 0.312392 | 0.247879 | 0.546525 | 0.821893 | 0.249547 |

E1 相对 E0 在两个数据集的五项测试均值均较高。E2 相对 E1 的差异按相同 dataset 和 seed 配对，见下表；这不意味着 E2 在未测数据或不同来源语义图上仍会有相同排序。

## 配对差值：E2 − E1

下列差值保持原始 scorer 标度。每项均值为三个种子配对差的算术均值；样本 SD 使用 ddof=1；正向数为差值严格大于零的种子数。负值、退步种子和不同选中步均原样保留。

### 验证集

| 数据集 | 指标 | 1111 | 2222 | 3333 | 均值 | 样本 SD | 正向数 |
|---|---|---:|---:|---:|---:|---:|---:|
| levir_mci | Bleu_4 | +0.006646 | -0.000504 | -0.031985 | -0.008614 | 0.020553 | 1/3 |
| levir_mci | METEOR | -0.001928 | +0.000476 | -0.010202 | -0.003885 | 0.005601 | 1/3 |
| levir_mci | ROUGE_L | -0.001705 | -0.000143 | -0.007450 | -0.003099 | 0.003848 | 0/3 |
| levir_mci | CIDEr | +0.001456 | -0.002960 | -0.014949 | -0.005484 | 0.008489 | 1/3 |
| levir_mci | SPICE | -0.001421 | +0.010112 | +0.023067 | +0.010586 | 0.012251 | 2/3 |
| second_cc | Bleu_4 | +0.001409 | +0.000649 | +0.004837 | +0.002299 | 0.002231 | 3/3 |
| second_cc | METEOR | +0.000883 | -0.002457 | +0.006626 | +0.001684 | 0.004594 | 2/3 |
| second_cc | ROUGE_L | -0.000521 | -0.005339 | +0.009628 | +0.001256 | 0.007640 | 1/3 |
| second_cc | CIDEr | +0.015522 | -0.020420 | +0.029657 | +0.008253 | 0.025818 | 2/3 |
| second_cc | SPICE | +0.005143 | -0.002032 | +0.013703 | +0.005605 | 0.007877 | 2/3 |

预注册候选晋级条件为：E2 的机制布局和旁路通过、E1 的旁路通过、CIDEr 平均差大于 0、至少 2/3 种子的 CIDEr 差大于 0、SPICE 平均差不小于 0。SECOND-CC 满足条件；LEVIR-MCI 的 CIDEr 平均差为负且仅 1/3 种子为正，因此不晋级。

### 测试集（仅描述，不反向用于选点）

| 数据集 | 指标 | 1111 | 2222 | 3333 | 均值 | 样本 SD | 正向数 |
|---|---|---:|---:|---:|---:|---:|---:|
| levir_mci | Bleu_4 | -0.006276 | -0.010655 | -0.021729 | -0.012887 | 0.007964 | 0/3 |
| levir_mci | METEOR | -0.004745 | -0.005136 | -0.004663 | -0.004848 | 0.000253 | 0/3 |
| levir_mci | ROUGE_L | -0.005626 | -0.001753 | -0.001112 | -0.002830 | 0.002443 | 0/3 |
| levir_mci | CIDEr | -0.017708 | -0.013843 | +0.009074 | -0.007492 | 0.014476 | 1/3 |
| levir_mci | SPICE | -0.002503 | +0.009231 | +0.023278 | +0.010002 | 0.012908 | 2/3 |
| second_cc | Bleu_4 | -0.008909 | -0.001597 | -0.005503 | -0.005336 | 0.003659 | 0/3 |
| second_cc | METEOR | -0.005787 | -0.003688 | -0.000455 | -0.003310 | 0.002686 | 0/3 |
| second_cc | ROUGE_L | -0.007605 | -0.003391 | -0.003392 | -0.004796 | 0.002433 | 0/3 |
| second_cc | CIDEr | -0.015465 | -0.010097 | -0.010559 | -0.012040 | 0.002975 | 0/3 |
| second_cc | SPICE | -0.008044 | -0.002277 | +0.007280 | -0.001013 | 0.007740 | 1/3 |

测试上 LEVIR-MCI 的 E2−E1 平均 BLEU-4 为 −0.012887，SPICE 为 +0.010002；SECOND-CC 的 BLEU-4 为 −0.005336、CIDEr 为 −0.012040。SECOND-CC 的验证晋级与测试退步并存，应一并报告，不能事后换点或只保留上升种子。

## 机制复验、冻结与剩余限制

12 个 E1/E2 选中权重均有 position_mechanism.json，真实加载所选训练后 checkpoint；12/12 的 finite、空行 Z=Q、非空邻居与 bypass 检查为真。E2 的布局响应 6/6 通过，最大绝对响应范围约 0.003736–0.023817，阈值 1e-7。E1 关闭位置编码，布局响应不是其晋级条件：5/6 记为 false；SECOND-CC/2222 记为 true（响应约 3.01e-6），不据此声称 E1 具有有效的位置机制。

协议审计 `experiments/server_20261109_1848/rsaca_local_position_20261109/audit.json`：protocol_audit_passed=true，errors=[]；冻结记录 `experiments/server_20261109_1848/rsaca_local_position_20261109/frozen.json`：validation_admitted=true，身份摘要 `4a2425b6ece34afb9b990f5ffcca2ed6ad50f74ef1cb7b81a1824794ad64f64f`。技术完整性与性能晋级是两个不同判断；冻结不等于 E2 在两个数据集均胜出。

测试 scorer 确实生成了 BLEU/METEOR/ROUGE-L/CIDEr/SPICE，Java 解析器运行成功；但 runtime/scorer_identity 的包元数据对 pycocoevalcap、pycocotools 显示 unavailable。测试使用服务器既有 PYTHONPATH overlay 中的实现，精确发行版号未由该身份记录解析，因此跨环境字节级复现仍需保留这份环境和源码。

本报告提交到 Git 的仅是 Markdown 汇总；原始 protocol、输入清单、验证曲线、checkpoint、预测、日志及逐项身份文件继续保存在上述服务器产物根目录，未因文档发布而覆盖或重新训练。语义来源生成链与 SECOND-CC 增强分组关系仍未核实；费用因服务器单价未知而不能核算。

## 原始记录定位

- 全局：`experiments/server_20261109_1848/rsaca_local_position_20261109/protocol.json`、`experiments/server_20261109_1848/rsaca_local_position_20261109/inputs_levir_mci.json`、`experiments/server_20261109_1848/rsaca_local_position_20261109/inputs_second_cc.json`、`experiments/server_20261109_1848/rsaca_local_position_20261109/audit.json`、`experiments/server_20261109_1848/rsaca_local_position_20261109/frozen.json`。
- 单项模板：`experiments/server_20261109_1848/rsaca_local_position_20261109/{arm}/{arm}_{dataset}_seed{seed}` 下的 best_snapshot_p1.json、run_summary.json、position_mechanism.json（E1/E2）、test_metrics.json、test_prediction_identity.json、test_score_identity.json。
- 运行日志：`/tmp/rsaca_position_train_20261109.log`、`/tmp/rsaca_position_test_20261110.log`；测试进程退出码 0。
