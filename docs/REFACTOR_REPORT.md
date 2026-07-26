# CARD 实验基础设施渐进式重构报告

## 1. 重构前审查

### 实际结构

训练与验证都集中在 train_card_spot.py，测试推理集中在 test_card_spot.py。配置由 configs/config_transformer.py 与 YAML 合并；数据通过 datasets/datasets.py 分派；三数据集当前主要共享 rcc_dataset_transformer_levir.py；CARD 与 DynamicSpeaker 分别位于 models/CARD.py 和 models/transformer_decoder.py。

项目在重构前已经具有 resolved config/hash、train JSONL、validation CSV、多个 selector、7.5/7.6 strict manifest 和 locked Test，并不是空白训练框架。

### 实际问题

1. Logger 是手写 print/file 封装，日志格式没有时间/级别/模块。
2. train、val、test 的结构化字段分散，旧 train JSONL 与 val/test CSV 不统一。
3. 已保存 resolved config，但缺少原始配置副本、完整命令、环境、Git 和运行状态。
4. 训练失败没有统一 run status；异常 traceback 不保证落盘。
5. seed 设置散落在训练入口，DataLoader worker 没有公共 helper。
6. 旧 checkpoint 字段兼容逻辑在 train/test 中重复，且没有统一可检查的 canonical view。
7. train/test CLI 重复且缺少安全的统一入口。
8. 通用汇总脚本面向固定论文清单，不适合递归扫描任意 run，也不能统一保留失败 run 和自动给出多 seed 样本标准差。
9. checkpoint selector 很强，但通用主/次指标、top-k 和加权入口不直观。
10. 没有一条命令把一批训练、validation、checkpoint 重载和入口检查串起来。
11. 默认输出仍是 flat experiments/{exp_name}，旧入口可以复用已有目录；为兼容不能直接改变。
12. checkpoint 实际只保存 CARD、speaker 和 model_cfg，不支持完整 resume。
13. RewardCriterion 存在但没有 SCST 训练调用链；AMP/DDP 也未实现。

## 2. 方案

采用增强现有系统的方式，而不是创建一套平行 engine：

- 保留模型、数据、loss、optimizer、scheduler、旧 CLI、YAML、checkpoint bytes 和论文脚本；
- 抽取与数学行为无关的 metrics、seed、runtime、config validation 和 checkpoint adapter；
- 旧入口增量写新工件，同时继续写所有旧工件；
- 新 tools 入口只包装旧脚本，远程论文脚本仍直接使用旧入口；
- 通用选择器只读 validation metrics；7.5/7.6 继续使用原严格 selector/manifest；
- 文档如实区分已实现、实验性、遗留未接入和未实现能力。

没有机械创建 datasets/models/engine 新目录，因为那会要求移动大段已经验证的训练逻辑，增加数学行为漂移风险。

## 3. 修改文件

### 新增

- utils/metrics.py：标准指标名、别名和 stage-aware MetricTracker。
- utils/seed.py：Python/NumPy/PyTorch seed 与 worker seed。
- utils/checkpointing.py：旧/标准 checkpoint 字段视图与 module prefix。
- utils/experiment_runtime.py：环境/Git、结构化指标、run state 和 traceback。
- utils/config_validation.py：resolved config 合法性检查。
- tools/_legacy_entry.py：统一入口公共命令构造。
- tools/train.py：安全训练包装器。
- tools/evaluate.py：validation 推理/评分包装器。
- tools/test.py：Test 推理/评分包装器。
- tools/score_predictions.py：调用现有 score_generation。
- tools/select_checkpoint.py：原 selector 委托及通用 validation-only 规则。
- tools/summarize_experiments.py：递归汇总、失败保留、多 seed mean/sample std。
- tools/smoke_test.py：一批训练/validation/checkpoint 重载 smoke。
- tests/test_experiment_refactor.py：公共组件和工具回归测试。
- docs/EXPERIMENT_GUIDE.md。
- docs/LOG_FORMAT.md。
- docs/REFACTOR_REPORT.md。

### 修改

- configs/config_transformer.py：增加实验元信息、deterministic/benchmark 和默认关闭的 worker seed 可选项。
- datasets/datasets.py：把十二个等价分支改为显式注册映射，保持原模块、类和 DataLoader 参数。
- train_card_spot.py：接入 config validation、统一 seed、结构化指标、run state、模型摘要和 checkpoint adapter。
- test_card_spot.py：接入 config validation、结构化推理记录、异常日志和 checkpoint adapter。
- utils/experiment_tracking.py：增量保存 original/resolved config、命令、环境与 Git。
- utils/logger.py：内部迁移到标准 logging，保留类名、方法、旧文件路径和 Visdom，并写 train.log/val.log。

### 删除

无。旧入口、脚本、配置和功能均保留。

## 4. 外部接口与行为

### 保持不变

- train_card_spot.py 和 test_card_spot.py 参数继续可用；
- YAML 与末尾 KEY VALUE opts 继续可用；
- CARD/DynamicSpeaker 参数、forward、loss 和梯度分支未修改；
- optimizer 和 StepLR 默认值未修改；
- Dataset 实现、增强、split 和预处理未修改；工厂只做等价映射去重；
- checkpoint 保存字典未修改；
- val_metrics.csv 和历史 selector 输入未修改；
- 7.5/7.6 validation-only、manifest、locked Test 未修改；
- 旧 checkpoint 仍可加载。

### 新增行为

- resolved config 会接受实验元信息、deterministic/benchmark 和 seed_workers 新键；
- 非法 semantic mode、detach ratio、负 loss weight、反向 warmup 区间更早报清晰错误；
- 新训练写更多复现工件和 run_summary；
- seed_workers 默认关闭，只有显式打开时才设置 DataLoader worker seed；
- 结构化日志记录 CUDA 显存，并把非有限数值规范为 null 以保持严格 JSON；
- 新 tools/train 默认拒绝非空目录；旧入口不强制此规则；
- 缺少 Visdom 时不再阻断训练，只禁用绘图并记录 warning。

没有弃用旧参数。

## 5. 行为一致性验证

已实际执行：

1. Python py_compile 覆盖所有新增公共模块、tools、训练/测试入口。
2. python -m unittest discover -s tests -v。
3. 结果：共执行 62 项，56 项通过，6 项因本地没有 PyTorch 跳过。
4. 原 validation Pareto、validation best CIDEr、7.6 stable window、strict manifest、locked Test immutable、三种子 sample std 测试全部通过。
5. 新增 11 项重构测试全部通过，覆盖：
   - 指标别名与 stage average；
   - 旧 checkpoint 字段；
   - original/resolved config、环境/Git；
   - JSONL/CSV 合法性；
   - run state；
   - 配置合法性和数据集注册映射；
   - 严格 JSON、阶段日志与非有限数值；
   - train/evaluate/test/smoke dry-run；
   - 相对输出路径解析、输入存在性与预测覆盖保护；
   - validation-only selector；
   - 空旧 CSV 回退、损坏工件保留、失败 run 和 sample std。
6. 新 tools 在真实 LEVIR-MCI/SECOND 配置上的 dry-run 通过。
7. 7.6 四个 shell 脚本 bash -n 通过。
8. run_7_6_followup_all.sh --dry_run 通过：
   - 12 个 LEVIR-CC speaker-only 候选；
   - MCI seeds 1111/2222/3333；
   - 5 项 validation-only 选择；
   - SECOND 同源三点、1000-step 稳定窗口；
   - 停在 manifest 审计之前的 dry-run，不触发训练或 Test。
9. git diff --check 通过，仅有 Windows LF/CRLF 提示。
10. 仓库内 13 份 YAML 全部完成独立合并与合法性检查。

本地未执行：

- Dataset 真实 batch；
- CARD/DynamicSpeaker forward；
- 单 batch backward；
- torch checkpoint 保存/重载；
- CIDEr/SPICE 实际 Java/COCO 运行。

原因是本地 D:/python 环境没有 PyTorch，且用户要求完整实验仅在远程 GPU。tools/smoke_test.py 已提供远程一条样本验证命令。不能以本地 dry-run 声称模型数值等价。

数学行为一致性的代码证据是：models、Dataset 实现、loss 计算块、optimizer、
scheduler、checkpoint save payload 均未修改；数据集工厂仍导入同一模块/类并传入
同一组默认 DataLoader 参数，train/test diff 只涉及配置检查、日志/状态/摘要和
等价的 checkpoint 字段提取。完整数值一致性仍需远程 smoke 证明。

## 6. checkpoint 与 resume 边界

附件期望标准 checkpoint 包含 optimizer、scheduler、scaler、random state 等字段，但最高优先级同时要求不改变原 checkpoint 保存结果。本次选择保留原 bytes，不擅自扩充 payload。

当前 checkpoint 只有：

~~~text
change_detector_state
speaker_state
model_cfg
~~~

因此 latest/best/epoch 标准化和完整 resume 没有伪装为已完成。utils/checkpointing.py 只提供兼容读取视图；若下一阶段需要完整 resume，应以显式新格式版本和兼容模式实现，并先做远程数值验证。

## 7. 风险与未解决问题

1. 远程尚未执行新的真实一批次 smoke。
2. 不能证明重构前后 tensor 输出逐元素一致，直到远程同 seed/checkpoint/input 对比完成。
3. 旧训练文件仍然较大；进一步拆 engine 会扩大风险，应在数值基准建立后进行。
4. 默认目录仍是 flat experiments/{exp_name}；只有新 tools/train 默认防复用。
5. 完整 resume、latest.pth、统一 best.pth、optimizer/scheduler/scaler/random state 尚未实现。
6. AMP、DDP、gradient accumulation、early stopping、SCST 尚未实现。
7. 通用 weighted selector 使用各指标在候选集内 min-max 标准化；论文 locked workflow 不应随意替换已注册规则。
8. environment.txt 的 pip freeze 会增加启动时间，但不访问网络。
9. Python 日志只规范由 Logger 发出的训练/验证统计；训练文件中大量逐样本 print 仍属遗留行为，未大规模改写以避免风险。
10. Windows 控制台可能错误显示中文路径，但 dry-run JSON 中路径对象未被模型逻辑使用；远程 Linux 不受此终端编码问题影响。

## 8. 后续建议

优先在远程执行：

~~~bash
python tools/smoke_test.py \
  --config configs/levir_mci/wcsg_card_final.yaml \
  --output-dir /tmp/card_refactor_smoke_mci
~~~

验证通过后，再考虑带格式版本号的新 checkpoint manager 和 engine 拆分。7.6 正式训练仍按 experiments/CARD_MODEL_EXPERIMENT_REFERENCE.md 的锁定流程执行。
