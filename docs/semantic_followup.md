# 验证阶段的后续实验入口

`scripts/run_semantic_followup.py` 实现修订方案 P0—P3，使用独立的 `semantic_controls_followup_20261005` 协议。原 `run_semantic_controls.py`、旧锁、36 运行矩阵、1000—10000 选点网格及15/15准入规则保留。本入口没有冻结/test命令；底层配置验证和 test admission 同样拒绝新协议的 test。

本次交付是代码及 CPU/合成工件测试。真实服务器复评分、完整解码、训练和人工标注均未运行，不改变上一批11/15、34/45结论。

## 登记与执行

从仓库根目录运行；复制 `configs/semantic_followup.example.json` 到独立位置填写。null/空候选/空门值/空资源是待登记项，不是推荐科研参数。

```bash
python scripts/run_semantic_followup.py environment > actual_environment.json
python scripts/run_semantic_followup.py register --spec /abs/followup.json --output /abs/new_runs/registration.json
python scripts/run_semantic_followup.py register --spec /abs/followup.json --output /abs/new_runs/registration.json --execute
python scripts/run_semantic_followup.py stage --registration /abs/new_runs/registration.json --stage P0
python scripts/run_semantic_followup.py stage --registration /abs/new_runs/registration.json --stage P0 --execute
```

`register`、`stage` 默认只读检查/展示：不创建目录、不导入模型运行环境、不启动科研子进程；会只读 Git 文件列表及源码哈希。`environment` 显式采集 torch/numpy 等运行环境。只有 `--execute` 才写登记或运行阶段。阶段目录必须不存在，不覆盖失败/成功记录。FAILED 返回非零；DIFFERENT 是完成的差异证据，不是复现通过。

stage预检列出缺失/非法参数、前置receipt状态与能构造的评分/解码/训练命令；BLOCKED会说明待填项。READY_FOR_EXECUTION_CHECKS仅说明静态预检通过，实际runtime/完整数据解析/checkpoint内容仍在执行时验证，不能作为PASS证据。

每个阶段必须先填完自身及前置阶段参数。封存后不可修改；补证或修改科学选择使用新登记、新 output_root，并重新核验前置证据。未完成参数不能启动对应科研工作。

- `output_root`：绝对路径的新目录，登记文件位于其中；全部输入工件位于目录外。
- `environments`：每个标签填写相应环境 `environment` 命令的完整 JSON。不同阶段可使用不同已登记环境；前置 receipt 按其原环境验证。记录 Python/torch/numpy/CUDA、评分包、Java版本/可执行文件哈希与确定性环境变量。
- `artifacts`：具名映射，如 `"cc_references": {"path": "/abs/references.json", "sha256": "实际SHA256"}`。使用实际文件，不以 audit 声明代替。初始化摘要、checkpoint、配置、预测、references、ID列表、评分资源和图像均如此。
- `datasets`：每条件登记完整解析配置及 `utils.semantic_control_audit.input_manifest(cfg, source_kind)` 在实际服务器生成的产物。`source_kind` 为 annotation/prediction/unknown。执行时重新解析并比对实际文件字节和输入映射；CARD 仅省略语义输入。
- `data_environment`：显式填写工厂支持的各数据根目录/feature目录变量、PYTORCH_GPU、NUM_WORKERS；生成配置必须匹配 manifest。不会从启动 shell 偷取未登记路径覆盖。
- `expected_ids`：非空、无重复的验证 ID JSON 列表；P1a/P1b 必须覆盖完整 val split，不取交集、不用少量例子充当完整复评。

## 阶段与进入条件

| 阶段 | 实际工作和产物 | 放行规则 |
|---|---|---|
| P0 | 实际工件/输入映射重hash；checkpoint checksum/metadata/step；实际初始化摘要；可选protocol_pairs比较旧/新源码声明、配置、runtime、输入哈希声明 | 身份检查PASS。记录声明与实际文件核验分开；不推断环境因果 |
| P1a | 同一保存预测+references调用项目scorer，全部五指标与原指标按登记绝对容差比较 | 所有job真执行、ID合法、全部指标匹配 |
| P1b | 实际checkpoint/config/输入/全val ID核验后，调用test_card_spot.py --split val完整解码并逐ID比文本 | 全量输出合法且变化比例不超登记阈值；fixed-forward不算完成 |
| P1c | 仅必要时按原CARD/RSACA做环境训练及原网格选点 | P1a/P1b均PASS则程序派生NOT_REQUIRED；否则登记训练条件。初始化实际摘要不同在更新前NOT_COMPARABLE；训练完成不自动解释因果 |
| P1d | 图像对、预测、references、抽样参数齐备可独立生成盲化包与单独解盲键 | READY_FOR_ANNOTATION，标签为null，人工分类NOT_RUN；不要求P0/评分/训练receipt |
| P2 | 登记单因素候选、常数g对照；同初始化、10000-step预算、原选点网格 | P0/P1a/P1b PASS；不强制不必要P1c。全部运行/选点/初始化/预测身份合格后机器选择候选 |
| P3 | 已选候选派生CARD、同输入plain_fusion、候选与匹配fixed_gate，完整4×3×3及配对统计 | 只产生验证统计；无test准入，不能复用不匹配旧G |

其他阶段同样使用 `stage --registration ... --stage P1a --execute`。
P1c的 `initialization_references` 用 `"levir_cc/2222/card": "参考实际初始化摘要工件名"` 等键，CARD/RSACA都需提供。仅相同seed不保证跨torch初始化相同，所以更新前逐参数摘要核对；不匹配直接阻断。直接加载共同初始权重不在当前实现中。

Receipt绑定登记内容、执行源码、实际运行环境、被使用输入和全部输出SHA256，以及执行结果、耗时、子进程命令/日志。失败保留记录。依赖重新检查输入/输出并由结果计算状态，手改状态字段不能放行；这是本地可审计流水线，不是针对有权篡改全部源码与工件的攻击者提供签名认证。
复评仍不同则先解释/修正来源，再建立新可重复基线登记；一次P1c训练完成不会自动放行P2。

## 科学参数与匹配对照

执行前登记容差、解码变化阈值、开发条件/seed、候选数量、常数g列表、抽样数/seed。
本入口只支持旧10000-step预算、1000间隔五指标等权log选点，预算仍须显式填写；其他预算/规则需另行修订协议。
P2选择分数为登记条件/seed选中点五指标log分数的均值，平分按候选名词典序。只用learned-g rsaca行选候选；常数g行是机制对照。

候选格式 `{"name":"dense_access","model":{"semantic_fusion_dense_local_access":true}}`。每候选最多改变一个支持因素：dense local access，或global_token_mode为all_mean/changed_mean；`model:{}`为原基线。P2的constant_gates是预登记、不重复的[0,1]列表；P3的匹配G保持原定义非空g=1。

固定门控为 `Z=Q+g*(LN(Q+gamma*U)-Q)`，空图g=0。默认常数1不增加state_dict项。dense access只开放有效未变化局部K/V及attention access，保留global/fallback、真实coverage、门控摘要、LN/残差与空图回退，ignore位置仍屏蔽。没有把门控移到语义增量上；那是涉及LN/残差公式的另一个实验。

P1b输出采用 `P1b/decode_N/evaluation/captions/val/`，连同cfg.exp_dir/exp_name、日志、summary、attention均在独立decode_N下。P1a登记实际评分资源（SPICE jars/模型等）并在独立目录执行；目标服务器仍须检查第三方缓存位置，输出隔离不等同历史环境复现。
P1d image_manifest格式为 `{"样本ID":{"before":"前图工件名","after":"后图工件名"}}`，必须覆盖登记ID集合。

## 本地测试

```bash
python -m pytest -q tests/test_semantic_followup.py tests/test_semantic_controls.py tests/test_semantic_control_admission.py tests/test_semantic_diagnostics.py tests/test_p1_checkpoint_integrity.py
```

小张量验证门控公式/梯度/空图/ignore/shared-init/default-state；阶段生命周期使用小工件和模拟scorer/decoder，不能冒充真实效果。旧测试覆盖原协议锁/准入/选点/checksum/诊断。CPU测试不证明服务器GPU环境复现成功。
