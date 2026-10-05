# PLAN-v3 实施与验证记录

实施基准：origin/V1 `01c27adf916447d187335f449ff4fe0cdf9d1241`；独立分支 `codex/v1-evidence-workflow`。B 为唯一代码修改者；等待 C 独立复审、A 验收，主会话随后提交/推送。

交付：独立 validation-only 登记/receipt/阶段 CLI；实际工件及旧新协议声明对照；固定预测评分、完整 val 解码、必要环境训练、独立盲化包、开发常数 g/单因素候选及匹配 36 矩阵入口。旧协议 runner/lock/15门槛未改。新模型选项默认保持原行为和 state 项。

关键边界：P1c 仅必要，actual initialization 摘要不一致在更新前拒绝；P1d 不要求评分/训练前置；P1b 的模型/配置/全val ID核验和派生输出目录隔离；P1a 同样检查全val集合，不能注册小子集冒充复评。新协议从低层禁止 test。训练阶段只复用旧10000步、1000间隔及旧选点规则。

实际测试工作目录：独立worktree根目录。每个测试进程使用默认 Python 3.9，并只读复用已有 `D:\实验\CARD\tmp\p1_controls_testenv\Lib\site-packages`（torch 2.8.0+cpu、pytest8.4.2）；未安装/修改全局环境。

```powershell
$env:PYTHONPATH='C:\Users\zhangyi\.codex\worktrees\v1-evidence-workflow\CARD;D:\实验\CARD\tmp\p1_controls_testenv\Lib\site-packages'
python -m pytest -q tests/test_semantic_followup.py tests/test_semantic_controls.py tests/test_semantic_control_admission.py tests/test_semantic_diagnostics.py tests/test_p1_checkpoint_integrity.py
python -m pytest -q tests/test_semantic_followup.py -k subset
git diff --check
python scripts/run_semantic_followup.py --help
python -m py_compile utils/semantic_followup.py scripts/run_semantic_followup.py models/CARD.py train_card_spot.py utils/config_validation.py utils/semantic_controls.py utils/semantic_control_audit.py scripts/select_best_snapshot_p1.py
```

首轮结果：合并回归 **86 passed, 1 skipped**（138.05秒），随后仅追加的全val子集拒绝测试 **1 passed, 17 deselected**（4.25秒）。一个skip为Windows缺少symlink权限，应在Linux补该已有用例；4条warning为已有torch attention mask类型弃用提示。diff --check、CLI help、py_compile均通过。

C预审第1轮指出两项边界并已修复：C-R3-01将scorer_error/parse_error/blocked/未知/空结果归为FAILED，只有真实评分match/different可作为完成证据；新用例验证receipt、P1c与CLI均阻断失败。C-R3-02补只读stage预检，给出缺失/非法参数、实际前置receipt状态及可构造命令；不写入、不导入模型/数据。对应修复后的 `python -m pytest -q tests/test_semantic_followup.py` **29 passed**（74.31秒），diff --check通过。旧相关4个文件此前69 passed、1 skipped，本轮修复没有改变它们的旧协议路径；C将对最终冻结版本独立复跑合并回归。

新测试覆盖小张量常数门值0/.25/1公式/梯度/空图、dense有效局部值访问及ignore屏蔽、默认state键兼容、四臂共享初始化、新协议test拒绝、合成文件/checkpoint的receipt生命周期、条件跳过P1c、独立P1d、P1b路径隔离、差异不放行、来源/输出/环境篡改拒绝、无副作用dryrun和具体预检反馈、待填常数阻断、初始摘要不同比较阻断、P3匹配36定义、P1a子集拒绝及评分失败不放行。

测试范围：评分器/完整decoder的阶段生命周期采用明确的合成替身；不代表COCO指标或真实解码已经跑过。P2/P3验证配置工厂、共享初始化、旧训练/选点回归，未启动整个真实训练矩阵。

NOT_RUN：真实服务器评分/推理/训练/正式test、实际新数据/初始张量的远端核验、人类错误标注、环境因果实验。P0实际摘要核验仍不能证明未取得的原始初始化张量。评分第三方缓存需服务器核查；本次不把输出隔离宣传为完整历史环境复现。未知科研参数在示例中保留待填。

建议提交摘要：`Add registered validation-only semantic follow-up workflow and matched gate controls`。
冻结清单与SHA256见同目录 `semantic_followup_snapshot.json`，不包含该清单自身。提交/推送由主会话在审查通过后完成。
