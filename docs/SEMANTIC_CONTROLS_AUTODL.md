# AutoDL：B / D / C0 / G 语义对照协议

新协议：`p1_semantic_controls_20260915`。只通过明确阶段执行；没有自动执行全部阶段的入口。代码交付不包含完整训练结果。

## 研究问题与固定配置

| 名称 / 参数 | 定义 | 解释范围 |
|---|---|---|
| B / `card` | RGB 特征 CARD，关闭语义和辅助分支 | 基础对照 |
| D / `plain_fusion` | 同一语义编码及投影，普通 dense cross-attention；无 sparse、global/fallback token、学习门控 | D−B：加入语义信息并通过普通融合使用的整体收益；不是单因素消融 |
| C0 / `rsaca` | whole-adapter sparse RSACA、学习门控、global/fallback token | C0−D：定制融合的额外收益 |
| G / `fixed_gate` | 与 C0 相同，仅输出非空图 gate=1、空图 gate=0 | C0−G：学习门控贡献；门控 MLP 保留但不接收损失梯度 |

三数据集统一 CARD 主体、decoder 主体、post-norm、gamma 初值 0.01/上限 0.5、语义 partial detach 0.5、Adam lr=2e-4、warmup=500、cosine min_lr_ratio=0.05、batch=32、10000 步、seeds=1111/2222/3333。B 无语义分支，detach 不生效。保留 CARD 原本 caption CE 和代码中的两个 0.001 约束项；mask/semantic/relation 辅助损失关闭。完整配置由 `utils/semantic_controls.py` 的单一模板生成。

仅数据路径、接口、词表、最大文本长度适配。LEVIR-CC 使用二值 diff → unknown-change 类；LEVIR-MCI 使用 diff label；SECOND-CC-AUG 使用成对 sem/A、sem/B。数据加载器保留原来的差分类别扩展（7→8），会记录实际模型配置。输入目录不证明其来源：预检可声明 `annotation`、`prediction` 或 `unknown`，仍需保留生成方法证据。AUG 原图分组尚无法从文件名可靠核实时明确标记缺口。

每个 seed 生成随机公共 CARD 和 decoder 初值；显式按名称、形状复制公共张量。另生成规范 C0，向 D/C0/G 复制共享语义张量。全部初始化隔离 Python、NumPy、Torch/CUDA RNG。禁止训练 checkpoint 迁移。每组保存逐参数形状、dtype、SHA256，冻结时检查公共参数、decoder 和共享语义参数一致。

## 在 AutoDL 仓库根目录执行

沿用服务器现有 Python 环境。不要整体安装旧 `requirements.txt` 中失效的本地 wheel 路径。以下所有命令均从仓库根目录执行；无需移动数据。

默认目录保持 `./Levir-CC`、`./LEVIR-MCI-dataset`、`./SECOND-CC-AUG`。保留 `PROJECT_DIR`、`LEVIR_CC_ROOT`、`LEVIR_MCI_ROOT`、`SECOND_CC_ROOT` 和 `LEVIR_CC_FEATURE_ROOT`、`LEVIR_MCI_FEATURE_ROOT`、`SECOND_CC_FEATURE_ROOT`，未设置时使用原目录。内部 features、pseudo_masks、images/.../label、sem/A 和 sem/B 解析复用数据加载器。`NUM_WORKERS` 默认 8，`PYTORCH_GPU` 默认 0；锁定后环境和路径需保持一致。

实际 preflight 要求 CUDA 可用、Java 在 PATH 中，并检查完整输入；不会自动安装依赖。`--root` 只能写入项目 `experiments/` 下的独立子目录，已有历史目录不会被当成新协议根目录。`--dry-run` 不要求 CUDA/Java 或数据可用。

```bash
# 1. 先看完整 36 次矩阵；不写实验目录、不读取数据、不执行训练。
python scripts/run_semantic_controls.py --stage train --dry-run

# 2. 只读复算 R2。即使旧 test_completed=false，也以实际评分工件报告完成状态。
python scripts/run_semantic_controls.py --stage audit \
  --historical-root experiments/p1_rsaca_20260914/paired_matrix_r2

# 3. 完整输入预检和锁定。源码须已提交；默认来源 unknown，不臆测来源。
python scripts/run_semantic_controls.py --stage preflight

# 如已核实来源，在首次 preflight 可追加：
# --semantic-source levir_cc=prediction --semantic-source levir_mci=annotation
# SECOND-CC-AUG 的实际语义来源也应核实后声明。

# 4. 训练：单个进程按数据集、seed、实验组顺序运行；请勿同时开启多个入口。
python scripts/run_semantic_controls.py --stage train

# 5. 全部训练完成后，greedy 验证完整网格选点，再审计和冻结。
python scripts/run_semantic_controls.py --stage select
python scripts/run_semantic_controls.py --stage audit
python scripts/run_semantic_controls.py --stage freeze

# 6. 仅 freeze 成功后执行。可重复调用，已完成且身份匹配的评分会跳过。
python scripts/run_semantic_controls.py --stage test
python scripts/run_semantic_controls.py --stage summary
python scripts/update_experiment_claim_reference.py
```

正式 `split=test` 测试即使直接调用 `test_card_spot.py` 也会执行同一套冻结准入：必须匹配协议锁、冻结清单、实际配置、输入清单、选定 checkpoint 及其 checksum/metadata，并且输出必须位于对应冻结运行目录。冻结前、checkpoint 不匹配、配置或输入变化、身份缺失或已有工件不一致都会在创建日志/数据加载器/模型之前拒绝。已有匹配预测但尚未评分时不会重复推理；应由统一入口继续评分。验证集诊断脚本仍是独立的 `split=val` 前向诊断，不要求新协议冻结。

不要用分号或忽略退出码的方式在 freeze 失败后强行执行测试。测试入口自身也会拒绝没有冻结凭据的运行。

训练、选点、测试和汇总支持筛选，例如：

```bash
python scripts/run_semantic_controls.py --stage train \
  --datasets levir_cc --seeds 1111 --arms card plain_fusion rsaca fixed_gate
python scripts/run_semantic_controls.py --stage select \
  --datasets levir_cc --seeds 1111
```

早期可部分筛查，但 preflight、协议审计及 freeze 总是针对完整 36 次矩阵，D/G 不要求改善；C0 必须三数据集五项验证均值全部高于 B。部分汇总不标记为完整稳定性结论。缺失/失败运行保留并排查，不自动清空或从已有权重继续训练。中断后已有预测且有匹配身份记录时可只补评分；缺少身份凭据则停止并列出问题，不能自动认证历史来源。

## 选点、冻结和统计

每个运行必须恰好含 1000、2000、…、10000 的十个验证 checkpoint，使用 greedy。分数为五指标 `mean(log(max(metric,1e-12)/reference))`；固定 reference 来自原 P1，所有数据集/组共同使用，平分选较早步。拒绝缺失、非有限或负值。reference 只平移同一公式分数，不改变 checkpoint 排序，不是实际比较基线。

训练请求配置、实际运行配置、源码 commit/执行文件哈希、输入清单和摘要、命令/环境、初始参数、验证预测/参考 ID、选点、checkpoint 校验和及内部 step/config 均参与审计。冻结保存这些工件的哈希，冻结后训练及选点关闭；测试前再次核查文件与输入解析，变动会被拒绝。历史 dirty 需结合可恢复源码/patch 和双方兼容性调查，本协议不据此要求历史全部重训。

新 `summary.json` 报告每组每 seed、均值、样本标准差，以及 D−B、C0−B、C0−D、C0−G 的逐 seed 配对差、差均值和样本标准差。分别记录 `mean_goal`、`positive_seed_metrics`、`validation_admitted`、`protocol_audit_passed`。不改旧 R2 的 acceptance 字段定义。三个 seed 不自动等于显著性证据；实际意义需结合差值和失败案例讨论。

冻结检查失败属于实验结论或证据缺失：先看 `audit.json`，补可补的记录或评分；配置确实不兼容时再决定新的独立实验。不得修改测试集口径、反复测试选模型、隐藏失败或擅自将 15/15 门槛降低。历史测试参与过开发的事实必须在论文中说明。

## 固定验证样本诊断

使用已有 checkpoint，默认在 CPU 上抽取固定 16 个验证样本；同样本 seed 可比较四组。此命令不评分、不训练，输出 gamma、gate、coverage、有效残差 L2 和相对幅度。它不是重新训练的消融。

```bash
python scripts/diagnose_semantic_controls.py \
  --cfg experiments/p1_semantic_controls_20260915/rsaca/rsaca_levir_cc_seed1111/request_config.json \
  --checkpoint experiments/p1_semantic_controls_20260915/rsaca/rsaca_levir_cc_seed1111/snapshots/rsaca_levir_cc_seed1111_checkpoint_1000.pt \
  --output experiments/p1_semantic_controls_20260915/diagnostic_rsaca_cc_1111.json \
  --sample-seed 1111 --count 16 --device cpu
```

D 的 coverage 为实现中 dense attention 的覆盖统计，不等同于语义变化面积；空图不应据此与 C0/G 的变化 coverage 混读。

## 历史依据、预算与验证边界

R2 原始评分复算为三数据集五项 **15/15 均值提升、41/45 逐 seed 提升**；旧状态标记不否定已完成的评分。其当前已上传的配置、初始化、源码恢复和 checkpoint 证据不足以自动认证为本协议运行，因此保持独立。新协议是 36 次完整矩阵；兼容复用必须有逐项证据，不把 R2 自动混入。

R2 已记录的 18 次训练合计约 7.47 小时（含训练过程中的验证，非纯 GPU 计算时间）；不能简单承诺新 36 次耗时。新初始化、D 的 dense 注意力、服务器负载、输入全文件哈希、Java/SPICE 和磁盘均会影响时间。先从首组 `execution_ledger.json` 与训练日志估算剩余工作，再安排串行运行。推理和评分分别记时，不能只计训练。

本次实施的本地测试使用隔离 CPU 环境，覆盖真实模型前向/反向、保存加载、初始化 RNG、配置差分、路径、选点与冻结规则和 R2 复算。AutoDL 的实际数据、CUDA/Java/SPICE 环境与完整训练需在服务器按上述命令验证；本地 CPU 测试不代表这些已经完成。

实施验证记录（2026-09-15）：新增协议测试 31 项、相关历史回归 24 项通过；另 1 项真实符号链接测试因本机 Windows 账户缺少权限而跳过，应在 Linux 上补验。13 个相关 Python 文件通过 AST 语法检查。使用隔离 Python 3.9 / PyTorch 2.8 CPU 环境；历史 shell 回归由 Git Bash 执行。现有 attention mask dtype 的 PyTorch 弃用提示保留，不在本次变更中改动历史计算。服务器环境验证：未执行；完整训练：未执行。
