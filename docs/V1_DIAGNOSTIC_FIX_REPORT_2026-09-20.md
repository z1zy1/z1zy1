# V1 语义诊断审查修复报告（2026-09-20）

## 基线与范围

实施基线：远端 V1 的 `d95e504752a37289bda347ecf48ed56ab3657d49`。
本次修改诊断工具、相关回归 fixture 和执行文档。模型、训练入口、选点生产脚本、
冻结准入实现及历史实验工件均未修改。保留 AutoDL 原路径约定。

## 修复对应关系

| 审查项 | 修复 |
|---|---|
| R1 多参考 COCO 被拒绝 | 参考 image_id 按覆盖集合核验，允许多描述；独立检查 annotation ID 重复。数值 ID 0 合法，预测重复/缺失/多余仍阻止评分。 |
| R2 audit.files 未识别 | 支持路径到 SHA256 的映射及 path/sha256 对象；检测冲突、错误摘要和 JSON 解析失败，不把注册哈希误归给 audit 本身。 |
| R3 评分日志无法解析、输出冲突 | 优先读取 metrics.json，兼容日志后多行 JSON；stdout/stderr 独立保存，组合报告按输入分配评分目录。run-dir 必须在该次诊断目录内。 |
| R4 前向配置核对不足 | 比较完整 model/train 字段及语义数据接口，覆盖 gamma/global token/warmup 等；缺失配置明确不可核验，冲突阻止推理。 |
| R5 选点证据核验不足 | 检查参考及摘要、规则、tie-break、候选网格/指标/分数/路径、选中指标及 checkpoint 内部身份。分数比较保留 1e-12 容差。 |
| R6 gate/coverage 嵌套数组被丢弃 | 支持嵌套数值，拒绝非有限值；必要 hook 缺失标记 incomplete，显式禁用才标为不适用。 |
| R7 输出覆盖历史 | CLI 在执行前拒绝历史目录内输出及非空输出位置；报告使用排他创建，已有文件不覆盖。路径解析包含符号链接解析。 |

额外修复：不再硬编码历史实验已完成/未测试结论；重复推理保留每次证据，
空预测不再表示复现成功；严格模式解码器加载失败直接报错；诊断隔离并恢复 RNG、
配置与工作目录。steps 只筛选曲线展示，完整网格核验保留。文本样例展示描述内容，
不把长度统计冒充对象/动作/关系错误分析。

## 证据边界

- 词表内容与历史执行源码的身份仍显式标为 unverified，不能凭当前提交或维度证明。
- 记录启动器运行环境、评分命令文件和 Java 信息；外部评分器环境/缓存相同尚未证明。
- 分数 match 仅表示数值在容差内一致。
- CSV 数值复算与 checkpoint 身份核验分开。`recomputed_statistics` 不能用于绕过准入。
- 对象、动作、属性、关系的错误归因需要参考描述及独立标注/解析协议。

## 验证

测试使用独立 Python 3.9 / PyTorch CPU 环境，未安装或修改 AutoDL 依赖。
命令分组如下（仓库根目录执行）：

```bash
python -m pytest tests/test_semantic_diagnostics.py -q
python -m pytest tests/test_semantic_controls.py tests/test_semantic_control_admission.py tests/test_p1_checkpoint_integrity.py tests/test_checkpointing.py -q
python -m pytest tests/test_unified_rsaca.py -q
```

诊断测试 23 项通过；B/D/C0/G、冻结准入和 checkpoint 回归 44 项通过、1 项跳过。跳过项为 Windows 无符号链接权限，需在 Linux 验证。合计 81 项通过、1 项跳过。
Bash 历史回归 14 项通过。初次运行误调用 Windows WSL bash，出现环境权限错误；
使用 Git Bash 获准执行后通过，未因此修改生产脚本或测试断言。
语法编译及 git diff --check 通过。

用户提供的 72 个历史验证文件只读复算：36 运行的选点元数据核对一致，
C0-B 均值提升仍为 11/15、逐 seed 提升 34/45。未提供 checkpoint 本体，
因此 authenticated selected_rows 为 0，数值归入 recomputed_statistics。
历史 R2 回归检查仍要求 15/15、41/45，并检查工件不变。

未执行 AutoDL/CUDA 环境验证、完整训练、完整真实 Java/SPICE 评分或正式测试。
本次代码修复不产生新的模型效果结论，也不使验证准入失败的实验自动获得测试资格。

## AutoDL 建议命令

先在独立诊断 checkout 更新 V1；原训练目录和 source lock 保持原样。
设置 EXP_ROOT 指向原服务器实验根目录（使用真实现有路径），再运行：

```bash
python scripts/diagnose_semantic_controls.py curves \
  --experiment-root "$EXP_ROOT" \
  --output-dir "experiments/analysis/controls_review_$(date +%Y%m%d_%H%M%S)"
```

若只拿到 CSV 与选点 JSON，另加 `--no-checkpoint-validation`；这会明确输出数值复算，
不会认证不存在的 checkpoint。每次使用新输出目录。评分与推理命令见
`docs/SEMANTIC_CONTROLS_AUTODL.md` 的诊断章节。
