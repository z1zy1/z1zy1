# PLAN-v5 修订与验收记录

基准为V1 `b7fd1074e2445ae534c7d01cd17bdb479d8863a7`。本轮修复review已复现的路径比较和重复hash问题，并补测试入口与后续真实验收说明。旧snapshot和历史实施记录保留，未把它们重写为当前状态。

变更范围：

- P1a/P1b及预检共用以真实执行PROJECT目录为基准的reference比较，同文件相对/绝对/含点路径可通过，异文件和hash不符仍拒绝。
- 旧input_manifest只在同步解析期间切换至PROJECT，finally恢复调用者cwd；不改登记配置、checkpoint或manifest字节。该上下文只面向单线程CLI。
- register局部复用验证后的Path，由每个正常外部工件3次hash降为1次；receipt复用同次验证所得digest比较登记和receipt身份，由源文件2次降为1次。没有跨阶段缓存；新调用和阶段结束仍fresh校验。
- `scripts/run_semantic_tests.py` 使用当前环境，从任意cwd定位仓库并运行相关CPU测试；不含本机依赖路径，不安装包。`docs/semantic_followup.md` 补完整val一组真实链路验收及按需外部存档步骤，未新增签章系统。

本轮实际环境：Windows，Python3.9.13、torch2.8.0+cpu、pytest8.4.2、numpy2.0.2、PyYAML6.0.3、h5py3.14.0、imageio2.37.2、Pillow11.3.0。只读复用既有测试依赖；没有安装/修改全局环境。

针对本轮边界执行：

```powershell
$env:PYTHONPATH='C:\Users\zhangyi\.codex\worktrees\v1-evidence-workflow\CARD;D:\实验\CARD\tmp\p1_controls_testenv\Lib\site-packages'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q -p no:cacheprovider tests/test_semantic_followup.py -k 'reference_paths or reference_mismatch or dataset_cwd or hash_once or stage_end'
```

结果：**11 passed, 29 deselected**（18.65秒）。覆盖从非项目cwd调用实际P1a/P1b handler、同文件三种路径表示、异文件预检/执行均拒绝、实际解析所见cwd与异常恢复、配置/manifest原字节保留、单次hash计数，以及下一次校验/阶段结束仍发现变更。scorer/decoder仍使用明确的合成替身，不代表真实科研计算。

另从 `C:\Users\zhangyi`（非项目cwd）验证便携完整入口，PYTHONPATH只提供已有外部依赖；项目路径由入口自行配置：

```powershell
$env:PYTHONPATH='D:\实验\CARD\tmp\p1_controls_testenv\Lib\site-packages'
$env:PYTHONDONTWRITEBYTECODE='1'
python 'C:\Users\zhangyi\.codex\worktrees\v1-evidence-workflow\CARD\scripts\run_semantic_tests.py'
```

结果：**109 passed, 1 skipped, 4 warnings**（141.08秒），退出0。skip仍为既有Windows symlink权限案例，warnings仍为既有torch mask类型提示。`git diff --check`通过。该命令实证便携入口无需从项目目录启动；此次使用已有环境不等于已验证所有平台的新环境安装。

C独立复验从 `D:\实验\CARD`（另一个非本worktree目录）运行同一便携入口，只复跑受影响文件；进程内加载既有依赖，不修改环境或安装包：

```powershell
python -c "import sys,os,runpy; sys.dont_write_bytecode=True; p=r'D:\实验\CARD\tmp\p1_controls_testenv\Lib\site-packages'; sys.path.insert(0,p); os.environ['PYTHONPATH']=p; s=r'C:\Users\zhangyi\.codex\worktrees\v1-evidence-workflow\CARD\scripts\run_semantic_tests.py'; sys.argv=[s,'-k','test_semantic_followup']; runpy.run_path(s,run_name='__main__')"
```

结果：**40 passed, 70 deselected**（66.67秒），退出0。C独立确认源码、测试、入口的冻结哈希均未改变；其后仅把这项实际结果补入本文及本轮快照。没有用合成测试代替真实服务器验收。

历史最终验收补记：PLAN-v3/R1当时由C独立执行相关5个测试文件，**98 passed, 1 skipped, 4 warnings**（168.31秒）；skip为Windows无symlink权限，warning为既有torch attention mask类型提示。旧实施记录中“C将复跑”是当时冻结点状态，本记录明确该轮独立验收已经完成，不能把历史状态误读为未测。此历史结果不替代本轮实际测试。

NOT_RUN：真实服务器P0→P1a→P1b全val链路、研究训练/正式test、GPU环境复现、人类标注、外部签章/可信时间锚。真实链路需要实际全val预测、references、checkpoint与metadata、输入manifest/源文件、评分资源和登记环境；缺材料则保持NOT_RUN。普通Git作者/commit时间不能单独证明可信预登记时序。

本轮冻结文件与SHA256见 `semantic_followup_revision_v5_snapshot.json`。快照不包含自身；本文不记录快照hash，避免循环。最终提交与推送由主会话在C审查/A验收后执行。
