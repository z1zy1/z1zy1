# 三个数据集的原始 CARD baseline 完整流程

此流程统一运行 LEVIR-CC、LEVIR-MCI、SECOND-CC 的原始 CARD：训练后只按验证集 CIDEr 选择 snapshot；CIDEr 完全相同时以验证集 SPICE 决胜。锁定后每个数据集仅测试一次。默认入口故意停在测试前，测试指标不会参与 snapshot 选择。

固定训练设置为：model=card、随机种子 1111、10000 steps、每 1000 steps 保存并验证、Adam lr=2e-4、batch size 32；不加载初始 checkpoint，不使用 change mask、语义图、辅助损失、feature reweight、content-word weighting 或 semantic fusion。选择前会审计 resolved_config.json，同名目录如果并非上述原始 CARD 会硬失败。

## 远程执行

~~~bash
cd /root/autodl-tmp/z1zy1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate card
export PYTHON=/root/miniconda3/envs/card/bin/python
export EXP_ROOT=/root/autodl-tmp/z1zy1/experiments
export LEVIR_CC_ROOT=/root/autodl-tmp/z1zy1/Levir-CC
export LEVIR_MCI_ROOT=/root/autodl-tmp/z1zy1/LEVIR-MCI-dataset
export SECOND_CC_ROOT=/root/autodl-tmp/z1zy1/SECOND-CC-AUG

bash scripts/run_card_baselines_all.sh --dry_run
bash scripts/run_card_baselines_all.sh
~~~

建议在 tmux 中运行最后一条正式命令。run_card_baselines_all.sh 完成训练、验证集 snapshot 选择和锁定，然后停止。先检查：

~~~bash
$PYTHON -m json.tool "$EXP_ROOT/card_baseline_locked_manifest.json"
~~~

确认三个 checkpoint 后运行单次测试：

~~~bash
bash scripts/run_card_baselines_test_locked.sh
~~~

流程默认不覆盖任何非空但未完成的实验目录。遇到这种目录时，先人工检查并移动/备份，或使用新的 EXP_ROOT；不要删除既有日志。--force 只用于显式重跑已经锁定的测试，不用于覆盖训练或改变 snapshot 选择。

## 结果文件

每个 baseline 源目录包含 val_metrics.csv、baseline_best_checkpoint.json 和权重。测试写入独立目录，不会覆盖源训练目录中已锁定哈希的 resolved_config：

~~~text
experiments/card_baseline_locked_tests/levir_cc/test_card_baseline_locked_result.json
experiments/card_baseline_locked_tests/levir_mci/test_card_baseline_locked_result.json
experiments/card_baseline_locked_tests/second_cc/test_card_baseline_locked_result.json
experiments/card_baseline_locked_manifest.json
experiments/card_baseline_test_summary.json
experiments/card_baseline_test_summary.csv
~~~

三个 baseline_best_checkpoint.json 含完整 8 项验证指标，可直接供 7.5 验证约束使用：

~~~bash
export LEVIR_CC_BASELINE_VAL_METRICS="$EXP_ROOT/card_levir_cc_baseline/baseline_best_checkpoint.json"
export LEVIR_MCI_BASELINE_VAL_METRICS="$EXP_ROOT/levir_mci_card_baseline/baseline_best_checkpoint.json"
export SECOND_CC_BASELINE_VAL_METRICS="$EXP_ROOT/second_cc_card_rgb_baseline/baseline_best_checkpoint.json"
~~~

失败日志分别为 card_baseline_train_failures.log、card_baseline_select_failures.log、card_baseline_test_failures.log，均只追加写入。
