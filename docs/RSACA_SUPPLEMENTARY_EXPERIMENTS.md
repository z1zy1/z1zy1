# CARD + RSACA 补充实验

2026-11-25 作者修订四模型正式测试准入。冻结须通过完整矩阵、训练完成、来源与初始化、输入与划分、选点及权重的技术审计。五指标方向、逐 seed 差值和正向均值数量继续报告。历史 O 验证为 11/15，S 为 14/15，两批当时均未冻结或执行正式测试。规则修订发生在看到 S/O 验证结果之后，R2 官方测试结果此前已可见。论文以“修订准入规则后的补充测试”描述新增结果。

## 环境与资产审查

在交付 SHA 对应的独立工作树根目录操作。依赖包括 Python、CUDA Torch、NumPy、h5py、ImageIO、PyYAML、pycocotools、pycocoevalcap 以及 Java。PTB、METEOR、SPICE 的 JAR 与模型资源进入评分器身份。Matplotlib 和 Pillow 可放在独立绘图环境。S/O 历史均为 Python 3.8.20。S 使用 Torch 1.10.0+cu113 与 NumPy 1.21.4，O 使用 Torch 1.8.0+cu111 与 NumPy 1.24.4。可选用 tmux 给每个系列单开会话，保留各阶段日志。

填写服务器绝对路径。ARCHIVE 是 11 月 2 日原件包中的 raw 目录，内含 O 和 S。FINAL3 是 11 月 3 日包顶层，含三个入口 run_evidence_map.csv、prediction_audit.csv 和 raw。OUT 是派生输出根。映射左侧为历史前缀，右侧为实际挂载，支持更具体的前缀。权重旁保留原始 checksum 和 metadata sidecar。

~~~bash
export ARCHIVE=/mnt/evidence/server_materials_20261102T063254Z/raw
export FINAL3=/mnt/evidence/core_final3_materials_20261103T093152Z
export OUT=/mnt/card-runs/rsaca-paper-supplement
export OLD_DATA=/mnt/card-original-data
export OLD_TRAIN=/mnt/card-original-checkouts/z1zy1_v1_accept_20260920
run_supplement() {
  python scripts/run_rsaca_supplement.py --root "$OUT" "$@"
}
run_history() {
  run_supplement --archive-root "$ARCHIVE" --final3-root "$FINAL3" \
    --map "/root/autodl-tmp/z1zy1=$OLD_DATA" \
    --map "/root/autodl-tmp/z1zy1_v1_accept_20260920=$OLD_TRAIN" "$@"
}
git rev-parse HEAD
run_history --stage audit --series O --metadata-only
run_history --stage audit --series S --metadata-only
run_supplement --stage test --dry-run
~~~

metadata_audit.json 记录历史选点、声明的权重身份和已映射时的本地存在状态。完整 audit 核对逐样本实际消费路径及内容、语义原图解码、配置、权重本体、内部 step、参数结构和严格加载。各组输出保存历史训练提交、当前评估提交、运行时及评分器身份。输入清单包括共享样本 ID、词表、参考、两期缓存和各臂实际消费的图。

## 历史矩阵与补充测试

先在 O 环境执行 audit、replay、freeze，再在 S 环境独立执行。replay 固定 LEVIR-CC RSACA seed 1111 的已选权重和完整 val。逐 ID 文本必须一致，五项原始分值逐项允许绝对差 1e-7，各次尝试和差异清单均保留。选点依据十点网格和五指标等权 log 比值，平分取早，参考向量为 (0.4375, 0.3377, 0.6942, 1.2299, 0.2607)。

~~~bash
series=O # O 结束后，在对应历史环境把这三条命令用于 S
run_history --series "$series" --stage audit
run_history --series "$series" --stage replay
run_history --series "$series" --stage freeze
~~~

按完整性、身份和重放结果选主系列，在任何新增正式 test 解码前登记。两批均合格时默认 O。仅 S 合格时指定 S 并记下原因。正式解码用官方 test、batch 1、greedy、eval、no_grad。两个 LEVIR 各 1,929 个 test ID，SECOND 1,227 个。预测与参考身份相符时，评分中断可续跑。解码和评分尝试写入 evaluation_attempts。孤立的新工件可通过 --retry-incomplete 保留到 failed_evaluations 再重试。

~~~bash
run_supplement --stage primary --series O
run_supplement --stage test --series O
run_supplement --stage summary --series O

# 切换至 S 的冻结环境后独立复评
run_supplement --stage test --series S
run_supplement --stage summary --series S

# 单运行恢复
run_supplement --stage test --series O --dataset levir_cc --arm rsaca \
  --seed 1111 --retry-incomplete
run_supplement --stage summary --series R2 --final3-root "$FINAL3"
~~~

直接用 test_card_spot.py 时传该运行的 test_config.json、已锁定 checkpoint、--split test 和 evaluations/<arm>/<run>/captions/test.json，沿用同一个准入检查。回到 wrapper 可继续打分。

## 两项消融和共同对照

CARD 对 RSACA 测给图和适配器的联合效果。Plain Fusion 对 RSACA 测同图输入的整体融合设计。Fixed Gate 对 RSACA 测学习门控和非空图满幅更新。Dense Local 仅开放有效未变化格，保留变化摘要和门控。No Global 仅移除附加全位置均值 K/V，保留 fallback。后归一化、无局部位置编码沿用论文配置。

训练配置沿用 32/64 train/val batch，10,000 步、每 1,000 步全量验证。Adam 学习率 2e-4、betas (0.9,0.999)、eps 1e-8、weight decay 0、warmup 500、cosine 最小比例 0.05。适配器 8 头、dropout 0.1、gamma 0.01/0.5、whole-adapter gate、部分梯度缩放 0.5。损失为 caption + 0.001 con + 0.001 ind，语义、mask 和 relation 辅助损失关闭。

new-ablation 为两项消融各 9 次和共享 RSACA 对照 9 次。new-full 为四模型 36 次加两项消融 18 次。当前训练入口采用同系列共同对照，历史对照达到完全匹配时可作为 18 次训练的预算情形。去 fallback 和额外固定门控系数没有注册到默认矩阵。

~~~bash
export LEVIR_CC_ROOT=/mnt/card-data/Levir-CC
export LEVIR_MCI_ROOT=/mnt/card-data/LEVIR-MCI-dataset
export SECOND_CC_ROOT=/mnt/card-data/SECOND-CC-AUG
export LEVIR_CC_FEATURE_ROOT="$LEVIR_CC_ROOT/features"
export LEVIR_MCI_FEATURE_ROOT="$LEVIR_MCI_ROOT/features"
export SECOND_CC_FEATURE_ROOT="$SECOND_CC_ROOT/features"
export CUDA_VISIBLE_DEVICES=0 NUM_WORKERS=8 PYTORCH_GPU=0

series=new-ablation # 主矩阵也需重训时用 new-full
run_supplement --stage preflight --series "$series"
run_supplement --stage train --series "$series"
run_supplement --stage select --series "$series"
run_supplement --stage audit --series "$series"
run_supplement --stage freeze --series "$series"
run_supplement --stage test --series "$series"
run_supplement --stage summary --series "$series"
~~~

训练中断保留原目录，补跑前审查失败键、配置、输入、初始化、环境及来源。审计与冻结始终基于预定全组，训练与测试可按数据、臂、seed 选择已登记的运行键。

## 统计、效率与定性图

reports 中的 val 和 test 分别保存逐运行、均值、逐 seed 配对及异常 CSV，亦生成 JSON、Markdown 和 LaTeX 表。报告有效/计划分母和样本 SD，n<2 时 SD 为空。三个 seed 用于描述性比较。R2 历史 test 从 score_result.json 复算，档案中未知的 checkpoint SHA 留空。

~~~bash
python scripts/plot_rsaca_supplement.py "$OUT/O/reports/test/statistics.json" \
  --output "$OUT/O/reports/test/seed_scores"

python scripts/measure_rsaca_supplement.py --dataset levir_cc \
  --output "$OUT/efficiency/levir_cc_params.json"
python scripts/measure_rsaca_supplement.py --series-root "$OUT/new-full" \
  --dataset levir_cc --seed 1111 --batch 1 --warmup 10 --repeats 30 --fixed-steps 10 \
  --output "$OUT/efficiency/levir_cc_cuda.json"

python scripts/visualize_paper_cases.py --supplement \
  --series-root "$OUT/O" --dataset levir_mci --run-seed 1111 --split test \
  --data-root "$LEVIR_MCI_ROOT" --sample-seed 7 --num-cases 4 \
  --output-dir "$OUT/O/cases/levir_mci_test_1111"

python scripts/visualize_paper_cases.py --supplement \
  --series-root "$OUT/O" --dataset levir_mci --run-seed 1111 --split val \
  --data-root "$LEVIR_MCI_ROOT" --sample-seed 7 --num-cases 4 \
  --reference "$LEVIR_MCI_ROOT/levir_mci_captions_reformat.json" \
  --input-manifest "$ARCHIVE/O/series/inputs_levir_mci.json" \
  --map "/root/autodl-tmp/z1zy1=$OLD_DATA" \
  --output-dir "$OUT/O/cases/levir_mci_val_1111"
~~~

延迟统计针对相同设备、样本、FP32、batch、预热与同步，覆盖预载特征上的实际 greedy、受控固定迭代数和适配器单独调用。报告实际描述长度、吞吐、峰值显存与重复观测。MAC 代理含 Linear、Conv2d、MultiheadAttention 的投影与点积，一次 MAC 记 2 FLOPs。函数式算子、归一化、softmax、激活、训练反传和优化器未计入。缓存 I/O、H2D、冻结 ResNet-101 和外部地图生产成本待实测。局部限制沿用密集注意力。Fixed Gate 保留 requires_grad 的门控 MLP，其权重在该路径上没有 caption loss 梯度。

定性输出使用源图和实际消费的语义图，SECOND 展示 A/B 两期图。可显式重复传 --id 或使用固定抽样 seed。版式包括所有参考与四条生成。原始文字和身份进入 cases.json，同时导出 PNG、PDF、SVG、Markdown、HTML。未知语义类别保持未知。

[本地审查、验证结果与预算](RSACA_SUPPLEMENTARY_AUDIT.md)
