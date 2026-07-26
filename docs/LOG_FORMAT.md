# CARD 日志与实验工件规范

## 1. 兼容原则

重构没有删除历史文件：

- train_log.txt：训练可读日志；
- eval_log.txt：验证/测试可读日志；
- train_log.jsonl：旧训练统计；
- val_metrics.csv：validation checkpoint 选择权威输入；
- test_metrics.csv：历史 Test 汇总兼容输入；
- cfg.json、resolved_config.json/yaml、args.txt、config_hash.txt。

新增 train.log、val.log、test.log、metrics.jsonl、metrics.csv、
run_summary.json、error.log、配置/命令/环境/Git 工件。历史 selector 和
7.5/7.6 manifest 仍读取原文件。

## 2. 可读日志

Logger 使用 Python 标准 logging，同时保留原类与方法。格式：

~~~text
2026-07-26 19:30:12 | INFO | logger | epoch=3 step=120 global_step=960 loss_total=2.3140 lr=0.0001
~~~

训练日志包含 epoch、batch step、global step、耗时、学习率、总损失、caption/CARD/辅助损失、有效 loss 权重、掩膜/语义监控指标和梯度调试字段。Visdom 存在时继续绘图；未安装时只禁用绘图，不伪造结果。

## 3. 标准指标名

caption 指标统一为：

~~~text
Bleu_1
Bleu_2
Bleu_3
Bleu_4
METEOR
ROUGE_L
CIDEr
SPICE
~~~

utils.metrics 会读取 BLEU4、bleu_4、rouge-l 等历史别名并转换。原始历史文件不会被改写。

## 4. metrics.jsonl

每行是独立合法 JSON。标准字段：

~~~json
{
  "timestamp": "2026-07-26T19:30:12+08:00",
  "stage": "val",
  "epoch": 12,
  "global_step": 9600,
  "step": 0,
  "learning_rate": 0.0001,
  "loss": 1.2345,
  "caption_loss": 1.1021,
  "mask_loss": 0.0312,
  "semantic_loss": 0.1012,
  "gpu_memory_mb": 8123.5,
  "Bleu_1": 0.82,
  "Bleu_2": 0.72,
  "Bleu_3": 0.63,
  "Bleu_4": 0.55,
  "METEOR": 0.39,
  "ROUGE_L": 0.74,
  "CIDEr": 1.34,
  "SPICE": 0.35,
  "duration_seconds": 12.3,
  "extras": {}
}
~~~

stage 只能是 train、val 或 test。gpu_memory_mb 在 CUDA 可用时记录当前已分配
显存，本地 CPU 环境为 null。未纳入稳定列的原调试字段进入 extras，例如
effective_lambda_mask、effective_lambda_semantic、gradient norm 和
balanced_score。NaN/Infinity 会写为 null，保证每行是严格 JSON。

## 5. metrics.csv

CSV 使用固定列：

~~~text
timestamp,stage,epoch,global_step,step,learning_rate,loss,
caption_loss,mask_loss,semantic_loss,gpu_memory_mb,duration_seconds,
Bleu_1,Bleu_2,Bleu_3,Bleu_4,METEOR,ROUGE_L,CIDEr,SPICE,
Mask_Precision,Mask_Recall,Mask_F1,Mask_IoU,Mask_mIoU,
Semantic_F1,Semantic_IoU,Semantic_mIoU,extras_json
~~~

extras_json 是合法 JSON 字符串。JSONL 是完整结构化日志，CSV 用于表格查看和快速分析。

## 6. run_summary.json

训练生命周期状态：

~~~text
initialized -> running -> completed
                       -> failed
                       -> interrupted
~~~

主要字段：

~~~json
{
  "experiment_name": "levir_mci_masksemantic_repro_seed1111",
  "dataset": "levir_mci",
  "run_id": "levir_mci_masksemantic_repro_seed1111",
  "seed": 1111,
  "status": "completed",
  "start_time": "...",
  "end_time": "...",
  "final_global_step": 10000,
  "checkpoint_selection_deferred": true,
  "test_completed": false,
  "failure_reason": ""
}
~~~

completed 表示训练循环结束，不表示 Test 完成。checkpoint_selection_deferred=true 表示最终 checkpoint 仍须 validation selector/manifest 锁定。推理与评分分别写 val_inference_completed、test_inference_completed、val_metrics_completed 和 test_metrics_completed。

## 7. 异常

未捕获异常写入 error.log，包含完整 traceback；run_summary 状态更新为 failed。KeyboardInterrupt 更新为 interrupted。测试入口也安装 traceback hook，但不会把训练状态改写成测试状态。

禁止捕获错误后写零指标继续运行。

## 8. 配置与复现

- config_original.yaml：用户指定 YAML 的原始副本；
- config_resolved.yaml：默认值、YAML、CLI/opts 合并后的最终配置；
- command.txt：训练启动命令；
- command_val.txt、command_test.txt：后续推理命令；
- environment.txt：系统、Python、pip freeze、PyTorch/CUDA/GPU；
- git_info.txt：commit、branch、dirty 标记与 status；
- model_summary.txt：总参数量、可训练参数量、设备、seed 和模型结构。

## 9. 如何排查

- NaN/Inf：检查 error.log 和 metrics.jsonl；结构化记录器不接受伪造指标。
- warmup：查看 extras.effective_lambda_mask 和 extras.effective_lambda_semantic。
- 学习率：查看 learning_rate。
- 某分支是否启用：核对 config_resolved.yaml 和 resolved key-switch 输出。
- checkpoint 选择：查看 checkpoint_selection.json/best_checkpoint.json 的 selection_metric_split 和 selection_uses_test_metrics。
- Test 是否锁定：查看 7.5/7.6 manifest，不以单独 test_metrics.csv 代替 provenance。
