# V1 Semantic Diagnostic Runtime Fix

本次修复仅涉及语义诊断工具和回归测试。

## 修复内容

- `train.global_step` 被明确视为 checkpoint 恢复的运行时状态，不再参与持久模型配置等价性比较。
- 严格模式从 checkpoint payload、文件名和完整性 metadata 独立核验步数；缺失或冲突会失败。兼容模式明确标记为 `unverified`。
- 显式配置值 `None` 不再被误判为缺失字段；真实缺少字段仍保持不可核验状态。
- 保存/加载测试改用项目已有 `load_checkpoint_file`，兼容 PyTorch 1.10，同时保留四组前向、反向、梯度和参数一致性断言。

## 验证

- 回归：`85 passed`，日志和 JUnit 位于 `/root/autodl-tmp/semantic_diag_gpu_20260921/pytest_fixed4.log` 与 `pytest_fixed4.xml`。
- 严格 GPU 推理：LEVIR-CC C0/rsaca seed1111 step9000，validation/greedy，16 个固定样本，repeat=2，无 shim、无 `--allow-unverified-checkpoint`；checkpoint integrity 和 step=9000 均 verified，预测及 hook 记录完全一致，RNG 前后指纹一致。
- 推理结果：`/root/autodl-tmp/semantic_diag_gpu_20260921/infer_strict2.json`。

本修复不改变模型结构、训练协议、历史实验工件或正式测试准入状态；Java/SPICE 评分阻塞仍未处理。
