#!/usr/bin/env python3
"""Regenerate the living experiment/claim reference from authoritative JSON results."""

import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, 'docs', 'EXPERIMENT_CLAIM_REFERENCE.md')
METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
SOURCE_FILES = [
    'experiments/card_baseline_test_summary.json',
    'experiments/paper_required_experiments_summary.json',
    'experiments/7_6_locked_test_summary.json',
    'experiments/second_cc_current_mci_test_summary.json',
]
FOLLOWUP_SUMMARY = 'experiments/reliability_sparse_rsaca_v1_whole_gate_retry/summary.json'
V1_SUMMARY = 'experiments/reliability_sparse_rsaca_v1/summary.json'

EXPERIMENT_NOTES = {
    'levir_mci_card_baseline': 'LEVIR-MCI 的审计基线；所有该数据集改进实验应与它比较。',
    'levir_mci_card_mask_loss': '仅增加 mask loss；BLEU-4、CIDEr 下降，只有 SPICE 极小提升，不支持整体有效。',
    'levir_mci_card_semantic_loss': 'BLEU-4 和 CIDEr 提升，但 SPICE 明显下降；说明语义辅助有潜力，但目标不平衡。',
    'levir_mci_card_mask_semantic': '单 seed 的 8 项指标均高于 CARD，是最强单次结果；三 seed 复现后优势未完全保持。',
    'levir_mci_card_mask_semantic_pd05': 'partial detach 改善 SPICE，但 BLEU-4 低于基线；属于混合结果。',
    'levir_mci_card_mask_semantic_pd05_noreweight': '与 pd05 数值完全相同，不能作为独立增益证据；需检查配置/权重是否实际生效。',
    'levir_mci_card_mask_semantic_pd05_reweight': 'BLEU-4 较强，但 SPICE 下降；feature reweight 不适合作为统一模型的固定组件。',
    'levir_mci_ours_weak_coupled_final': '与 pd05/noreweight 数值完全相同，且 BLEU-4 下降；不能作为最终统一模型证据。',
    'second_cc_card_rgb_baseline': '旧论文汇总使用 checkpoint 7000；最终比较应改用锁定 CARD checkpoint 9000。',
    'second_cc_card_semantic_aux': 'CIDEr、SPICE 略升但 BLEU-4 下降；单独语义辅助不足以证明全面改进。',
    'second_cc_card_semantic_crossattn': '单次结果在主要指标上最均衡，是 RSACA 统一方案的直接结构依据。',
    'second_cc_card_semantic_hardgate': 'BLEU-4 提升但 SPICE 大幅下降；硬门控破坏语义质量，不纳入统一方案。',
    'second_cc_ours_weak_coupled_final': 'BLEU-4 较高，但 SPICE 明显下降；不能支持整体优于 CARD 的主张。',
    'second_cc_mmodalcc_comparison': '缺少外部结果，不能进入定量主表或结论。',
}


def load(relative_path, required=True):
    path = os.path.join(ROOT, relative_path)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(path)
        return None
    with open(path, encoding='utf-8-sig') as handle:
        return json.load(handle)


def fmt(value, signed=False):
    if value in (None, ''):
        return '-'
    return ('%+.4f' if signed else '%.4f') % float(value)


def metric_row(name, metrics):
    return '| %s | %s |' % (name, ' | '.join(fmt(metrics.get(metric)) for metric in METRICS))


def verdict(metrics, baseline):
    available = [metric for metric in METRICS if metrics.get(metric) not in (None, '')]
    if not available:
        return '无可分析结果'
    improvements = sum(float(metrics[metric]) > float(baseline[metric]) for metric in available)
    if improvements == len(available):
        return '%d/%d 提升，当前结果支持' % (improvements, len(available))
    if improvements >= 5:
        return '%d/%d 提升，但证据混合' % (improvements, len(available))
    return '%d/%d 提升，不支持整体优于基线' % (improvements, len(available))


def unified_conclusion_lines(unified):
    if unified is None:
        return [
            '- **目前尚不能声称三数据集统一有效。**',
            '- 统一 RSACA 的 scratch 三 seed 锁定测试尚未完成，不能用探索性、迁移初始化或单 seed 结果替代主实验。',
            '- SECOND-CC 最新 MCI-transfer cross-attention 三 seed 均值 8/8 指标提升，是结构潜力的补充证据，但 transfer initialization 会混合结构增益与迁移增益。',
        ]

    levir_cc = unified['datasets']['levir_cc']['delta']
    levir_mci = unified['datasets']['levir_mci']['delta']
    second_cc = unified['datasets']['second_cc']['delta']
    return [
        '- **统一 RSACA scratch 三 seed 锁定测试已完成，但尚不能声称三数据集统一有效。**',
        '- 当前 `experiments/unified_rsaca/summary.json` 包含 9 次 RSACA 运行；CARD 对照仍是每数据集 1 次锁定运行，而不是第 3 节要求的 CARD 三 seed 配对矩阵。因此现有 delta 是对单次审计基线的比较，尚不能作为完整 `3 x 2 x 3 = 18` 主实验的最终统计结论。',
        '- LEVIR-CC 均值仅 ROUGE-L、SPICE 提升；BLEU-1 至 BLEU-4、METEOR、CIDEr 均低于 CARD（B4 %s，CIDEr %s）。' % (fmt(levir_cc['Bleu_4'], True), fmt(levir_cc['CIDEr'], True)),
        '- LEVIR-MCI 三 seed 均值 8/8 指标高于 CARD（B4 %s，CIDEr %s，SPICE %s），支持该数据集上的有效性。' % (fmt(levir_mci['Bleu_4'], True), fmt(levir_mci['CIDEr'], True), fmt(levir_mci['SPICE'], True)),
        '- SECOND-CC 三 seed 均值 7/8 指标提升，但 SPICE 低于 CARD %s，未达到全指标统一验收标准。' % fmt(second_cc['SPICE'], True),
        '- 因此统一验收为未通过；MCI-transfer 结果仍只能作为独立迁移证据，不能补足 LEVIR-CC 或 SECOND-CC 的主实验缺口。',
    ]


def followup_candidate_lines():
    summary = load(FOLLOWUP_SUMMARY, required=False)
    if summary is None:
        whole_line = (
            'V1 whole-adapter 候选 `run_reliability_sparse_rsaca_v1_whole_gate.sh` 将门控作用于完整 RSACA 适配器，并加入输入相关全局语义 token；无变化样本的门控被强制为零。该候选尚无锁定结果，不能进入主结果表。'
        )
    else:
        datasets = summary.get('datasets', {})
        cc = datasets.get('levir_cc', {})
        mci = datasets.get('levir_mci', {})
        second = datasets.get('second_cc', {})
        whole_line = (
            'V1 whole-adapter 候选已完成独立三数据集三 seed 锁定测试，但总体验收仍为 `acceptance_passed=false`：'
            'LEVIR-CC 的 B4/CIDEr 均值仍低于 CARD（%s、%s），而 LEVIR-MCI 与 SECOND-CC 的均值 8/8 指标提升。'
            '该结果是候选证据，不能改写统一主张。'
            % (fmt(cc.get('delta', {}).get('Bleu_4'), True), fmt(cc.get('delta', {}).get('CIDEr'), True))
        )
    return [
        whole_line,
        '新增待验证候选 `run_reliability_sparse_rsaca_v1_prenorm_changed_global.sh` 使用 `context_pre_norm`、`gamma_max=0.1`、gate bias `-2.5` 和 changed-only global token；其输出目录独立，锁定测试前必须先完成验证集筛选。',
    ]


def v1_candidate_line():
    summary = load(V1_SUMMARY, required=False)
    if summary is None:
        return 'V1 reliability-gated sparse RSACA 尚无锁定结果，不能进入主结果表。'
    cc = summary.get('datasets', {}).get('levir_cc', {})
    return (
        'V1 reliability-gated sparse RSACA 已完成三数据集三 seed 锁定测试，但总体验收为 `%s`；'
        'LEVIR-CC B4/CIDEr 均值变化为 %s/%s，不能作为三数据集统一有效的证据。'
        % (
            summary.get('acceptance_passed', False),
            fmt(cc.get('delta', {}).get('Bleu_4'), True),
            fmt(cc.get('delta', {}).get('CIDEr'), True),
        )
    )


def main():
    baseline_payload = load('experiments/card_baseline_test_summary.json')
    paper_rows = load('experiments/paper_required_experiments_summary.json')
    locked = load('experiments/7_6_locked_test_summary.json')
    second_current = load('experiments/second_cc_current_mci_test_summary.json')
    unified = load('experiments/unified_rsaca/summary.json', required=False)
    source_paths = [os.path.join(ROOT, path) for path in SOURCE_FILES]
    if unified is not None:
        source_paths.append(os.path.join(ROOT, 'experiments/unified_rsaca/summary.json'))
    if os.path.exists(os.path.join(ROOT, FOLLOWUP_SUMMARY)):
        source_paths.append(os.path.join(ROOT, FOLLOWUP_SUMMARY))
    if os.path.exists(os.path.join(ROOT, V1_SUMMARY)):
        source_paths.append(os.path.join(ROOT, V1_SUMMARY))
    source_snapshot = datetime.fromtimestamp(max(os.path.getmtime(path) for path in source_paths), timezone.utc)
    baselines = {row['dataset']: {metric: row[metric] for metric in METRICS}
                 for row in baseline_payload['results']}

    lines = [
        '# 实验主张与结果分析参考文档', '',
        '> 本文档是模型实现、实验设计和论文结论的统一语义来源。修改 CARD/RSACA、数据输入、训练协议、选点策略、测试结果或论文主张后，必须运行 `python scripts/update_experiment_claim_reference.py` 并复核结论。', '',
        '结果快照时间：`%s`' % source_snapshot.strftime('%Y-%m-%d %H:%M:%S UTC'), '',
        '## 1. 原始论文主张', '',
        '论文希望提出一个基于 CARD 改进的统一模型，并证明该模型在 LEVIR-CC、LEVIR-MCI、SECOND-CC 三个数据集上均优于原始 CARD，从而证明改进方法具有跨数据集有效性。这里的“统一模型”指主体结构、语义融合机制、损失项和核心超参数一致；允许数据路径、词表、序列长度和语义标签来源随数据集变化，并为每个数据集分别训练 checkpoint。', '',
        '该主张只有在三个数据集都完成同协议、多随机种子、验证集选点和锁定测试后才能成立。单 seed、测试集选点、不同基线 checkpoint 或数据集专用结构都不能单独支撑原主张。', '',
        '## 2. 当前结论', '',
        *unified_conclusion_lines(unified), '',
        '## 3. 实现原主张的建议', '',
        '统一候选采用 **CARD + Residual Semantic Cross-Attention Adapter（RSACA）**：残差 cross-attention、`gamma_init=0.01`、`gamma_max=0.5`、partial detach `0.5`；关闭 auxiliary mask loss、semantic caption loss、hard gate 和 feature reweight。SECOND-CC 使用成对语义图；LEVIR-CC 与 LEVIR-MCI 使用显式标记的 diff-only 语义图接口。', '',
        '当前锁定结果使用 `semantic_fusion_norm_mode=legacy_post_norm`。审计发现该实现即使在 `gamma=0` 时仍执行 `LayerNorm(query)`，因此不是严格恒等残差；而锁定 checkpoint 的实际 gamma 仅约 0.016-0.025。`context_pre_norm` 将归一化移到语义 context 分支，使 `gamma=0` 时输出严格等于原 CARD 特征。它目前只是针对 LEVIR-CC 退化的待验证结构假设，必须先做验证集驱动的小规模消融，再按 scratch 三 seed 锁定协议复验，不能据此改写现有结果。', '',
        '新增的 V1 候选为 **reliability-gated sparse RSACA**：只把变化位置提供给语义 cross-attention 的 K/V，并为无变化样本加入一个可学习回退 token；连续可靠性门控以视觉变化摘要、稀疏语义摘要和变化覆盖率缩放语义残差。成对语义图按 `before != after` 判定变化，diff-only 输入按非零类别判定变化。该规则在三数据集共享，保留锁定 RSACA 的其余训练设置。完整入口：`bash scripts/run_reliability_sparse_rsaca_v1.sh --stage all`；候选筛选阶段仅可运行 `preflight`、`train` 和 `select`。', '',
        v1_candidate_line(), '',
        *followup_candidate_lines(), '',
        '主实验必须从 scratch 分别训练 CARD 与 RSACA，避免用 MCI 初始化混淆结构增益。固定 3 个 seed（1111、2222、3333），形成 `3 datasets x 2 models x 3 seeds = 18` 次主实验。MCI-transfer 结果只能作为独立迁移实验。', '',
        '验收标准：每个数据集的主要指标均值不低于 CARD；至少 BLEU-4、CIDEr、SPICE 的方向一致；报告每 seed、均值、样本标准差；checkpoint 只能根据验证集选择，测试集仅运行一次锁定评估。', '',
        '完整统一实验入口：`bash scripts/run_unified_rsaca_experiments.sh --stage all`。', '',
        '## 4. 审计 CARD 基线', '',
        '| 数据集 | %s |' % ' | '.join(METRICS),
        '|---|%s|' % '|'.join(['---:'] * len(METRICS)),
    ]
    for dataset in ('levir_cc', 'levir_mci', 'second_cc'):
        lines.append(metric_row(dataset, baselines[dataset]))
    lines.extend(['', '这些数值来自 `experiments/card_baseline_test_summary.json`，是本文档所有 delta 的默认比较基线。它们各自仅对应 1 次 CARD 锁定运行；在补齐 CARD seeds 1111、2222、3333 前，只能作为审计基线，不能替代多 seed 对照均值和样本标准差。SECOND-CC 旧论文汇总中的 checkpoint 7000 与这里的锁定 checkpoint 9000 不一致，最终论文必须使用后者。', '',
        '## 5. 当前锁定结果总览', '',
        '| 数据集/方案 | seeds | B4 delta | CIDEr delta | SPICE delta | 提升指标数 | 分析 |',
        '|---|---:|---:|---:|---:|---:|---|'])
    for dataset in ('levir_cc', 'levir_mci'):
        item = locked['dataset_aggregates'][dataset]
        delta = item['mean_deltas_vs_card_baseline']
        analysis = 'SPICE 增益不能抵消 BLEU 系列下降，当前候选失败。' if dataset == 'levir_cc' else '大多数指标改善，但 BLEU-4 未复现单 seed 优势，结论仍为混合。'
        lines.append('| %s / 7.6 locked | %d | %s | %s | %s | %d/8 | %s |' % (dataset, item['n'], fmt(delta['Bleu_4'], True), fmt(delta['CIDEr'], True), fmt(delta['SPICE'], True), sum(delta[m] > 0 for m in METRICS), analysis))
    delta = second_current['delta']
    lines.append('| second_cc / current MCI-transfer cross-attn | %d | %s | %s | %s | %d/8 | 三 seed 均值全面提升；需用 scratch 对照分离迁移收益。 |' % (second_current['seed_count'], fmt(delta['Bleu_4'], True), fmt(delta['CIDEr'], True), fmt(delta['SPICE'], True), sum(delta[m] > 0 for m in METRICS)))
    failed = locked['dataset_aggregates']['second_cc']['mean_deltas_vs_card_baseline']
    lines.append('| second_cc / 7.6 legacy locked | 1 | %s | %s | %s | %d/8 | 与最新复现实验冲突，属于旧配置/评估链路失败结果，不作为最终模型证据。 |' % (fmt(failed['Bleu_4'], True), fmt(failed['CIDEr'], True), fmt(failed['SPICE'], True), sum(failed[m] > 0 for m in METRICS)))
    lines.extend(['', '## 6. 每项论文实验结果分析', '',
        '| 数据集 | 实验 | B4 | CIDEr | SPICE | 相对审计基线 | 结论说明 |', '|---|---|---:|---:|---:|---|---|'])
    for row in paper_rows:
        metrics = {metric: row.get(metric) for metric in METRICS}
        base = baselines.get(row['dataset'])
        if row['experiment'].endswith('card_baseline'):
            result_verdict = '参考基线，不判定增益'
        elif row['experiment'] == 'second_cc_card_rgb_baseline':
            result_verdict = '旧参考基线，不用于最终比较'
        else:
            result_verdict = verdict(metrics, base) if base else '无审计基线'
        lines.append('| %s | `%s` | %s | %s | %s | %s | %s |' % (row['dataset'], row['experiment'], fmt(row.get('Bleu_4')), fmt(row.get('CIDEr')), fmt(row.get('SPICE')), result_verdict, EXPERIMENT_NOTES.get(row['experiment'], '需要补充人工分析。')))
    lines.extend(['', '### LEVIR-MCI 三 seed 复现', '',
        '单 seed `levir_mci_card_mask_semantic` 的 8/8 提升不能直接作为最终结论。三 seed 锁定均值为：', '',
        '| 方案 | %s |' % ' | '.join(METRICS), '|---|%s|' % '|'.join(['---:'] * len(METRICS)),
        metric_row('mask+semantic 3-seed mean', locked['dataset_aggregates']['levir_mci']['metric_mean']),
        metric_row('delta vs CARD', locked['dataset_aggregates']['levir_mci']['mean_deltas_vs_card_baseline']), '',
        '结论：BLEU-3 delta 约为 0，BLEU-4 为负；只能称为多数指标改善，不能称为所有指标稳定优于 CARD。', '',
        '### SECOND-CC 三 seed 复现', '', '| seed | B4 | CIDEr | SPICE | 8/8 超过基线 |', '|---:|---:|---:|---:|---|'])
    for row in second_current['seed_results']:
        lines.append('| %s | %s | %s | %s | %s |' % (row['seed'], fmt(row['Bleu_4']), fmt(row['CIDEr']), fmt(row['SPICE']), '是' if row['all_metrics_strictly_above_baseline'] else '否'))
    lines.extend(['', '结论：均值 8/8 提升，2/3 seeds 全指标通过；seed 3333 的 SPICE 低于基线。该结果支持方案潜力，但因为使用 MCI 初始化，不能替代 scratch 统一主实验。', '',
        '## 7. 统一 RSACA 实验状态', ''])
    if unified is None:
        lines.extend(['统一 RSACA 的全矩阵结果尚未生成。当前状态是代码、输入预检、训练/选点/锁定测试/汇总脚本已经具备，但不能提前写入性能结论。', '', '预期结果文件：`experiments/unified_rsaca/summary.json`。生成后重新运行本文档更新脚本，结果将被纳入此节。'])
    else:
        lines.extend(['| 数据集 | mean 8/8 above CARD | B4 delta | CIDEr delta | SPICE delta |', '|---|---|---:|---:|---:|'])
        for dataset, item in unified['datasets'].items():
            lines.append('| %s | %s | %s | %s | %s |' % (dataset, '是' if item['mean_all_metrics_strictly_above_baseline'] else '否', fmt(item['delta']['Bleu_4'], True), fmt(item['delta']['CIDEr'], True), fmt(item['delta']['SPICE'], True)))
        lines.append('')
        lines.append('总体验收：**%s**。' % ('通过' if unified['acceptance_passed'] else '未通过'))
    if unified is None:
        writing_allowed = '当前可以写：提出残差语义交叉注意力改进；SECOND-CC 上存在 MCI-transfer 多 seed 增益；LEVIR-MCI 多数指标改善；消融表明 hard gate 和 feature reweight 会造成指标权衡。'
    else:
        writing_allowed = '当前可以写：提出残差语义交叉注意力改进；统一 RSACA 在 LEVIR-MCI 的 scratch 三 seed 锁定均值 8/8 指标高于 CARD；其在 LEVIR-CC 和 SECOND-CC 呈现明确的指标权衡；消融表明 hard gate 和 feature reweight 会造成指标权衡。'
    lines.extend(['', '## 8. 论文写作边界', '',
        writing_allowed, '',
        '当前不能写：统一模型已在三个数据集全部优于 CARD；所有指标均显著提升；改进完全来自模型结构；LEVIR-CC 或 SECOND-CC 已验证成功；已完成 `3 x 2 x 3 = 18` 次完整主实验。', '',
        '只有统一 RSACA scratch 矩阵达到第 3 节验收标准后，才可以把主结论升级为“三数据集均有效”。', '',
        '## 9. 同步规则', '',
        '以下修改必须同步本文档：模型结构或默认开关；语义标签来源和类别映射；训练步数、seed、初始化、学习率；验证选点与测试锁定策略；任一实验指标或有效性判断；论文主张和对外表述。', '',
        '```bash', 'python scripts/update_experiment_claim_reference.py', '```', '',
        '更新后检查 `git diff -- docs/EXPERIMENT_CLAIM_REFERENCE.md`，确认自动表格和人工结论一致，再提交代码。', '',
        '## 10. 权威数据源', '',
        '- `experiments/card_baseline_test_summary.json`：三数据集 CARD 锁定基线。',
        '- `experiments/paper_required_experiments_summary.json`：论文要求的单项实验结果。',
        '- `experiments/7_6_locked_test_summary.json`：LEVIR-CC、LEVIR-MCI 和旧 SECOND-CC 锁定结果。',
        '- `experiments/second_cc_current_mci_test_summary.json`：SECOND-CC 的 MCI-transfer 三 seed 结果，只能作为迁移证据。',
        '- `experiments/unified_rsaca/summary.json`：统一 RSACA scratch 三 seed 矩阵结果；存在时作为当前统一结论的直接依据。', ''])
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(OUTPUT)


if __name__ == '__main__':
    main()
