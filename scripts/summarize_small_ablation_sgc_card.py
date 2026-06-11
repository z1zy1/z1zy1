import argparse
import csv
import json
import os


DEFAULT_EXPERIMENTS = [
    'lmask003_lsem005_pd05_warmup_no_reweight',
    'lmask003_lsem005_pd05_rw01_warmup',
    'lmask003_lsem003_pd05_rw01_warmup',
]

BASELINE_TEST = {
    'Bleu_1': 0.828,
    'Bleu_2': 0.728,
    'Bleu_3': 0.641,
    'Bleu_4': 0.565,
    'METEOR': 0.391,
    'ROUGE_L': 0.746,
    'CIDEr': 1.348,
    'SPICE': 0.336,
}

SUMMARY_FIELDS = [
    'exp_name',
    'recommended_snapshot',
    'Bleu_1',
    'Bleu_2',
    'Bleu_3',
    'Bleu_4',
    'METEOR',
    'ROUGE_L',
    'CIDEr',
    'SPICE',
    'delta_Bleu_4',
    'delta_METEOR',
    'delta_ROUGE_L',
    'delta_CIDEr',
    'delta_SPICE',
    'all_above_test_baseline',
    'spice_above_baseline',
    'caption_metrics_above_baseline_count',
    'main_failure_metric',
    'recommendation',
]

CAPTION_CORE_METRICS = ['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
TEST_METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize SGC-CARD small ablation results.')
    parser.add_argument('--exp_root', default='./experiments')
    parser.add_argument('--exp_names', nargs='*', default=DEFAULT_EXPERIMENTS)
    parser.add_argument('--output_csv', default=None)
    parser.add_argument('--output_txt', default=None)
    return parser.parse_args()


def to_float(value, default=0.0):
    if value is None or value == '':
        return default
    return float(value)


def parse_bool(value):
    return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y')


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def score_test_row(row):
    bleu4_norm = to_float(row.get('Bleu_4')) / BASELINE_TEST['Bleu_4']
    cider_norm = to_float(row.get('CIDEr')) / BASELINE_TEST['CIDEr']
    spice_norm = to_float(row.get('SPICE')) / BASELINE_TEST['SPICE']
    meteor_norm = to_float(row.get('METEOR')) / BASELINE_TEST['METEOR']
    rouge_norm = to_float(row.get('ROUGE_L')) / BASELINE_TEST['ROUGE_L']
    spice_only_penalty = max(0.0, spice_norm - min(bleu4_norm, cider_norm))
    return (
        0.30 * cider_norm
        + 0.25 * bleu4_norm
        + 0.20 * spice_norm
        + 0.15 * meteor_norm
        + 0.10 * rouge_norm
        - 0.15 * spice_only_penalty
    )


def select_recommended_test_row(test_rows):
    if not test_rows:
        return None, 'missing_test_results'

    for row in test_rows:
        row['_score'] = score_test_row(row)

    all_above = [row for row in test_rows if parse_bool(row.get('all_above_test_baseline'))]
    if all_above:
        return max(all_above, key=lambda row: row['_score']), 'all_test_metrics_above_baseline'

    caption_ok = [
        row for row in test_rows
        if all(to_float(row.get(metric), -float('inf')) >= BASELINE_TEST[metric] for metric in CAPTION_CORE_METRICS)
    ]
    if caption_ok:
        return max(caption_ok, key=lambda row: (to_float(row.get('SPICE')) - BASELINE_TEST['SPICE'], row['_score'])), 'caption_metrics_pass_spice_best'

    return max(test_rows, key=lambda row: row['_score']), 'best_balanced_available_snapshot'


def build_summary_row(exp_name, exp_root):
    exp_path = os.path.join(exp_root, exp_name)
    compare_rows = load_csv(os.path.join(exp_path, 'snapshot_compare_report.csv'))
    test_rows = load_csv(os.path.join(exp_path, 'test_top_snapshots_summary.csv'))
    best_v2 = load_json(os.path.join(exp_path, 'best_snapshot_v2.json'))

    recommended, reason = select_recommended_test_row(test_rows)
    if recommended is None:
        return {
            'exp_name': exp_name,
            'recommended_snapshot': best_v2.get('best_snapshot', ''),
            'main_failure_metric': 'missing_test_results',
            'recommendation': 'Run test_top_snapshots_sgc_card.sh and compare_snapshot_results.py for this experiment.',
        }

    deltas = {
        metric: to_float(recommended.get(metric)) - BASELINE_TEST[metric]
        for metric in TEST_METRICS
        if metric in BASELINE_TEST
    }
    caption_pass_count = sum(1 for metric in CAPTION_CORE_METRICS if deltas.get(metric, -float('inf')) >= 0)
    negative_deltas = {metric: delta for metric, delta in deltas.items() if delta < 0}
    main_failure_metric = min(negative_deltas, key=negative_deltas.get) if negative_deltas else 'none'

    recommendation = reason
    if reason == 'caption_metrics_pass_spice_best' and deltas.get('SPICE', -1.0) < 0:
        recommendation = 'Caption metrics pass; SPICE is the remaining bottleneck.'
    elif reason == 'best_balanced_available_snapshot' and deltas.get('SPICE', 0.0) >= 0 and (
        deltas.get('Bleu_4', 0.0) < 0 or deltas.get('CIDEr', 0.0) < 0
    ):
        recommendation = 'SPICE improved but BLEU-4/CIDEr fell; do not use as final model without another checkpoint/filter check.'

    row = {
        'exp_name': exp_name,
        'recommended_snapshot': recommended.get('snapshot_path', ''),
        'all_above_test_baseline': recommended.get('all_above_test_baseline', ''),
        'spice_above_baseline': str(deltas.get('SPICE', -float('inf')) >= 0),
        'caption_metrics_above_baseline_count': caption_pass_count,
        'main_failure_metric': main_failure_metric,
        'recommendation': recommendation,
    }
    for metric in TEST_METRICS:
        row[metric] = recommended.get(metric, '')
    for metric in ['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']:
        row['delta_' + metric] = deltas.get(metric, '')

    if compare_rows and not row['recommended_snapshot']:
        row['recommended_snapshot'] = compare_rows[0].get('snapshot_path', '')
    return row


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in SUMMARY_FIELDS})


def best_row(rows):
    usable = [row for row in rows if row.get('Bleu_4') not in (None, '')]
    if not usable:
        return None
    all_above = [row for row in usable if parse_bool(row.get('all_above_test_baseline'))]
    if all_above:
        return max(all_above, key=lambda row: score_test_row(row))
    caption_ok = [
        row for row in usable
        if all(to_float(row.get(metric), -float('inf')) >= BASELINE_TEST[metric] for metric in CAPTION_CORE_METRICS)
    ]
    if caption_ok:
        return max(caption_ok, key=lambda row: (to_float(row.get('SPICE')) - BASELINE_TEST['SPICE'], score_test_row(row)))
    return max(usable, key=score_test_row)


def row_by_exp(rows, exp_name):
    for row in rows:
        if row.get('exp_name') == exp_name:
            return row
    return None


def delta(row, metric):
    if row is None:
        return None
    return to_float(row.get(metric), 0.0) - BASELINE_TEST[metric]


def yes_no_unknown(condition):
    if condition is None:
        return 'unknown'
    return 'yes' if condition else 'no'


def write_report(path, rows):
    chosen = best_row(rows)
    no_rw = row_by_exp(rows, 'lmask003_lsem005_pd05_warmup_no_reweight')
    rw01 = row_by_exp(rows, 'lmask003_lsem005_pd05_rw01_warmup')
    lsem003 = row_by_exp(rows, 'lmask003_lsem003_pd05_rw01_warmup')

    feature_reweight_harmful = None
    if no_rw is not None and rw01 is not None:
        no_rw_spice = delta(no_rw, 'SPICE')
        rw01_spice = delta(rw01, 'SPICE')
        no_rw_caption = to_float(no_rw.get('caption_metrics_above_baseline_count'))
        rw01_caption = to_float(rw01.get('caption_metrics_above_baseline_count'))
        feature_reweight_harmful = no_rw_spice > rw01_spice and no_rw_caption >= rw01_caption - 1

    alpha_too_strong = None
    if rw01 is not None:
        alpha_too_strong = parse_bool(rw01.get('all_above_test_baseline')) or to_float(rw01.get('caption_metrics_above_baseline_count')) >= 3

    lsem_too_strong = None
    if rw01 is not None and lsem003 is not None:
        lsem_too_strong = score_test_row(lsem003) > score_test_row(rw01)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Small Ablation SGC-CARD Report\n')
        f.write('==============================\n\n')
        for row in rows:
            f.write(
                '%s: snapshot=%s Bleu_4=%s METEOR=%s ROUGE_L=%s CIDEr=%s SPICE=%s all_above=%s main_failure=%s\n'
                % (
                    row.get('exp_name', ''),
                    row.get('recommended_snapshot', ''),
                    row.get('Bleu_4', ''),
                    row.get('METEOR', ''),
                    row.get('ROUGE_L', ''),
                    row.get('CIDEr', ''),
                    row.get('SPICE', ''),
                    row.get('all_above_test_baseline', ''),
                    row.get('main_failure_metric', ''),
                )
            )

        f.write('\nConclusion:\n')
        f.write('- Is feature reweight harmful? %s\n' % yes_no_unknown(feature_reweight_harmful))
        f.write('- Is alpha=0.2 too strong? %s (rw01 must be compared against the previous rw02 run for a firm answer)\n' % yes_no_unknown(alpha_too_strong))
        f.write('- Is lsem=0.005 too strong? %s\n' % yes_no_unknown(lsem_too_strong))
        if chosen is not None:
            f.write('- Which experiment should be used for final testing/reporting? %s\n' % chosen.get('exp_name', ''))
            f.write('  snapshot: %s\n' % chosen.get('recommended_snapshot', ''))
        else:
            f.write('- Which experiment should be used for final testing/reporting? unknown, no usable test rows found.\n')
        f.write('- Should we continue with another round of ablation? ')
        if chosen is not None and parse_bool(chosen.get('all_above_test_baseline')):
            f.write('no, first verify this checkpoint on the final reporting protocol.\n')
        else:
            f.write('yes, if none of these rows reaches all test baselines; focus on the main_failure_metric shown above.\n')


def main():
    args = parse_args()
    output_csv = args.output_csv or os.path.join(args.exp_root, 'small_ablation_sgc_card_summary.csv')
    output_txt = args.output_txt or os.path.join(args.exp_root, 'small_ablation_sgc_card_report.txt')
    rows = [build_summary_row(exp_name, args.exp_root) for exp_name in args.exp_names]
    write_csv(output_csv, rows)
    write_report(output_txt, rows)
    print('Wrote summary CSV: %s' % output_csv)
    print('Wrote summary report: %s' % output_txt)


if __name__ == '__main__':
    main()
