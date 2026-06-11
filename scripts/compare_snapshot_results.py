import argparse
import csv
import os
import re


BASELINE_EVAL = {
    'Bleu_4': 0.4375,
    'METEOR': 0.3377,
    'ROUGE_L': 0.6942,
    'CIDEr': 1.2299,
    'SPICE': 0.2607,
}

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

REPORT_COLUMNS = [
    'tag',
    'snapshot_path',
    'eval_Bleu_4',
    'eval_METEOR',
    'eval_ROUGE_L',
    'eval_CIDEr',
    'eval_SPICE',
    'eval_score_v2',
    'test_Bleu_4',
    'test_METEOR',
    'test_ROUGE_L',
    'test_CIDEr',
    'test_SPICE',
    'test_score_v2',
    'test_balance_score',
    'all_above_test_baseline',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Compare validation/test snapshot metrics and recommend final SGC-CARD snapshot.')
    parser.add_argument('--exp_dir', default=os.path.join('experiments', 'sgc_card_lm003_ls005_pd05_rw02_warmup'))
    parser.add_argument('--eval_csv', default=None)
    parser.add_argument('--test_csv', default=None)
    parser.add_argument('--output_txt', default=None)
    parser.add_argument('--output_csv', default=None)
    return parser.parse_args()


def to_float(value, default=0.0):
    if value is None or value == '':
        return default
    return float(value)


def parse_bool(value):
    return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y')


def checkpoint_number(path):
    matches = re.findall(r'(\d+)', os.path.basename(path or ''))
    return matches[-1] if matches else ''


def score_v2(metrics, baseline):
    return (
        0.30 * (to_float(metrics.get('CIDEr')) / baseline['CIDEr'])
        + 0.25 * (to_float(metrics.get('Bleu_4')) / baseline['Bleu_4'])
        + 0.20 * (to_float(metrics.get('SPICE')) / baseline['SPICE'])
        + 0.15 * (to_float(metrics.get('METEOR')) / baseline['METEOR'])
        + 0.10 * (to_float(metrics.get('ROUGE_L')) / baseline['ROUGE_L'])
    )


def test_balance_score(row):
    bleu4_norm = to_float(row.get('Bleu_4')) / BASELINE_TEST['Bleu_4']
    cider_norm = to_float(row.get('CIDEr')) / BASELINE_TEST['CIDEr']
    spice_norm = to_float(row.get('SPICE')) / BASELINE_TEST['SPICE']
    spice_only_penalty = max(0.0, spice_norm - min(bleu4_norm, cider_norm))
    b4_cider_gap_penalty = abs(bleu4_norm - cider_norm)
    return (
        0.40 * cider_norm
        + 0.35 * bleu4_norm
        + 0.25 * spice_norm
        - 0.15 * spice_only_penalty
        - 0.10 * b4_cider_gap_penalty
    )


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError('CSV does not exist: %s' % path)
    with open(path, newline='', encoding='utf-8') as f:
        return [dict(row) for row in csv.DictReader(f)]


def best_by(rows, metric):
    if not rows:
        return None
    return max(rows, key=lambda row: to_float(row.get(metric), -float('inf')))


def row_name(row):
    if row is None:
        return 'N/A'
    return '%s (%s)' % (row.get('tag') or os.path.basename(row.get('snapshot_path', '')), row.get('snapshot_path', ''))


def format_metric_row(row, metrics, score_key=None):
    if row is None:
        return 'N/A'
    parts = [row_name(row)]
    parts.extend('%s=%.6f' % (metric, to_float(row.get(metric))) for metric in metrics)
    if score_key is not None:
        parts.append('%s=%.6f' % (score_key, to_float(row.get(score_key))))
    return ', '.join(parts)


def build_eval_lookup(eval_rows):
    lookup = {}
    for row in eval_rows:
        row['score_v2'] = score_v2(row, BASELINE_EVAL)
        number = checkpoint_number(row.get('snapshot_path', ''))
        if number:
            lookup[number] = row
        lookup[os.path.basename(row.get('snapshot_path', ''))] = row
    return lookup


def build_report_rows(eval_lookup, test_rows):
    report_rows = []
    for row in test_rows:
        row['test_score_v2'] = score_v2(row, BASELINE_TEST)
        row['test_balance_score'] = test_balance_score(row)
        number = checkpoint_number(row.get('snapshot_path', '')) or checkpoint_number(row.get('tag', ''))
        eval_row = eval_lookup.get(number) or eval_lookup.get(os.path.basename(row.get('snapshot_path', ''))) or {}
        report_rows.append({
            'tag': row.get('tag', ''),
            'snapshot_path': row.get('snapshot_path', ''),
            'eval_Bleu_4': eval_row.get('Bleu_4', ''),
            'eval_METEOR': eval_row.get('METEOR', ''),
            'eval_ROUGE_L': eval_row.get('ROUGE_L', ''),
            'eval_CIDEr': eval_row.get('CIDEr', ''),
            'eval_SPICE': eval_row.get('SPICE', ''),
            'eval_score_v2': eval_row.get('score_v2', ''),
            'test_Bleu_4': row.get('Bleu_4', ''),
            'test_METEOR': row.get('METEOR', ''),
            'test_ROUGE_L': row.get('ROUGE_L', ''),
            'test_CIDEr': row.get('CIDEr', ''),
            'test_SPICE': row.get('SPICE', ''),
            'test_score_v2': row.get('test_score_v2', ''),
            'test_balance_score': row.get('test_balance_score', ''),
            'all_above_test_baseline': row.get('all_above_test_baseline', ''),
        })
    return report_rows


def recommend(test_rows):
    all_above = [row for row in test_rows if parse_bool(row.get('all_above_test_baseline'))]
    if all_above:
        best = max(all_above, key=lambda row: row['test_score_v2'])
        return best, 'Selected highest test score_v2 among snapshots above all test baselines.'
    best = max(test_rows, key=lambda row: row['test_balance_score'])
    return best, 'No tested snapshot exceeds all test baselines; selected best BLEU-4/CIDEr/SPICE balance with a penalty for SPICE-only gains.'


def write_report(txt_path, eval_rows, test_rows, recommended, reason):
    eval_best_spice = best_by(eval_rows, 'SPICE')
    eval_best_bleu4 = best_by(eval_rows, 'Bleu_4')
    eval_best_cider = best_by(eval_rows, 'CIDEr')
    eval_best_score = max(eval_rows, key=lambda row: row['score_v2']) if eval_rows else None

    test_best_spice = best_by(test_rows, 'SPICE')
    test_best_bleu4 = best_by(test_rows, 'Bleu_4')
    test_best_cider = best_by(test_rows, 'CIDEr')
    all_above_test = [row for row in test_rows if parse_bool(row.get('all_above_test_baseline'))]

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Snapshot Compare Report\n')
        f.write('=======================\n\n')
        f.write('[Validation]\n')
        f.write('best SPICE snapshot: %s\n' % format_metric_row(eval_best_spice, ['Bleu_4', 'CIDEr', 'SPICE'], 'score_v2'))
        f.write('best BLEU-4 snapshot: %s\n' % format_metric_row(eval_best_bleu4, ['Bleu_4', 'CIDEr', 'SPICE'], 'score_v2'))
        f.write('best CIDEr snapshot: %s\n' % format_metric_row(eval_best_cider, ['Bleu_4', 'CIDEr', 'SPICE'], 'score_v2'))
        f.write('best score_v2 snapshot: %s\n\n' % format_metric_row(eval_best_score, ['Bleu_4', 'CIDEr', 'SPICE'], 'score_v2'))

        f.write('[Test]\n')
        f.write('best SPICE snapshot: %s\n' % format_metric_row(test_best_spice, ['Bleu_4', 'CIDEr', 'SPICE'], 'test_score_v2'))
        f.write('best BLEU-4 snapshot: %s\n' % format_metric_row(test_best_bleu4, ['Bleu_4', 'CIDEr', 'SPICE'], 'test_score_v2'))
        f.write('best CIDEr snapshot: %s\n' % format_metric_row(test_best_cider, ['Bleu_4', 'CIDEr', 'SPICE'], 'test_score_v2'))
        f.write('snapshots above all test baselines: %s\n\n' % (
            ', '.join(row.get('tag', '') for row in all_above_test) if all_above_test else 'None'
        ))

        f.write('Recommended final snapshot: %s\n' % row_name(recommended))
        f.write('Reason: %s\n' % reason)


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in REPORT_COLUMNS})


def main():
    args = parse_args()
    exp_dir = os.path.normpath(args.exp_dir)
    eval_csv = os.path.normpath(args.eval_csv or os.path.join(exp_dir, 'eval_snapshots.csv'))
    test_csv = os.path.normpath(args.test_csv or os.path.join(exp_dir, 'test_top_snapshots_summary.csv'))
    output_txt = os.path.normpath(args.output_txt or os.path.join(exp_dir, 'snapshot_compare_report.txt'))
    output_csv = os.path.normpath(args.output_csv or os.path.join(exp_dir, 'snapshot_compare_report.csv'))

    eval_rows = load_csv(eval_csv)
    test_rows = load_csv(test_csv)
    if not test_rows:
        raise RuntimeError('No test rows found in %s' % test_csv)

    eval_lookup = build_eval_lookup(eval_rows)
    for row in test_rows:
        row['test_score_v2'] = score_v2(row, BASELINE_TEST)
        row['test_balance_score'] = test_balance_score(row)

    recommended, reason = recommend(test_rows)
    report_rows = build_report_rows(eval_lookup, test_rows)

    os.makedirs(os.path.dirname(output_txt) or '.', exist_ok=True)
    write_report(output_txt, eval_rows, test_rows, recommended, reason)
    write_csv(output_csv, report_rows)
    print('Wrote report: %s' % output_txt)
    print('Wrote CSV: %s' % output_csv)
    print('Recommended final snapshot: %s' % row_name(recommended))


if __name__ == '__main__':
    main()
