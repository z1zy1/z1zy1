import argparse
import csv
import json
import os
import shutil


BASELINE_EVAL = {
    'Bleu_4': 0.4375,
    'METEOR': 0.3377,
    'ROUGE_L': 0.6942,
    'CIDEr': 1.2299,
    'SPICE': 0.2607,
}

METRIC_COLUMNS = [
    'Bleu_1',
    'Bleu_2',
    'Bleu_3',
    'Bleu_4',
    'METEOR',
    'ROUGE_L',
    'CIDEr',
    'SPICE',
]


def parse_args():
    default_exp_dir = os.path.join('experiments', 'sgc_card_lm003_ls005_pd05_rw02_warmup')
    parser = argparse.ArgumentParser(description='Select the best SGC-CARD snapshot from eval_snapshots.csv.')
    parser.add_argument('--csv', default=os.path.join(default_exp_dir, 'eval_snapshots.csv'))
    parser.add_argument('--output_json', default=os.path.join(default_exp_dir, 'best_snapshot.json'))
    parser.add_argument('--copy_path', default=os.path.join(default_exp_dir, 'best_balanced.pth'))
    parser.add_argument('--best_cider_path', default=None)
    parser.add_argument('--best_spice_path', default=None)
    return parser.parse_args()


def parse_bool(value):
    return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y')


def to_float(value, default=0.0):
    if value is None or value == '':
        return default
    return float(value)


def balanced_score(row):
    return (
        0.25 * (to_float(row.get('CIDEr')) / BASELINE_EVAL['CIDEr'])
        + 0.25 * (to_float(row.get('SPICE')) / BASELINE_EVAL['SPICE'])
        + 0.20 * (to_float(row.get('Bleu_4')) / BASELINE_EVAL['Bleu_4'])
        + 0.15 * (to_float(row.get('METEOR')) / BASELINE_EVAL['METEOR'])
        + 0.15 * (to_float(row.get('ROUGE_L')) / BASELINE_EVAL['ROUGE_L'])
    )


def not_above_baseline_metrics(row):
    missing = []
    for metric, baseline in BASELINE_EVAL.items():
        if to_float(row.get(metric), default=-float('inf')) <= baseline:
            missing.append(metric)
    return missing


def load_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError('CSV does not exist: %s' % path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError('CSV has no snapshot rows: %s' % path)

    normalized = []
    for row in rows:
        row = dict(row)
        if 'balanced_score' not in row or row['balanced_score'] == '':
            row['balanced_score'] = balanced_score(row)
        else:
            row['balanced_score'] = to_float(row['balanced_score'])
        if 'all_above_baseline' not in row or row['all_above_baseline'] == '':
            row['all_above_baseline'] = len(not_above_baseline_metrics(row)) == 0
        else:
            row['all_above_baseline'] = parse_bool(row['all_above_baseline'])
        normalized.append(row)
    return normalized


def copy_snapshot(src, dst):
    if not src:
        return
    if not os.path.exists(src):
        raise FileNotFoundError('Selected snapshot does not exist: %s' % src)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    shutil.copy2(src, dst)


def row_metrics(row):
    return {metric: to_float(row.get(metric)) for metric in METRIC_COLUMNS if row.get(metric) not in (None, '')}


def main():
    args = parse_args()
    rows = load_rows(args.csv)

    all_above_rows = [row for row in rows if row['all_above_baseline']]
    candidates = all_above_rows if all_above_rows else rows
    best = max(candidates, key=lambda row: row['balanced_score'])

    if all_above_rows:
        selection_reason = 'Selected highest balanced_score among snapshots above all validation baselines.'
    else:
        selection_reason = 'No snapshot exceeded all validation baselines; selected highest balanced_score.'

    not_above = not_above_baseline_metrics(best)
    payload = {
        'best_snapshot': best['snapshot_path'],
        'selection_reason': selection_reason,
        'metrics': row_metrics(best),
        'balanced_score': float(best['balanced_score']),
        'all_above_baseline': bool(best['all_above_baseline']),
        'not_above_baseline_metrics': not_above,
    }

    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    copy_snapshot(best['snapshot_path'], args.copy_path)

    best_cider_path = args.best_cider_path or os.path.join(os.path.dirname(args.copy_path), 'best_cider.pth')
    best_spice_path = args.best_spice_path or os.path.join(os.path.dirname(args.copy_path), 'best_spice.pth')
    copy_snapshot(max(rows, key=lambda row: to_float(row.get('CIDEr'), -float('inf')))['snapshot_path'], best_cider_path)
    copy_snapshot(max(rows, key=lambda row: to_float(row.get('SPICE'), -float('inf')))['snapshot_path'], best_spice_path)

    print(json.dumps(payload, indent=2))
    print('Copied balanced snapshot to %s' % args.copy_path)


if __name__ == '__main__':
    main()
