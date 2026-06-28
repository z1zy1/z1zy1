import argparse
import csv
import glob
import json
import os
import re


METRICS = ['Bleu_4', 'METEOR', 'CIDEr', 'SPICE']
ALL_METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
DEFAULT_BASELINE = {
    'Bleu_4': 0.0,
    'METEOR': 0.0,
    'CIDEr': 0.0,
    'SPICE': 0.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Select best checkpoint with SPICE-constrained balanced score.')
    parser.add_argument('--exp_dir', required=True, help='Experiment directory.')
    parser.add_argument('--metrics', default=None, help='Validation metrics CSV or JSON. Defaults to <exp_dir>/val_metrics.csv or eval_snapshots.csv.')
    parser.add_argument('--baseline_metrics', default=None, help='Baseline metrics JSON/CSV/TXT.')
    parser.add_argument('--strategy', default='spice_constrained_balanced', choices=['best_cider', 'best_spice', 'balanced', 'spice_constrained_balanced'])
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--output_txt', default=None)
    return parser.parse_args()


def to_float(value, default=0.0):
    if value in (None, ''):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def metric_dict_from_payload(payload):
    if isinstance(payload, dict):
        if 'metrics' in payload and isinstance(payload['metrics'], dict):
            return {key: to_float(value) for key, value in payload['metrics'].items() if key in ALL_METRICS}
        return {key: to_float(value) for key, value in payload.items() if key in ALL_METRICS}
    return {}


def load_baseline(path):
    if not path:
        return dict(DEFAULT_BASELINE)
    if not os.path.exists(path):
        raise FileNotFoundError('Baseline metrics file does not exist: %s' % path)
    if path.lower().endswith('.json'):
        baseline = metric_dict_from_payload(load_json(path))
    else:
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        baseline = {key: to_float(rows[0].get(key)) for key in ALL_METRICS} if rows else {}
    result = dict(DEFAULT_BASELINE)
    result.update({key: value for key, value in baseline.items() if key in ALL_METRICS})
    return result


def resolve_default_metrics(exp_dir):
    for name in ('val_metrics.csv', 'eval_snapshots.csv'):
        path = os.path.join(exp_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError('No validation metrics found. Expected val_metrics.csv or eval_snapshots.csv under %s.' % exp_dir)


def checkpoint_number(path):
    matches = re.findall(r'(\d+)', os.path.basename(path or ''))
    return matches[-1] if matches else None


def first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return os.path.normpath(path)
    return None


def resolve_checkpoint(raw_path, exp_dir, row):
    raw_path = (raw_path or row.get('checkpoint_path') or row.get('snapshot_path') or '').strip()
    candidates = []
    if raw_path:
        candidates.extend([
            raw_path,
            os.path.abspath(raw_path),
            os.path.join(exp_dir, raw_path),
            os.path.join(exp_dir, os.path.basename(raw_path)),
            os.path.join(exp_dir, 'snapshots', os.path.basename(raw_path)),
            os.path.join(exp_dir, 'checkpoints', os.path.basename(raw_path)),
        ])
    found = first_existing(candidates)
    if found:
        return found
    number = checkpoint_number(raw_path) or row.get('iter') or row.get('step') or row.get('epoch')
    patterns = []
    if number:
        for folder in ('snapshots', 'checkpoints', ''):
            for ext in ('*.pt', '*.pth'):
                patterns.append(os.path.join(exp_dir, folder, '*%s*%s' % (number, ext[1:])))
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = sorted({os.path.normpath(path) for path in matches if os.path.exists(path)})
    return matches[0] if matches else os.path.normpath(raw_path)


def load_rows(metrics_path, exp_dir):
    if metrics_path.lower().endswith('.json'):
        payload = load_json(metrics_path)
        raw_rows = payload if isinstance(payload, list) else payload.get('checkpoints', payload.get('rows', []))
    else:
        with open(metrics_path, newline='', encoding='utf-8') as f:
            raw_rows = list(csv.DictReader(f))
    rows = []
    for raw in raw_rows:
        row = dict(raw)
        row['checkpoint_path'] = resolve_checkpoint(row.get('checkpoint_path') or row.get('snapshot_path'), exp_dir, row)
        row['metrics'] = {key: to_float(row.get(key)) for key in ALL_METRICS if row.get(key) not in (None, '')}
        rows.append(row)
    rows = [row for row in rows if row.get('checkpoint_path') and row.get('metrics')]
    if not rows:
        raise RuntimeError('No checkpoint metric rows found in %s.' % metrics_path)
    return rows


def minmax_norm(rows, metric):
    values = [to_float(row['metrics'].get(metric)) for row in rows]
    mn, mx = min(values), max(values)
    if mx == mn:
        return {id(row): 0.0 for row in rows}
    return {id(row): (to_float(row['metrics'].get(metric)) - mn) / (mx - mn) for row in rows}


def add_balanced_scores(rows):
    norms = {metric: minmax_norm(rows, metric) for metric in METRICS}
    for row in rows:
        row['balanced_score'] = sum(0.25 * norms[metric][id(row)] for metric in METRICS)


def select_rows(rows, baseline, strategy):
    add_balanced_scores(rows)
    relaxed = False
    candidates = rows
    if strategy == 'best_cider':
        key = lambda row: to_float(row['metrics'].get('CIDEr'))
    elif strategy == 'best_spice':
        key = lambda row: to_float(row['metrics'].get('SPICE'))
    else:
        if strategy == 'spice_constrained_balanced':
            candidates = [
                row for row in rows
                if to_float(row['metrics'].get('SPICE')) >= to_float(baseline.get('SPICE'))
                and to_float(row['metrics'].get('CIDEr')) >= to_float(baseline.get('CIDEr')) - 0.005
                and to_float(row['metrics'].get('Bleu_4')) >= to_float(baseline.get('Bleu_4')) - 0.005
            ]
            if not candidates:
                relaxed = True
                candidates = [
                    row for row in rows
                    if to_float(row['metrics'].get('SPICE')) >= to_float(baseline.get('SPICE')) - 0.005
                ]
            if not candidates:
                relaxed = True
                candidates = rows
        key = lambda row: row['balanced_score']
    best = sorted(candidates, key=lambda row: (key(row), to_float(row['metrics'].get('SPICE')), to_float(row['metrics'].get('CIDEr'))), reverse=True)[0]
    return best, candidates, relaxed


def main():
    args = parse_args()
    exp_dir = os.path.normpath(args.exp_dir)
    metrics_path = os.path.normpath(args.metrics or resolve_default_metrics(exp_dir))
    output_json = os.path.normpath(args.output_json or os.path.join(exp_dir, 'best_checkpoint.json'))
    output_txt = os.path.normpath(args.output_txt or os.path.join(exp_dir, 'best_checkpoint.txt'))
    baseline = load_baseline(args.baseline_metrics)
    rows = load_rows(metrics_path, exp_dir)
    best, candidates, relaxed = select_rows(rows, baseline, args.strategy)
    payload = {
        'selected_checkpoint_path': best['checkpoint_path'],
        'selected_epoch': best.get('epoch', ''),
        'selected_step': best.get('step', best.get('iter', '')),
        'selected_metrics': best['metrics'],
        'baseline_metrics': baseline,
        'selection_strategy': args.strategy,
        'constraints_relaxed': relaxed,
        'balanced_score': best.get('balanced_score'),
        'metrics_file': metrics_path,
        'candidate_count': len(candidates),
    }
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(payload['selected_checkpoint_path'] + '\n')
        f.write(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()