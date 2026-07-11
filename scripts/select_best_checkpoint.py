import argparse
import csv
import glob
import json
import os
import re


METRICS = ['Bleu_4', 'METEOR', 'CIDEr', 'SPICE']
ALL_METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
STRICT_METRICS = ['Bleu_4', 'CIDEr', 'SPICE']
STRICT_STRATEGIES = {'strict_spice', 'strict_spice_constrained_balanced', 'strict_nearest_balanced'}
DEFAULT_BASELINE = {
    'Bleu_4': 0.4375,
    'METEOR': 0.3377,
    'CIDEr': 1.2299,
    'SPICE': 0.2607,
}
DEFAULT_BASELINE_BY_DATASET = {
    'levir_mci': {
        'Bleu_4': 0.562,
        'CIDEr': 1.338,
        'SPICE': 0.336,
    },
    'second_cc': {
        'Bleu_4': 0.2837,
        'CIDEr': 0.7793,
        'SPICE': 0.2541,
    },
}
DEFAULT_BASELINE_EXPERIMENTS = {
    'levir_mci': 'levir_mci_card_baseline',
    'second_cc': 'second_cc_card_rgb_baseline',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Select best checkpoint with SPICE-constrained balanced score.')
    parser.add_argument('--exp_dir', required=True, help='Experiment directory.')
    parser.add_argument('--candidate_exp_dir', action='append', default=[], help='Additional experiment directory whose validation checkpoints join the candidate pool. Repeat as needed.')
    parser.add_argument('--metrics', default=None, help='Validation metrics CSV or JSON. Defaults to <exp_dir>/val_metrics.csv or eval_snapshots.csv.')
    parser.add_argument('--baseline_metrics', default=None, help='Baseline metrics JSON/CSV/TXT. If omitted, uses the built-in validation baseline.')
    parser.add_argument('--summary_csv', default=os.path.join('experiments', 'paper_required_experiments_summary.csv'), help='Optional paper summary CSV used to infer dataset baseline metrics.')
    parser.add_argument('--baseline_exp_name', default=None, help='Optional baseline experiment row name in --summary_csv.')
    parser.add_argument('--strategy', default='spice_constrained_balanced', choices=['best_cider', 'best_spice', 'balanced', 'spice_constrained_balanced', 'strict_spice_constrained_balanced', 'strict_spice', 'strict_nearest_balanced'])
    parser.add_argument('--strict_bleu4', type=float, default=0.562, help='BLEU-4 threshold for strict LEVIR-MCI reselect.')
    parser.add_argument('--strict_cider', type=float, default=1.338, help='CIDEr threshold for strict LEVIR-MCI reselect.')
    parser.add_argument('--strict_spice', type=float, default=0.336, help='SPICE threshold for strict LEVIR-MCI reselect.')
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
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


def load_first_csv_row(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def metric_dict_from_payload(payload):
    if isinstance(payload, dict):
        if 'metrics' in payload and isinstance(payload['metrics'], dict):
            return {key: to_float(value) for key, value in payload['metrics'].items() if key in ALL_METRICS}
        return {key: to_float(value) for key, value in payload.items() if key in ALL_METRICS}
    return {}


def load_summary_baseline(summary_csv, dataset_name='', baseline_exp_name=None):
    if not summary_csv or not os.path.exists(summary_csv):
        return {}, ''
    dataset_name = str(dataset_name or '').lower()
    target_exp = baseline_exp_name or DEFAULT_BASELINE_EXPERIMENTS.get(dataset_name)
    with open(summary_csv, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    candidates = []
    for row in rows:
        exp_name = str(row.get('exp_name') or row.get('experiment') or '')
        dataset = str(row.get('dataset') or '').lower()
        if target_exp and exp_name == target_exp:
            candidates.append(row)
        elif not target_exp and dataset_name and dataset == dataset_name and 'baseline' in exp_name:
            candidates.append(row)
    for row in candidates:
        metrics = {key: to_float(row.get(key), None) for key in ALL_METRICS if row.get(key) not in (None, '')}
        metrics = {key: value for key, value in metrics.items() if value is not None}
        if any(key in metrics for key in ('Bleu_4', 'CIDEr', 'SPICE')):
            source = row.get('exp_name') or row.get('experiment') or summary_csv
            return metrics, source
    return {}, ''


def load_baseline(path=None, dataset_name='', summary_csv=None, baseline_exp_name=None):
    dataset_key = str(dataset_name or '').lower()
    result = dict(DEFAULT_BASELINE)
    result.update(DEFAULT_BASELINE_BY_DATASET.get(dataset_key, {}))
    source = 'built_in_dataset:%s' % dataset_key if dataset_key in DEFAULT_BASELINE_BY_DATASET else 'built_in_default'
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError('Baseline metrics file does not exist: %s' % path)
        source = path
    if path and path.lower().endswith('.json'):
        baseline = metric_dict_from_payload(load_json(path))
    elif path:
        baseline = {key: to_float(load_first_csv_row(path).get(key)) for key in ALL_METRICS}
    else:
        baseline, inferred_source = load_summary_baseline(summary_csv, dataset_name, baseline_exp_name)
        if inferred_source:
            source = 'summary:%s' % inferred_source
    result.update({key: value for key, value in baseline.items() if key in ALL_METRICS})
    return result, source


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
            patterns.extend([
                os.path.join(exp_dir, folder, '*%s*.pt' % number),
                os.path.join(exp_dir, folder, '*%s*.pth' % number),
            ])
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
        with open(metrics_path, newline='', encoding='utf-8-sig') as f:
            raw_rows = list(csv.DictReader(f))
    rows = []
    for raw in raw_rows:
        row = dict(raw)
        row['checkpoint_path'] = resolve_checkpoint(row.get('checkpoint_path') or row.get('snapshot_path'), exp_dir, row)
        row['metrics'] = {key: to_float(row.get(key)) for key in ALL_METRICS if row.get(key) not in (None, '')}
        row['source_exp_dir'] = os.path.normpath(exp_dir)
        row['source_exp_name'] = os.path.basename(os.path.normpath(exp_dir))
        row['metrics_path'] = os.path.normpath(metrics_path)
        if row.get('checkpoint_path') and row.get('metrics'):
            rows.append(row)
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


def add_strict_constraint_scores(rows, strict_thresholds):
    thresholds = strict_thresholds or {}
    for row in rows:
        shortfall = {}
        normalized_shortfall = 0.0
        violation_count = 0
        for metric in STRICT_METRICS:
            threshold = to_float(thresholds.get(metric))
            value = to_float(row['metrics'].get(metric))
            deficit = max(0.0, threshold - value)
            shortfall[metric] = deficit
            if deficit > 0:
                violation_count += 1
                normalized_shortfall += deficit / max(abs(threshold), 1e-12)
        row['strict_constraints_met'] = violation_count == 0
        row['threshold_shortfall'] = shortfall
        row['constraint_violation_count'] = violation_count
        row['normalized_threshold_shortfall'] = normalized_shortfall


def select_rows(rows, baseline, strategy, strict_thresholds=None):
    add_balanced_scores(rows)
    relaxed = False
    candidates = rows
    if strategy == 'best_cider':
        key = lambda row: to_float(row['metrics'].get('CIDEr'))
    elif strategy == 'best_spice':
        key = lambda row: to_float(row['metrics'].get('SPICE'))
    else:
        if strategy in STRICT_STRATEGIES:
            add_strict_constraint_scores(rows, strict_thresholds)
            exact_candidates = [row for row in rows if row['strict_constraints_met']]
            if exact_candidates:
                candidates = exact_candidates
                key = lambda row: row['balanced_score']
            elif strategy == 'strict_nearest_balanced':
                relaxed = True
                candidates = rows
                key = lambda row: (
                    -row['constraint_violation_count'],
                    -row['normalized_threshold_shortfall'],
                    row['balanced_score'],
                )
            else:
                return None, [], False
        elif strategy == 'spice_constrained_balanced':
            candidates = [
                row for row in rows
                if to_float(row['metrics'].get('SPICE')) >= to_float(baseline.get('SPICE'))
                and to_float(row['metrics'].get('CIDEr')) >= to_float(baseline.get('CIDEr')) - 0.005
                and to_float(row['metrics'].get('Bleu_4')) >= to_float(baseline.get('Bleu_4'))
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
        if strategy not in STRICT_STRATEGIES:
            key = lambda row: row['balanced_score']
    best = sorted(
        candidates,
        key=lambda row: (key(row), to_float(row['metrics'].get('SPICE')), to_float(row['metrics'].get('CIDEr'))),
        reverse=True,
    )[0]
    return best, candidates, relaxed


def load_resolved_config(exp_dir):
    path = os.path.join(exp_dir, 'resolved_config.json')
    if not os.path.exists(path):
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def infer_exp_name(exp_dir, resolved):
    return str(resolved.get('exp_name') or os.path.basename(os.path.normpath(exp_dir)))


def infer_dataset_name(resolved):
    data = resolved.get('data', {}) if isinstance(resolved, dict) else {}
    return str(data.get('dataset') or resolved.get('dataset_name') or '')


def infer_dataset_name_from_exp_dir(exp_dir, resolved):
    dataset = infer_dataset_name(resolved)
    if dataset:
        return dataset
    name = os.path.basename(os.path.normpath(exp_dir)).lower()
    if name.startswith('levir_mci'):
        return 'levir_mci'
    if name.startswith('second_cc'):
        return 'second_cc'
    return ''


def selected_epoch_or_step(row):
    for key in ('epoch', 'step', 'iter'):
        value = row.get(key)
        if value not in (None, ''):
            return value
    return checkpoint_number(row.get('checkpoint_path')) or ''


def main():
    args = parse_args()
    exp_dir = os.path.normpath(args.exp_dir)
    output_json = os.path.normpath(args.output_json or os.path.join(exp_dir, 'best_checkpoint.json'))
    output_txt = os.path.normpath(args.output_txt or os.path.join(exp_dir, 'best_checkpoint.txt'))

    candidate_exp_dirs = []
    for candidate_dir in [exp_dir] + list(args.candidate_exp_dir):
        normalized = os.path.normpath(candidate_dir)
        if normalized not in candidate_exp_dirs:
            candidate_exp_dirs.append(normalized)

    rows = []
    metrics_paths = []
    for index, candidate_dir in enumerate(candidate_exp_dirs):
        metrics_path = os.path.normpath(
            args.metrics if index == 0 and args.metrics else resolve_default_metrics(candidate_dir)
        )
        rows.extend(load_rows(metrics_path, candidate_dir))
        metrics_paths.append(metrics_path)

    resolved = load_resolved_config(exp_dir)
    dataset_name = infer_dataset_name_from_exp_dir(exp_dir, resolved)
    baseline, baseline_source = load_baseline(
        args.baseline_metrics,
        dataset_name=dataset_name,
        summary_csv=args.summary_csv,
        baseline_exp_name=args.baseline_exp_name,
    )
    strict_thresholds = {
        'Bleu_4': args.strict_bleu4,
        'CIDEr': args.strict_cider,
        'SPICE': args.strict_spice,
    }
    best, candidates, relaxed = select_rows(rows, baseline, args.strategy, strict_thresholds=strict_thresholds)
    if best is None:
        payload = {
            'exp_name': infer_exp_name(exp_dir, resolved),
            'dataset_name': dataset_name,
            'selected_checkpoint': '',
            'selected_checkpoint_path': '',
            'selected_epoch_or_step': '',
            'selected_epoch': '',
            'selected_step': '',
            'selection_strategy': args.strategy,
            'status': 'no_valid_checkpoint',
            'failure_reason': 'no_checkpoint_satisfies_strict_constraints',
            'notes': 'no checkpoint satisfies strict SPICE/CIDEr/BLEU constraints',
            'strict_constraints': strict_thresholds,
            'baseline_metrics': baseline,
            'baseline_source': baseline_source,
            'selected_val_metrics': {},
            'selected_metrics': {},
            'balanced_score': None,
            'metrics_file': metrics_paths[0] if len(metrics_paths) == 1 else '',
            'metrics_files': metrics_paths,
            'candidate_exp_dirs': candidate_exp_dirs,
            'candidate_count': 0,
            'all_candidate_count': len(rows),
        }
        os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write('no_checkpoint_satisfies_strict_constraints\n')
            f.write(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    epoch_or_step = selected_epoch_or_step(best)
    uses_strict_constraints = args.strategy in STRICT_STRATEGIES
    strict_constraints_met = best.get('strict_constraints_met') if uses_strict_constraints else None
    notes = ''
    if args.strategy == 'strict_nearest_balanced' and not strict_constraints_met:
        notes = 'no checkpoint satisfies all strict constraints; selected nearest balanced checkpoint'
    payload = {
        'exp_name': infer_exp_name(exp_dir, resolved),
        'dataset_name': dataset_name,
        'selected_checkpoint': best['checkpoint_path'],
        'selected_checkpoint_path': best['checkpoint_path'],
        'selected_epoch_or_step': epoch_or_step,
        'selected_epoch': best.get('epoch', ''),
        'selected_step': best.get('step', best.get('iter', '')),
        'selection_strategy': args.strategy,
        'status': 'done',
        'notes': notes,
        'constraints_relaxed': relaxed,
        'strict_constraints': strict_thresholds if uses_strict_constraints else {},
        'strict_constraints_met': strict_constraints_met,
        'constraint_violation_count': best.get('constraint_violation_count'),
        'threshold_shortfall': best.get('threshold_shortfall', {}),
        'normalized_threshold_shortfall': best.get('normalized_threshold_shortfall'),
        'baseline_metrics': baseline,
        'baseline_source': baseline_source,
        'selected_source_exp_dir': best.get('source_exp_dir', exp_dir),
        'selected_source_exp_name': best.get('source_exp_name', infer_exp_name(exp_dir, resolved)),
        'selected_val_metrics': best['metrics'],
        'selected_metrics': best['metrics'],
        'balanced_score': best.get('balanced_score'),
        'metrics_file': best.get('metrics_path', metrics_paths[0]),
        'metrics_files': metrics_paths,
        'candidate_exp_dirs': candidate_exp_dirs,
        'candidate_count': len(candidates),
        'all_candidate_count': len(rows),
    }
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(payload['selected_checkpoint'] + '\n')
        f.write(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
