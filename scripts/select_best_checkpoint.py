import argparse
import csv
import glob
import json
import math
import os
import re
import statistics


METRICS = ['Bleu_4', 'METEOR', 'CIDEr', 'SPICE']
ALL_METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
STRICT_METRICS = ['Bleu_4', 'CIDEr', 'SPICE']
DEFAULT_PROTECTED_METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
STRICT_STRATEGIES = {'strict_spice', 'strict_spice_constrained_balanced', 'strict_nearest_balanced'}
BASELINE_VALIDATION_STRATEGIES = {'val_baseline_pareto', 'val_baseline_stable_window'}
VALIDATION_ONLY_STRATEGIES = {'validation_best_cider'} | BASELINE_VALIDATION_STRATEGIES
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
    parser.add_argument('--strategy', default='spice_constrained_balanced', choices=['best_cider', 'validation_best_cider', 'best_spice', 'balanced', 'spice_constrained_balanced', 'strict_spice_constrained_balanced', 'strict_spice', 'strict_nearest_balanced', 'val_baseline_pareto', 'val_baseline_stable_window'])
    parser.add_argument('--protected_metrics', default=','.join(DEFAULT_PROTECTED_METRICS), help='Comma-separated validation metrics that may not fall below baseline-tolerance.')
    parser.add_argument('--baseline_tolerance', type=float, default=0.0, help='Uniform non-negative validation baseline tolerance.')
    parser.add_argument('--min_spice_gain', type=float, default=None, help='Optional minimum selected validation SPICE gain over the explicit baseline.')
    parser.add_argument('--require_audited_validation_baseline', action='store_true', help='Require validation-only CARD baseline provenance metadata (recommended for locked workflows).')
    parser.add_argument('--stability_window', type=int, default=3, help='Odd validation window size for val_baseline_stable_window (default: 3).')
    parser.add_argument('--expected_step_gap', type=int, default=0, help='Required adjacent validation step gap for val_baseline_stable_window.')
    parser.add_argument('--strict_bleu4', type=float, default=0.562, help='BLEU-4 threshold for strict LEVIR-MCI reselect.')
    parser.add_argument('--strict_cider', type=float, default=1.338, help='CIDEr threshold for strict LEVIR-MCI reselect.')
    parser.add_argument('--strict_spice', type=float, default=0.336, help='SPICE threshold for strict LEVIR-MCI reselect.')
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--output_txt', default=None)
    args = parser.parse_args()
    if args.strategy in BASELINE_VALIDATION_STRATEGIES and not args.baseline_metrics:
        parser.error('--baseline_metrics is required for %s; summary/test thresholds are never used.' % args.strategy)
    if not math.isfinite(args.baseline_tolerance) or args.baseline_tolerance < 0:
        parser.error('--baseline_tolerance must be finite and non-negative.')
    if args.min_spice_gain is not None and not math.isfinite(args.min_spice_gain):
        parser.error('--min_spice_gain must be finite when provided.')
    if args.stability_window < 3 or args.stability_window % 2 == 0:
        parser.error('--stability_window must be an odd integer >= 3.')
    if args.strategy == 'val_baseline_stable_window' and args.expected_step_gap <= 0:
        parser.error('--expected_step_gap must be provided and positive for val_baseline_stable_window.')
    return args


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


def strict_finite_float(value):
    if isinstance(value, bool):
        raise ValueError('boolean is not a validation metric')
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError('validation metric is not finite')
    return numeric


def metric_dict_from_payload(payload, strict_finite=False):
    if isinstance(payload, dict):
        containers = [
            payload[field]
            for field in ('metrics', 'selected_val_metrics', 'selected_metrics')
            if isinstance(payload.get(field), dict)
        ]
        containers = containers or [payload]

        def parse_container(source):
            parsed = {}
            for key, value in source.items():
                if key not in ALL_METRICS:
                    continue
                if strict_finite:
                    try:
                        parsed[key] = strict_finite_float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError('Invalid validation baseline metric %s: %s' % (key, exc))
                else:
                    parsed[key] = to_float(value)
            return parsed

        result = parse_container(containers[0])
        if strict_finite:
            for other in containers[1:]:
                if parse_container(other) != result:
                    raise ValueError('Validation baseline artifact contains conflicting metric containers.')
        return result
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


def load_baseline(path=None, dataset_name='', summary_csv=None, baseline_exp_name=None,
                  explicit_only=False, strict_finite=False,
                  require_validation_artifact=False):
    dataset_key = str(dataset_name or '').lower()
    result = {} if explicit_only else dict(DEFAULT_BASELINE)
    if not explicit_only:
        result.update(DEFAULT_BASELINE_BY_DATASET.get(dataset_key, {}))
    source = 'built_in_dataset:%s' % dataset_key if dataset_key in DEFAULT_BASELINE_BY_DATASET else 'built_in_default'
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError('Baseline metrics file does not exist: %s' % path)
        source = path
    if path and path.lower().endswith('.json'):
        payload = load_json(path)
        if require_validation_artifact:
            if not isinstance(payload, dict):
                raise ValueError('Validation baseline artifact must be a JSON object: %s' % path)
            required = {
                'status': 'done',
                'selection_uses_test_metrics': False,
                'selection_metric_split': 'validation',
            }
            invalid = [key for key, expected in required.items() if payload.get(key) != expected]
            if invalid:
                raise ValueError(
                    'Baseline artifact is not an audited validation-only selection (%s): %s'
                    % (', '.join(invalid), path)
                )
            strategy = payload.get('selection_strategy')
            if strategy != 'validation_best_cider':
                raise ValueError('CARD baseline artifact must use validation_best_cider: %s' % path)
            identity = ' '.join(str(payload.get(key, '')) for key in (
                'exp_name', 'selected_source_exp_name', 'candidate_exp_dirs'))
            if 'baseline' not in identity.lower():
                raise ValueError('Baseline artifact does not identify a baseline experiment: %s' % path)
            if 'test' in os.path.basename(path).lower():
                raise ValueError('Test result JSON cannot be used as validation baseline: %s' % path)
        baseline = metric_dict_from_payload(payload, strict_finite=strict_finite)
    elif path:
        raw = load_first_csv_row(path)
        baseline = {}
        for key in ALL_METRICS:
            if raw.get(key) in (None, ''):
                continue
            if require_validation_artifact:
                raise ValueError('Audited validation baseline must be a JSON selection artifact: %s' % path)
            if strict_finite:
                try:
                    baseline[key] = strict_finite_float(raw.get(key))
                except (TypeError, ValueError) as exc:
                    raise ValueError('Invalid validation baseline metric %s: %s' % (key, exc))
            else:
                baseline[key] = to_float(raw.get(key))
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
        row['metrics'] = {}
        row['invalid_metrics'] = {}
        for key in ALL_METRICS:
            raw_value = row.get(key)
            if raw_value in (None, ''):
                continue
            try:
                row['metrics'][key] = strict_finite_float(raw_value)
            except (TypeError, ValueError) as exc:
                row['invalid_metrics'][key] = str(exc)
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


def parse_protected_metrics(value):
    metrics = []
    for item in str(value or '').split(','):
        metric = item.strip()
        if metric and metric not in metrics:
            metrics.append(metric)
    unknown = [metric for metric in metrics if metric not in ALL_METRICS or metric == 'SPICE']
    if unknown:
        raise ValueError('Unknown/invalid protected metrics: %s' % ', '.join(unknown))
    if not metrics:
        raise ValueError('--protected_metrics must contain at least one metric.')
    return metrics


def pareto_front(rows, metrics):
    front = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            at_least_as_good = all(
                to_float(other['metrics'].get(metric), -float('inf'))
                >= to_float(candidate['metrics'].get(metric), -float('inf'))
                for metric in metrics
            )
            strictly_better = any(
                to_float(other['metrics'].get(metric), -float('inf'))
                > to_float(candidate['metrics'].get(metric), -float('inf'))
                for metric in metrics
            )
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def select_val_baseline_pareto(rows, baseline, protected_metrics, tolerance,
                               min_spice_gain=None):
    add_balanced_scores(rows)
    feasible = []
    required_candidate_metrics = list(protected_metrics) + ['SPICE']
    for row in rows:
        if not has_complete_finite_metrics(row):
            continue
        if not os.path.isfile(row.get('checkpoint_path', '')):
            continue
        deltas = {
            metric: to_float(row['metrics'].get(metric)) - to_float(baseline.get(metric))
            for metric in required_candidate_metrics
        }
        row['baseline_deltas'] = deltas
        row['protected_relative_margins'] = {
            metric: deltas[metric] / max(abs(to_float(baseline.get(metric))), 1e-12)
            for metric in protected_metrics
        }
        row['min_protected_relative_margin'] = min(row['protected_relative_margins'].values())
        row['spice_gain'] = deltas['SPICE']
        spice_ok = min_spice_gain is None or deltas['SPICE'] >= min_spice_gain
        if all(deltas[metric] >= -tolerance for metric in protected_metrics) and spice_ok:
            feasible.append(row)
    if not feasible:
        return None, [], []
    front = pareto_front(feasible, list(protected_metrics) + ['SPICE'])
    best = sorted(
        front,
        key=lambda row: (
            row['spice_gain'],
            row['min_protected_relative_margin'],
            row['balanced_score'],
            to_float(row['metrics'].get('CIDEr')),
        ),
        reverse=True,
    )[0]
    return best, feasible, front


def has_complete_finite_metrics(row):
    metrics = row.get('metrics', {})
    for metric in ALL_METRICS:
        if metric not in metrics:
            return False
        try:
            if not math.isfinite(float(metrics[metric])):
                return False
        except (TypeError, ValueError):
            return False
    return True


def numeric_step(row):
    value = selected_epoch_or_step(row)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def metric_aggregates(rows):
    mean = {}
    std = {}
    worst = {}
    for metric in ALL_METRICS:
        values = [float(row['metrics'][metric]) for row in rows]
        mean[metric] = statistics.mean(values)
        std[metric] = statistics.stdev(values) if len(values) > 1 else 0.0
        worst[metric] = min(values)
    return mean, std, worst


def select_val_baseline_stable_window(rows, baseline, protected_metrics, tolerance,
                                      window_size, expected_step_gap,
                                      min_spice_gain=None):
    '''Select the centre checkpoint of the strongest safe contiguous validation window.

    A window never crosses experiment boundaries. Its checkpoint steps must be
    adjacent at the explicitly required validation interval, every
    member must contain all eight finite metrics, and every member (not merely
    the mean) must preserve all protected baseline metrics.
    '''
    add_balanced_scores(rows)
    by_source = {}
    for row in rows:
        if not has_complete_finite_metrics(row):
            continue
        if not os.path.isfile(row.get('checkpoint_path', '')):
            continue
        step = numeric_step(row)
        if step is None:
            continue
        row['_numeric_step'] = step
        by_source.setdefault(row.get('source_exp_dir', ''), []).append(row)

    stable_windows = []
    for source, source_rows in by_source.items():
        source_rows = sorted(source_rows, key=lambda row: row['_numeric_step'])
        steps = [row['_numeric_step'] for row in source_rows]
        if len(steps) != len(set(steps)):
            raise ValueError('Duplicate validation checkpoint steps in %s.' % source)
        if len(steps) < window_size:
            continue
        for start in range(0, len(source_rows) - window_size + 1):
            members = source_rows[start:start + window_size]
            member_steps = [row['_numeric_step'] for row in members]
            member_gaps = [right - left for left, right in zip(member_steps, member_steps[1:])]
            if any(gap != expected_step_gap for gap in member_gaps):
                continue
            member_deltas = []
            safe = True
            for member in members:
                deltas = {
                    metric: float(member['metrics'][metric]) - float(baseline[metric])
                    for metric in protected_metrics + ['SPICE']
                }
                member_deltas.append(deltas)
                if any(deltas[metric] < -tolerance for metric in protected_metrics):
                    safe = False
                    break
                if min_spice_gain is not None and deltas['SPICE'] < min_spice_gain:
                    safe = False
                    break
            if not safe:
                continue
            mean, std, worst = metric_aggregates(members)
            centre = dict(members[window_size // 2])
            centre_deltas = {
                metric: float(centre['metrics'][metric]) - float(baseline[metric])
                for metric in protected_metrics + ['SPICE']
            }
            if min_spice_gain is not None:
                if centre_deltas['SPICE'] < min_spice_gain:
                    continue
                if mean['SPICE'] - float(baseline['SPICE']) < min_spice_gain:
                    continue
            centre['baseline_deltas'] = centre_deltas
            centre['protected_relative_margins'] = {
                metric: centre_deltas[metric] / max(abs(float(baseline[metric])), 1e-12)
                for metric in protected_metrics
            }
            centre['min_protected_relative_margin'] = min(
                centre['protected_relative_margins'].values()
            )
            centre['spice_gain'] = centre_deltas['SPICE']
            window_relative_margins = [
                deltas[metric] / max(abs(float(baseline[metric])), 1e-12)
                for deltas in member_deltas
                for metric in protected_metrics
            ]
            centre['stability_window'] = {
                'size': window_size,
                'source_exp_dir': source,
                'source_exp_name': centre.get('source_exp_name', os.path.basename(source)),
                'expected_step_gap': expected_step_gap,
                'member_steps': member_steps,
                'member_checkpoints': [member['checkpoint_path'] for member in members],
                'member_metrics': [member['metrics'] for member in members],
                'member_metric_deltas': member_deltas,
                'metric_mean': mean,
                'metric_sample_std': std,
                'metric_worst': worst,
                'all_members_complete_finite': True,
                'all_members_preserve_validation_baseline': True,
                'all_members_preserve_spice_gain': True,
                'min_protected_relative_margin': min(window_relative_margins),
            }
            stable_windows.append(centre)

    if not stable_windows:
        return None, []
    best = sorted(
        stable_windows,
        key=lambda row: (
            row['stability_window']['metric_mean']['SPICE'],
            row['stability_window']['min_protected_relative_margin'],
            row['stability_window']['metric_worst']['SPICE'],
            row['stability_window']['metric_mean']['CIDEr'],
            -row['stability_window']['metric_sample_std']['SPICE'],
            float(row['metrics']['SPICE']),
        ),
        reverse=True,
    )[0]
    return best, stable_windows


def select_rows(rows, baseline, strategy, strict_thresholds=None):
    add_balanced_scores(rows)
    relaxed = False
    candidates = rows
    if strategy == 'validation_best_cider':
        candidates = [row for row in rows if has_complete_finite_metrics(row)]
        if not candidates:
            return None, [], False
        key = lambda row: to_float(row['metrics'].get('CIDEr'))
    elif strategy == 'best_cider':
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
    protected_metrics = parse_protected_metrics(args.protected_metrics)
    if args.strategy == 'validation_best_cider':
        baseline, baseline_source = {}, 'not_used'
    else:
        baseline, baseline_source = load_baseline(
            args.baseline_metrics,
            dataset_name=dataset_name,
            summary_csv=args.summary_csv,
            baseline_exp_name=args.baseline_exp_name,
            explicit_only=args.strategy in BASELINE_VALIDATION_STRATEGIES,
            strict_finite=args.strategy in BASELINE_VALIDATION_STRATEGIES,
            require_validation_artifact=args.require_audited_validation_baseline,
        )
    if args.strategy in BASELINE_VALIDATION_STRATEGIES:
        missing = [metric for metric in protected_metrics + ['SPICE'] if metric not in baseline]
        if missing:
            raise ValueError('Baseline metrics file is missing required metrics: %s' % ', '.join(missing))
    strict_thresholds = {
        'Bleu_4': args.strict_bleu4,
        'CIDEr': args.strict_cider,
        'SPICE': args.strict_spice,
    }
    pareto = []
    stable_windows = []
    if args.strategy == 'val_baseline_pareto':
        best, candidates, pareto = select_val_baseline_pareto(
            rows,
            baseline,
            protected_metrics,
            args.baseline_tolerance,
            args.min_spice_gain,
        )
        relaxed = False
    elif args.strategy == 'val_baseline_stable_window':
        best, stable_windows = select_val_baseline_stable_window(
            rows,
            baseline,
            protected_metrics,
            args.baseline_tolerance,
            args.stability_window,
            args.expected_step_gap,
            args.min_spice_gain,
        )
        candidates = stable_windows
        relaxed = False
    else:
        best, candidates, relaxed = select_rows(
            rows,
            baseline,
            args.strategy,
            strict_thresholds=strict_thresholds,
        )
    if best is None:
        is_val_pareto = args.strategy == 'val_baseline_pareto'
        is_val_baseline = args.strategy in BASELINE_VALIDATION_STRATEGIES
        if args.strategy == 'validation_best_cider':
            failure_reason = 'no_checkpoint_with_complete_validation_metrics'
        elif args.strategy == 'val_baseline_stable_window':
            failure_reason = 'no_stable_validation_window_preserves_baseline'
        elif is_val_pareto:
            failure_reason = 'no_checkpoint_preserves_validation_baseline'
        else:
            failure_reason = 'no_checkpoint_satisfies_strict_constraints'
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
            'failure_reason': failure_reason,
            'notes': failure_reason,
            'strict_constraints': strict_thresholds,
            'baseline_metrics': baseline,
            'baseline_source': baseline_source,
            'protected_metrics': protected_metrics if is_val_baseline else [],
            'baseline_tolerance': args.baseline_tolerance if is_val_baseline else None,
            'min_spice_gain': args.min_spice_gain if is_val_baseline else None,
            'selected_metric_deltas': {},
            'pareto_front_count': 0,
            'stable_window_count': 0,
            'stability_window_size': args.stability_window if args.strategy == 'val_baseline_stable_window' else None,
            'expected_step_gap': args.expected_step_gap if args.strategy == 'val_baseline_stable_window' else None,
            'stability_window': {},
            'feasible_candidate_count': 0,
            'selection_uses_test_metrics': False if args.strategy in VALIDATION_ONLY_STRATEGIES else None,
            'selection_metric_split': 'validation' if args.strategy in VALIDATION_ONLY_STRATEGIES else None,
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
            f.write(failure_reason + '\n')
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
        'protected_metrics': protected_metrics if args.strategy in BASELINE_VALIDATION_STRATEGIES else [],
        'baseline_tolerance': args.baseline_tolerance if args.strategy in BASELINE_VALIDATION_STRATEGIES else None,
        'min_spice_gain': args.min_spice_gain if args.strategy in BASELINE_VALIDATION_STRATEGIES else None,
        'selected_metric_deltas': best.get('baseline_deltas', {}),
        'protected_relative_margins': best.get('protected_relative_margins', {}),
        'min_protected_relative_margin': best.get('min_protected_relative_margin'),
        'spice_gain': best.get('spice_gain'),
        'pareto_front_count': len(pareto),
        'stable_window_count': len(stable_windows),
        'stability_window_size': args.stability_window if args.strategy == 'val_baseline_stable_window' else None,
        'expected_step_gap': args.expected_step_gap if args.strategy == 'val_baseline_stable_window' else None,
        'stability_window': best.get('stability_window', {}),
        'feasible_candidate_count': len(candidates) if args.strategy in BASELINE_VALIDATION_STRATEGIES else None,
        'selection_uses_test_metrics': False if args.strategy in VALIDATION_ONLY_STRATEGIES else None,
        'selection_metric_split': 'validation' if args.strategy in VALIDATION_ONLY_STRATEGIES else None,
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
