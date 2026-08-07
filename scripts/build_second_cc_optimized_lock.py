#!/usr/bin/env python3
"""Build or verify a validation-only lock for the optimized SECOND-CC sweep."""

import argparse
import datetime
import hashlib
import json
import math
import os
import statistics


METRICS = ('Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE')


def canonical(path):
    return os.path.realpath(os.path.abspath(os.path.normpath(path)))


def load_json(path):
    with open(path, encoding='utf-8-sig') as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: %s' % path)
    return value


def metric_map(raw, label):
    result = {}
    for metric in METRICS:
        value = raw.get(metric) if isinstance(raw, dict) else None
        if isinstance(value, bool):
            raise ValueError('%s has invalid %s.' % (label, metric))
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError('%s is missing %s.' % (label, metric))
        if not math.isfinite(value):
            raise ValueError('%s has non-finite %s.' % (label, metric))
        result[metric] = value
    return result


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_lock(path):
    path = canonical(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return {'path': path, 'size': os.path.getsize(path), 'sha256': sha256(path)}


def parse_words(value, cast=str):
    return tuple(cast(item) for item in value.replace(',', ' ').split() if item)


def audit_selection(path, baseline, selection_strategy):
    payload = load_json(path)
    required = {
        'status': 'done',
        'selection_strategy': selection_strategy,
        'selection_uses_test_metrics': False,
        'selection_metric_split': 'validation',
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError('%s has %s=%r, expected %r.' % (path, key, payload.get(key), expected))
    if selection_strategy == 'val_baseline_stable_window':
        if payload.get('stability_window_size') != 3 or payload.get('expected_step_gap') != 1000:
            raise ValueError('%s does not use the required 3x1000 stability window.' % path)
        if int(payload.get('stable_window_count', 0)) < 1:
            raise ValueError('%s has no stable validation window.' % path)
    metrics = metric_map(payload.get('selected_val_metrics'), path)
    bad = {metric: metrics[metric] - baseline[metric] for metric in METRICS
           if metrics[metric] <= baseline[metric]}
    if bad:
        raise ValueError('%s does not strictly beat validation baseline: %s' % (path, bad))
    checkpoint = canonical(payload.get('selected_checkpoint') or '')
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError('Selected checkpoint missing: %s' % checkpoint)
    source_dir = canonical(payload.get('selected_source_exp_dir') or os.path.dirname(path))
    source_config = canonical(os.path.join(source_dir, 'resolved_config.json'))
    if not os.path.isfile(source_config):
        raise FileNotFoundError('Resolved source config missing: %s' % source_config)
    return payload, metrics, checkpoint, source_dir, source_config


def build(args):
    exp_root = canonical(args.exp_root)
    run_root = canonical(args.run_root)
    baseline_path = canonical(args.baseline)
    baseline_payload = load_json(baseline_path)
    baseline = metric_map(
        baseline_payload.get('selected_val_metrics') or baseline_payload.get('selected_metrics'),
        baseline_path,
    )
    specs = parse_words(args.specs)
    seeds = parse_words(args.seeds, int)
    eligible = []
    rejected = {}
    for spec in specs:
        rows = []
        try:
            for seed in seeds:
                exp_name = '%s_%s_seed%d' % (args.run_prefix, spec, seed)
                selection_path = canonical(os.path.join(run_root, exp_name, 'best_checkpoint.json'))
                payload, metrics, checkpoint, source_dir, source_config = audit_selection(
                    selection_path, baseline, args.selection_strategy
                )
                relative = [(metrics[metric] - baseline[metric]) / max(abs(baseline[metric]), 1e-12)
                            for metric in METRICS]
                rows.append({
                    'seed': seed,
                    'exp_name': exp_name,
                    'source_exp_dir': source_dir,
                    'source_config': source_config,
                    'source_config_lock': file_lock(source_config),
                    'selection': file_lock(selection_path),
                    'checkpoint': file_lock(checkpoint),
                    'selected_step': payload.get('selected_step'),
                    'validation_metrics': metrics,
                    'min_relative_margin': min(relative),
                })
        except (FileNotFoundError, ValueError) as exc:
            rejected[spec] = str(exc)
            continue
        mean = {metric: statistics.mean(row['validation_metrics'][metric] for row in rows)
                for metric in METRICS}
        eligible.append({
            'spec': spec,
            'rows': rows,
            'validation_mean': mean,
            'minimum_seed_metric_relative_margin': min(row['min_relative_margin'] for row in rows),
        })
    if not eligible:
        raise SystemExit('No learning-rate spec passed the three-seed stable-window guard: %s' % rejected)
    winner = max(eligible, key=lambda item: (
        item['minimum_seed_metric_relative_margin'],
        item['validation_mean']['SPICE'],
        item['validation_mean']['CIDEr'],
    ))
    locks = []
    for row in winner['rows']:
        target = '%s_locked_%s_seed%d' % (args.run_prefix, winner['spec'], row['seed'])
        locks.append({
            **row,
            'lock_id': '%s_seed%d' % (winner['spec'], row['seed']),
            'target_exp': target,
            'test_result': canonical(os.path.join(exp_root, target, 'test_optimized_locked_result.json')),
        })
    output = canonical(args.output)
    manifest = {
        'status': 'validation_locked',
        'selection_uses_test_metrics': False,
        'selection_metric_split': 'validation',
        'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'dataset': 'second_cc',
        'dataset_root': canonical(args.dataset_root),
        'run_root': run_root,
        'baseline': {'artifact': file_lock(baseline_path), 'metrics': baseline},
        'selection_rule': 'maximize worst seed/metric relative margin, then mean SPICE and CIDEr',
        'checkpoint_selection_strategy': args.selection_strategy,
        'selected_spec': winner['spec'],
        'validation_mean': winner['validation_mean'],
        'minimum_seed_metric_relative_margin': winner['minimum_seed_metric_relative_margin'],
        'rejected_specs': rejected,
        'locks': locks,
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print('Locked %s with %d seeds: %s' % (winner['spec'], len(locks), output))


def verify(path):
    manifest = load_json(path)
    if manifest.get('status') != 'validation_locked' or manifest.get('selection_uses_test_metrics') is not False:
        raise ValueError('Manifest is not a validation-only lock: %s' % path)
    if len(manifest.get('locks', ())) < 3:
        raise ValueError('Manifest must contain at least three seed locks.')
    for record in (manifest['baseline']['artifact'],):
        if file_lock(record['path']) != record:
            raise ValueError('Locked artifact changed: %s' % record['path'])
    for lock in manifest['locks']:
        for key in ('selection', 'checkpoint', 'source_config_lock'):
            record = lock[key]
            if file_lock(record['path']) != record:
                raise ValueError('Locked artifact changed: %s' % record['path'])
    print('Verified validation-only lock: %s' % canonical(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify')
    parser.add_argument('--exp_root', default='./experiments')
    parser.add_argument('--run_root', default='./experiments/second_cc_optimized_runs')
    parser.add_argument('--run_prefix', default='second_cc_opt')
    parser.add_argument('--dataset_root', default='./SECOND-CC-AUG')
    parser.add_argument('--baseline', default='./experiments/second_cc_card_rgb_baseline/baseline_best_checkpoint.json')
    parser.add_argument('--specs', default='lr15e5 lr20e5 lr25e5')
    parser.add_argument('--seeds', default='1111 2222 3333')
    parser.add_argument('--selection_strategy', default='val_baseline_stable_window',
                        choices=('val_baseline_stable_window', 'val_baseline_pareto'))
    parser.add_argument('--output', default='./experiments/second_cc_optimized_lock.json')
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        build(args)


if __name__ == '__main__':
    main()
