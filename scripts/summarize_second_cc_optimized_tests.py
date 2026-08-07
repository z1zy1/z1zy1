#!/usr/bin/env python3
"""Summarize immutable SECOND-CC locked tests across seeds."""

import argparse
import csv
import json
import math
import os
import statistics


METRICS = ('Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE')


def load(path):
    with open(path, encoding='utf-8-sig') as handle:
        return json.load(handle)


def canonical(path):
    return os.path.realpath(os.path.abspath(os.path.normpath(path)))


def metrics(raw, label):
    result = {}
    for metric in METRICS:
        try:
            value = float(raw[metric])
        except (KeyError, TypeError, ValueError):
            raise ValueError('%s missing metric %s.' % (label, metric))
        if not math.isfinite(value):
            raise ValueError('%s has non-finite %s.' % (label, metric))
        result[metric] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='./experiments/second_cc_optimized_lock.json')
    parser.add_argument('--baseline_summary', default='./experiments/card_baseline_test_summary.json')
    parser.add_argument('--output_json', default='./experiments/second_cc_optimized_test_summary.json')
    parser.add_argument('--output_csv', default='./experiments/second_cc_optimized_test_summary.csv')
    args = parser.parse_args()
    manifest = load(args.manifest)
    baseline_summary = load(args.baseline_summary)
    baseline_row = next(row for row in baseline_summary['results'] if row['dataset'] == 'second_cc')
    baseline = metrics(baseline_row, 'SECOND-CC test baseline')
    rows = []
    group_rows = {'change': [], 'nochange': []}
    for lock in manifest['locks']:
        result_path = lock['test_result']
        result = load(result_path)
        actual = canonical(result.get('snapshot_path') or '')
        expected = canonical(lock['checkpoint']['path'])
        if actual != expected:
            raise ValueError('Locked checkpoint mismatch: %s != %s' % (actual, expected))
        values = metrics(result.get('metrics'), result_path)
        row = {'seed': lock['seed'], 'checkpoint': expected, **values}
        row['all_metrics_strictly_above_baseline'] = all(values[name] > baseline[name] for name in METRICS)
        rows.append(row)
        for group in group_rows:
            group_values = metrics(result.get('group_metrics', {}).get(group, {}).get('metrics'), '%s %s' % (result_path, group))
            group_rows[group].append(group_values)
    mean = {name: statistics.mean(row[name] for row in rows) for name in METRICS}
    std = {name: statistics.stdev(row[name] for row in rows) if len(rows) > 1 else 0.0 for name in METRICS}
    delta = {name: mean[name] - baseline[name] for name in METRICS}
    group_mean = {
        group: {name: statistics.mean(row[name] for row in values) for name in METRICS}
        for group, values in group_rows.items()
    }
    seed_passes = sum(row['all_metrics_strictly_above_baseline'] for row in rows)
    payload = {
        'status': 'done',
        'selection_uses_test_metrics': False,
        'manifest': canonical(args.manifest),
        'selected_spec': manifest['selected_spec'],
        'baseline': baseline,
        'seed_results': rows,
        'mean': mean,
        'sample_std': std,
        'delta': delta,
        'group_mean': group_mean,
        'seed_passes': seed_passes,
        'seed_count': len(rows),
        'mean_all_metrics_strictly_above_baseline': all(delta[name] > 0 for name in METRICS),
        'at_least_two_seeds_strictly_above_baseline': seed_passes >= 2,
    }
    payload['acceptance_passed'] = (
        payload['mean_all_metrics_strictly_above_baseline']
        and payload['at_least_two_seeds_strictly_above_baseline']
    )
    os.makedirs(os.path.dirname(canonical(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    fields = ['seed', 'checkpoint', *METRICS, 'all_metrics_strictly_above_baseline']
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('SECOND-CC locked test acceptance: %s (%d/%d seeds pass)' % (
        'PASS' if payload['acceptance_passed'] else 'FAIL', seed_passes, len(rows)))


if __name__ == '__main__':
    main()
