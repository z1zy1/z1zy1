#!/usr/bin/env python3
"""Summarize validation-locked RSACA test results without selecting on test metrics."""

import argparse
import csv
import json
import os
import statistics


METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_root', default='experiments/unified_rsaca')
    parser.add_argument('--baseline', default='experiments/card_baseline_test_summary.json')
    parser.add_argument('--datasets', default='levir_cc,levir_mci,second_cc')
    parser.add_argument('--seeds', default='1111,2222,3333')
    parser.add_argument('--output', default='experiments/unified_rsaca_summary.json')
    return parser.parse_args()


def load_json(path):
    with open(path, encoding='utf-8-sig') as handle:
        return json.load(handle)


def canonical(path):
    return os.path.realpath(os.path.abspath(os.path.normpath(path)))


def main():
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(',') if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    baseline_payload = load_json(args.baseline)
    baselines = {
        row['dataset']: {metric: float(row[metric]) for metric in METRICS}
        for row in baseline_payload['results']
    }
    payload = {
        'status': 'done',
        'selection_uses_test_metrics': False,
        'model': 'CARD + Residual Semantic Cross-Attention Adapter (RSACA)',
        'run_root': canonical(args.run_root),
        'baseline': canonical(args.baseline),
        'datasets': {},
    }
    csv_rows = []
    for dataset in datasets:
        if dataset not in baselines:
            raise KeyError('Missing CARD baseline for dataset %s in %s' % (dataset, args.baseline))
        rows = []
        for seed in seeds:
            exp_dir = os.path.join(args.run_root, 'unified_rsaca_%s_seed%d' % (dataset, seed))
            selection_path = os.path.join(exp_dir, 'best_snapshot_for_paper.json')
            result_path = os.path.join(exp_dir, 'test_unified_locked_result.json')
            selection = load_json(selection_path)
            result = load_json(result_path)
            selected = selection['best_snapshot']
            tested = result['snapshot_path']
            if canonical(selected) != canonical(tested):
                raise RuntimeError(
                    'Locked checkpoint mismatch for %s seed %d: selected=%s tested=%s'
                    % (dataset, seed, selected, tested)
                )
            metrics = {metric: float(result['metrics'][metric]) for metric in METRICS}
            row = {
                'dataset': dataset,
                'seed': seed,
                'checkpoint': canonical(selected),
                'selection_json': canonical(selection_path),
                'test_result': canonical(result_path),
                **metrics,
            }
            row['all_metrics_strictly_above_baseline'] = all(
                metrics[metric] > baselines[dataset][metric] for metric in METRICS
            )
            rows.append(row)
            csv_rows.append(row)
        mean = {metric: statistics.mean(row[metric] for row in rows) for metric in METRICS}
        sample_std = {
            metric: statistics.stdev(row[metric] for row in rows) if len(rows) > 1 else None
            for metric in METRICS
        }
        delta = {metric: mean[metric] - baselines[dataset][metric] for metric in METRICS}
        payload['datasets'][dataset] = {
            'baseline': baselines[dataset],
            'seed_results': rows,
            'mean': mean,
            'sample_std': sample_std,
            'delta': delta,
            'seed_passes': sum(row['all_metrics_strictly_above_baseline'] for row in rows),
            'seed_count': len(rows),
            'mean_all_metrics_strictly_above_baseline': all(value > 0 for value in delta.values()),
        }
    payload['acceptance_passed'] = all(
        item['mean_all_metrics_strictly_above_baseline']
        for item in payload['datasets'].values()
    )
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    csv_path = os.path.splitext(args.output)[0] + '.csv'
    fieldnames = [
        'dataset', 'seed', 'checkpoint', 'selection_json', 'test_result', *METRICS,
        'all_metrics_strictly_above_baseline',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
