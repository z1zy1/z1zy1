#!/usr/bin/env python3
"""Summarize an immutable, validation-selected paired CARD/RSACA test matrix."""

import argparse
import csv
import json
import os
import statistics


METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
PRIMARY_METRICS = ['Bleu_4', 'CIDEr', 'SPICE']
ALLOWED_ARM_DIFFERENCE_PREFIXES = (
    'data.allow_missing_pseudo_mask', 'data.semantic_', 'data.use_semantic_maps',
    'model.num_semantic_classes', 'model.semantic_', 'train.semantic_',
    'train.use_semantic_cross_attention', 'train.use_semantic_partial_detach',
)
IGNORED_CONFIG_PATHS = {'exp_dir', 'exp_name'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-root', required=True)
    parser.add_argument('--datasets', default='levir_cc,levir_mci,second_cc')
    parser.add_argument('--seeds', default='1111,2222,3333')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def canonical(path):
    return os.path.realpath(os.path.abspath(os.path.normpath(path)))


def load_json(path):
    with open(path, encoding='utf-8-sig') as handle:
        return json.load(handle)


def flatten(value, prefix=''):
    if isinstance(value, dict):
        flattened = {}
        for key, item in value.items():
            child = '%s.%s' % (prefix, key) if prefix else key
            flattened.update(flatten(item, child))
        return flattened
    return {prefix: value}


def allowed_difference(path):
    return path in IGNORED_CONFIG_PATHS or path.startswith(ALLOWED_ARM_DIFFERENCE_PREFIXES)


def verify_resolved_configs(card_path, rsaca_path):
    card = flatten(load_json(card_path))
    rsaca = flatten(load_json(rsaca_path))
    differences = []
    for key in sorted(set(card) | set(rsaca)):
        if card.get(key) != rsaca.get(key) and not allowed_difference(key):
            differences.append(key)
    if differences:
        raise RuntimeError(
            'Paired resolved configs differ outside the semantic-arm allowlist: %s' % ', '.join(differences[:12])
        )
    if card.get('model.semantic_input_mode') != 'none':
        raise RuntimeError('CARD arm is not semantic_input_mode=none: %s' % card_path)
    if rsaca.get('model.semantic_input_mode') != 'cross_attention':
        raise RuntimeError('RSACA arm is not semantic_input_mode=cross_attention: %s' % rsaca_path)


def load_arm(pair_root, arm, dataset, seed):
    exp_dir = os.path.join(pair_root, arm, '%s_%s_seed%d' % (arm, dataset, seed))
    selection_path = os.path.join(exp_dir, 'best_snapshot_for_paper.json')
    test_path = os.path.join(exp_dir, 'test_paired_locked_result.json')
    resolved_path = os.path.join(exp_dir, 'resolved_config.json')
    for path in (selection_path, test_path, resolved_path):
        if not os.path.isfile(path):
            raise FileNotFoundError('Missing paired artifact: %s' % path)
    selection = load_json(selection_path)
    result = load_json(test_path)
    selected = selection.get('best_snapshot', '')
    tested = result.get('snapshot_path', '')
    if not selected or not os.path.isfile(selected):
        raise RuntimeError('Selected checkpoint is missing: %s' % selected)
    if canonical(selected) != canonical(tested):
        raise RuntimeError('Locked checkpoint mismatch in %s: selected=%s tested=%s' % (exp_dir, selected, tested))
    metrics = result.get('metrics', {})
    missing = [metric for metric in METRICS if metric not in metrics]
    if missing:
        raise RuntimeError('Test metrics missing in %s: %s' % (test_path, ', '.join(missing)))
    return {
        'exp_dir': canonical(exp_dir),
        'selection_json': canonical(selection_path),
        'test_result': canonical(test_path),
        'resolved_config': canonical(resolved_path),
        'checkpoint': canonical(selected),
        'metrics': {metric: float(metrics[metric]) for metric in METRICS},
    }


def mean(rows, metric):
    return statistics.mean(row[metric] for row in rows)


def main():
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(',') if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    protocol_path = os.path.join(args.pair_root, 'paired_protocol_lock.json')
    if not os.path.isfile(protocol_path):
        raise FileNotFoundError('Missing paired protocol lock: %s' % protocol_path)
    protocol = load_json(protocol_path)
    if protocol.get('datasets') != datasets or protocol.get('seeds') != seeds:
        raise RuntimeError('Summary dataset/seed arguments must exactly match the paired protocol lock.')

    payload = {
        'status': 'done',
        'selection_uses_test_metrics': False,
        'pair_root': canonical(args.pair_root),
        'protocol_lock': canonical(protocol_path),
        'protocol': protocol,
        'candidate': 'whole_adapter_gate_v1_rsaca',
        'control': 'card',
        'datasets': {},
    }
    csv_rows = []
    for dataset in datasets:
        paired_rows = []
        for seed in seeds:
            card = load_arm(args.pair_root, 'card', dataset, seed)
            rsaca = load_arm(args.pair_root, 'rsaca', dataset, seed)
            verify_resolved_configs(card['resolved_config'], rsaca['resolved_config'])
            delta = {metric: rsaca['metrics'][metric] - card['metrics'][metric] for metric in METRICS}
            row = {
                'dataset': dataset,
                'seed': seed,
                'card': card,
                'rsaca': rsaca,
                'delta_rsaca_minus_card': delta,
                'all_metrics_strictly_improved': all(delta[metric] > 0.0 for metric in METRICS),
                'primary_metrics_noninferior': all(delta[metric] >= 0.0 for metric in PRIMARY_METRICS),
            }
            paired_rows.append(row)
            csv_rows.append({
                'dataset': dataset, 'seed': seed, 'card_checkpoint': card['checkpoint'],
                'rsaca_checkpoint': rsaca['checkpoint'], **{
                    'card_' + metric: card['metrics'][metric] for metric in METRICS
                }, **{
                    'rsaca_' + metric: rsaca['metrics'][metric] for metric in METRICS
                }, **{
                    'delta_' + metric: delta[metric] for metric in METRICS
                },
            })
        card_means = {metric: mean([row['card']['metrics'] for row in paired_rows], metric) for metric in METRICS}
        rsaca_means = {metric: mean([row['rsaca']['metrics'] for row in paired_rows], metric) for metric in METRICS}
        deltas = {metric: rsaca_means[metric] - card_means[metric] for metric in METRICS}
        delta_std = {
            metric: statistics.stdev(row['delta_rsaca_minus_card'][metric] for row in paired_rows)
            if len(paired_rows) > 1 else 0.0
            for metric in METRICS
        }
        payload['datasets'][dataset] = {
            'seed_results': paired_rows,
            'card_mean': card_means,
            'rsaca_mean': rsaca_means,
            'delta_rsaca_minus_card': deltas,
            'paired_delta_sample_std': delta_std,
            'seed_count': len(paired_rows),
            'all_seed_primary_metrics_noninferior': all(row['primary_metrics_noninferior'] for row in paired_rows),
            'mean_primary_metrics_noninferior': all(deltas[metric] >= 0.0 for metric in PRIMARY_METRICS),
            'mean_all_metrics_strictly_improved': all(deltas[metric] > 0.0 for metric in METRICS),
        }
    payload['acceptance_passed'] = all(
        item['mean_all_metrics_strictly_improved'] and item['all_seed_primary_metrics_noninferior']
        for item in payload['datasets'].values()
    )
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
    csv_path = os.path.splitext(args.output)[0] + '.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]) if csv_rows else ['dataset', 'seed'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
