#!/usr/bin/env python3
"""Compare validation-selected checkpoints from a paired RSACA screen."""

import argparse
import json
import os


METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
NON_SPICE_METRICS = [metric for metric in METRICS if metric != 'SPICE']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root', required=True)
    parser.add_argument('--control-root', required=True)
    parser.add_argument('--dataset', default='levir_cc')
    parser.add_argument('--seeds', default='3333 1111')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def load_selected_metrics(root, dataset, seed):
    path = os.path.join(root, 'unified_rsaca_%s_seed%s' % (dataset, seed), 'best_snapshot_for_paper.json')
    if not os.path.isfile(path):
        raise FileNotFoundError('Validation selection not found: %s' % path)
    with open(path, encoding='utf-8-sig') as handle:
        payload = json.load(handle)
    best = payload.get('best', {})
    checkpoint = best.get('snapshot_path', '')
    if not checkpoint or not os.path.isfile(checkpoint):
        raise FileNotFoundError('Selected checkpoint is missing: %s' % checkpoint)
    metrics = best.get('metrics', {})
    missing = [metric for metric in METRICS if metric not in metrics]
    if missing:
        raise ValueError('Selection %s lacks metrics: %s' % (path, ', '.join(missing)))
    return checkpoint, {metric: float(metrics[metric]) for metric in METRICS}


def main():
    args = parse_args()
    seeds = args.seeds.split()
    if not seeds:
        raise ValueError('At least one seed is required')

    records = []
    for seed in seeds:
        candidate_checkpoint, candidate = load_selected_metrics(args.candidate_root, args.dataset, seed)
        control_checkpoint, control = load_selected_metrics(args.control_root, args.dataset, seed)
        delta = {metric: candidate[metric] - control[metric] for metric in METRICS}
        records.append({
            'seed': int(seed),
            'candidate_checkpoint': candidate_checkpoint,
            'control_checkpoint': control_checkpoint,
            'candidate': candidate,
            'control': control,
            'delta_candidate_minus_control': delta,
            'primary_b4_cider_improved': all(delta[metric] > 0.0 for metric in ('Bleu_4', 'CIDEr')),
            'all_non_spice_metrics_improved': all(delta[metric] > 0.0 for metric in NON_SPICE_METRICS),
        })

    payload = {
        'dataset': args.dataset,
        'candidate_root': os.path.abspath(args.candidate_root),
        'control_root': os.path.abspath(args.control_root),
        'seeds': [int(seed) for seed in seeds],
        'selection_uses_test_metrics': False,
        'per_seed': records,
        'screen': {
            'all_seeds_primary_b4_cider_improved': all(row['primary_b4_cider_improved'] for row in records),
            'all_seeds_all_non_spice_metrics_improved': all(row['all_non_spice_metrics_improved'] for row in records),
            'locked_test_recommended': all(row['all_non_spice_metrics_improved'] for row in records),
            'rule': 'Run no locked test unless every screened seed strictly improves all seven non-SPICE validation metrics over its paired control.',
        },
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
