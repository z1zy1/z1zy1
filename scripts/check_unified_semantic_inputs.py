#!/usr/bin/env python3
"""Preflight all semantic inputs used by the unified RSACA experiment."""

import argparse
import json
import os


SUFFIXES = ('.png', '.jpg', '.jpeg', '.npy')


def candidates(root, split, phase, filename):
    stem = os.path.splitext(filename)[0]
    names = [filename] + [stem + suffix for suffix in SUFFIXES]
    bases = []
    if split and phase:
        bases.extend([os.path.join(root, split, phase), os.path.join(root, phase, split)])
    if split:
        bases.append(os.path.join(root, split))
    if phase:
        bases.append(os.path.join(root, phase))
    bases.append(root)
    return [os.path.join(base, name) for base in bases for name in names]


def first_existing(paths):
    return next((path for path in paths if os.path.isfile(path)), None)


def check_dataset(name, root):
    split_path = os.path.join(root, 'splits.json')
    with open(split_path, encoding='utf-8-sig') as handle:
        split_data = json.load(handle)
    missing = []
    checked = 0
    for index, filename in split_data['idx_to_filename'].items():
        split = split_data['idx_to_split'][str(index)]
        if name == 'levir_cc':
            specs = [(os.path.join(root, 'pseudo_masks'), '', 'semantic_diff')]
        elif name == 'levir_mci':
            specs = [(os.path.join(root, 'images'), 'label', 'semantic_diff')]
        else:
            specs = [(root, 'sem/A', 'semantic_before'), (root, 'sem/B', 'semantic_after')]
        for semantic_root, phase, role in specs:
            if not first_existing(candidates(semantic_root, split, phase, filename)):
                missing.append({'filename': filename, 'split': split, 'role': role})
        checked += 1
    return {'dataset': name, 'root': os.path.abspath(root), 'checked': checked, 'missing': missing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--levir_cc_root', default='./Levir-CC')
    parser.add_argument('--levir_mci_root', default='./LEVIR-MCI-dataset')
    parser.add_argument('--second_cc_root', default='./SECOND-CC-AUG')
    parser.add_argument('--output', default='experiments/unified_rsaca_semantic_preflight.json')
    args = parser.parse_args()
    results = [
        check_dataset('levir_cc', args.levir_cc_root),
        check_dataset('levir_mci', args.levir_mci_root),
        check_dataset('second_cc', args.second_cc_root),
    ]
    payload = {
        'status': 'passed' if all(not item['missing'] for item in results) else 'failed',
        'datasets': results,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))
    if payload['status'] != 'passed':
        raise SystemExit('Unified semantic input preflight failed; see %s.' % args.output)


if __name__ == '__main__':
    main()
