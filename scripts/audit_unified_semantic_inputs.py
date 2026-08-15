#!/usr/bin/env python3
"""Audit semantic-map coverage and label diversity before a unified run."""

import argparse
import glob
import json
import os

import numpy as np
from imageio.v2 import imread


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--levir_cc_root', default='./Levir-CC')
    parser.add_argument('--levir_mci_root', default='./LEVIR-MCI-dataset')
    parser.add_argument('--second_cc_root', default='./SECOND-CC-AUG')
    parser.add_argument('--sample_limit', type=int, default=256)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def load_array(path):
    array = imread(path)
    if array.ndim == 3:
        array = array[..., :3]
    return np.asarray(array)


def summarize(paths, change_fn, sample_limit):
    paths = sorted(paths)
    sampled = paths[:max(0, int(sample_limit))]
    coverage = []
    unique = set()
    for path in sampled:
        array = load_array(path)
        coverage.append(float(change_fn(array).mean()))
        if array.ndim == 2:
            unique.update(int(value) for value in np.unique(array).tolist())
        else:
            unique.update(tuple(int(value) for value in row) for row in np.unique(array.reshape(-1, array.shape[-1]), axis=0))
        if len(unique) > 64:
            unique = set(list(unique)[:64])
    return {
        'files_found': len(paths),
        'files_sampled': len(sampled),
        'mean_change_coverage': float(np.mean(coverage)) if coverage else None,
        'empty_change_fraction': float(np.mean(np.asarray(coverage) == 0.0)) if coverage else None,
        'unique_values_sample': sorted(repr(value) for value in unique),
    }


def single_map_change(array):
    if array.ndim == 2:
        return array > 0
    return np.any(array != 0, axis=-1)


def paired_summary(root, sample_limit):
    before = {}
    after = {}
    for path in glob.glob(os.path.join(root, '*', 'sem', 'A', '*')):
        before[os.path.basename(path)] = path
    for path in glob.glob(os.path.join(root, '*', 'sem', 'B', '*')):
        after[os.path.basename(path)] = path
    names = sorted(set(before) & set(after))
    sampled = names[:max(0, int(sample_limit))]
    coverage = []
    unique_before = set()
    unique_after = set()
    for name in sampled:
        before_array = load_array(before[name])
        after_array = load_array(after[name])
        if before_array.shape != after_array.shape:
            continue
        coverage.append(float(np.any(before_array != after_array, axis=-1).mean()))
        unique_before.update(tuple(int(value) for value in row) for row in np.unique(before_array.reshape(-1, before_array.shape[-1]), axis=0))
        unique_after.update(tuple(int(value) for value in row) for row in np.unique(after_array.reshape(-1, after_array.shape[-1]), axis=0))
    return {
        'before_files': len(before),
        'after_files': len(after),
        'paired_files': len(names),
        'missing_before': len(set(after) - set(before)),
        'missing_after': len(set(before) - set(after)),
        'files_sampled': len(sampled),
        'mean_change_coverage': float(np.mean(coverage)) if coverage else None,
        'empty_change_fraction': float(np.mean(np.asarray(coverage) == 0.0)) if coverage else None,
        'unique_before_sample': sorted(repr(value) for value in list(unique_before)[:64]),
        'unique_after_sample': sorted(repr(value) for value in list(unique_after)[:64]),
    }


def main():
    args = parse_args()
    report = {
        'sample_limit': int(args.sample_limit),
        'datasets': {
            'levir_cc': summarize(
                glob.glob(os.path.join(args.levir_cc_root, 'pseudo_masks', '*', '*')),
                single_map_change,
                args.sample_limit,
            ),
            'levir_mci': summarize(
                glob.glob(os.path.join(args.levir_mci_root, 'images', '*', 'label', '*')),
                single_map_change,
                args.sample_limit,
            ),
            'second_cc': paired_summary(args.second_cc_root, args.sample_limit),
        },
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
