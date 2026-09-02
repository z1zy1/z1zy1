#!/usr/bin/env python3
"""Create or validate the immutable protocol lock for a paired CARD/RSACA run."""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-root', required=True)
    parser.add_argument('--git-commit', required=True)
    parser.add_argument('--source-digest', required=True)
    parser.add_argument('--source-status-digest', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--cuda-visible-devices', default='')
    parser.add_argument('--pytorch-gpu', default='0')
    parser.add_argument('--omp-num-threads', default='')
    parser.add_argument('--num-workers', required=True)
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--seeds', required=True)
    parser.add_argument('--rsaca-arm', required=True)
    return parser.parse_args()


def canonical(path):
    return os.path.realpath(os.path.abspath(os.path.normpath(path)))


def main():
    args = parse_args()
    payload = {
        'schema_version': 1,
        'pair_root': canonical(args.pair_root),
        'git_commit': args.git_commit,
        'source_digest_sha256': args.source_digest,
        'source_status_digest_sha256': args.source_status_digest,
        'python': canonical(args.python) if os.path.exists(args.python) else args.python,
        'cuda_visible_devices': args.cuda_visible_devices,
        'pytorch_gpu': args.pytorch_gpu,
        'omp_num_threads': args.omp_num_threads,
        'num_workers': args.num_workers,
        'datasets': args.datasets.split(),
        'seeds': [int(value) for value in args.seeds.split()],
        'arms': {'card': 'semantic_input_mode=none', 'rsaca': args.rsaca_arm},
        'selection': 'validation_only: paper_balanced_no_spice',
        'test_policy': 'one immutable test per validation-selected checkpoint',
    }
    path = os.path.join(args.pair_root, 'paired_protocol_lock.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig') as handle:
            existing = json.load(handle)
        if existing != payload:
            raise RuntimeError(
                'Paired protocol lock differs from this invocation: %s. '
                'Use a new PAIR_ROOT rather than mixing code, environment, seeds, or workers.' % path
            )
        print('Validated paired protocol lock: %s' % path)
        return
    os.makedirs(args.pair_root, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print('Created paired protocol lock: %s' % path)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
