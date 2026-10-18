#!/usr/bin/env python3
"""Require exact initial detector and decoder parameter identity for a pair."""
import argparse
import json
import os
import sys

PROTOCOL = 'exploratory_norm_context_pre_20261015'


def load(path):
    with open(path, encoding='utf-8') as handle:
        value = json.load(handle)
    if value.get('protocol_id') != PROTOCOL:
        raise ValueError('%s has the wrong protocol_id' % path)
    return value


def compare(left, right):
    if left.get('seed') != right.get('seed'):
        raise ValueError('paired initialization summaries have different seeds')
    differences = []
    for module in ('change_detector', 'speaker'):
        a, b = left.get(module), right.get(module)
        if not isinstance(a, dict) or not isinstance(b, dict) or set(a) != set(b):
            differences.append('%s parameter names differ' % module)
            continue
        for name in sorted(a):
            if a[name] != b[name]:
                differences.append('%s.%s differs' % (module, name))
    return {'protocol_id': PROTOCOL, 'seed': left['seed'],
            'matched': not differences, 'differences': differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('c0_summary')
    parser.add_argument('n1_summary')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(load(args.c0_summary), load(args.n1_summary))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['matched'] else 1


if __name__ == '__main__':
    sys.exit(main())
