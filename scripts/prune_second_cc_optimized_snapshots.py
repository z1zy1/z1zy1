#!/usr/bin/env python3
"""Plan or apply removal of unlocked snapshots from the optimized sweep."""

import argparse
import json
import os

from build_second_cc_optimized_lock import canonical, load_json, verify


def within(path, root):
    try:
        return os.path.commonpath((canonical(path), canonical(root))) == canonical(root)
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='./experiments/second_cc_optimized_lock.json')
    parser.add_argument('--output', default='./experiments/second_cc_optimized_prune_plan.json')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    verify(args.manifest)
    manifest = load_json(args.manifest)
    run_root = canonical(manifest['run_root'])
    keep = {canonical(lock['checkpoint']['path']) for lock in manifest['locks']}
    candidates = []
    for root, _, files in os.walk(run_root):
        if os.path.basename(root) != 'snapshots':
            continue
        for name in files:
            if not name.lower().endswith(('.pt', '.pth')):
                continue
            path = canonical(os.path.join(root, name))
            if not within(path, run_root):
                raise ValueError('Snapshot escaped run root: %s' % path)
            if path not in keep:
                candidates.append(path)
    candidates.sort()
    payload = {
        'status': 'applied' if args.apply else 'planned',
        'manifest': canonical(args.manifest),
        'run_root': run_root,
        'kept_snapshots': sorted(keep),
        'delete_snapshots': candidates,
        'delete_count': len(candidates),
        'reclaim_bytes': sum(os.path.getsize(path) for path in candidates),
    }
    if args.apply:
        for path in candidates:
            os.unlink(path)
    output = canonical(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print('%s %d unlocked snapshots (%.2f GiB): %s' % (
        'Deleted' if args.apply else 'Planned',
        len(candidates),
        payload['reclaim_bytes'] / float(1024 ** 3),
        output,
    ))


if __name__ == '__main__':
    main()
