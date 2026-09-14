#!/usr/bin/env python3
"""Select one validation checkpoint with the frozen P1 five-metric rule.

P1 selection is deliberately separate from the legacy paper selector.  The
reference values are supplied once (normally from the P1 CARD validation
pilot), recorded in the output, and never inferred from test metrics.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.checkpoint_integrity import atomic_copy_checkpoint, atomic_write_text, validate_checkpoint_file

METRICS = ('Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE')
DEFAULT_REFERENCE = {
    'Bleu_4': 0.4375, 'METEOR': 0.3377, 'ROUGE_L': 0.6942,
    'CIDEr': 1.2299, 'SPICE': 0.2607,
}


def resolve(path, exp_dir, expected_iter):
    snapshot_dir = os.path.realpath(os.path.join(exp_dir, 'snapshots'))
    candidates = [path, os.path.join(exp_dir, path), os.path.join(snapshot_dir, os.path.basename(path))]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and not os.path.islink(candidate):
            resolved = os.path.abspath(os.path.normpath(candidate))
            if os.path.commonpath((snapshot_dir, os.path.realpath(resolved))) != snapshot_dir:
                continue
            match = re.search(r'_checkpoint_(\d+)\.(?:pt|pth)$', os.path.basename(resolved))
            if not match or int(match.group(1)) != expected_iter:
                continue
            try:
                digest = validate_checkpoint_file(
                    resolved, require_checksum=True, require_metadata=True,
                    expected_step=expected_iter)
            except ValueError:
                continue
            return resolved, digest, os.path.getsize(resolved)
    return None, None, None


def finite_metric(row, name):
    raw = row.get(name, '')
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError('missing/non-numeric %s in validation row' % name)
    if not math.isfinite(value):
        raise ValueError('non-finite %s in validation row' % name)
    if value < 0:
        raise ValueError('negative %s in validation row' % name)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-dir', required=True)
    parser.add_argument('--csv', default=None)
    parser.add_argument('--output-json', default=None)
    parser.add_argument('--copy-path', default=None)
    parser.add_argument('--reference-json', default=None)
    args = parser.parse_args()
    exp_dir = os.path.abspath(args.exp_dir)
    csv_path = os.path.abspath(args.csv or os.path.join(exp_dir, 'val_metrics.csv'))
    output_path = os.path.abspath(args.output_json or os.path.join(exp_dir, 'best_snapshot_p1.json'))
    copy_path = os.path.abspath(args.copy_path or os.path.join(exp_dir, 'best_for_p1.pth'))
    if os.path.commonpath((exp_dir, output_path)) != exp_dir:
        raise ValueError('selection output must remain inside exp_dir')
    if os.path.commonpath((exp_dir, copy_path)) != exp_dir:
        raise ValueError('selected checkpoint copy must remain inside exp_dir')
    snapshot_dir = os.path.realpath(os.path.join(exp_dir, 'snapshots'))
    if os.path.commonpath((snapshot_dir, os.path.realpath(copy_path))) == snapshot_dir:
        raise ValueError('selected checkpoint copy must remain outside snapshots directory')
    reference = dict(DEFAULT_REFERENCE)
    if args.reference_json:
        with open(args.reference_json, encoding='utf-8-sig') as handle:
            supplied = json.load(handle)
        # The frozen protocol stores references under ``reference``; accepting
        # a top-level mapping also keeps this utility usable with small pilot
        # fixtures without changing the scoring rule.
        supplied = supplied.get('reference', supplied)
        reference.update({key: float(supplied[key]) for key in METRICS})
    if any(not math.isfinite(reference[key]) or reference[key] <= 0 for key in METRICS):
        raise ValueError('P1 reference values must be finite and strictly positive')
    reference_sha = hashlib.sha256(json.dumps(reference, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    rows = []
    with open(csv_path, newline='', encoding='utf-8-sig') as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            if not any(str(value or '').strip() for value in row.values()):
                continue
            try:
                iteration = int(row.get('iter', ''))
            except (TypeError, ValueError):
                raise ValueError('row %d has invalid iter' % row_number)
            snapshot, snapshot_sha256, snapshot_size = resolve(
                row.get('snapshot_path', ''), exp_dir, iteration)
            if not snapshot:
                raise ValueError(
                    'row %d has no valid checksummed snapshot for iter %d' % (row_number, iteration))
            metrics = {name: finite_metric(row, name) for name in METRICS}
            score = sum(math.log(max(metrics[name], 1e-12) / reference[name]) for name in METRICS) / len(METRICS)
            rows.append({
                'row': row_number, 'iter': iteration, 'snapshot_path': snapshot,
                'snapshot_sha256': snapshot_sha256, 'snapshot_size_bytes': snapshot_size,
                'metrics': metrics, 'score': score,
            })
    if not rows:
        raise RuntimeError('no validation rows in %s' % csv_path)
    actual_steps = [row['iter'] for row in rows]
    expected_steps = list(range(1000, 10001, 1000))
    if sorted(actual_steps) != expected_steps:
        raise ValueError(
            'P1 validation grid must contain each step 1000..10000 exactly once; got %s'
            % sorted(actual_steps))
    # Higher score wins; ties are resolved by the earlier validation step.
    best = sorted(rows, key=lambda item: (-item['score'], item['iter']))[0]
    os.makedirs(os.path.dirname(copy_path), exist_ok=True)
    selected_sha256 = atomic_copy_checkpoint(
        best['snapshot_path'], copy_path, require_checksum=True,
        require_metadata=True, expected_step=best['iter'])
    payload = {
        'protocol_id': 'p1_rsaca_20260913',
        'selection_metric': 'five_metric_equal_weight_log',
        'selection_strategy': 'five_metric_equal_weight_log',
        'selection_metric_split': 'validation',
        'selection': {'split': 'validation', 'rule': 'five_metric_equal_weight_log', 'reference': reference, 'reference_sha256': reference_sha, 'tie_break': 'earlier_step'},
        'exp_dir': exp_dir, 'csv': os.path.abspath(csv_path), 'best_snapshot': best['snapshot_path'],
        'best_snapshot_sha256': selected_sha256,
        'best_snapshot_size_bytes': best['snapshot_size_bytes'],
        'copy_path': os.path.abspath(copy_path),
        'best': best, 'candidates': rows,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    atomic_write_text(output_path, json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
