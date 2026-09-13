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
import shutil

METRICS = ('Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE')
DEFAULT_REFERENCE = {
    'Bleu_4': 0.4375, 'METEOR': 0.3377, 'ROUGE_L': 0.6942,
    'CIDEr': 1.2299, 'SPICE': 0.2607,
}


def checkpoint_number(path):
    found = re.findall(r'(\d+)', os.path.basename(path or ''))
    return int(found[-1]) if found else None


def resolve(path, exp_dir):
    candidates = [path, os.path.join(exp_dir, path), os.path.join(exp_dir, 'snapshots', os.path.basename(path))]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.normpath(candidate)
    number = checkpoint_number(path)
    if number is not None:
        for name in os.listdir(os.path.join(exp_dir, 'snapshots')) if os.path.isdir(os.path.join(exp_dir, 'snapshots')) else ():
            if str(number) in name and name.endswith(('.pt', '.pth')):
                return os.path.join(exp_dir, 'snapshots', name)
    return None


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
    csv_path = args.csv or os.path.join(exp_dir, 'val_metrics.csv')
    output_path = args.output_json or os.path.join(exp_dir, 'best_snapshot_p1.json')
    copy_path = args.copy_path or os.path.join(exp_dir, 'best_for_p1.pth')
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
            snapshot = resolve(row.get('snapshot_path', ''), exp_dir)
            if not snapshot:
                raise ValueError('row %d has no existing snapshot_path' % row_number)
            metrics = {name: finite_metric(row, name) for name in METRICS}
            score = sum(math.log(max(metrics[name], 1e-12) / reference[name]) for name in METRICS) / len(METRICS)
            rows.append({'row': row_number, 'iter': int(float(row.get('iter', 0))), 'snapshot_path': snapshot, 'metrics': metrics, 'score': score})
    if not rows:
        raise RuntimeError('no validation rows in %s' % csv_path)
    # Higher score wins; ties are resolved by the earlier validation step.
    best = sorted(rows, key=lambda item: (-item['score'], item['iter']))[0]
    os.makedirs(os.path.dirname(copy_path), exist_ok=True)
    if os.path.lexists(copy_path):
        os.remove(copy_path)
    try:
        os.symlink(os.path.abspath(best['snapshot_path']), copy_path)
    except OSError:
        shutil.copy2(best['snapshot_path'], copy_path)
    payload = {
        'protocol_id': 'p1_rsaca_20260913',
        'selection_metric': 'five_metric_equal_weight_log',
        'selection_strategy': 'five_metric_equal_weight_log',
        'selection_metric_split': 'validation',
        'selection': {'split': 'validation', 'rule': 'five_metric_equal_weight_log', 'reference': reference, 'reference_sha256': reference_sha, 'tie_break': 'earlier_step'},
        'exp_dir': exp_dir, 'csv': os.path.abspath(csv_path), 'best_snapshot': best['snapshot_path'], 'copy_path': os.path.abspath(copy_path),
        'best': best, 'candidates': rows,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
