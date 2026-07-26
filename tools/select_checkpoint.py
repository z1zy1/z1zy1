"""Traceable validation-only checkpoint selection.

Without --primary-metric/--weights this delegates to the established selector
and therefore preserves its default SPICE-constrained behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.select_best_checkpoint import load_rows, resolve_default_metrics
from utils.metrics import canonical_metric_name


def _step(row) -> int:
    for key in ('step', 'iter', 'epoch'):
        value = row.get(key)
        if value not in (None, ''):
            return int(float(value))
    return 0


def _parse_weights(raw: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for item in (raw or '').split(','):
        if not item.strip():
            continue
        if '=' not in item:
            raise ValueError('Weighted metric must use NAME=WEIGHT: %s' % item)
        name, weight = item.split('=', 1)
        name = canonical_metric_name(name.strip())
        numeric = float(weight)
        if not math.isfinite(numeric):
            raise ValueError('Metric weight must be finite: %s' % item)
        if numeric < 0:
            raise ValueError('Metric weight must be non-negative: %s' % item)
        result[name] = numeric
    if raw and not result:
        raise ValueError('No valid weighted metrics were provided.')
    if result and not any(weight > 0 for weight in result.values()):
        raise ValueError('At least one metric weight must be positive.')
    return result


def _normalized(rows, metric: str) -> Dict[int, float]:
    values = [float(row['metrics'][metric]) for row in rows]
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        return {id(row): 0.0 for row in rows}
    return {
        id(row): (float(row['metrics'][metric]) - minimum) / (maximum - minimum)
        for row in rows
    }


def _generic_select(args) -> int:
    run_dir = os.path.normpath(args.run_dir)
    metrics_path = os.path.normpath(args.metrics or resolve_default_metrics(run_dir))
    rows = load_rows(metrics_path, run_dir)
    weights = _parse_weights(args.weights)
    primary = canonical_metric_name(args.primary_metric) if args.primary_metric else None
    secondary = canonical_metric_name(args.secondary_metric) if args.secondary_metric else None
    required = list(weights) if weights else [name for name in (primary, secondary) if name]
    rows = [
        row for row in rows
        if _step(row) >= args.minimum_step and all(name in row['metrics'] for name in required)
    ]
    if not rows:
        raise RuntimeError('No validation checkpoint has all requested metrics after minimum-step filtering.')

    if weights:
        norms = {metric: _normalized(rows, metric) for metric in weights}
        for row in rows:
            row['_selection_score'] = sum(
                weight * norms[metric][id(row)] for metric, weight in weights.items()
            )
        if args.mode == 'max':
            sort_key = lambda row: (-row['_selection_score'], -_step(row))
        else:
            sort_key = lambda row: (row['_selection_score'], -_step(row))
        rule = {
            'type': 'weighted_minmax',
            'weights': weights,
            'mode': args.mode,
            'tie_breaking_rule': 'prefer the later validation step',
        }
    else:
        for row in rows:
            row['_selection_score'] = float(row['metrics'][primary])
        def _secondary_score(row):
            return (
                float(row['metrics'].get(secondary, row['_selection_score']))
                if secondary else row['_selection_score']
            )
        if args.mode == 'max':
            sort_key = lambda row: (
                -row['_selection_score'],
                -_secondary_score(row),
                -_step(row),
            )
        else:
            sort_key = lambda row: (
                row['_selection_score'],
                _secondary_score(row),
                -_step(row),
            )
        rule = {
            'type': 'primary_secondary',
            'primary_metric': primary,
            'secondary_metric': secondary,
            'mode': args.mode,
            'tie_breaking_rule': (
                'primary metric, then secondary metric, then later validation step'
            ),
        }
    ranked = sorted(rows, key=sort_key)
    top = ranked[:args.top_k]
    candidates: List[Dict[str, object]] = []
    for rank, row in enumerate(ranked, 1):
        candidates.append({
            'rank': rank,
            'checkpoint': row['checkpoint_path'],
            'step': _step(row),
            'metrics': row['metrics'],
            'selection_score': row['_selection_score'],
        })
    payload = {
        'status': 'done',
        'run_dir': run_dir,
        'metrics_file': metrics_path,
        'selection_metric_split': 'validation',
        'selection_uses_test_metrics': False,
        'rule': rule,
        'minimum_step': args.minimum_step,
        'top_k': args.top_k,
        'selected_checkpoint': top[0]['checkpoint_path'],
        'selected_step': _step(top[0]),
        'selected_val_metrics': top[0]['metrics'],
        'top_k_checkpoints': [row['checkpoint_path'] for row in top],
        'candidates': candidates,
        'selection_reason': (
            'ranked validation checkpoints using the recorded rule and explicit '
            'tie-breaking; test metrics were not read'
        ),
    }
    output = os.path.normpath(args.output or os.path.join(run_dir, 'checkpoint_selection.json'))
    if not args.dry_run:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        with open(output, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(payload, output=output), indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Select checkpoints using validation metrics only.')
    parser.add_argument('--run-dir', '--exp_dir', dest='run_dir', required=True)
    parser.add_argument('--metrics', default=None)
    parser.add_argument('--primary-metric', default=None)
    parser.add_argument('--secondary-metric', default=None)
    parser.add_argument('--mode', choices=['max', 'min'], default='max')
    parser.add_argument('--weights', default='')
    parser.add_argument('--minimum-step', '--minimum-epoch', type=int, default=0)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--output', default=None)
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    if args.top_k <= 0:
        parser.error('--top-k must be positive.')
    if args.secondary_metric and not args.primary_metric:
        parser.error('--secondary-metric requires --primary-metric.')
    if not args.primary_metric and not args.weights:
        if (
            args.mode != 'max'
            or args.minimum_step != 0
            or args.top_k != 1
        ):
            parser.error(
                '--mode, --minimum-step, and --top-k are generic selector options; '
                'provide --primary-metric or --weights.'
            )
        output = args.output or os.path.join(args.run_dir, 'checkpoint_selection.json')
        command = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '..', 'scripts', 'select_best_checkpoint.py'),
            '--exp_dir',
            args.run_dir,
            '--output_json',
            output,
        ] + extra
        print(json.dumps({'command': command, 'delegated_to_legacy_selector': True}, indent=2))
        if args.dry_run:
            return 0
        return subprocess.call(command, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    return _generic_select(args)


if __name__ == '__main__':
    raise SystemExit(main())
