"""Recursively summarize CARD runs, including failures and multi-seed stats."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from typing import Dict, Iterable, List, Mapping, Optional

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics import CAPTION_METRICS, normalize_metrics


SUMMARY_FIELDS = [
    'dataset', 'experiment_name', 'experiment_group', 'model_name', 'run_id', 'seed',
    'status', 'failure_reason', 'best_epoch', 'checkpoint',
] + list(CAPTION_METRICS) + [
    'val_%s' % name for name in CAPTION_METRICS
] + [
    'mask_loss_weight', 'semantic_loss_weight', 'partial_detach', 'reweight',
    'fusion_type', 'gate_type', 'config_path', 'run_dir',
]


def _load_json(path: str) -> Optional[object]:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding='utf-8-sig') as handle:
        return json.load(handle)


def _load_config(run_dir: str):
    candidates = (
        'resolved_config.json',
        'config_resolved.yaml',
        'resolved_config.yaml',
        'cfg.json',
    )
    for name in candidates:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            continue
        if path.lower().endswith('.json'):
            payload = _load_json(path)
        else:
            with open(path, encoding='utf-8') as handle:
                payload = yaml.safe_load(handle)
        if isinstance(payload, dict):
            return path, payload
    return '', {}


def _nested(payload: Mapping[str, object], path: str, default=''):
    current = payload
    for key in path.split('.'):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _read_last_csv(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    with open(path, newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    return dict(rows[-1]) if rows else {}


def _find_test_metrics(run_dir: str) -> Dict[str, object]:
    csv_values = _read_last_csv(os.path.join(run_dir, 'test_metrics.csv'))
    if csv_values:
        normalized = normalize_metrics(csv_values)
        has_numeric_metric = False
        for name in CAPTION_METRICS:
            try:
                has_numeric_metric = math.isfinite(float(normalized.get(name, '')))
            except (TypeError, ValueError):
                has_numeric_metric = False
            if has_numeric_metric:
                break
        if has_numeric_metric:
            return normalized
    for name in ('test_metrics.json', 'locked_test_result.json', 'test_result.json'):
        payload = _load_json(os.path.join(run_dir, name))
        if not isinstance(payload, dict):
            continue
        values = payload.get('metrics', payload)
        if isinstance(values, dict):
            return normalize_metrics(values)
    return {}


def _find_selection(run_dir: str) -> Dict[str, object]:
    for name in (
        'checkpoint_selection.json',
        'best_checkpoint.json',
        'paper_best_checkpoint.json',
    ):
        payload = _load_json(os.path.join(run_dir, name))
        if isinstance(payload, dict):
            return payload
    return {}


def _discover_runs(root: str) -> Iterable[str]:
    markers = {
        'resolved_config.json', 'config_resolved.yaml', 'cfg.json',
        'run_summary.json', 'val_metrics.csv', 'test_metrics.csv',
    }
    for current, directories, files in os.walk(root):
        directories[:] = [
            name for name in directories
            if name not in {'.git', '__pycache__', 'snapshots', 'checkpoints'}
        ]
        if markers.intersection(files):
            yield current


def _experiment_group(name: str, config: Mapping[str, object]) -> str:
    explicit = config.get('experiment_group')
    if explicit:
        return str(explicit)
    return re.sub(r'(?:[_-]seed)?(?:1111|2222|3333|\d{4,10})$', '', name).rstrip('_-') or name


def _row_from_run(run_dir: str) -> Dict[str, object]:
    config_path, config = _load_config(run_dir)
    summary = _load_json(os.path.join(run_dir, 'run_summary.json')) or {}
    selection = _find_selection(run_dir)
    metrics = _find_test_metrics(run_dir)
    selected_val_metrics = normalize_metrics(
        selection.get('selected_val_metrics')
        or selection.get('selected_metrics')
        or {}
    )
    experiment_name = str(
        summary.get('experiment_name')
        or config.get('exp_name')
        or os.path.basename(run_dir)
    )
    seed = summary.get('seed', _nested(config, 'train.seed', ''))
    dataset = str(summary.get('dataset') or _nested(config, 'data.dataset', ''))
    checkpoint = (
        selection.get('selected_checkpoint')
        or selection.get('selected_checkpoint_path')
        or summary.get('best_checkpoint')
        or metrics.get('checkpoint_path')
        or ''
    )
    status = str(summary.get('status') or '')
    if not status:
        status = 'completed' if any(name in metrics for name in CAPTION_METRICS) else 'incomplete'
    failure_reason = str(summary.get('failure_reason') or selection.get('failure_reason') or '')
    if status == 'failed' and not failure_reason:
        failure_reason = 'run_summary marks the experiment failed without a recorded reason'
    semantic_mode = str(_nested(config, 'model.semantic_input_mode', 'none'))
    row = {key: '' for key in SUMMARY_FIELDS}
    row.update({
        'dataset': dataset,
        'experiment_name': experiment_name,
        'experiment_group': _experiment_group(experiment_name, config),
        'model_name': _nested(config, 'model.type', ''),
        'run_id': summary.get('run_id', os.path.basename(run_dir)),
        'seed': seed,
        'status': status,
        'failure_reason': failure_reason,
        'best_epoch': selection.get(
            'selected_epoch',
            selection.get(
                'selected_step',
                selection.get('selected_iter', summary.get('best_epoch', '')),
            ),
        ),
        'checkpoint': checkpoint,
        'mask_loss_weight': _nested(config, 'train.lambda_mask', ''),
        'semantic_loss_weight': _nested(config, 'train.lambda_semantic', ''),
        'partial_detach': _nested(config, 'train.use_semantic_partial_detach', False),
        'reweight': _nested(config, 'train.use_feature_reweight', False),
        'fusion_type': semantic_mode if semantic_mode in ('early_fusion', 'cross_attention', 'weak_coupled') else 'none',
        'gate_type': 'hard_gate' if semantic_mode == 'hard_gate' else 'none',
        'config_path': config_path,
        'run_dir': run_dir,
    })
    for name in CAPTION_METRICS:
        row[name] = metrics.get(name, '')
        row['val_%s' % name] = selected_val_metrics.get(name, '')
    return row


def _numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[object, List[Dict[str, object]]] = {}
    for row in rows:
        key = (row['dataset'], row['experiment_group'])
        groups.setdefault(key, []).append(row)
    aggregate_rows: List[Dict[str, object]] = []
    for (dataset, group), members in groups.items():
        valid = [
            row for row in members
            if row['status'] == 'completed'
            and all(_numeric(row[name]) is not None for name in CAPTION_METRICS)
        ]
        distinct_seeds = {str(row['seed']) for row in valid if str(row['seed'])}
        if len(valid) < 2 or len(distinct_seeds) != len(valid):
            continue
        for aggregate_name in ('mean', 'std'):
            output = {key: '' for key in SUMMARY_FIELDS}
            output.update({
                'dataset': dataset,
                'experiment_name': group,
                'experiment_group': group,
                'run_id': aggregate_name,
                'seed': 'multiple',
                'status': 'aggregate',
                'run_dir': os.path.commonpath([str(row['run_dir']) for row in valid]),
            })
            for metric in CAPTION_METRICS:
                values = [_numeric(row[metric]) for row in valid]
                output[metric] = (
                    statistics.mean(values)
                    if aggregate_name == 'mean'
                    else statistics.stdev(values)
                )
                val_values = [_numeric(row['val_%s' % metric]) for row in valid]
                if all(value is not None for value in val_values):
                    output['val_%s' % metric] = (
                        statistics.mean(val_values)
                        if aggregate_name == 'mean'
                        else statistics.stdev(val_values)
                    )
            aggregate_rows.append(output)
    return aggregate_rows


def summarize(input_root: str) -> List[Dict[str, object]]:
    rows = []
    for path in sorted(set(_discover_runs(input_root))):
        try:
            rows.append(_row_from_run(path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError, yaml.YAMLError) as exc:
            row = {key: '' for key in SUMMARY_FIELDS}
            name = os.path.basename(path)
            row.update({
                'experiment_name': name,
                'experiment_group': name,
                'run_id': name,
                'status': 'failed',
                'failure_reason': 'summary artifact error: %s' % exc,
                'run_dir': path,
            })
            rows.append(row)
    return rows + _aggregate(rows)


def write_outputs(rows, csv_path: str, json_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in SUMMARY_FIELDS})
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False, allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser(description='Summarize experiment runs recursively.')
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-json', default=None)
    args = parser.parse_args()
    output_json = args.output_json or os.path.splitext(args.output_csv)[0] + '.json'
    rows = summarize(os.path.normpath(args.input_root))
    write_outputs(rows, os.path.normpath(args.output_csv), os.path.normpath(output_json))
    print(json.dumps({'rows': len(rows), 'output_csv': args.output_csv, 'output_json': output_json}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
