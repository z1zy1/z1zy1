#!/usr/bin/env python3
"""Resolve and audit the validation-locked LEVIR-CC source used by 7.6."""

import argparse
import csv
import json
import math
import os
import sys


METRICS = (
    'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
    'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE',
)
PROTECTED = METRICS[:-1]


def canonical(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path or ''))))


def load_json(path):
    with open(path, encoding='utf-8-sig') as handle:
        return json.load(handle)


def dotted(payload, path, default=None):
    value = payload
    for key in path.split('.'):
        value = value.get(key, default) if isinstance(value, dict) else default
    return value


def require_equal(payload, path, expected, context):
    actual = dotted(payload, path)
    if isinstance(expected, float):
        try:
            valid = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            valid = False
    else:
        valid = actual == expected
    if not valid:
        raise ValueError(
            '%s mismatch: %s=%r expected %r' % (context, path, actual, expected)
        )


def validate_selection(selection, path):
    if not isinstance(selection, dict) or selection.get('status') != 'done':
        raise ValueError('LEVIR-CC source selection is not done: %s' % path)
    if selection.get('selection_uses_test_metrics') is not False:
        raise ValueError('LEVIR-CC source selection is not validation-only: %s' % path)
    if selection.get('selection_metric_split') != 'validation':
        raise ValueError('LEVIR-CC source selection split is not validation: %s' % path)
    if selection.get('selection_strategy') not in (
        'val_baseline_pareto', 'val_baseline_stable_window'
    ):
        raise ValueError('LEVIR-CC source selection strategy is not approved: %s' % path)


def finite_metrics(payload, context):
    if not isinstance(payload, dict):
        raise ValueError('%s has no metrics object.' % context)
    result = {}
    for metric in METRICS:
        value = payload.get(metric)
        if isinstance(value, bool):
            raise ValueError('%s has boolean %s.' % (context, metric))
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError('%s is missing valid %s.' % (context, metric))
        if not math.isfinite(value):
            raise ValueError('%s has non-finite %s.' % (context, metric))
        result[metric] = value
    return result


def audit_validation_row(selection, checkpoint, source_dir):
    metrics_path = canonical(selection.get('metrics_file'))
    if os.path.dirname(metrics_path) != source_dir:
        raise ValueError('LEVIR-CC source validation metrics are outside the source experiment.')
    if os.path.basename(metrics_path) not in ('val_metrics.csv', 'eval_snapshots.csv'):
        raise ValueError('LEVIR-CC source metrics are not a validation artifact: %s' % metrics_path)
    selected = finite_metrics(selection.get('selected_val_metrics'), 'LEVIR-CC selection')
    with open(metrics_path, newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    matches = []
    for row in rows:
        raw = row.get('checkpoint_path') or row.get('snapshot_path') or ''
        candidates = (
            raw,
            os.path.join(source_dir, raw),
            os.path.join(source_dir, 'snapshots', os.path.basename(raw)),
        )
        resolved = next(
            (canonical(path) for path in candidates if path and os.path.isfile(path)), ''
        )
        if resolved == checkpoint:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError('LEVIR-CC checkpoint must map to exactly one validation row.')
    observed = finite_metrics(matches[0], 'LEVIR-CC validation row')
    for metric in METRICS:
        if not math.isclose(observed[metric], selected[metric], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('LEVIR-CC selected metric differs from validation row: %s' % metric)

    baseline_path = canonical(selection.get('baseline_source'))
    if os.path.basename(baseline_path) != 'baseline_best_checkpoint.json':
        raise ValueError('LEVIR-CC source selection has no audited CARD baseline artifact.')
    baseline_payload = load_json(baseline_path)
    validate_baseline = {
        'status': 'done',
        'selection_strategy': 'validation_best_cider',
        'selection_uses_test_metrics': False,
        'selection_metric_split': 'validation',
    }
    for key, expected in validate_baseline.items():
        if baseline_payload.get(key) != expected:
            raise ValueError('LEVIR-CC CARD baseline provenance mismatch at %s.' % key)
    baseline = finite_metrics(
        baseline_payload.get('selected_val_metrics'), 'LEVIR-CC CARD validation baseline'
    )
    for metric in PROTECTED:
        if selected[metric] < baseline[metric] - 1e-12:
            raise ValueError('LEVIR-CC source selection drops protected %s.' % metric)


def resolve_from_manifest(path):
    path = canonical(path)
    manifest = load_json(path)
    if manifest.get('status') != 'validation_locked':
        raise ValueError('LEVIR-CC source manifest is not validation_locked: %s' % path)
    if manifest.get('selection_uses_test_metrics') is not False:
        raise ValueError('LEVIR-CC source manifest is not validation-only: %s' % path)
    entry = manifest.get('datasets', {}).get('levir_cc', {})
    if entry.get('status') != 'done':
        raise ValueError('LEVIR-CC source manifest entry is not done: %s' % path)
    selection_path = canonical(entry.get('selection_json'))
    if not os.path.isfile(selection_path):
        raise FileNotFoundError('LEVIR-CC selection JSON is missing: %s' % selection_path)
    selection = load_json(selection_path)
    validate_selection(selection, selection_path)
    checkpoint = canonical(entry.get('selected_checkpoint'))
    source_dir = canonical(entry.get('selected_source_exp_dir'))
    source_name = str(entry.get('selected_source_exp_name') or '')
    source_config = canonical(entry.get('source_config') or entry.get('source_resolved_config'))
    selected_checkpoint = canonical(
        selection.get('selected_checkpoint_path') or selection.get('selected_checkpoint')
    )
    if selected_checkpoint != checkpoint:
        raise ValueError('Manifest checkpoint does not match LEVIR-CC validation selection.')
    if canonical(selection.get('selected_source_exp_dir')) != source_dir:
        raise ValueError('Manifest source directory does not match LEVIR-CC selection.')
    audit_validation_row(selection, checkpoint, source_dir)
    return checkpoint, source_dir, source_name, source_config


def resolve_from_selection(path, explicit_checkpoint):
    path = canonical(path)
    selection = load_json(path)
    validate_selection(selection, path)
    checkpoint = canonical(
        selection.get('selected_checkpoint_path') or selection.get('selected_checkpoint')
    )
    if checkpoint != canonical(explicit_checkpoint):
        raise ValueError('Explicit checkpoint does not match LEVIR-CC validation selection.')
    source_dir = canonical(selection.get('selected_source_exp_dir'))
    source_name = str(selection.get('selected_source_exp_name') or '')
    audit_validation_row(selection, checkpoint, source_dir)
    sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
    from resolve_experiment_config import resolve_experiment_config
    source_config, _, _ = resolve_experiment_config(source_dir)
    return checkpoint, source_dir, source_name, canonical(source_config)


def audit_paths(checkpoint, source_dir, source_config, source_name, expected_name):
    if source_name != expected_name:
        raise ValueError(
            'LEVIR-CC source name mismatch: %s expected %s' % (source_name, expected_name)
        )
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError('LEVIR-CC locked checkpoint is missing: %s' % checkpoint)
    if not os.path.isdir(source_dir):
        raise FileNotFoundError('LEVIR-CC source experiment is missing: %s' % source_dir)
    if os.path.commonpath((checkpoint, source_dir)) != source_dir:
        raise ValueError('LEVIR-CC checkpoint is outside its selected source experiment.')
    if not os.path.isfile(source_config):
        raise FileNotFoundError('LEVIR-CC source config is missing: %s' % source_config)


def audit_config(config, allow_registered_legacy=False):
    expected = {
        'model.enable_aux_mask': True,
        'model.semantic_input_mode': 'none',
        'train.use_semantic_aux': True,
        'train.use_semantic_partial_detach': True,
        'train.semantic_detach_ratio': 0.5,
        'train.use_feature_reweight': True,
        'train.detach_reweight_mask': True,
        'train.reweight_alpha': 0.2,
        'train.use_aux_warmup': True,
        'train.aux_warmup_start_ratio': 0.30,
        'train.aux_warmup_end_ratio': 0.70,
        'train.lambda_mask': 0.003,
        'train.lambda_semantic': 0.005,
        'train.mask_loss_type': 'bce_dice',
        'train.semantic_loss_type': 'multilabel_bce',
        'data.allow_missing_pseudo_mask': True,
    }
    for path, expected_value in expected.items():
        if (
            allow_registered_legacy
            and path == 'model.semantic_input_mode'
            and dotted(config, path) in (None, '')
        ):
            # The registered 7.5 source predates explicit tracking of this
            # switch. Its missing value is the historical default ``none``.
            continue
        require_equal(config, path, expected_value, 'LEVIR-CC source config')
    # This registered legacy run predates explicit model.type tracking. Its
    # complete SGC signature and state dictionaries are audited below.
    if dotted(config, 'model.type') not in (None, '', 'sgc_card'):
        raise ValueError('Unexpected model.type in legacy LEVIR-CC source config.')
    if dotted(config, 'data.dataset') not in (
        'rcc_dataset_transformer_levir', 'levir_cc', 'levir-cc'
    ):
        raise ValueError('Unexpected data.dataset in legacy LEVIR-CC source config.')


def audit_checkpoint(checkpoint, config, allow_registered_legacy=False):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('PyTorch is required for source checkpoint audit: %s' % exc)
    try:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location='cpu')
    if not isinstance(payload, dict):
        raise ValueError('LEVIR-CC source checkpoint payload is not a dictionary.')
    for state_name in ('change_detector_state', 'speaker_state'):
        if not isinstance(payload.get(state_name), dict) or not payload[state_name]:
            raise ValueError('LEVIR-CC source checkpoint has no %s.' % state_name)
    model_cfg = payload.get('model_cfg')
    if not isinstance(model_cfg, dict):
        raise ValueError('LEVIR-CC source checkpoint has no model_cfg mapping.')
    for path in (
        'data.dataset', 'model.type', 'model.enable_aux_mask',
        'model.semantic_input_mode', 'train.use_semantic_aux',
        'train.use_semantic_partial_detach', 'train.semantic_detach_ratio',
        'train.use_feature_reweight', 'train.reweight_alpha',
    ):
        checkpoint_value = dotted(model_cfg, path)
        config_value = dotted(config, path)
        if allow_registered_legacy and path == 'model.type':
            checkpoint_value = checkpoint_value or 'sgc_card'
            config_value = config_value or 'sgc_card'
        if allow_registered_legacy and path == 'model.semantic_input_mode':
            checkpoint_value = checkpoint_value or 'none'
            config_value = config_value or 'none'
        if checkpoint_value != config_value:
            raise ValueError('Checkpoint model_cfg disagrees with source config at %s.' % path)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--manifest')
    group.add_argument('--selection_json')
    parser.add_argument('--checkpoint')
    parser.add_argument('--expected_source_exp', required=True)
    args = parser.parse_args()
    if args.selection_json and not args.checkpoint:
        parser.error('--selection_json requires --checkpoint')
    if args.manifest:
        resolved = resolve_from_manifest(args.manifest)
    else:
        resolved = resolve_from_selection(args.selection_json, args.checkpoint)
    checkpoint, source_dir, source_name, source_config = resolved
    audit_paths(checkpoint, source_dir, source_config, source_name, args.expected_source_exp)
    config = load_json(source_config)
    registered_legacy = source_name == 'sgc_card_lm003_ls005_pd05_rw02_warmup'
    audit_config(config, allow_registered_legacy=registered_legacy)
    audit_checkpoint(
        checkpoint, config, allow_registered_legacy=registered_legacy
    )
    print(checkpoint)
    print(source_config)
    print(source_dir)


if __name__ == '__main__':
    main()
