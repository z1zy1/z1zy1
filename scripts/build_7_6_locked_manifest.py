#!/usr/bin/env python3
'''Build and verify the validation-only 7.6 five-lock manifest.'''

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import copy
import re
import statistics
import tempfile

from resolve_experiment_config import resolve_experiment_config
from build_card_baseline_manifest import (
    DEFAULT_EXPERIMENTS as CARD_BASELINE_EXPERIMENTS,
    audit_resolved_config as audit_card_baseline_config,
    build_entry as build_card_baseline_entry,
    verify_manifest as verify_card_baseline_manifest,
)
from resolve_7_6_levir_cc_source import (
    audit_checkpoint as audit_levir_cc_source_checkpoint,
    audit_config as audit_levir_cc_source_config,
    audit_paths as audit_levir_cc_source_paths,
    resolve_from_manifest as resolve_levir_cc_source_manifest,
)


METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
PROTECTED = METRICS[:-1]
MCI_SEEDS = (1111, 2222, 3333)
INVENTORY_CACHE = {}


def canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.normpath(path))))


def is_within(path, root):
    try:
        return os.path.commonpath([canonical(path), canonical(root)]) == canonical(root)
    except ValueError:
        return False


def load_json(path):
    with open(path, encoding='utf-8-sig') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError('Expected JSON object: %s' % path)
    return payload


def finite_metric_map(raw, label):
    if not isinstance(raw, dict):
        raise ValueError('%s metrics must be an object.' % label)
    result = {}
    for metric in METRICS:
        value = raw.get(metric)
        if isinstance(value, bool):
            raise ValueError('%s %s is boolean.' % (label, metric))
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError('%s is missing/invalid %s.' % (label, metric))
        if not math.isfinite(value):
            raise ValueError('%s %s is not finite.' % (label, metric))
        result[metric] = value
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(role, path):
    path = canonical(path)
    if not os.path.isfile(path):
        raise FileNotFoundError('%s file missing: %s' % (role, path))
    stat = os.stat(path)
    return {
        'role': role,
        'path': path,
        'size': stat.st_size,
        'sha256': sha256_file(path),
    }


def directory_inventory(path):
    path = canonical(path)
    if path in INVENTORY_CACHE:
        return copy.deepcopy(INVENTORY_CACHE[path])
    if not os.path.isdir(path):
        raise FileNotFoundError('Feature directory missing: %s' % path)
    digest = hashlib.sha256()
    count = 0
    total_size = 0
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            full_path = os.path.join(root, name)
            stat = os.stat(full_path)
            relative = os.path.relpath(full_path, path).replace(os.sep, '/')
            digest.update(('%s\0%d\0%d\n' % (relative, stat.st_size, stat.st_mtime_ns)).encode('utf-8'))
            count += 1
            total_size += stat.st_size
    if count == 0:
        raise ValueError('Feature directory is empty: %s' % path)
    result = {
        'path': path,
        'file_count': count,
        'total_size': total_size,
        'inventory_sha256': digest.hexdigest(),
    }
    INVENTORY_CACHE[path] = copy.deepcopy(result)
    return result


def nested_get(payload, dotted, default=None):
    value = payload
    for part in dotted.split('.'):
        if isinstance(value, dict):
            value = value.get(part, default)
        else:
            try:
                value = getattr(value, part)
            except (AttributeError, KeyError):
                return default
        if value is default:
            return default
    return value


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def as_int(value, label):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError('%s must be an integer, got %r.' % (label, value))


def resolve_data_path(raw, project_dir):
    raw = str(raw or '')
    if not raw:
        return ''
    return canonical(raw if os.path.isabs(raw) else os.path.join(project_dir, raw))


def audit_vocab_file(path, configured_vocab_size):
    payload = load_json(path)
    source = payload
    if isinstance(payload.get('word_to_idx'), dict):
        source = payload['word_to_idx']
    elif isinstance(payload.get('word2idx'), dict):
        source = payload['word2idx']
    elif isinstance(payload.get('idx_to_word'), (dict, list)):
        source = payload['idx_to_word']
    if isinstance(source, list):
        indices = list(range(len(source)))
    elif isinstance(source, dict):
        values_are_indices = all(
            not isinstance(value, bool) and isinstance(value, (int, float, str))
            and str(value).lstrip('-').isdigit()
            for value in source.values()
        )
        if values_are_indices:
            indices = [int(value) for value in source.values()]
        elif all(str(key).lstrip('-').isdigit() for key in source):
            indices = [int(key) for key in source]
        else:
            raise ValueError('Cannot infer vocabulary indices from %s.' % path)
    else:
        raise ValueError('Unsupported vocabulary structure: %s' % path)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError('Vocabulary indices are empty or duplicated: %s' % path)
    if sorted(indices) != list(range(len(indices))):
        raise ValueError('Vocabulary indices must be contiguous 0..N-1: %s' % path)
    configured_vocab_size = as_int(configured_vocab_size, 'configured vocab_size')
    if len(indices) != configured_vocab_size:
        raise ValueError('Vocabulary size mismatch: file=%d config=%d (%s).' % (
            len(indices), configured_vocab_size, path))
    return {
        'path': canonical(path),
        'size': len(indices),
        'min_index': min(indices),
        'max_index': max(indices),
        'indices_contiguous': True,
    }


def audit_speaker_vocab_shapes(speaker_state, vocab_size, share_wd_cls_weight):
    embed = [
        (key, value) for key, value in speaker_state.items()
        if key.endswith('core.embed.weight')
    ]
    if len(embed) != 1:
        raise ValueError('Expected exactly one speaker core.embed.weight, found %d.' % len(embed))
    embed_key, embed_tensor = embed[0]
    embed_shape = tuple(getattr(embed_tensor, 'shape', ()))
    if not embed_shape or embed_shape[0] != vocab_size:
        raise ValueError('Embedding vocabulary dimension mismatch: %s shape=%r expected=%d.' % (
            embed_key, embed_shape, vocab_size))
    output = [
        (key, value) for key, value in speaker_state.items()
        if key.endswith('logit.weight') or key.endswith('logit.decoder.weight')
    ]
    if output:
        bad = [(key, tuple(getattr(value, 'shape', ()))) for key, value in output
               if not getattr(value, 'shape', ()) or int(value.shape[0]) != vocab_size]
        if bad:
            raise ValueError('Output vocabulary dimension mismatch: %r expected=%d.' % (bad, vocab_size))
    elif not share_wd_cls_weight:
        raise ValueError('Non-shared speaker checkpoint has no logit output weight.')
    return {
        'vocab_size': vocab_size,
        'embedding_key': embed_key,
        'embedding_shape': list(embed_shape),
        'output_weight_shapes': {
            key: list(tuple(getattr(value, 'shape', ()))) for key, value in output
        },
        'share_wd_cls_weight': bool(share_wd_cls_weight),
    }


def audit_baseline(path, dataset, exp_root, baseline_manifest_entry):
    expected_exp = CARD_BASELINE_EXPERIMENTS[dataset]
    expected_dir = canonical(os.path.join(exp_root, expected_exp))
    expected_path = canonical(os.path.join(expected_dir, 'baseline_best_checkpoint.json'))
    if canonical(path) != expected_path:
        raise ValueError('%s baseline artifact must be exactly %s.' % (dataset, expected_path))
    payload = load_json(path)
    required = {
        'status': 'done',
        'selection_strategy': 'validation_best_cider',
        'selection_uses_test_metrics': False,
        'selection_metric_split': 'validation',
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError('%s baseline %s=%r, expected %r.' % (dataset, key, payload.get(key), expected))
    identity = ' '.join(str(payload.get(key, '')) for key in (
        'exp_name', 'selected_source_exp_name', 'candidate_exp_dirs'))
    if 'baseline' not in identity.lower():
        raise ValueError('%s baseline artifact has no baseline identity.' % dataset)
    if 'test' in os.path.basename(path).lower():
        raise ValueError('Test JSON cannot be a validation baseline: %s' % path)
    source_dir = canonical(payload.get('selected_source_exp_dir') or '')
    if source_dir != expected_dir:
        raise ValueError('%s baseline selection source is not its canonical CARD experiment.' % dataset)
    config_path = canonical(os.path.join(expected_dir, 'resolved_config.json'))
    audit_card_baseline_config(config_path, dataset)
    rebuilt = build_card_baseline_entry(exp_root, dataset, expected_exp)
    for key in ('selection_json', 'selected_checkpoint', 'resolved_config'):
        if canonical(rebuilt[key]) != canonical(baseline_manifest_entry[key]):
            raise ValueError('%s CARD baseline manifest binding changed for %s.' % (dataset, key))
    metrics = finite_metric_map(
        payload.get('selected_val_metrics') or payload.get('selected_metrics'),
        '%s baseline' % dataset,
    )
    manifest_metrics = finite_metric_map(
        baseline_manifest_entry.get('selected_val_metrics'),
        '%s CARD baseline manifest' % dataset,
    )
    if metrics != manifest_metrics:
        raise ValueError('%s baseline selection metrics differ from CARD baseline manifest.' % dataset)
    return {
        'path': canonical(path),
        'metrics': metrics,
        'artifact': file_record('%s_baseline_validation' % dataset, path),
        'selected_checkpoint': canonical(rebuilt['selected_checkpoint']),
        'resolved_config': canonical(rebuilt['resolved_config']),
    }


def read_validation_row(metrics_path, checkpoint, selected_metrics):
    if 'test' in os.path.basename(metrics_path).lower():
        raise ValueError('Validation metrics filename contains test: %s' % metrics_path)
    if os.path.basename(metrics_path) not in ('val_metrics.csv', 'eval_snapshots.csv'):
        raise ValueError('Unexpected validation metrics artifact: %s' % metrics_path)
    with open(metrics_path, newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    matches = []
    checkpoint = canonical(checkpoint)
    checkpoint_step = os.path.basename(checkpoint).rsplit('_', 1)[-1].split('.')[0]
    for row in rows:
        raw = row.get('checkpoint_path') or row.get('snapshot_path') or ''
        candidates = [raw, os.path.join(os.path.dirname(metrics_path), raw),
                      os.path.join(os.path.dirname(metrics_path), 'snapshots', os.path.basename(raw))]
        row_path = next((canonical(path) for path in candidates if path and os.path.isfile(path)), '')
        row_step = str(row.get('iter') or row.get('step') or row.get('epoch') or '')
        if row_path == checkpoint or (not row_path and row_step == checkpoint_step):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError('Expected exactly one validation row for %s in %s; found %d.' % (
            checkpoint, metrics_path, len(matches)))
    row_metrics = finite_metric_map(matches[0], 'selected validation row')
    for metric in METRICS:
        if not math.isclose(row_metrics[metric], selected_metrics[metric], rel_tol=0, abs_tol=1e-12):
            raise ValueError('Selection/validation row mismatch for %s.' % metric)
    return row_metrics


def load_checkpoint_cfg_audit(checkpoint, resolved, dataset, expected_seed, vocab_size):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('PyTorch is required for checkpoint payload audit: %s' % exc)
    try:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location='cpu')
    if not isinstance(payload, dict):
        raise ValueError('Checkpoint payload is not a dict: %s' % checkpoint)
    for state_key in ('change_detector_state', 'speaker_state'):
        if not isinstance(payload.get(state_key), dict) or not payload[state_key]:
            raise ValueError('Checkpoint missing non-empty %s: %s' % (state_key, checkpoint))
    model_cfg = payload.get('model_cfg')
    if model_cfg is None:
        raise ValueError('Checkpoint missing model_cfg: %s' % checkpoint)
    checks = {
        'data.dataset': dataset,
        'train.seed': expected_seed,
        'model.type': nested_get(resolved, 'model.type'),
        'model.semantic_input_mode': nested_get(resolved, 'model.semantic_input_mode'),
        'model.transformer_decoder.vocab_size': nested_get(resolved, 'model.transformer_decoder.vocab_size'),
        'model.transformer_decoder.seq_length': nested_get(resolved, 'model.transformer_decoder.seq_length'),
        'data.eval_anno_path': nested_get(resolved, 'data.eval_anno_path'),
        'data.vocab_json': nested_get(resolved, 'data.vocab_json'),
        'data.h5_label_file': nested_get(resolved, 'data.h5_label_file'),
        'data.default_feature_dir': nested_get(resolved, 'data.default_feature_dir'),
        'data.semantic_feature_dir': nested_get(resolved, 'data.semantic_feature_dir'),
    }
    observed = {}
    for key, expected in checks.items():
        if expected in (None, ''):
            continue
        actual = nested_get(model_cfg, key)
        observed[key] = actual
        if str(actual) != str(expected):
            raise ValueError('Checkpoint model_cfg mismatch %s: %r != %r.' % (key, actual, expected))
    vocab_shape_audit = audit_speaker_vocab_shapes(
        payload['speaker_state'],
        vocab_size,
        as_bool(nested_get(resolved, 'model.transformer_decoder.share_wd_cls_weight', False)),
    )
    return {
        'model_cfg_checks': observed,
        'change_detector_state_keys': len(payload['change_detector_state']),
        'speaker_state_keys': len(payload['speaker_state']),
        'speaker_vocab_shape_audit': vocab_shape_audit,
    }


def audit_data(resolved, expected_root, project_dir):
    data = resolved.get('data', {})
    dataset = str(data.get('dataset', '')).lower()
    artifacts = []
    resolved_files = {}
    for key in ('eval_anno_path', 'vocab_json', 'h5_label_file', 'splits_json'):
        path = resolve_data_path(data.get(key), project_dir)
        if not path:
            raise ValueError('Source config missing data.%s.' % key)
        if not is_within(path, expected_root):
            raise ValueError('data.%s is outside expected root: %s' % (key, path))
        artifacts.append(file_record('data.%s' % key, path))
        resolved_files[key] = path
    feature_dirs = []
    for key in ('default_feature_dir', 'semantic_feature_dir'):
        raw = data.get(key)
        if not raw:
            continue
        path = resolve_data_path(raw, project_dir)
        if not is_within(path, expected_root):
            raise ValueError('data.%s is outside expected root: %s' % (key, path))
        inventory = directory_inventory(path)
        inventory['role'] = 'data.%s' % key
        if all(item['path'] != inventory['path'] for item in feature_dirs):
            feature_dirs.append(inventory)
    if not feature_dirs:
        raise ValueError('Source config has no feature directory.')
    vocab_audit = audit_vocab_file(
        resolved_files['vocab_json'],
        nested_get(resolved, 'model.transformer_decoder.vocab_size'),
    )
    pseudo_mask_dirs = []
    if dataset == 'levir_mci':
        pseudo_root = resolve_data_path(data.get('pseudo_mask_root'), project_dir)
        if not pseudo_root or not is_within(pseudo_root, expected_root):
            raise ValueError('LEVIR-MCI pseudo_mask_root is missing/outside dataset root.')
        pseudo_inventory = directory_inventory(pseudo_root)
        pseudo_inventory['role'] = 'data.pseudo_mask_root'
        pseudo_mask_dirs.append(pseudo_inventory)
    semantic_map_dirs = []
    if as_bool(data.get('use_semantic_maps')):
        semantic_root = resolve_data_path(data.get('semantic_map_root'), project_dir)
        if not semantic_root or not is_within(semantic_root, expected_root):
            raise ValueError('Semantic map root is missing/outside dataset root.')
        for phase_key in ('semantic_before_phase', 'semantic_after_phase'):
            phase = str(data.get(phase_key) or '')
            phase_path = canonical(os.path.join(semantic_root, phase))
            if not phase or not is_within(phase_path, expected_root):
                raise ValueError('data.%s is missing/outside dataset root.' % phase_key)
            inventory = directory_inventory(phase_path)
            inventory['role'] = 'data.semantic_map_root/%s' % phase_key
            semantic_map_dirs.append(inventory)
    changeflag_name = {
        'levir_mci': 'LevirCCcaptions.json',
        'second_cc': 'SECOND-CC-AUG.json',
    }.get(dataset)
    if changeflag_name:
        changeflag_path = canonical(os.path.join(expected_root, changeflag_name))
        artifacts.append(file_record('data.changeflag_json', changeflag_path))
    semantic_tag_file = resolve_data_path(nested_get(resolved, 'train.semantic_tag_file'), project_dir)
    if as_bool(nested_get(resolved, 'train.use_semantic_aux')):
        if not semantic_tag_file or not is_within(semantic_tag_file, project_dir):
            raise ValueError('Semantic tag file is missing/outside project root.')
        artifacts.append(file_record('train.semantic_tag_file', semantic_tag_file))
    return {
        'artifacts': artifacts,
        'feature_directories': feature_dirs,
        'vocab_audit': vocab_audit,
        'pseudo_mask_directories': pseudo_mask_dirs,
        'semantic_map_directories': semantic_map_dirs,
        'resolved_paths': dict(resolved_files, **{
            'default_feature_dir': resolve_data_path(data.get('default_feature_dir'), project_dir),
            'semantic_feature_dir': resolve_data_path(data.get('semantic_feature_dir'), project_dir),
            'semantic_map_root': resolve_data_path(data.get('semantic_map_root'), project_dir),
            'semantic_tag_file': semantic_tag_file,
        }),
    }


def audit_levir_cc_init_provenance(exp_root):
    manifest_path = canonical(os.path.join(exp_root, '7_5_locked_manifest.json'))
    checkpoint, source_dir, source_name, source_config = resolve_levir_cc_source_manifest(manifest_path)
    expected_name = 'sgc_card_lm003_ls005_pd05_rw02_warmup'
    audit_levir_cc_source_paths(
        checkpoint, source_dir, source_config, source_name, expected_name)
    config = load_json(source_config)
    audit_levir_cc_source_config(config)
    audit_levir_cc_source_checkpoint(checkpoint, config)
    entry = load_json(manifest_path)['datasets']['levir_cc']
    selection_path = canonical(entry['selection_json'])
    return {
        'manifest': file_record('levir_cc_init_7_5_manifest', manifest_path),
        'selection': file_record('levir_cc_init_7_5_selection', selection_path),
        'checkpoint': file_record('levir_cc_init_checkpoint', checkpoint),
        'source_config': file_record('levir_cc_init_source_config', source_config),
        'source_exp_dir': source_dir,
        'source_exp_name': source_name,
    }


def config_interval(resolved):
    for key in ('train.eval_interval', 'train.snapshot_interval', 'train.save_interval'):
        value = nested_get(resolved, key, 0)
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def mci_method_signature(resolved):
    expected = {
        'model.type': 'sgc_card',
        'data.dataset': 'levir_mci',
        'data.use_change_mask': True,
        'data.mask_type': 'multiclass',
        'model.enable_aux_mask': True,
        'train.use_semantic_aux': True,
        'model.semantic_input_mode': 'aux',
        'train.lambda_mask': 0.003,
        'train.lambda_semantic': 0.005,
        'train.use_feature_reweight': False,
        'train.finetune_decoder_only': False,
        'train.use_semantic_partial_detach': False,
        'train.semantic_detach_ratio': 0.5,
        'train.mask_loss_type': 'ce_dice',
        'train.semantic_loss_type': 'multilabel_bce',
        'train.use_aux_warmup': True,
        'train.aux_warmup_start_ratio': 0.30,
        'train.aux_warmup_end_ratio': 0.70,
        'train.max_iter': 10000,
        'train.total_steps': 10000,
        'train.snapshot_interval': 1000,
        'train.save_interval': 1000,
        'train.eval_interval': 1000,
        'train.optim.type': 'adam',
        'train.optim.lr': 0.0002,
        'train.optim.weight_decay': 0.0,
        'train.optim.step_size': 17,
        'train.optim.gamma': 0.1,
        'data.train.batch_size': 32,
    }
    critical = {}
    for key, expected_value in expected.items():
        actual = nested_get(resolved, key)
        if isinstance(expected_value, bool):
            actual = as_bool(actual)
        elif isinstance(expected_value, float):
            actual = float(actual)
        elif isinstance(expected_value, int):
            actual = int(actual)
        if actual != expected_value:
            raise ValueError('MCI reproducibility config mismatch %s: %r != %r.' % (
                key, actual, expected_value))
        critical[key] = actual
    if nested_get(resolved, 'train.init_checkpoint') not in (None, ''):
        raise ValueError('MCI reproducibility run must train from scratch (init_checkpoint must be empty).')
    if config_interval(resolved) != 1000:
        raise ValueError('MCI reproducibility validation interval must be 1000.')
    normalized = copy.deepcopy(resolved)
    normalized.pop('exp_name', None)
    normalized.pop('exp_dir', None)
    normalized.pop('logger', None)
    if isinstance(normalized.get('train'), dict):
        normalized['train'].pop('seed', None)
    serialized = json.dumps(normalized, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return {
        'normalized_config_sha256': hashlib.sha256(serialized.encode('utf-8')).hexdigest(),
        'critical_values': critical,
    }


def require_close(resolved, key, expected, label):
    value = nested_get(resolved, key)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('%s %s is not numeric: %r.' % (label, key, value))
    if not math.isclose(value, float(expected), rel_tol=0, abs_tol=1e-12):
        raise ValueError('%s %s=%r, expected %r.' % (label, key, value, expected))


def audit_selected_method(resolved, dataset, source_name):
    model_type = nested_get(resolved, 'model.type')
    legacy_cc = dataset == 'levir_cc' and source_name == 'sgc_card_lm003_ls005_pd05_rw02_warmup'
    legacy_second = dataset == 'second_cc' and source_name in {
        'second_cc_crossattn_pd07_lsem0000',
        'second_cc_crossattn_pd08_lsem0000',
        'second_cc_crossattn_pd09_lsem0000',
        'second_cc_card_semantic_crossattn',
    }
    if legacy_cc or legacy_second:
        if model_type not in (None, '', 'sgc_card'):
            raise ValueError('%s has an unexpected legacy model.type: %r.' % (source_name, model_type))
    elif model_type != 'sgc_card':
        raise ValueError('%s selected model.type must be sgc_card.' % source_name)
    if dataset == 'levir_cc' and source_name == 'sgc_card_lm003_ls005_pd05_rw02_warmup':
        compatibility_view = copy.deepcopy(resolved)
        compatibility_view.setdefault('data', {})['dataset'] = 'levir_cc'
        compatibility_view.setdefault('model', {})['type'] = 'sgc_card'
        audit_levir_cc_source_config(compatibility_view)
    elif dataset == 'levir_cc' and source_name.startswith('levir_cc_decft_'):
        audit_levir_cc_source_config(resolved)
        match = re.fullmatch(r'levir_cc_decft_cw(100|101|102)_s(10|20)_lr(5e7|1e6)', source_name)
        if not match:
            raise ValueError('Unexpected LEVIR-CC grid source name: %s' % source_name)
        weight = int(match.group(1)) / 100.0
        steps = int(match.group(2))
        lr = 0.0000005 if match.group(3) == '5e7' else 0.000001
        if not as_bool(nested_get(resolved, 'train.finetune_decoder_only')):
            raise ValueError('%s must enable decoder-only fine-tuning.' % source_name)
        if nested_get(resolved, 'train.init_checkpoint') in (None, ''):
            raise ValueError('%s must record its initialization checkpoint.' % source_name)
        if not as_bool(nested_get(resolved, 'train.use_content_word_weight')):
            raise ValueError('%s must enable content-word weighting.' % source_name)
        if not as_bool(nested_get(resolved, 'train.normalize_content_word_weights')):
            raise ValueError('%s must normalize content-word weights.' % source_name)
        require_close(resolved, 'train.content_word_weight', weight, source_name)
        require_close(resolved, 'train.finetune_steps', steps, source_name)
        require_close(resolved, 'train.total_steps', steps, source_name)
        require_close(resolved, 'train.optim.lr', lr, source_name)
        for key in ('train.save_interval', 'train.eval_interval', 'train.snapshot_interval'):
            require_close(resolved, key, 5, source_name)
    if dataset == 'second_cc':
        if not as_bool(nested_get(resolved, 'data.use_semantic_maps')):
            raise ValueError('%s must use semantic maps.' % source_name)
        if nested_get(resolved, 'model.semantic_input_mode') != 'cross_attention':
            raise ValueError('%s must use semantic cross-attention.' % source_name)
        require_close(resolved, 'model.num_semantic_classes', 7, source_name)
        if source_name.startswith('second_cc_crossattn_pd'):
            match = re.fullmatch(r'second_cc_crossattn_pd(07|08|09)_lsem0000', source_name)
            if not match:
                raise ValueError('Unexpected SECOND-CC source name: %s' % source_name)
            ratio = int(match.group(1)) / 10.0
            if not as_bool(nested_get(resolved, 'train.use_semantic_partial_detach')):
                raise ValueError('%s must enable partial detach.' % source_name)
            require_close(resolved, 'train.semantic_detach_ratio', ratio, source_name)
            require_close(resolved, 'train.lambda_semantic', 0.0, source_name)
            if as_bool(nested_get(resolved, 'train.use_semantic_aux')):
                raise ValueError('%s must disable semantic auxiliary loss.' % source_name)
        elif source_name == 'second_cc_card_semantic_crossattn':
            if as_bool(nested_get(resolved, 'train.use_semantic_partial_detach')):
                raise ValueError('%s must not use partial detach.' % source_name)
            if not as_bool(nested_get(resolved, 'train.use_semantic_aux')):
                raise ValueError('%s must enable semantic auxiliary loss.' % source_name)
            require_close(resolved, 'train.lambda_semantic', 0.005, source_name)
        else:
            raise ValueError('Unregistered SECOND-CC selected source: %s' % source_name)


def audit_selection(spec, baselines, roots, project_dir):
    selection_path = canonical(spec['selection_json'])
    selection = load_json(selection_path)
    expected_strategy = spec['strategy']
    required = {
        'status': 'done',
        'selection_strategy': expected_strategy,
        'selection_uses_test_metrics': False,
        'selection_metric_split': 'validation',
    }
    for key, expected in required.items():
        if selection.get(key) != expected:
            raise ValueError('%s selection %s=%r, expected %r.' % (
                spec['lock_id'], key, selection.get(key), expected))
    baseline = baselines[spec['dataset']]
    if canonical(selection.get('baseline_source', '')) != baseline['path']:
        raise ValueError('%s did not use the locked CARD validation baseline.' % spec['lock_id'])
    selected_metrics = finite_metric_map(selection.get('selected_val_metrics'), spec['lock_id'])
    declared_deltas = finite_metric_map(selection.get('selected_metric_deltas'), '%s deltas' % spec['lock_id'])
    deltas = {
        metric: selected_metrics[metric] - baseline['metrics'][metric]
        for metric in METRICS
    }
    for metric in METRICS:
        if not math.isclose(declared_deltas[metric], deltas[metric], rel_tol=0, abs_tol=1e-12):
            raise ValueError('%s declared delta mismatch for %s.' % (spec['lock_id'], metric))
    if any(deltas[metric] < -1e-12 for metric in PROTECTED):
        raise ValueError('%s drops a protected validation metric below baseline.' % spec['lock_id'])
    if deltas['SPICE'] < -1e-12:
        raise ValueError('%s does not preserve validation SPICE baseline.' % spec['lock_id'])
    checkpoint = canonical(selection.get('selected_checkpoint_path') or selection.get('selected_checkpoint') or '')
    source_dir = canonical(selection.get('selected_source_exp_dir') or '')
    allowed_source_dirs = [canonical(path) for path in spec['allowed_source_dirs']]
    candidate_dirs = [canonical(path) for path in selection.get('candidate_exp_dirs', [])]
    if set(candidate_dirs) != set(allowed_source_dirs) or len(candidate_dirs) != len(allowed_source_dirs):
        raise ValueError('%s candidate source set differs from the preregistered pool.' % spec['lock_id'])
    if source_dir not in allowed_source_dirs:
        raise ValueError('%s selected source is outside the preregistered pool.' % spec['lock_id'])
    if os.path.basename(source_dir) == CARD_BASELINE_EXPERIMENTS[spec['dataset']]:
        raise ValueError('%s must not fall back to CARD baseline checkpoint.' % spec['lock_id'])
    if not os.path.isdir(source_dir):
        raise FileNotFoundError('Selected source experiment missing: %s' % source_dir)
    snapshots_dir = os.path.join(source_dir, 'snapshots')
    if not os.path.isfile(checkpoint) or not is_within(checkpoint, snapshots_dir):
        raise ValueError('Selected checkpoint is not a source snapshot: %s' % checkpoint)
    metrics_path = canonical(selection.get('metrics_file') or '')
    if os.path.dirname(metrics_path) != source_dir:
        raise ValueError('Selected metrics file is not in source experiment: %s' % metrics_path)
    read_validation_row(metrics_path, checkpoint, selected_metrics)
    source_config, resolved, source_artifact = resolve_experiment_config(source_dir)
    config_dataset = str(nested_get(resolved, 'data.dataset', '')).lower()
    allowed_config_datasets = {spec['dataset']}
    if spec['lock_id'] == 'levir_cc' and os.path.basename(source_dir) == 'sgc_card_lm003_ls005_pd05_rw02_warmup':
        allowed_config_datasets.add('rcc_dataset_transformer_levir')
    if config_dataset not in allowed_config_datasets:
        raise ValueError('%s source config dataset mismatch: %s.' % (spec['lock_id'], config_dataset))
    seed = as_int(nested_get(resolved, 'train.seed'), '%s train.seed' % spec['lock_id'])
    if spec['seed'] is not None and seed != spec['seed']:
        raise ValueError('%s source seed mismatch: %s.' % (spec['lock_id'], seed))
    if spec['dataset'] == 'levir_mci' and source_dir != canonical(spec['required_source_dir']):
        raise ValueError('%s must select only its own seed experiment.' % spec['lock_id'])
    audit_selected_method(resolved, spec['dataset'], os.path.basename(source_dir))
    levir_cc_init_provenance = {}
    if spec['dataset'] == 'levir_cc':
        levir_cc_init_provenance = audit_levir_cc_init_provenance(os.path.dirname(source_dir))
    if spec['dataset'] == 'levir_cc' and os.path.basename(source_dir).startswith('levir_cc_decft_'):
        init_checkpoint = canonical(nested_get(resolved, 'train.init_checkpoint') or '')
        expected_init = canonical(levir_cc_init_provenance['checkpoint']['path'])
        if init_checkpoint != expected_init:
            raise ValueError('%s init checkpoint differs from the exact 7.5 validation lock.' % spec['lock_id'])
    data_audit = audit_data(resolved, roots[spec['dataset']], project_dir)
    payload_audit = load_checkpoint_cfg_audit(
        checkpoint,
        resolved,
        config_dataset,
        seed,
        data_audit['vocab_audit']['size'],
    )
    method_signature = mci_method_signature(resolved) if spec['dataset'] == 'levir_mci' else {}

    stability = selection.get('stability_window') or {}
    window_records = []
    if expected_strategy == 'val_baseline_stable_window':
        if stability.get('size') != 3:
            raise ValueError('SECOND stability window must contain exactly 3 checkpoints.')
        if stability.get('all_members_complete_finite') is not True:
            raise ValueError('SECOND stability window is not marked finite/complete.')
        if stability.get('all_members_preserve_validation_baseline') is not True:
            raise ValueError('SECOND stability window does not preserve protected baselines.')
        if stability.get('all_members_preserve_spice_gain') is not True:
            raise ValueError('SECOND stability window contains a SPICE-regressing member.')
        gap = as_int(stability.get('expected_step_gap'), 'SECOND stability gap')
        if gap != 1000 or config_interval(resolved) != gap:
            raise ValueError('SECOND stability gap does not match source validation interval.')
        steps = [as_int(value, 'SECOND member step') for value in stability.get('member_steps', [])]
        if len(steps) != 3 or any(right - left != gap for left, right in zip(steps, steps[1:])):
            raise ValueError('SECOND stability members are not contiguous at the audited interval.')
        members = stability.get('member_checkpoints') or []
        if len(members) != 3 or checkpoint != canonical(members[1]):
            raise ValueError('SECOND selected checkpoint is not the centre stability member.')
        if len({canonical(member) for member in members}) != 3:
            raise ValueError('SECOND stability window checkpoints must be unique.')
        if canonical(stability.get('source_exp_dir', '')) != source_dir:
            raise ValueError('SECOND stability window crosses source experiments.')
        declared_member_metrics = stability.get('member_metrics') or []
        declared_member_deltas = stability.get('member_metric_deltas') or []
        if len(declared_member_metrics) != 3 or len(declared_member_deltas) != 3:
            raise ValueError('SECOND stability window must record metrics/deltas for all members.')
        audited_member_metrics = []
        for index, member in enumerate(members):
            member = canonical(member)
            if not os.path.isfile(member) or not is_within(member, snapshots_dir):
                raise ValueError('SECOND window member is outside source snapshots: %s' % member)
            member_metrics = finite_metric_map(
                declared_member_metrics[index], 'SECOND stability member %d' % index)
            read_validation_row(metrics_path, member, member_metrics)
            member_deltas = finite_metric_map(
                declared_member_deltas[index], 'SECOND stability delta %d' % index)
            for metric in METRICS:
                computed = member_metrics[metric] - baseline['metrics'][metric]
                if not math.isclose(computed, member_deltas[metric], rel_tol=0, abs_tol=1e-12):
                    raise ValueError('SECOND window member %d delta mismatch for %s.' % (index, metric))
                if metric in PROTECTED and computed < -1e-12:
                    raise ValueError('SECOND window member %d drops protected %s.' % (index, metric))
                if metric == 'SPICE' and computed < -1e-12:
                    raise ValueError('SECOND window member %d drops SPICE.' % index)
            audited_member_metrics.append(member_metrics)
            window_records.append(file_record('stability_member_%d' % index, member))
        computed_mean = {metric: statistics.mean(row[metric] for row in audited_member_metrics) for metric in METRICS}
        computed_std = {metric: statistics.stdev(row[metric] for row in audited_member_metrics) for metric in METRICS}
        computed_worst = {metric: min(row[metric] for row in audited_member_metrics) for metric in METRICS}
        for field, computed in (
            ('metric_mean', computed_mean),
            ('metric_sample_std', computed_std),
            ('metric_worst', computed_worst),
        ):
            declared = finite_metric_map(stability.get(field), 'SECOND stability %s' % field)
            for metric in METRICS:
                if not math.isclose(computed[metric], declared[metric], rel_tol=0, abs_tol=1e-12):
                    raise ValueError('SECOND stability %s mismatch for %s.' % (field, metric))

    artifacts = [
        file_record('selection_json', selection_path),
        file_record('selected_checkpoint', checkpoint),
        file_record('source_config', source_config),
        file_record('validation_metrics', metrics_path),
    ] + data_audit['artifacts']
    if levir_cc_init_provenance:
        artifacts.extend(
            levir_cc_init_provenance[key]
            for key in ('manifest', 'selection', 'checkpoint', 'source_config')
        )
    target_path = canonical(os.path.join(os.path.dirname(selection_path)))
    return {
        'lock_id': spec['lock_id'],
        'dataset': spec['dataset'],
        'seed': spec['seed'],
        'target_exp': os.path.basename(target_path),
        'selection_json': selection_path,
        'selection_strategy': expected_strategy,
        'selected_checkpoint': checkpoint,
        'selected_source_exp_dir': source_dir,
        'selected_source_exp_name': os.path.basename(source_dir),
        'source_config': canonical(source_config),
        'source_config_artifact': source_artifact,
        'metrics_file': metrics_path,
        'selected_val_metrics': selected_metrics,
        'selected_metric_deltas': deltas,
        'test_result': canonical(os.path.join(target_path, 'test_7_6_locked_result.json')),
        'semantic_fusion_gamma_max': nested_get(resolved, 'model.semantic_fusion_gamma_max', 0.0),
        'method_signature': method_signature,
        'checkpoint_payload_audit': payload_audit,
        'data_audit': data_audit,
        'levir_cc_init_provenance': levir_cc_init_provenance,
        'stability_window': stability,
        'stability_member_artifacts': window_records,
        'artifacts': artifacts,
    }


def default_specs(exp_root):
    cc_sources = [
        'sgc_card_lm003_ls005_pd05_rw02_warmup',
    ] + [
        'levir_cc_decft_cw%d_s%d_%s' % (weight, steps, lr)
        for weight in (100, 101, 102)
        for steps in (10, 20)
        for lr in ('lr5e7', 'lr1e6')
    ]
    specs = [
        {
            'lock_id': 'levir_cc', 'dataset': 'levir_cc', 'seed': None,
            'target': 'levir_cc_7_6_val_pareto_locked', 'strategy': 'val_baseline_pareto',
            'allowed_sources': cc_sources,
        },
    ]
    for seed in MCI_SEEDS:
        source = 'levir_mci_masksemantic_repro_seed%d' % seed
        specs.append({
            'lock_id': 'levir_mci_seed%d' % seed,
            'dataset': 'levir_mci', 'seed': seed,
            'target': '%s_7_6_val_locked' % source,
            'strategy': 'val_baseline_pareto',
            'required_source_dir': os.path.join(exp_root, source),
            'allowed_sources': [source],
        })
    specs.append({
        'lock_id': 'second_cc', 'dataset': 'second_cc', 'seed': None,
        'target': 'second_cc_7_6_stable_locked',
        'strategy': 'val_baseline_stable_window',
        'allowed_sources': [
            'second_cc_crossattn_pd07_lsem0000',
            'second_cc_crossattn_pd08_lsem0000',
            'second_cc_crossattn_pd09_lsem0000',
            'second_cc_card_semantic_crossattn',
        ],
    })
    for spec in specs:
        spec['selection_json'] = os.path.join(exp_root, spec['target'], 'best_checkpoint.json')
        spec['allowed_source_dirs'] = [os.path.join(exp_root, name) for name in spec['allowed_sources']]
    return specs


def verify_artifact(record):
    actual = file_record(record['role'], record['path'])
    for key in ('path', 'size', 'sha256'):
        if actual[key] != record.get(key):
            raise ValueError('Artifact changed for %s (%s).' % (record['role'], key))


def verify_manifest(path):
    # A build immediately verifies its temporary manifest in the same process.
    # Never let the build-time directory cache stand in for verification-time I/O.
    INVENTORY_CACHE.clear()
    manifest = load_json(path)
    if manifest.get('status') != 'validation_locked' or manifest.get('selection_uses_test_metrics') is not False:
        raise ValueError('7.6 manifest is not validation locked.')
    locks = manifest.get('locks')
    if not isinstance(locks, list) or len(locks) != 5:
        raise ValueError('7.6 manifest must contain exactly five locks.')
    ids = [lock.get('lock_id') for lock in locks]
    if len(ids) != len(set(ids)):
        raise ValueError('7.6 lock IDs are not unique.')
    mci = [lock for lock in locks if lock.get('dataset') == 'levir_mci']
    dataset_counts = {dataset: sum(lock.get('dataset') == dataset for lock in locks)
                      for dataset in ('levir_cc', 'levir_mci', 'second_cc')}
    if dataset_counts != {'levir_cc': 1, 'levir_mci': 3, 'second_cc': 1}:
        raise ValueError('7.6 lock roles must be exactly 1 LEVIR-CC, 3 LEVIR-MCI, 1 SECOND-CC.')
    if sorted(lock.get('seed') for lock in mci) != list(MCI_SEEDS):
        raise ValueError('7.6 MCI seed set must be 1111/2222/3333 exactly once.')
    if len({canonical(lock['selected_source_exp_dir']) for lock in mci}) != 3:
        raise ValueError('7.6 MCI seed source experiments must be distinct.')
    signatures = [lock.get('method_signature') for lock in mci]
    if not signatures or any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError('7.6 MCI method configs differ beyond seed/experiment identity.')
    baselines = manifest.get('baseline_artifacts')
    if not isinstance(baselines, dict) or set(baselines) != {'levir_cc', 'levir_mci', 'second_cc'}:
        raise ValueError('7.6 manifest must bind exactly three CARD validation baselines.')
    card_manifest_record = manifest.get('card_baseline_manifest')
    if not isinstance(card_manifest_record, dict):
        raise ValueError('7.6 manifest is missing CARD baseline manifest binding.')
    verify_artifact(card_manifest_record)
    card_manifest = verify_card_baseline_manifest(card_manifest_record['path'])
    for dataset, baseline in baselines.items():
        verify_artifact(baseline['artifact'])
        finite_metric_map(baseline['metrics'], 'manifest baseline')
        expected_entry = card_manifest['datasets'][dataset]
        if canonical(baseline['path']) != canonical(expected_entry['selection_json']):
            raise ValueError('%s baseline selection binding changed.' % dataset)
        if canonical(baseline['selected_checkpoint']) != canonical(expected_entry['selected_checkpoint']):
            raise ValueError('%s baseline checkpoint binding changed.' % dataset)
    for lock in locks:
        if not lock.get('artifacts') or not lock.get('data_audit', {}).get('artifacts'):
            raise ValueError('%s has an empty artifact/data audit.' % lock.get('lock_id'))
        if not lock.get('checkpoint_payload_audit', {}).get('speaker_vocab_shape_audit'):
            raise ValueError('%s has no checkpoint vocabulary-shape audit.' % lock.get('lock_id'))
        for artifact in lock.get('artifacts', []) + lock.get('stability_member_artifacts', []):
            verify_artifact(artifact)
        directories = (
            lock.get('data_audit', {}).get('feature_directories', [])
            + lock.get('data_audit', {}).get('pseudo_mask_directories', [])
            + lock.get('data_audit', {}).get('semantic_map_directories', [])
        )
        for inventory in directories:
            current = directory_inventory(inventory['path'])
            for key in ('path', 'file_count', 'total_size', 'inventory_sha256'):
                if current[key] != inventory.get(key):
                    raise ValueError('Feature inventory changed for %s (%s).' % (inventory['path'], key))
        if not is_within(lock['test_result'], os.path.join(os.path.dirname(path), lock['target_exp'])):
            raise ValueError('Locked test result path escapes target experiment: %s' % lock['test_result'])
        selection = load_json(lock['selection_json'])
        if selection.get('selection_uses_test_metrics') is not False:
            raise ValueError('Selection changed to non-validation-only: %s' % lock['selection_json'])
        if canonical(selection.get('selected_checkpoint_path') or selection.get('selected_checkpoint') or '') != canonical(lock['selected_checkpoint']):
            raise ValueError('Selection checkpoint changed: %s' % lock['selection_json'])
        finite_metric_map(lock.get('selected_val_metrics'), '%s locked validation' % lock['lock_id'])
        finite_metric_map(lock.get('selected_metric_deltas'), '%s locked deltas' % lock['lock_id'])
        if lock['dataset'] == 'second_cc':
            if len(lock.get('stability_member_artifacts', [])) != 3:
                raise ValueError('SECOND lock must contain three stability member artifacts.')
        elif lock.get('stability_member_artifacts'):
            raise ValueError('%s unexpectedly has stability members.' % lock['lock_id'])

    exp_root = canonical(manifest.get('exp_root') or '')
    project_dir = canonical(manifest.get('project_dir') or '')
    if exp_root != canonical(os.path.dirname(path)):
        raise ValueError('Manifest exp_root does not match manifest location.')
    roots = manifest.get('dataset_roots')
    if not isinstance(roots, dict) or set(roots) != {'levir_cc', 'levir_mci', 'second_cc'}:
        raise ValueError('Manifest dataset_roots must contain exactly three datasets.')
    roots = {dataset: canonical(root) for dataset, root in roots.items()}
    fresh_baselines = {
        dataset: audit_baseline(
            baseline['path'],
            dataset,
            exp_root,
            card_manifest['datasets'][dataset],
        )
        for dataset, baseline in baselines.items()
    }
    for dataset in fresh_baselines:
        for field in ('path', 'metrics', 'artifact', 'selected_checkpoint', 'resolved_config'):
            if baselines[dataset].get(field) != fresh_baselines[dataset].get(field):
                raise ValueError('%s stored baseline audit differs from recomputed audit (%s).' % (
                    dataset, field))
    fresh_locks = [
        audit_selection(spec, fresh_baselines, roots, project_dir)
        for spec in default_specs(exp_root)
    ]
    stored_by_id = {lock['lock_id']: lock for lock in locks}
    fields_to_recompute = (
        'dataset', 'seed', 'target_exp', 'selection_json', 'selection_strategy',
        'selected_checkpoint', 'selected_source_exp_dir', 'selected_source_exp_name',
        'source_config', 'source_config_artifact', 'metrics_file',
        'selected_val_metrics', 'selected_metric_deltas', 'test_result',
        'semantic_fusion_gamma_max', 'method_signature', 'checkpoint_payload_audit',
        'data_audit', 'levir_cc_init_provenance', 'stability_window',
        'stability_member_artifacts', 'artifacts',
    )
    for fresh in fresh_locks:
        stored = stored_by_id.get(fresh['lock_id'])
        if stored is None:
            raise ValueError('Missing stored lock: %s' % fresh['lock_id'])
        for field in fields_to_recompute:
            if stored.get(field) != fresh.get(field):
                raise ValueError('%s stored audit differs from recomputed audit (%s).' % (
                    fresh['lock_id'], field))
    return manifest


def build_manifest(args):
    exp_root = canonical(args.exp_root)
    project_dir = canonical(args.project_dir)
    card_manifest_path = canonical(os.path.join(exp_root, 'card_baseline_locked_manifest.json'))
    card_manifest = verify_card_baseline_manifest(card_manifest_path)
    baseline_paths = {
        'levir_cc': args.levir_cc_baseline,
        'levir_mci': args.levir_mci_baseline,
        'second_cc': args.second_cc_baseline,
    }
    roots = {
        'levir_cc': canonical(args.levir_cc_root),
        'levir_mci': canonical(args.levir_mci_root),
        'second_cc': canonical(args.second_cc_root),
    }
    baselines = {
        dataset: audit_baseline(
            path,
            dataset,
            exp_root,
            card_manifest['datasets'][dataset],
        )
        for dataset, path in baseline_paths.items()
    }
    locks = [audit_selection(spec, baselines, roots, project_dir) for spec in default_specs(exp_root)]
    mci_signatures = [lock['method_signature'] for lock in locks if lock['dataset'] == 'levir_mci']
    if len(mci_signatures) != 3 or any(signature != mci_signatures[0] for signature in mci_signatures[1:]):
        raise ValueError('MCI seed configs are not method-equivalent.')
    manifest = {
        'status': 'validation_locked',
        'selection_uses_test_metrics': False,
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'project_dir': project_dir,
        'exp_root': exp_root,
        'dataset_roots': roots,
        'baseline_artifacts': baselines,
        'card_baseline_manifest': file_record('card_baseline_locked_manifest', card_manifest_path),
        'locks': locks,
    }
    output = canonical(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.7_6_manifest_', suffix='.json', dir=os.path.dirname(output))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        verify_manifest(temporary)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', default='./experiments')
    parser.add_argument('--project_dir', default='.')
    parser.add_argument('--output', default='./experiments/7_6_locked_manifest.json')
    parser.add_argument('--levir_cc_root', default='./Levir-CC')
    parser.add_argument('--levir_mci_root', default='./LEVIR-MCI-dataset')
    parser.add_argument('--second_cc_root', default='./SECOND-CC-AUG')
    parser.add_argument('--levir_cc_baseline', default='./experiments/card_levir_cc_baseline/baseline_best_checkpoint.json')
    parser.add_argument('--levir_mci_baseline', default='./experiments/levir_mci_card_baseline/baseline_best_checkpoint.json')
    parser.add_argument('--second_cc_baseline', default='./experiments/second_cc_card_rgb_baseline/baseline_best_checkpoint.json')
    parser.add_argument('--verify', default='')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verify:
        manifest = verify_manifest(args.verify)
        print('Verified 7.6 validation-locked manifest: %s (%d locks)' % (args.verify, len(manifest['locks'])))
    else:
        manifest = build_manifest(args)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
