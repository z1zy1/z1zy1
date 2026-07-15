import csv
import hashlib
import json
import os
import sys
import time

import yaml


KEY_SWITCHES = [
    'dataset_name',
    'exp_name',
    'use_aux_mask',
    'use_aux_semantic',
    'use_semantic_cross_attention',
    'use_partial_detach',
    'semantic_detach_ratio',
    'semantic_fusion_gamma_max',
    'use_feature_reweight',
    'use_semantic_hard_gate',
    'lambda_mask',
    'lambda_semantic',
    'aux_warmup_start_ratio',
    'aux_warmup_end_ratio',
    'selection_strategy',
    'init_checkpoint',
    'learning_rate',
    'finetune_steps',
    'max_iter',
    'save_interval',
    'eval_interval',
    'snapshot_interval',
    'use_content_word_weight',
    'content_word_weight',
    'normalize_content_word_weights',
    'lambda_mask_warmup',
    'lambda_semantic_warmup',
    'checkpoint_path',
]


def cfg_to_plain(value):
    if isinstance(value, dict):
        return {str(key): cfg_to_plain(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [cfg_to_plain(item) for item in value]
    return value


def _getattr_bool(obj, name, default=False):
    return bool(getattr(obj, name, default))


def sync_wcsg_config_aliases(cfg):
    """Keep legacy CARD flags and WCSG-CARD flag names in lockstep."""
    model_aux_mask = _getattr_bool(cfg.model, 'enable_aux_mask', False) or _getattr_bool(cfg.model, 'use_aux_mask', False)
    train_aux_mask = _getattr_bool(cfg.train, 'use_aux_mask', False)
    aux_mask = model_aux_mask or train_aux_mask
    cfg.model.enable_aux_mask = aux_mask
    cfg.model.use_aux_mask = aux_mask
    cfg.train.use_aux_mask = aux_mask

    aux_semantic = _getattr_bool(cfg.train, 'use_semantic_aux', False) or _getattr_bool(cfg.train, 'use_aux_semantic', False)
    cfg.train.use_semantic_aux = aux_semantic
    cfg.train.use_aux_semantic = aux_semantic

    partial_detach = _getattr_bool(cfg.train, 'use_semantic_partial_detach', False) or _getattr_bool(cfg.train, 'use_partial_detach', False)
    cfg.train.use_semantic_partial_detach = partial_detach
    cfg.train.use_partial_detach = partial_detach

    semantic_mode = str(getattr(cfg.model, 'semantic_input_mode', 'none')).lower()
    use_cross = _getattr_bool(cfg.train, 'use_semantic_cross_attention', False) or semantic_mode == 'cross_attention'
    use_hard_gate = _getattr_bool(cfg.train, 'use_semantic_hard_gate', False) or semantic_mode == 'hard_gate'
    if use_hard_gate:
        cfg.model.semantic_input_mode = 'hard_gate'
    elif use_cross:
        cfg.model.semantic_input_mode = 'cross_attention'
    cfg.train.use_semantic_cross_attention = use_cross
    cfg.train.use_semantic_hard_gate = use_hard_gate

    return cfg


def comparable_config_dict(cfg):
    plain = cfg_to_plain(cfg)
    plain.pop('exp_name', None)
    plain.pop('exp_dir', None)
    if 'logger' in plain and isinstance(plain['logger'], dict):
        # Display windows do not affect the model/loss/eval path.
        plain['logger'].pop('display_id', None)
    return plain


def stable_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def key_switch_summary(cfg, checkpoint_path=''):
    semantic_mode = str(getattr(cfg.model, 'semantic_input_mode', 'none')).lower()
    train_optim = getattr(cfg.train, 'optim', None)
    return {
        'use_aux_mask': bool(getattr(cfg.model, 'enable_aux_mask', False)),
        'use_aux_semantic': bool(getattr(cfg.train, 'use_semantic_aux', False)),
        'use_semantic_cross_attention': bool(getattr(cfg.train, 'use_semantic_cross_attention', False)) or semantic_mode == 'cross_attention',
        'use_semantic_hard_gate': bool(getattr(cfg.train, 'use_semantic_hard_gate', False)) or semantic_mode == 'hard_gate',
        'use_feature_reweight': bool(getattr(cfg.train, 'use_feature_reweight', False)),
        'use_partial_detach': bool(getattr(cfg.train, 'use_semantic_partial_detach', False)),
        'semantic_detach_ratio': float(getattr(cfg.train, 'semantic_detach_ratio', 0.0)),
        'semantic_fusion_gamma_max': float(getattr(cfg.model, 'semantic_fusion_gamma_max', 0.0)),
        'lambda_mask': float(getattr(cfg.train, 'lambda_mask', 0.0)),
        'lambda_semantic': float(getattr(cfg.train, 'lambda_semantic', 0.0)),
        'aux_warmup_start_ratio': float(getattr(cfg.train, 'aux_warmup_start_ratio', 0.0)),
        'aux_warmup_end_ratio': float(getattr(cfg.train, 'aux_warmup_end_ratio', 0.0)),
        'selection_strategy': str(getattr(cfg.train, 'selection_strategy', 'spice_constrained_balanced')),
        'init_checkpoint': str(getattr(cfg.train, 'init_checkpoint', '') or ''),
        'learning_rate': float(getattr(train_optim, 'lr', 0.0)),
        'finetune_steps': int(getattr(cfg.train, 'finetune_steps', 0) or 0),
        'max_iter': int(getattr(cfg.train, 'max_iter', 0) or 0),
        'save_interval': int(getattr(cfg.train, 'save_interval', 0) or 0),
        'eval_interval': int(getattr(cfg.train, 'eval_interval', 0) or 0),
        'snapshot_interval': int(getattr(cfg.train, 'snapshot_interval', 0) or 0),
        'use_content_word_weight': bool(
            getattr(cfg.train, 'use_content_word_weight', False)
            or getattr(cfg.train, 'use_content_word_weighted_ce', False)
        ),
        'content_word_weight': float(getattr(cfg.train, 'content_word_weight', 1.0)),
        'normalize_content_word_weights': bool(getattr(cfg.train, 'normalize_content_word_weights', False)),
        'lambda_mask_warmup': bool(getattr(cfg.train, 'use_mask_warmup', False) or getattr(cfg.train, 'use_aux_warmup', False)),
        'lambda_semantic_warmup': bool(getattr(cfg.train, 'use_semantic_warmup', False) or getattr(cfg.train, 'use_aux_warmup', False)),
        'dataset_name': str(getattr(cfg.data, 'dataset', '')),
        'exp_name': str(getattr(cfg, 'exp_name', '')),
        'checkpoint_path': checkpoint_path or '',
    }


def _append_lines(path, lines):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


def warn_identical_comparable_config(output_dir, cfg, comparable_hash, log_path=None):
    exp_dir = os.path.normpath(getattr(cfg, 'exp_dir', os.path.dirname(output_dir) or '.'))
    exp_name = str(getattr(cfg, 'exp_name', os.path.basename(output_dir)))
    index_path = os.path.join(exp_dir, '.resolved_config_hashes.json')
    try:
        with open(index_path, encoding='utf-8') as f:
            index = json.load(f)
    except Exception:
        index = {}

    records = index.get(comparable_hash, [])
    other_names = sorted({item.get('exp_name') for item in records if item.get('exp_name') and item.get('exp_name') != exp_name})
    warning = None
    if other_names:
        warning = 'WARNING: different exp_name but identical resolved config: %s and %s' % (exp_name, ', '.join(other_names))
        print(warning)
        if log_path:
            _append_lines(log_path, [warning])

    records.append({
        'exp_name': exp_name,
        'output_dir': os.path.normpath(output_dir),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    index[comparable_hash] = records
    try:
        os.makedirs(exp_dir, exist_ok=True)
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print('Warning: could not update config hash index %s: %s' % (index_path, exc))
    return warning


def save_resolved_config(output_dir, cfg, args=None, checkpoint_path='', phase='train', log_path=None):
    os.makedirs(output_dir, exist_ok=True)
    plain = cfg_to_plain(cfg)
    full_hash = stable_hash(plain)
    comparable_hash = stable_hash(comparable_config_dict(cfg))

    with open(os.path.join(output_dir, 'resolved_config.json'), 'w', encoding='utf-8') as f:
        json.dump(plain, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, 'resolved_config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(plain, f, allow_unicode=True, sort_keys=True)
    with open(os.path.join(output_dir, 'config_hash.txt'), 'w', encoding='utf-8') as f:
        f.write('full_config_hash=%s\n' % full_hash)
        f.write('comparable_config_hash=%s\n' % comparable_hash)

    args_path = os.path.join(output_dir, 'args.txt')
    with open(args_path, 'w', encoding='utf-8') as f:
        f.write('phase=%s\n' % phase)
        f.write('argv=%s\n' % ' '.join(sys.argv))
        if args is not None:
            try:
                args_dict = vars(args)
            except TypeError:
                args_dict = {}
            for key in sorted(args_dict):
                f.write('%s=%s\n' % (key, args_dict[key]))

    lines = ['Resolved WCSG-CARD key switches:']
    summary = key_switch_summary(cfg, checkpoint_path=checkpoint_path)
    for key in KEY_SWITCHES:
        lines.append('  %s: %s' % (key, summary.get(key)))
    lines.append('USING HARD GATE: %s' % str(summary['use_semantic_hard_gate']).lower())
    lines.append('USING FEATURE REWEIGHT: %s' % str(summary['use_feature_reweight']).lower())
    lines.append('full_config_hash: %s' % full_hash)
    lines.append('comparable_config_hash: %s' % comparable_hash)
    for line in lines:
        print(line)
    if log_path:
        _append_lines(log_path, lines)
    warning = warn_identical_comparable_config(output_dir, cfg, comparable_hash, log_path=log_path)
    return {
        'full_config_hash': full_hash,
        'comparable_config_hash': comparable_hash,
        'key_switches': summary,
        'warning': warning,
    }


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    clean = cfg_to_plain(record)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + '\n')


def write_single_row_csv(path, row, fieldnames=None, append=False):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted(row.keys())
    write_header = (not append) or (not os.path.exists(path)) or os.path.getsize(path) == 0
    mode = 'a' if append else 'w'
    with open(path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in fieldnames})
