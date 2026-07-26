"""Validation for the resolved CARD experiment configuration."""

from __future__ import annotations

import math


SEMANTIC_INPUT_MODES = {
    'none', 'aux', 'early_fusion', 'cross_attention', 'hard_gate', 'weak_coupled',
}


def _finite_number(value, path, *, minimum=None, maximum=None):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError('%s must be numeric, got %r.' % (path, value))
    if not math.isfinite(numeric):
        raise ValueError('%s must be finite, got %r.' % (path, value))
    if minimum is not None and numeric < minimum:
        raise ValueError('%s must be >= %s, got %s.' % (path, minimum, numeric))
    if maximum is not None and numeric > maximum:
        raise ValueError('%s must be <= %s, got %s.' % (path, maximum, numeric))
    return numeric


def validate_resolved_config(cfg, *, phase='train'):
    """Fail early with clear messages while preserving all legacy defaults."""
    if not str(getattr(cfg, 'exp_name', '') or '').strip():
        raise ValueError('exp_name must be a non-empty string.')
    if not str(getattr(cfg.data, 'dataset', '') or '').strip():
        raise ValueError('data.dataset must be a non-empty string.')
    semantic_mode = str(getattr(cfg.model, 'semantic_input_mode', 'none')).lower()
    if semantic_mode not in SEMANTIC_INPUT_MODES:
        raise ValueError(
            'model.semantic_input_mode=%r is invalid; expected one of %s.'
            % (semantic_mode, sorted(SEMANTIC_INPUT_MODES))
        )
    _finite_number(
        getattr(cfg.train, 'semantic_detach_ratio', 0.0),
        'train.semantic_detach_ratio',
        minimum=0.0,
        maximum=1.0,
    )
    for name in ('lambda_mask', 'lambda_semantic'):
        _finite_number(getattr(cfg.train, name, 0.0), 'train.%s' % name, minimum=0.0)
    for name in ('aux_warmup_start_ratio', 'aux_warmup_end_ratio'):
        _finite_number(getattr(cfg.train, name, 0.0), 'train.%s' % name, minimum=0.0, maximum=1.0)
    start = float(getattr(cfg.train, 'aux_warmup_start_ratio', 0.0))
    end = float(getattr(cfg.train, 'aux_warmup_end_ratio', 0.0))
    if start > end:
        raise ValueError('train.aux_warmup_start_ratio may not exceed aux_warmup_end_ratio.')
    if phase == 'train':
        if int(getattr(cfg.train, 'max_iter', 0)) <= 0:
            raise ValueError('train.max_iter must be positive.')
        if int(getattr(cfg.train, 'snapshot_interval', 0)) <= 0:
            raise ValueError('train.snapshot_interval must be positive.')
    if bool(getattr(cfg.train, 'use_semantic_detach', False)) and bool(
        getattr(cfg.train, 'use_semantic_partial_detach', False)
    ):
        raise ValueError('Full semantic detach and partial detach cannot both be enabled.')
    return cfg
