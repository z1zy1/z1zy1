"""Validation for the resolved CARD experiment configuration."""

from __future__ import annotations

import math


SEMANTIC_INPUT_MODES = {
    'none', 'aux', 'early_fusion', 'cross_attention', 'hard_gate', 'weak_coupled',
}
SEMANTIC_FUSION_NORM_MODES = {'legacy_post_norm', 'context_pre_norm'}


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
    if semantic_mode in ('early_fusion', 'cross_attention', 'hard_gate'):
        if int(getattr(cfg.data, 'num_semantic_classes', 0) or 0) <= 0:
            raise ValueError(
                'data.num_semantic_classes must be positive for semantic input mode %r.'
                % semantic_mode
            )
    if bool(getattr(cfg.data, 'semantic_diff_only', False)) and not str(
        getattr(cfg.data, 'semantic_diff_root', '') or ''
    ).strip():
        raise ValueError('data.semantic_diff_root is required when semantic_diff_only=True.')
    unknown_class = int(getattr(cfg.data, 'semantic_unknown_change_class', 0))
    semantic_classes = int(getattr(cfg.data, 'num_semantic_classes', 0) or 0)
    if bool(getattr(cfg.data, 'semantic_diff_binary', False)) and not (
        0 <= unknown_class < semantic_classes
    ):
        raise ValueError(
            'data.semantic_unknown_change_class must be in [0, num_semantic_classes) '
            'for binary semantic difference maps.'
        )
    _finite_number(
        getattr(cfg.model, 'semantic_fusion_gamma_init', 0.1),
        'model.semantic_fusion_gamma_init',
    )
    _finite_number(
        getattr(cfg.model, 'semantic_fusion_gamma_max', 0.0),
        'model.semantic_fusion_gamma_max',
        minimum=0.0,
    )
    fusion_norm_mode = str(
        getattr(cfg.model, 'semantic_fusion_norm_mode', 'legacy_post_norm')
    ).lower()
    if fusion_norm_mode not in SEMANTIC_FUSION_NORM_MODES:
        raise ValueError(
            'model.semantic_fusion_norm_mode=%r is invalid; expected one of %s.'
            % (fusion_norm_mode, sorted(SEMANTIC_FUSION_NORM_MODES))
        )
    _finite_number(
        getattr(cfg.model, 'semantic_fusion_reliability_gate_bias', -1.5),
        'model.semantic_fusion_reliability_gate_bias',
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
