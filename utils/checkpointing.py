"""Explicit compatibility helpers for CARD checkpoint payloads.

Legacy scripts remain the serialization authority. These helpers expose
canonical fields without changing existing checkpoint bytes or defaults.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple


MODEL_STATE_KEYS = ('model_state_dict', 'change_detector_state', 'change_detector')
SPEAKER_STATE_KEYS = ('speaker_state_dict', 'speaker_state', 'speaker')
OPTIMIZER_STATE_KEYS = ('optimizer_state_dict', 'optimizer_state', 'optimizer')
SCHEDULER_STATE_KEYS = ('scheduler_state_dict', 'scheduler_state', 'scheduler')


def _first(payload: Mapping[str, object], keys: Tuple[str, ...]) -> object:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def normalize_checkpoint_payload(payload: Mapping[str, object]) -> Dict[str, object]:
    """Expose canonical fields while preserving every original key."""
    if not isinstance(payload, Mapping):
        raise TypeError('checkpoint payload must be a mapping, got %s.' % type(payload).__name__)
    normalized = dict(payload)
    normalized.setdefault('model_state_dict', _first(payload, MODEL_STATE_KEYS))
    normalized.setdefault('speaker_state_dict', _first(payload, SPEAKER_STATE_KEYS))
    normalized.setdefault('optimizer_state_dict', _first(payload, OPTIMIZER_STATE_KEYS))
    normalized.setdefault('scheduler_state_dict', _first(payload, SCHEDULER_STATE_KEYS))
    normalized.setdefault('scaler_state_dict', payload.get('scaler_state'))
    normalized.setdefault('global_step', payload.get('iter', payload.get('step', 0)))
    normalized.setdefault('epoch', payload.get('epoch', 0))
    normalized.setdefault('best_metrics', payload.get('best_metrics', {}))
    normalized.setdefault('config', payload.get('config', payload.get('model_cfg')))
    return normalized


def split_model_states(payload: Mapping[str, object]) -> Tuple[object, object]:
    """Return CARD and speaker states with an explicit compatibility error."""
    normalized = normalize_checkpoint_payload(payload)
    model_state = normalized.get('model_state_dict')
    speaker_state = normalized.get('speaker_state_dict')
    if model_state is None or speaker_state is None:
        raise KeyError(
            'checkpoint must contain CARD and speaker weights; accepted keys are %s and %s.'
            % (MODEL_STATE_KEYS, SPEAKER_STATE_KEYS)
        )
    return model_state, speaker_state


def strip_module_prefix(state_dict: Mapping[str, object]) -> Dict[str, object]:
    """Remove a DataParallel module prefix without hiding key mismatch."""
    return {
        (str(key)[7:] if str(key).startswith('module.') else str(key)): value
        for key, value in state_dict.items()
    }


def load_checkpoint_file(path: str, map_location: Optional[str] = 'cpu') -> Dict[str, object]:
    """Load and normalize a checkpoint; PyTorch is imported lazily."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError('Loading a checkpoint requires PyTorch.') from exc
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    return normalize_checkpoint_payload(payload)
