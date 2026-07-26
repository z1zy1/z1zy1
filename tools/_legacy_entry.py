"""Shared command construction for behavior-preserving tool entry points."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, Iterable, List

import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_config(path: str) -> Dict[str, object]:
    with open(path, encoding='utf-8') as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError('Config must contain a YAML mapping: %s' % path)
    return payload


def config_output_dir(payload: Dict[str, object]) -> str:
    exp_dir = str(payload.get('exp_dir', './experiments'))
    exp_name = str(payload.get('exp_name', '') or '')
    if not exp_name:
        raise ValueError('Config must define exp_name or the wrapper must receive --run-name.')
    return resolve_project_path(os.path.join(exp_dir, exp_name))


def resolve_project_path(path: str) -> str:
    """Resolve legacy relative output paths against the repository root."""
    path = os.path.normpath(path)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def require_new_output(path: str, allow_existing: bool) -> None:
    if os.path.exists(path) and not os.path.isdir(path):
        raise FileExistsError('Experiment output path exists and is not a directory: %s' % path)
    if allow_existing or not os.path.isdir(path):
        return
    if os.listdir(path):
        raise FileExistsError(
            'Refusing to reuse non-empty experiment directory %s. '
            'Use a new --run-name or explicitly pass --allow-existing.' % path
        )


def require_input_file(path: str, label: str, *, dry_run: bool = False) -> None:
    if dry_run:
        return
    if not os.path.isfile(path):
        raise FileNotFoundError('%s does not exist or is not a file: %s' % (label, path))


def require_new_file(path: str, label: str, allow_existing: bool) -> None:
    if path and os.path.exists(path) and not allow_existing:
        raise FileExistsError('Refusing to overwrite existing %s: %s' % (label, path))


def run_command(command: Iterable[str], dry_run: bool = False) -> int:
    command = [str(item) for item in command]
    print(json.dumps({'cwd': PROJECT_ROOT, 'command': command}, indent=2, ensure_ascii=False))
    if dry_run:
        return 0
    return subprocess.call(command, cwd=PROJECT_ROOT)


def legacy_train_command(config: str, extra: Iterable[str]) -> List[str]:
    return [sys.executable, os.path.join(PROJECT_ROOT, 'train_card_spot.py'), '--cfg', config] + list(extra)


def legacy_eval_command(
    config: str,
    checkpoint: str,
    split: str,
    extra: Iterable[str],
) -> List[str]:
    return [
        sys.executable,
        os.path.join(PROJECT_ROOT, 'test_card_spot.py'),
        '--cfg',
        config,
        '--checkpoint',
        checkpoint,
        '--split',
        split,
    ] + list(extra)
