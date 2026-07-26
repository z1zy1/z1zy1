"""Experiment metadata, structured metrics, and lifecycle artifacts."""

from __future__ import annotations

import atexit
import csv
import json
import logging
import math
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime
from typing import Dict, Mapping, Optional

from utils.metrics import CAPTION_METRICS, normalize_metrics


STRUCTURED_FIELDS = [
    'timestamp', 'stage', 'epoch', 'global_step', 'step', 'learning_rate',
    'loss', 'caption_loss', 'mask_loss', 'semantic_loss', 'gpu_memory_mb',
    'duration_seconds',
] + list(CAPTION_METRICS) + [
    'Mask_Precision', 'Mask_Recall', 'Mask_F1', 'Mask_IoU', 'Mask_mIoU',
    'Semantic_F1', 'Semantic_IoU', 'Semantic_mIoU', 'extras_json',
]


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def current_gpu_memory_mb() -> Optional[float]:
    """Return allocated CUDA memory without requiring CUDA on local test hosts."""
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.memory_allocated()) / (1024.0 ** 2)


def create_stage_logger(run_dir: str, stage: str, additional_log_paths=()):
    """Create one console logger and stage log while retaining legacy log files."""
    stage = str(stage).lower()
    if stage not in ('train', 'val', 'test'):
        raise ValueError('Logger stage must be train, val, or test.')
    os.makedirs(run_dir, exist_ok=True)
    logger_name = 'card.%s.%s' % (stage, abs(hash(os.path.abspath(run_dir))))
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    paths = [os.path.join(run_dir, '%s.log' % stage)] + list(additional_log_paths)
    for path in dict.fromkeys(os.path.normpath(item) for item in paths):
        file_handler = logging.FileHandler(path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def to_json_safe(value):
    """Convert common experiment values to strict, portable JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    item = getattr(value, 'item', None)
    if callable(item):
        try:
            return to_json_safe(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _run_git(project_root: str, args) -> str:
    try:
        result = subprocess.run(
            ['git', '-C', project_root] + list(args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ''
    return result.stdout.strip()


def collect_git_info(project_root: str = '.') -> Dict[str, object]:
    root = os.path.abspath(project_root)
    status = _run_git(root, ['status', '--porcelain'])
    return {
        'root': root,
        'commit': _run_git(root, ['rev-parse', 'HEAD']),
        'branch': _run_git(root, ['branch', '--show-current']),
        'dirty': bool(status),
        'status': status.splitlines() if status else [],
    }


def collect_environment() -> Dict[str, object]:
    info: Dict[str, object] = {
        'timestamp': timestamp(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'cwd': os.getcwd(),
    }
    try:
        freeze = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        info['pip_freeze'] = freeze.stdout.splitlines()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        info['pip_freeze_error'] = str(exc)
    try:
        import torch
    except ModuleNotFoundError:
        info['torch_available'] = False
        return info
    info.update({
        'torch_available': True,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': getattr(torch.version, 'cuda', None),
        'cudnn_version': torch.backends.cudnn.version(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpus': [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
    })
    try:
        import torchvision
        info['torchvision_version'] = torchvision.__version__
    except (ModuleNotFoundError, RuntimeError):
        info['torchvision_version'] = None
    return info


def write_reproducibility_artifacts(
    run_dir: str,
    *,
    project_root: str = '.',
    command: Optional[str] = None,
) -> Dict[str, object]:
    """Write command, environment, and Git artifacts without probing datasets."""
    os.makedirs(run_dir, exist_ok=True)
    environment = collect_environment()
    git_info = collect_git_info(project_root)
    if command is None:
        command = subprocess.list2cmdline([sys.executable] + sys.argv)

    with open(os.path.join(run_dir, 'command.txt'), 'w', encoding='utf-8') as handle:
        handle.write(command.rstrip() + '\n')
    with open(os.path.join(run_dir, 'environment.txt'), 'w', encoding='utf-8') as handle:
        for key, value in environment.items():
            rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            handle.write('%s=%s\n' % (key, rendered))
    with open(os.path.join(run_dir, 'git_info.txt'), 'w', encoding='utf-8') as handle:
        for key, value in git_info.items():
            rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            handle.write('%s=%s\n' % (key, rendered))
    return {'environment': environment, 'git': git_info, 'command': command}


def update_run_summary(run_dir: str, **fields: object) -> Dict[str, object]:
    """Merge non-lifecycle fields into an existing run summary atomically."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, 'run_summary.json')
    try:
        with open(path, encoding='utf-8') as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        payload = {'status': 'initialized', 'start_time': None}
    if not isinstance(payload, dict):
        raise ValueError('run_summary.json must contain a JSON object: %s' % path)
    payload.update(to_json_safe(fields))
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)
    return payload


def install_error_hook(run_dir: str):
    """Install a traceback-preserving hook without changing run lifecycle."""
    os.makedirs(run_dir, exist_ok=True)
    error_path = os.path.join(run_dir, 'error.log')
    previous = sys.excepthook

    def _hook(exc_type, exc_value, exc_traceback):
        rendered = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        with open(error_path, 'a', encoding='utf-8') as handle:
            handle.write(rendered)
            if not rendered.endswith('\n'):
                handle.write('\n')
        previous(exc_type, exc_value, exc_traceback)

    sys.excepthook = _hook
    return previous


class StructuredMetricLogger:
    """Append canonical JSONL and stable-column CSV records."""

    def __init__(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(run_dir, 'metrics.jsonl')
        self.csv_path = os.path.join(run_dir, 'metrics.csv')

    def log(
        self,
        stage: str,
        values: Mapping[str, object],
        *,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        step: Optional[int] = None,
        duration_seconds: Optional[float] = None,
    ) -> Dict[str, object]:
        stage = str(stage).lower()
        if stage not in ('train', 'val', 'test'):
            raise ValueError('Structured metric stage must be train, val, or test.')
        values = to_json_safe(normalize_metrics(values))
        record: Dict[str, object] = {
            'timestamp': timestamp(),
            'stage': stage,
            'epoch': epoch,
            'global_step': global_step,
            'step': step,
            'learning_rate': values.get('learning_rate', values.get('lr')),
            'loss': values.get('loss', values.get('loss_total', values.get('total_loss'))),
            'caption_loss': values.get('caption_loss', values.get('loss_caption', values.get('loss_cap'))),
            'mask_loss': values.get('mask_loss', values.get('loss_mask')),
            'semantic_loss': values.get('semantic_loss', values.get('loss_semantic', values.get('loss_sem'))),
            'gpu_memory_mb': values.get('gpu_memory_mb', current_gpu_memory_mb()),
            'duration_seconds': duration_seconds,
        }
        for name in CAPTION_METRICS:
            if name in values:
                record[name] = values[name]
        auxiliary_aliases = {
            'Mask_Precision': ('Mask_Precision', 'mask_precision'),
            'Mask_Recall': ('Mask_Recall', 'mask_recall'),
            'Mask_F1': ('Mask_F1', 'mask_f1'),
            'Mask_IoU': ('Mask_IoU', 'mask_iou'),
            'Mask_mIoU': ('Mask_mIoU', 'mask_miou'),
            'Semantic_F1': ('Semantic_F1', 'semantic_f1'),
            'Semantic_IoU': ('Semantic_IoU', 'semantic_iou'),
            'Semantic_mIoU': ('Semantic_mIoU', 'semantic_miou'),
        }
        consumed = set(record) | set(CAPTION_METRICS) | {
            'lr',
            'loss_total',
            'total_loss',
            'loss_caption',
            'loss_cap',
            'cap_loss',
            'loss_mask',
            'loss_semantic',
            'loss_sem',
        }
        for output_name, aliases in auxiliary_aliases.items():
            for alias in aliases:
                if alias in values:
                    record[output_name] = values[alias]
                    consumed.update(aliases)
                    break
        extras = {key: value for key, value in values.items() if key not in consumed}
        record['extras'] = extras

        with open(self.jsonl_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        csv_row = {key: record.get(key, '') for key in STRUCTURED_FIELDS}
        csv_row['extras_json'] = json.dumps(
            extras, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        write_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=STRUCTURED_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(csv_row)
        return record


class RunStateManager:
    """Maintain run_summary.json and preserve failure tracebacks."""

    VALID_STATES = {'initialized', 'running', 'completed', 'failed', 'interrupted'}

    def __init__(
        self,
        run_dir: str,
        *,
        experiment_name: str,
        dataset: str,
        seed: int,
        run_id: Optional[str] = None,
    ) -> None:
        self.run_dir = os.path.abspath(run_dir)
        self.summary_path = os.path.join(self.run_dir, 'run_summary.json')
        self.error_path = os.path.join(self.run_dir, 'error.log')
        self._finished = False
        self._old_hook = None
        os.makedirs(self.run_dir, exist_ok=True)
        self.payload: Dict[str, object] = {
            'experiment_name': experiment_name,
            'dataset': dataset,
            'run_id': run_id or os.path.basename(self.run_dir),
            'seed': int(seed),
            'status': 'initialized',
            'start_time': timestamp(),
            'end_time': None,
            'best_epoch': None,
            'best_checkpoint': '',
            'selection_metric': '',
            'test_completed': False,
            'failure_reason': '',
        }
        self._write()

    def start(self) -> None:
        self.update(status='running')

    def update(self, *, status: Optional[str] = None, **fields: object) -> None:
        if status is not None:
            if status not in self.VALID_STATES:
                raise ValueError('Invalid experiment status: %s' % status)
            self.payload['status'] = status
        self.payload.update(to_json_safe(fields))
        if status in ('completed', 'failed', 'interrupted'):
            self.payload['end_time'] = timestamp()
            self._finished = True
        self._write()

    def complete(self, **fields: object) -> None:
        self.update(status='completed', **fields)

    def install_failure_hooks(self) -> None:
        if self._old_hook is not None:
            return
        self._old_hook = sys.excepthook

        def _hook(exc_type, exc_value, exc_traceback):
            rendered = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            with open(self.error_path, 'a', encoding='utf-8') as handle:
                handle.write(rendered)
                if not rendered.endswith('\n'):
                    handle.write('\n')
            state = 'interrupted' if issubclass(exc_type, KeyboardInterrupt) else 'failed'
            self.update(status=state, failure_reason=str(exc_value))
            self._old_hook(exc_type, exc_value, exc_traceback)

        sys.excepthook = _hook

        def _at_exit():
            if not self._finished and self.payload.get('status') == 'running':
                self.update(status='interrupted', failure_reason='process exited before completion')

        atexit.register(_at_exit)

    def _write(self) -> None:
        temporary = self.summary_path + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(self.payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, self.summary_path)
