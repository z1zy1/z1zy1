"""Run a one-sample legacy train/validation/checkpoint smoke workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools._legacy_entry import PROJECT_ROOT, legacy_eval_command, legacy_train_command


def _print_command(label, command):
    print(json.dumps({'stage': label, 'cwd': PROJECT_ROOT, 'command': command}, indent=2, ensure_ascii=False))


def _check_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError('Invalid JSONL at %s:%d: %s' % (path, line_number, exc))


def main() -> int:
    parser = argparse.ArgumentParser(description='Run a one-batch CARD smoke workflow.')
    parser.add_argument('--config', '--cfg', dest='config', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--config-only', action='store_true')
    args, extra = parser.parse_known_args()

    config = os.path.abspath(args.config)
    if not os.path.exists(config):
        parser.error('Config does not exist: %s' % config)
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(tempfile.gettempdir(), 'card_smoke_run')
    )
    run_name = os.path.basename(os.path.normpath(output_dir))
    smoke_opts = [
        'data.train.batch_size', '1',
        'data.train.max_samples', '1',
        'data.val.batch_size', '1',
        'data.val.max_samples', '1',
        'data.test.batch_size', '1',
        'data.test.max_samples', '1',
        'data.num_workers', '0',
        'train.log_interval', '1',
    ]
    train_extra = [
        '--output_dir', output_dir,
        '--max_iter', '1',
        '--snapshot_interval', '1',
    ] + list(extra) + smoke_opts
    train_command = legacy_train_command(config, train_extra)
    train_command[0] = args.python
    checkpoint = os.path.join(output_dir, 'snapshots', '%s_checkpoint_1.pt' % run_name)
    eval_command = legacy_eval_command(
        config,
        checkpoint,
        'val',
        list(extra) + smoke_opts,
    )
    eval_command[0] = args.python
    _print_command('train_one_batch_and_validate', train_command)
    _print_command('reload_checkpoint_and_validate_entry', eval_command)
    if args.dry_run or args.config_only:
        return 0

    torch_check = subprocess.call([args.python, '-c', 'import torch'], cwd=PROJECT_ROOT)
    if torch_check != 0:
        raise RuntimeError(
            'PyTorch is unavailable in %s. Run this smoke test in the remote CARD environment.'
            % args.python
        )
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError('Smoke output directory must be absent or empty: %s' % output_dir)
    result = subprocess.call(train_command, cwd=PROJECT_ROOT)
    if result != 0:
        return result
    if not os.path.exists(checkpoint):
        raise FileNotFoundError('One-step training did not create the expected checkpoint: %s' % checkpoint)
    result = subprocess.call(eval_command, cwd=PROJECT_ROOT)
    if result != 0:
        return result

    required = (
        'config_original.yaml',
        'config_resolved.yaml',
        'command.txt',
        'environment.txt',
        'git_info.txt',
        'model_summary.txt',
        'run_summary.json',
        'metrics.jsonl',
        'metrics.csv',
    )
    missing = [name for name in required if not os.path.exists(os.path.join(output_dir, name))]
    if missing:
        raise RuntimeError('Smoke workflow is missing required artifacts: %s' % ', '.join(missing))
    _check_jsonl(os.path.join(output_dir, 'metrics.jsonl'))
    with open(os.path.join(output_dir, 'run_summary.json'), encoding='utf-8') as handle:
        summary = json.load(handle)
    print(json.dumps({
        'status': 'passed',
        'output_dir': output_dir,
        'checkpoint': checkpoint,
        'run_summary_status': summary.get('status'),
        'verified_artifacts': list(required),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
