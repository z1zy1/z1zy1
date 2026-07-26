"""Unified locked-test launcher backed by the legacy inference path."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools._legacy_entry import (
    legacy_eval_command,
    require_input_file,
    require_new_file,
    run_command,
)


def main() -> int:
    parser = argparse.ArgumentParser(description='Run test inference for one preselected checkpoint.')
    parser.add_argument('--config', '--cfg', dest='config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--result-json', default=None)
    parser.add_argument('--annotation', default=None)
    parser.add_argument('--metrics-json', default=None)
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--allow-existing', action='store_true')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    if args.annotation and not args.result_json:
        parser.error('--annotation requires --result-json so the scorer can locate predictions.')
    config = os.path.abspath(args.config)
    checkpoint = os.path.abspath(args.checkpoint)
    require_input_file(config, 'Config')
    require_input_file(checkpoint, 'Checkpoint', dry_run=args.dry_run)
    if args.annotation:
        require_input_file(
            os.path.abspath(args.annotation),
            'Annotation',
            dry_run=args.dry_run,
        )
    require_new_file(
        os.path.abspath(args.result_json) if args.result_json else '',
        'test prediction file',
        args.allow_existing,
    )
    forwarded = list(extra)
    if args.result_json:
        forwarded.extend(['--result_json', args.result_json])
    if args.gpu is not None:
        forwarded.extend(['--gpu', str(args.gpu)])
    command = legacy_eval_command(
        config,
        checkpoint,
        'test',
        forwarded,
    )
    result = run_command(command, dry_run=args.dry_run)
    if result != 0 or not args.annotation:
        return result
    run_dir = args.run_dir or os.path.dirname(os.path.abspath(args.result_json))
    score_command = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'score_predictions.py'),
        '--annotation', os.path.abspath(args.annotation),
        '--predictions', os.path.abspath(args.result_json),
        '--stage', 'test',
        '--run-dir', os.path.abspath(run_dir),
        '--checkpoint', checkpoint,
    ]
    if args.metrics_json:
        score_command.extend(['--output-json', os.path.abspath(args.metrics_json)])
    if args.allow_existing:
        score_command.append('--allow-existing')
    return run_command(score_command, dry_run=args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
