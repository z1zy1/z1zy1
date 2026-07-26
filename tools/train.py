"""Unified, backward-compatible training launcher."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools._legacy_entry import (
    config_output_dir,
    legacy_train_command,
    load_config,
    require_new_output,
    resolve_project_path,
    run_command,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Launch the legacy CARD trainer with safer run tracking.')
    parser.add_argument('--config', '--cfg', dest='config', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--output-root', default=None)
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--allow-existing', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_known_args()


def main() -> int:
    args, extra = parse_args()
    config = os.path.abspath(args.config)
    payload = load_config(config)
    if args.output_dir:
        output_dir = resolve_project_path(args.output_dir)
    elif args.run_name:
        output_root = args.output_root or str(payload.get('exp_dir', './experiments'))
        output_dir = resolve_project_path(os.path.join(output_root, args.run_name))
    else:
        output_dir = config_output_dir(payload)
    require_new_output(output_dir, args.allow_existing)
    forwarded = ['--output_dir', output_dir]
    if args.seed is not None:
        forwarded.extend(['--seed', str(args.seed)])
    forwarded.extend(extra)
    return run_command(legacy_train_command(config, forwarded), dry_run=args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
