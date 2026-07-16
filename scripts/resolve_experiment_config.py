#!/usr/bin/env python3
'''Resolve new and legacy JSON config artifacts from a source experiment.'''

import argparse
import json
import os


REQUIRED_SECTIONS = ('model', 'data', 'train')


def _load_config(path):
    try:
        with open(path, encoding='utf-8-sig') as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise ValueError('Invalid JSON config %s: %s' % (path, exc))
    if not isinstance(payload, dict):
        raise ValueError('Experiment config must be a JSON object: %s' % path)
    missing = [key for key in REQUIRED_SECTIONS if not isinstance(payload.get(key), dict)]
    if missing:
        raise ValueError('%s is missing object section(s): %s' % (path, ', '.join(missing)))
    return payload


def resolve_experiment_config(exp_dir):
    '''Prefer resolved_config.json; use post-merge cfg.json only when absent.'''
    exp_dir = os.path.abspath(os.path.normpath(exp_dir))
    for artifact in ('resolved_config.json', 'cfg.json'):
        path = os.path.join(exp_dir, artifact)
        if os.path.isfile(path):
            return os.path.normpath(path), _load_config(path), artifact
    raise FileNotFoundError(
        'Source experiment has neither resolved_config.json nor cfg.json: %s' % exp_dir
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True)
    args = parser.parse_args()
    path, _, artifact = resolve_experiment_config(args.exp_dir)
    print(json.dumps({'status': 'resolved', 'config_path': path,
                      'config_artifact': artifact}, indent=2))


if __name__ == '__main__':
    main()
