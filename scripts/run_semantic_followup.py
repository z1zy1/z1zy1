#!/usr/bin/env python
"""Validation-only registered follow-up; dry-run is the default."""
import argparse
import json
from pathlib import Path
import sys

if '--execute' not in sys.argv:
    sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.semantic_followup import STAGES, register, execute_stage, runtime_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('environment', help='Print actual Python/package/CUDA identity for registration')
    seal = commands.add_parser('register', help='Validate a plan; --execute seals it without running experiments')
    seal.add_argument('--spec', required=True)
    seal.add_argument('--output', required=True)
    seal.add_argument('--execute', action='store_true')
    stage = commands.add_parser('stage', help='Inspect a stage; --execute performs its actual work')
    stage.add_argument('--registration', required=True)
    stage.add_argument('--stage', choices=STAGES, required=True)
    stage.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.command == 'environment':
        result = runtime_identity()
    elif args.command == 'register':
        result = register(args.spec, args.output, args.execute)
    else:
        result = execute_stage(args.registration, args.stage, args.execute)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result.get('status') == 'FAILED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
