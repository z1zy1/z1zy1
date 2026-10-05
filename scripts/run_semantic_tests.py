#!/usr/bin/env python
"""Run the portable CPU follow-up regression suite using the active environment."""
import os
from pathlib import Path
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    os.chdir(project)
    try:
        import pytest
    except ImportError as exc:
        raise SystemExit('pytest is missing from the active Python environment; see docs/semantic_followup.md') from exc
    return pytest.main(['-q', '-p', 'no:cacheprovider',
        'tests/test_semantic_followup.py', 'tests/test_semantic_controls.py',
        'tests/test_semantic_control_admission.py', 'tests/test_semantic_diagnostics.py',
        'tests/test_p1_checkpoint_integrity.py'] + sys.argv[1:])


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
