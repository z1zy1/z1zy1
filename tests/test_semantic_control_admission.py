import copy
import json
from pathlib import Path

import pytest

from utils.attr_dict import AttrDict
from utils.checkpoint_integrity import atomic_save_checkpoint
from utils.experiment_tracking import cfg_to_plain
import utils.semantic_control_audit as audit
from utils.semantic_controls import PROTOCOL, configuration


def _node(value):
    if isinstance(value, dict):
        return AttrDict({key: _node(item) for key, item in value.items()})
    return value


def _fixture(tmp_path, monkeypatch):
    root = tmp_path / PROTOCOL
    run = root / 'rsaca' / 'rsaca_levir_cc_seed1111'
    snapshot = run / 'snapshots' / 'rsaca_levir_cc_seed1111_checkpoint_1000.pt'
    snapshot.parent.mkdir(parents=True)
    atomic_save_checkpoint({'global_step': 1000, 'model_state_dict': {}, 'speaker_state_dict': {}},
                           str(snapshot), immutable=True, write_checksum=True, expected_step=1000)
    cfg = configuration(Path(__file__).parents[1], root, 'levir_cc', 1111, 'rsaca', {})
    cfg.exp_dir = str(root / 'rsaca')
    cfg.exp_name = run.name
    cfg = _node(cfg)
    selection = {'protocol_id': PROTOCOL, 'selection_metric_split': 'validation',
                 'best_snapshot': str(snapshot), 'best_snapshot_sha256': audit.sha256_file(snapshot),
                 'best': {'iter': 1000}}
    run.mkdir(exist_ok=True)
    (run / 'best_snapshot_p1.json').write_text(json.dumps(selection))
    manifest = {'source_kind': 'unknown', 'samples': [], 'missing': []}
    (root / 'inputs_levir_cc.json').write_text(json.dumps(manifest))
    identity = {'commit': 'test', 'execution_files': {}, 'execution_sha256': 'test', 'dirty_execution': False}
    lock = {'protocol_id': PROTOCOL, 'source': identity,
            'configurations': {str(run.relative_to(root)): cfg_to_plain(cfg)}}
    (root / 'protocol.json').write_text(json.dumps(lock))
    frozen = {'protocol_id': PROTOCOL, 'source': identity,
              'validation_admitted': True,
              'files': {str(run / 'best_snapshot_p1.json'): 'unused'}}
    (root / 'frozen.json').write_text(json.dumps(frozen))
    frozen_hash = audit.sha256_file(root / 'frozen.json')
    (root / 'frozen_identity.json').write_text(json.dumps({'sha256': frozen_hash}))
    monkeypatch.setattr(audit, 'source_identity', lambda project: identity)
    monkeypatch.setattr(audit, 'verify_frozen_files', lambda frozen: None)
    monkeypatch.setattr(audit, 'input_manifest', lambda cfg, source: manifest)
    return cfg, snapshot, run / 'test_output' / 'captions' / 'controls_locked' / 'sc_results.json'


def test_admission_rejects_without_frozen_before_output(tmp_path):
    protocol_root = tmp_path / PROTOCOL
    cfg = _node(configuration(Path(__file__).parents[1], protocol_root, 'levir_cc', 1111, 'rsaca', {}))
    cfg.exp_dir = str(protocol_root / 'rsaca')
    cfg.exp_name = 'rsaca_levir_cc_seed1111'
    with pytest.raises(ValueError, match='requires protocol.json'):
        audit.enforce_test_admission(Path(__file__).parents[1], cfg, '/missing.pt',
                                     tmp_path / 'out' / 'sc_results.json')
    assert not (tmp_path / 'out').exists()


def test_admission_accepts_frozen_and_rejects_wrong_checkpoint(tmp_path, monkeypatch):
    cfg, snapshot, output = _fixture(tmp_path, monkeypatch)
    assert audit.enforce_test_admission(Path(__file__).parents[1], cfg, snapshot, output) == 'new'
    wrong = snapshot.with_name('other_checkpoint_1000.pt')
    atomic_save_checkpoint({'global_step': 1000, 'model_state_dict': {}, 'speaker_state_dict': {}},
                           str(wrong), immutable=True, write_checksum=True, expected_step=1000)
    with pytest.raises(ValueError, match='not the frozen selected checkpoint'):
        audit.enforce_test_admission(Path(__file__).parents[1], cfg, wrong, output)


def test_admission_rejects_config_change_and_source_change(tmp_path, monkeypatch):
    cfg, snapshot, output = _fixture(tmp_path, monkeypatch)
    cfg.train.seed = 2222
    with pytest.raises(ValueError, match='configuration'):
        audit.enforce_test_admission(Path(__file__).parents[1], cfg, snapshot, output)
