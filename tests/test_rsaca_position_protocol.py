"""Small synthetic files exercise the real registration/selection/admission code."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts import run_rsaca_position as runner
from utils import rsaca_position_protocol as protocol
from utils.semantic_control_audit import read, write, prediction_identity, enforce_test_admission
from utils.semantic_controls import (POSITION_PROTOCOL, POSITION_ARMS, POSITION_DATASETS,
    P1_PROTOCOLS, CONTROL_PROTOCOLS, METRICS, SEEDS, REFERENCE, position_configuration)
from utils.checkpoint_integrity import atomic_save_checkpoint, sha256_file
from utils.experiment_tracking import cfg_to_plain, stable_hash

ROOT = Path(__file__).resolve().parents[1]


def configs(root, dataset='levir_mci', seed=1111, arm='E1'):
    return position_configuration(ROOT, root, dataset, seed, arm, {})


def test_registered_configs_and_single_factor(tmp_path):
    assert POSITION_PROTOCOL in P1_PROTOCOLS and POSITION_PROTOCOL in CONTROL_PROTOCOLS
    for dataset in POSITION_DATASETS:
        for seed in SEEDS:
            c0, c1, c2 = (configs(tmp_path, dataset, seed, a) for a in POSITION_ARMS)
            protocol.verify_pair_configuration(c1, c2)
            assert not c0.data.use_semantic_maps and c0.model.semantic_input_mode == 'none'
            assert c1.model.semantic_fusion_local_position == 'none'
            assert c2.model.semantic_fusion_local_position == 'sincos2d_kv'
            c2.train.semantic_detach_ratio = .4
            with pytest.raises(ValueError, match='beyond'):
                protocol.verify_pair_configuration(c1, c2)


def test_common_manifest_does_not_require_semantic_columns(tmp_path, monkeypatch):
    common = {'artifacts': {'reference': {'sha256': 'ref'}}, 'samples': [
        {'split': 'val', 'index': '1', 'sample_id': 'x.png', 'visual': [{'sha256': 'a'}, {'sha256': 'b'}]}], 'missing': []}
    registered = copy.deepcopy(common)
    registered['samples'][0]['semantic'] = [{'sha256': 'sem'}]
    cfg = configs(tmp_path, arm='E0')
    monkeypatch.setattr(protocol, 'input_manifest', lambda *args: common)
    assert protocol.verify_manifest(cfg, registered) == common
    cfg.data.use_semantic_maps = True
    with pytest.raises(ValueError, match='Semantic'):
        protocol.verify_manifest(cfg, registered)
    common['samples'][0]['visual'][0]['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='Common'):
        protocol.verify_manifest(configs(tmp_path, arm='E0'), registered)


@pytest.fixture
def complete(tmp_path, monkeypatch):
    root = tmp_path/POSITION_PROTOCOL
    source = {'commit': 'synthetic', 'dirty_execution': False}
    lock = {'protocol_id': POSITION_PROTOCOL, 'source': source, 'runtime': {'scorer': 'synthetic'},
            'reference': REFERENCE, 'selection_steps': list(range(1000, 10001, 1000)), 'selection_tie_break': 'earlier_step',
            'configurations': {}, 'input_hashes': {}}
    manifests = {}
    for d in POSITION_DATASETS:
        reference = tmp_path/(d+'_reference.json')
        write(reference, {'annotations': [{'image_id': 'val.png', 'caption': 'a'}, {'image_id': 'test.png', 'caption': 'a'}]})
        manifest = {'source_kind': 'unknown', 'artifacts': {'reference': {'path': str(reference), 'sha256': sha256_file(reference)}},
                    'samples': [{'split': split, 'index': str(i), 'sample_id': split+'.png',
                                 'visual': [{'sha256': 'visualA'}, {'sha256': 'visualB'}],
                                 'semantic': [{'sha256': 'semantic'}]} for i, split in enumerate(('val', 'test'))], 'missing': []}
        manifests[d] = manifest
        write(root/('inputs_'+d+'.json'), manifest)
        lock['input_hashes'][d] = stable_hash(manifest)
    for d, seed, arm, run in protocol.matrix(root):
        cfg = configs(root, d, seed, arm)
        cfg.data.eval_anno_path = manifests[d]['artifacts']['reference']['path']
        request = cfg_to_plain(cfg)
        lock['configurations'][str(run.relative_to(root))] = request
        for file in ('request_config.json', 'resolved_config.json', 'actual_model_config.json'):
            write(run/file, request)
        write(run/'controls_provenance.json', {'source': source, 'input_hash': lock['input_hashes'][d]})
        state = {'public': torch.ones(1)}
        if arm != 'E0':
            state['adapter'] = torch.ones(1)
        tensor_record = {'shape': [1], 'dtype': 'torch.float32', 'sha256': 'same-initial'}
        write(run/'initial_parameter_summary.json', {'protocol_id': POSITION_PROTOCOL, 'seed': seed,
            'change_detector': {k: tensor_record for k in state}, 'speaker': {'decoder': tensor_record}})
        write(run/'run_summary.json', {'status': 'completed', 'final_global_step': 10000})
        (run/'command.txt').write_text('synthetic; no training')
        (run/'environment.txt').write_text('CPU fixture')
        prediction = run/'val_prediction.json'
        write(prediction, [{'image_id': 'val.png', 'caption': 'a'}])
        identity = prediction_identity(prediction, cfg.data.eval_anno_path, ['val.png'])
        identity.update(prediction=str(prediction), reference=cfg.data.eval_anno_path)
        write(run/'validation_prediction_identity.json', {step: identity for step in range(1000, 10001, 1000)})
        rows = []
        for step in range(1000, 10001, 1000):
            path = run/'snapshots'/('run_checkpoint_%d.pt' % step)
            atomic_save_checkpoint({'global_step': step, 'config': request, 'model_state_dict': state,
                'speaker_state_dict': {'decoder': torch.ones(1), 'core.position_enc.pe': torch.zeros(cfg.model.transformer_decoder.seq_length, cfg.model.transformer_decoder.word_dim)}},
                str(path), immutable=True, write_checksum=True, expected_step=step)
            # One dataset regresses; the other improves. Preserve both and all seeds.
            score = (0.9 if d == 'levir_mci' else 1.1) if arm == 'E2' else 1.0
            rows.append(dict(iter=step, snapshot_path=str(path), **{m: score for m in METRICS}))
        with (run/'val_metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['iter', 'snapshot_path', *METRICS])
            writer.writeheader()
            writer.writerows(rows)
    write(root/'protocol.json', lock)
    # Run the actual selector over all candidate grids, with checksum/metadata checks.
    for d, seed, arm, run in protocol.matrix(root):
        subprocess.run([sys.executable, str(ROOT/'scripts/select_best_snapshot_p1.py'), '--exp-dir', str(run),
            '--protocol-id', POSITION_PROTOCOL], check=True, stdout=subprocess.DEVNULL)
        if arm != 'E0':
            selected = read(run/'best_snapshot_p1.json')
            write(run/'position_mechanism.json', {'checkpoint_sha256': selected['best_snapshot_sha256'],
                'finite': True, 'layout_passed': not (d == 'levir_mci' and seed == 1111), 'bypass_passed': True})
    monkeypatch.setattr(runner.base, 'actual_config_check', lambda request, actual: None)
    monkeypatch.setattr(runner, 'verify_inputs', lambda root, lock: None)
    monkeypatch.setattr(protocol, 'source_identity', lambda project: source)
    monkeypatch.setattr(protocol, 'runtime_identity', lambda: lock['runtime'])
    monkeypatch.setattr(protocol, 'input_manifest', lambda cfg, source: (
        dict(manifests[cfg.data.dataset], samples=[{k: v for k, v in r.items() if k != 'semantic'}
             for r in manifests[cfg.data.dataset]['samples']]) if not cfg.data.use_semantic_maps else manifests[cfg.data.dataset]))
    return root, lock


def frozen_run(complete, dataset='levir_mci', arm='E0', seed=1111):
    root, lock = complete
    run = root/arm/('%s_%s_seed%d' % (arm, dataset, seed))
    cfg = configs(root, dataset, seed, arm)
    cfg.data.eval_anno_path = lock['configurations'][str(run.relative_to(root))]['data']['eval_anno_path']
    selected = read(run/'best_snapshot_p1.json')
    return cfg, run, selected, run/'test_output/captions/controls_locked/sc_results.json'


def test_nine_per_dataset_freeze_regression_mechanism_failure_and_reentry(complete, monkeypatch):
    root, lock = complete
    runner.freeze(root, lock)
    frozen = read(root/'frozen.json')
    assert all(d['technical_complete'] and len(d['runs']) == 9 for d in frozen['datasets'].values())
    assert frozen['datasets']['levir_mci']['advancement']['advance'] is False
    assert frozen['datasets']['second_cc']['advancement']['advance'] is True
    assert len(frozen['datasets']['levir_mci']['runs']) == 9
    cfg, run, selected, output = frozen_run(complete)
    assert enforce_test_admission(ROOT, cfg, selected['best_snapshot'], output) == 'new'
    assert not output.parent.exists()
    with pytest.raises(ValueError, match='fixed frozen test path'):
        enforce_test_admission(ROOT, cfg, selected['best_snapshot'], run/'val_output/result.json')
    with pytest.raises(ValueError, match='configuration'):
        changed = copy.deepcopy(cfg)
        changed.model.semantic_fusion_local_position = 'sincos2d_kv'
        enforce_test_admission(ROOT, changed, selected['best_snapshot'], output)
    with pytest.raises(ValueError, match='not the frozen selected'):
        enforce_test_admission(ROOT, cfg, run/'snapshots/run_checkpoint_2000.pt', output)
    write(output, [{'image_id': 'test.png', 'caption': 'a'}])
    with pytest.raises(ValueError, match='lack identity'):
        enforce_test_admission(ROOT, cfg, selected['best_snapshot'], output)
    identity = prediction_identity(output, cfg.data.eval_anno_path, ['test.png'])
    identity.update(checkpoint_sha256=selected['best_snapshot_sha256'], scorer_identity=lock['runtime'])
    assert protocol.record_position_prediction(cfg, output, selected['best_snapshot']) == identity
    assert enforce_test_admission(ROOT, cfg, selected['best_snapshot'], output) == 'prediction_only'
    write(run/'test_metrics.json', {'metrics': {m: 1.0 for m in METRICS}})
    write(run/'test_score_identity.json', {'score_sha256': sha256_file(run/'test_metrics.json'), 'identity': identity})
    # Real inference/scoring update this mutable summary. It must not break the freeze.
    write(run/'run_summary.json', {'status': 'completed', 'final_global_step': 10000, 'test_completed': True})
    before = {str(p): sha256_file(p) for p in run.rglob('*') if p.is_file()}
    assert enforce_test_admission(ROOT, cfg, selected['best_snapshot'], output) == 'complete'
    assert enforce_test_admission(ROOT, cfg, selected['best_snapshot'], output) == 'complete'
    assert before == {str(p): sha256_file(p) for p in run.rglob('*') if p.is_file()}
    monkeypatch.setattr(runner.base, 'execute', lambda *a: pytest.fail('complete test executed again'))
    runner.base.test_one(root, frozen, 'levir_mci', 1111, 'E0', run)
    for claimed in (POSITION_PROTOCOL, 'p1_rsaca_20260913'):
        with pytest.raises(ValueError, match='frozen|identity'):
            protocol.guard_position_selection(run, claimed)
    changed = copy.deepcopy(cfg)
    changed.train.protocol_id = 'legacy'
    with pytest.raises(ValueError, match='identity'):
        enforce_test_admission(ROOT, changed, selected['best_snapshot'], output)
    with pytest.raises(ValueError, match='Validation cannot'):
        protocol.guard_output_route(cfg, 'val', output)
    with pytest.raises(ValueError, match='frozen'):
        protocol.guard_position_training(cfg)


@pytest.mark.parametrize('fault', ('seed', 'sha', 'config', 'validation_grid', 'init', 'nonfinite'))
def test_technical_failures_do_not_freeze_incomplete_dataset(complete, fault):
    root, lock = complete
    run = root/'E2/E2_levir_mci_seed3333'
    if fault == 'seed':
        (run/'initial_parameter_summary.json').unlink()
    elif fault == 'sha':
        selected = read(run/'best_snapshot_p1.json')
        Path(selected['best_snapshot']).write_bytes(b'corruption')
    elif fault == 'config':
        cfg = read(run/'actual_model_config.json')
        cfg['train']['semantic_detach_ratio'] = .4
        write(run/'actual_model_config.json', cfg)
    elif fault == 'validation_grid':
        ids = read(run/'validation_prediction_identity.json')
        ids.pop('10000')
        write(run/'validation_prediction_identity.json', ids)
    elif fault == 'init':
        params = read(run/'initial_parameter_summary.json')
        params['change_detector']['adapter']['sha256'] = 'wrong'
        write(run/'initial_parameter_summary.json', params)
    else:
        m = read(run/'position_mechanism.json')
        m['finite'] = False
        write(run/'position_mechanism.json', m)
    runner.freeze(root, lock)
    frozen = read(root/'frozen.json')
    assert not frozen['datasets']['levir_mci']['technical_complete']
    assert frozen['datasets']['second_cc']['technical_complete']
    cfg, run, selected, output = frozen_run(complete)
    with pytest.raises(ValueError, match='not technically frozen'):
        enforce_test_admission(ROOT, cfg, selected['best_snapshot'], output)
    assert not output.parent.exists()


def test_unfrozen_and_wrong_protocol_output_have_no_side_effects(tmp_path):
    root = tmp_path/POSITION_PROTOCOL
    cfg = configs(root)
    output = root/'E1/E1_levir_mci_seed1111/test_output/captions/controls_locked/sc_results.json'
    with pytest.raises(ValueError, match='technical freeze'):
        enforce_test_admission(ROOT, cfg, '/missing.pt', output)
    assert not root.exists()
    cfg.train.protocol_id = 'legacy'
    with pytest.raises(ValueError, match='identity'):
        protocol.guard_output_route(cfg, 'val', output)
    assert not root.exists()


@pytest.mark.parametrize('ids', (['test.png', 'test.png'], [], ['outside.png']))
def test_prediction_identity_rejects_duplicate_missing_or_foreign_ids(tmp_path, ids):
    reference = tmp_path/'reference.json'
    prediction = tmp_path/'prediction.json'
    write(reference, {'annotations': [{'image_id': 'test.png', 'caption': 'a'}]})
    write(prediction, [{'image_id': identity, 'caption': 'a'} for identity in ids])
    with pytest.raises(ValueError, match='Prediction IDs'):
        prediction_identity(prediction, reference, ['test.png'])


@pytest.mark.parametrize('seed', SEEDS)
def test_new_protocol_real_models_shared_initialization_and_rng(tmp_path, seed):
    from models.CARD import CARD
    from models.transformer_decoder import DynamicSpeaker
    from utils.semantic_controls import build_control_models, parameter_summary
    from utils.seed import capture_rng_state, restore_rng_state
    torch.set_num_threads(1)
    summaries = {}
    for arm in POSITION_ARMS:
        cfg = configs(tmp_path, seed=seed, arm=arm)
        enc, dec = cfg.model.transformer_encoder, cfg.model.transformer_decoder
        enc.feat_dim = enc.att_dim = enc.emb_dim = dec.att_dim = dec.word_dim = 16
        enc.att_head = dec.att_head = cfg.model.semantic_fusion_heads = 2
        enc.att_layer = dec.att_layer = 1
        dec.vocab_size, dec.seq_length = 13, 6
        cfg.model.num_semantic_classes = 7 if arm == 'E0' else 8
        rng = capture_rng_state()
        detector, speaker = build_control_models(cfg, CARD, DynamicSpeaker)
        following = torch.rand(3)
        restore_rng_state(rng)
        assert torch.equal(following, torch.rand(3))
        summaries[arm] = dict(protocol_id=POSITION_PROTOCOL, seed=seed,
            change_detector=parameter_summary(detector), speaker=parameter_summary(speaker))
    protocol.verify_initialization(summaries)


def test_training_aliases_preserve_registered_configuration(tmp_path):
    import ast
    from utils.experiment_tracking import sync_wcsg_config_aliases
    tree = ast.parse((ROOT/'train_card_spot.py').read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                 node.name in ('_positive_int', 'apply_train_step_aliases')]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'train_step_aliases', 'exec'), namespace)
    root = tmp_path/POSITION_PROTOCOL
    configurations = {}
    for d, seed, arm, run in protocol.matrix(root):
        cfg = configs(root, d, seed, arm)
        configurations[str(run.relative_to(root))] = cfg_to_plain(cfg)
    write(root/'protocol.json', {'protocol_id': POSITION_PROTOCOL, 'configurations': configurations})
    for d, seed, arm, run in protocol.matrix(root):
        cfg = configs(root, d, seed, arm)
        sync_wcsg_config_aliases(cfg)
        namespace['apply_train_step_aliases'](cfg)
        protocol.guard_position_training(cfg)
