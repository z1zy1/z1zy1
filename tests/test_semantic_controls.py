import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from models.CARD import CARD, SemanticCrossAttentionFusion
from models.transformer_decoder import DynamicSpeaker
from utils.semantic_controls import (ARMS, DATASETS, SEEDS, PROTOCOL, METRICS,
                                     configuration, build_control_models, parameter_summary)
from utils.semantic_control_audit import (historical_report, verify_initialization,
    statistics_report, verify_frozen_files, write, read, input_manifest, prediction_identity)
from utils.checkpoint_integrity import atomic_save_checkpoint, sha256_file
from utils.seed import capture_rng_state, restore_rng_state, seeded_initialization

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location('controls_runner', ROOT/'scripts/run_semantic_controls.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def config(arm, seed=1111, dataset='levir_cc'):
    return configuration(ROOT, 'experiments/testing_controls', dataset, seed, arm, {})


def small(cfg):
    cfg = copy.deepcopy(cfg)
    enc, dec = cfg.model.transformer_encoder, cfg.model.transformer_decoder
    enc.feat_dim = 16
    enc.att_dim = enc.emb_dim = dec.att_dim = 16
    enc.att_head = dec.att_head = 2
    enc.att_layer = dec.att_layer = 1
    dec.word_dim = 16
    dec.vocab_size, dec.seq_length = 13, 6
    cfg.model.semantic_fusion_heads = 2
    cfg.model.num_semantic_classes = 8 if cfg.data.use_semantic_maps else 7
    return cfg


@pytest.mark.parametrize('seed', SEEDS)
def test_explicit_shared_initialization_and_rng(seed):
    summaries = {}
    for arm in ARMS:
        state = capture_rng_state()
        cfg = config(arm, seed)
        cfg.model.num_semantic_classes = 7 if arm == 'card' else 8
        detector, speaker = build_control_models(cfg, CARD, DynamicSpeaker)
        following = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng_state(state)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        assert following[:2] == expected[:2]
        assert torch.equal(following[2], expected[2])
        summaries[arm] = dict(protocol_id=PROTOCOL, seed=seed,
                              change_detector=parameter_summary(detector), speaker=parameter_summary(speaker))
    verify_initialization(summaries)
    broken = copy.deepcopy(summaries)
    name = next(iter(broken['plain_fusion']['speaker']))
    broken['plain_fusion']['speaker'][name]['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='Decoder'):
        verify_initialization(broken)


def test_fixed_gate_empty_nonempty_gradient():
    fusion = SemanticCrossAttentionFusion(8, 8, num_heads=2, dropout=0,
        use_sparse_change_tokens=True, use_reliability_gate=True,
        use_global_semantic_token=True, gate_whole_adapter=True, fixed_nonempty_gate=True)
    x = torch.randn(2, 4, 8, requires_grad=True)
    semantic = torch.zeros(2, 2, 2, dtype=torch.long)
    semantic[1, 0, 0] = 1
    y = fusion(x, semantic_diff=semantic, detach_ratio=0.5)
    assert torch.equal(y[0], x[0])
    assert fusion.last_reliability_gate.tolist() == [[0.0], [1.0]]
    (y*torch.randn_like(y)).sum().backward()
    assert all(p.grad is None for p in fusion.reliability_gate.parameters())
    assert fusion.attention.in_proj_weight.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('arm', ARMS)
def test_real_model_forward_backward_save_load(tmp_path, arm):
    cfg = small(config(arm))
    detector, speaker = build_control_models(cfg, CARD, DynamicSpeaker)
    before, after = torch.randn(2, 16, 14, 14), torch.randn(2, 16, 14, 14)
    semantic = torch.randint(0, 7, (2, 28, 28)) if arm != 'card' else None
    output = detector(before, after, semantic_diff=semantic)
    seq = torch.randint(1, 12, (2, 5))
    loss, logits, _ = speaker._forward(output[0], seq, torch.ones_like(seq), labels_with_ignore=seq)
    assert torch.isfinite(loss)
    loss.backward()
    assert detector.img[0].weight.grad.abs().sum() > 0
    assert speaker.logit.weight.grad.abs().sum() > 0
    path = tmp_path/'model_checkpoint_1000.pt'
    atomic_save_checkpoint({'change_detector_state': detector.state_dict(), 'speaker_state': speaker.state_dict()},
                           str(path), immutable=True, write_checksum=True, expected_step=1000)
    restored_detector, restored_speaker = build_control_models(cfg, CARD, DynamicSpeaker)
    saved = torch.load(path, weights_only=True)
    restored_detector.load_state_dict(saved['change_detector_state'], strict=True)
    restored_speaker.load_state_dict(saved['speaker_state'], strict=True)
    detector.eval()
    restored_detector.eval()
    with torch.no_grad():
        assert torch.equal(detector(before, after, semantic_diff=semantic)[0],
                           restored_detector(before, after, semantic_diff=semantic)[0])


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix+key
        if isinstance(item, dict):
            result.update(flatten(item, name+'.'))
        else:
            result[name] = item
    return result


def test_single_factor_g_and_dense_d():
    c, g, d = (flatten(config(a)) for a in ('rsaca', 'fixed_gate', 'plain_fusion'))
    assert {k for k in c if c[k] != g[k]} == {'exp_name', 'exp_dir', 'model.semantic_fusion_fixed_nonempty_gate'}
    assert {k for k in c if c[k] != d[k]} == {'exp_name', 'exp_dir',
        'model.semantic_fusion_sparse_change_tokens', 'model.semantic_fusion_reliability_gate',
        'model.semantic_fusion_global_token', 'model.semantic_fusion_gate_whole_adapter'}
    for dataset in DATASETS:
        cfg = config('rsaca', dataset=dataset)
        c0 = config('rsaca')
        assert cfg.train == c0.train
        for section in ('transformer_encoder',):
            assert cfg.model[section] == c0.model[section]


def test_config_can_be_loaded_by_existing_cli(tmp_path):
    from configs.config_transformer import cfg as defaults, merge_cfg_from_file
    from utils.config_validation import validate_resolved_config
    original = copy.deepcopy(defaults)
    try:
        for d in DATASETS:
            for arm in ARMS:
                path = tmp_path/'request.json'
                write(path, config(arm, dataset=d))
                merge_cfg_from_file(str(path))
                validate_resolved_config(defaults)
                assert defaults.train.protocol_id == PROTOCOL
    finally:
        defaults.clear()
        defaults.update(original)


def test_r2_recompute_keeps_historical_acceptance():
    root = ROOT/'experiments/p1_rsaca_20260914/paired_matrix_r2'
    before = {str(p): sha256_file(p) for p in root.rglob('*') if p.is_file()}
    report = historical_report(root)
    assert report['statistics']['positive_mean_metrics'] == 15
    assert report['statistics']['positive_seed_metrics'] == 41
    assert report['statistics']['paired_seed_metrics'] == 45
    assert sum(r['scoring_completed_from_artifact'] for r in report['runs']) == 18
    assert not any(r['new_protocol_reuse_proven'] for r in report['runs'])
    assert before == {str(p): sha256_file(p) for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('stage', ('audit', 'preflight', 'train', 'select', 'freeze', 'test', 'summary'))
def test_dry_run_filters_no_writes_and_server_roots(tmp_path, stage):
    root = tmp_path/'absent'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', LEVIR_CC_ROOT='/server/custom/cc',
               LEVIR_CC_FEATURE_ROOT='/server/features/cc')
    result = subprocess.run([sys.executable, str(ROOT/'scripts/run_semantic_controls.py'),
        '--stage', stage, '--root', str(root), '--datasets', 'levir_cc', '--arms', 'plain_fusion', 'fixed_gate',
        '--seeds', '2222', '--dry-run'], env=env, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    assert len(payload['runs']) == 2
    assert not root.exists()
    cfg = payload['runs'][0]['configuration']
    assert Path(cfg['data']['default_feature_dir']).as_posix() == '/server/features/cc'
    assert Path(cfg['data']['semantic_diff_root']).as_posix() == '/server/custom/cc/pseudo_masks'


def grid(tmp_path, omit=None):
    rows = []
    for step in range(1000, 10001, 1000):
        if step == omit:
            continue
        path = tmp_path/'snapshots'/('run_checkpoint_%d.pt' % step)
        atomic_save_checkpoint({'global_step': step}, str(path), immutable=True,
                               write_checksum=True, expected_step=step)
        rows.append(dict(iter=step, snapshot_path=str(path), **{m: 1.0 for m in METRICS}))
    with (tmp_path/'val_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['iter', 'snapshot_path', *METRICS])
        writer.writeheader()
        writer.writerows(rows)


def test_selector_grid_tie_nan(tmp_path):
    grid(tmp_path)
    command = [sys.executable, str(ROOT/'scripts/select_best_snapshot_p1.py'), '--exp-dir', str(tmp_path), '--protocol-id', PROTOCOL]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    selected = read(tmp_path/'best_snapshot_p1.json')
    assert selected['best']['iter'] == 1000 and selected['protocol_id'] == PROTOCOL
    path = tmp_path/'val_metrics.csv'
    path.write_text(path.read_text().replace('1.0', 'nan', 1))
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_selector_itself_rejects_frozen_protocol(tmp_path):
    run = tmp_path/'card'/'card_levir_cc_seed1111'
    run.mkdir(parents=True)
    write(tmp_path/'frozen.json', {})
    result = subprocess.run([sys.executable, str(ROOT/'scripts/select_best_snapshot_p1.py'),
        '--exp-dir', str(run), '--protocol-id', PROTOCOL], capture_output=True, text=True)
    assert result.returncode != 0 and 'reselection is forbidden' in result.stderr


def test_freeze_requires_complete_audit_and_positive_means(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'verify_inputs', lambda *args: None)
    for audit_passed, goal in ((False, True), (True, False), (True, None)):
        monkeypatch.setattr(runner, 'run_audit', lambda *args: {
            'protocol_audit_passed': audit_passed, 'validation': {'mean_goal': goal}})
        with pytest.raises(ValueError, match='Test blocked'):
            runner.freeze(tmp_path, {})
        assert not (tmp_path/'frozen.json').exists()
    file = tmp_path/'checkpoint'
    file.write_text('original')
    monkeypatch.setattr(runner, 'run_audit', lambda *args: {
        'protocol_audit_passed': True, 'validation': {'mean_goal': True}, 'files': {str(file): sha256_file(file)}})
    write(tmp_path/'protocol.json', {})
    for d in DATASETS:
        write(tmp_path/('inputs_'+d+'.json'), {})
    runner.freeze(tmp_path, {'source': {}})
    runner.load_frozen(tmp_path, {'source': {}})
    with pytest.raises(ValueError, match='Frozen'):
        runner.assert_unfrozen(tmp_path)
    verify_frozen_files(read(tmp_path/'frozen.json'))
    file.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        verify_frozen_files(read(tmp_path/'frozen.json'))
    with pytest.raises((OSError, ValueError)):
        runner.load_frozen(tmp_path/'absent', {'source': {}})


def test_validation_random_stream_isolation():
    state = capture_rng_state()
    with seeded_initialization(42):
        random.random()
        np.random.rand(20)
        torch.rand(100)
    after = torch.rand(3)
    restore_rng_state(state)
    assert torch.equal(after, torch.rand(3))


def test_preflight_refuses_historical_directory_and_cpu(tmp_path, monkeypatch):
    historical = tmp_path/'historical'
    historical.mkdir()
    (historical/'old_result.json').write_text('{}')
    with pytest.raises(ValueError, match='Nonempty directory'):
        runner.preflight(historical, {})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    absent = tmp_path/'new_protocol'
    with pytest.raises(RuntimeError, match='requires CUDA'):
        runner.preflight(absent, {})
    assert not absent.exists()


def dataset_fixture(tmp_path, dataset='levir_cc'):
    import h5py
    from utils.semantic_controls import ENV_ROOTS
    cfg = configuration(ROOT, tmp_path/'runs', dataset, 1111, 'rsaca', {ENV_ROOTS[dataset]: str(tmp_path/'data')})
    root = tmp_path/'data'
    root.mkdir(parents=True)
    splits = {'train': [0], 'val': [1], 'test': [2], 'idx_to_filename': {}, 'idx_to_split': {}}
    refs = []
    for i, split in enumerate(('train', 'val', 'test')):
        name = split+'.png'
        splits['idx_to_filename'][str(i)] = name
        splits['idx_to_split'][str(i)] = split
        refs.append({'image_id': name, 'caption': 'a building changed', 'id': i})
        for phase in (cfg.data.default_phase, cfg.data.semantic_phase):
            path = Path(cfg.data.default_feature_dir)/split/phase/(name+'.npy')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.ones((16, 14, 14), dtype=np.float32))
        for phase in (('sem/A', 'sem/B') if dataset == 'second_cc' else (cfg.data.semantic_diff_phase,)):
            path = Path(cfg.data.semantic_map_root if dataset == 'second_cc' else cfg.data.semantic_diff_root)/split/phase/(split+'.npy')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.ones((28, 28), dtype=np.uint8))
    write(cfg.data.splits_json, splits)
    write(cfg.data.vocab_json, {'<NULL>': 0, '<START>': 1, '<END>': 2, 'building': 3})
    write(cfg.data.eval_anno_path, {'annotations': refs})
    write(cfg.data.caption_json, {'images': []})
    with h5py.File(cfg.data.h5_label_file, 'w') as handle:
        handle['labels'] = np.ones((3, 6), dtype=np.int32)
        handle['label_start_idx'] = np.arange(3)
        handle['label_end_idx'] = np.arange(3)
    return cfg


@pytest.mark.parametrize('dataset', DATASETS)
def test_input_manifest_same_interfaces_and_id_mismatch(tmp_path, dataset):
    cfg = dataset_fixture(tmp_path, dataset)
    manifest = input_manifest(cfg, 'unknown')
    assert not manifest['missing']
    assert len(manifest['samples']) == 3
    for arm in ('plain_fusion', 'fixed_gate'):
        other = copy.deepcopy(cfg)
        other.model = config(arm, dataset=dataset).model
        assert input_manifest(other, 'unknown') == manifest
    changed = copy.deepcopy(manifest)
    file = Path(manifest['samples'][0]['semantic'][0]['path'])
    file.write_bytes(file.read_bytes()+b'changed')
    assert input_manifest(cfg, 'unknown') != changed
    splits = read(cfg.data.splits_json)
    splits['idx_to_split']['0'] = 'val'
    write(cfg.data.splits_json, splits)
    with pytest.raises(ValueError, match='split/sample_id'):
        input_manifest(cfg)


def test_actual_runtime_config_accepts_only_data_adaptation(tmp_path):
    cfg = dataset_fixture(tmp_path)
    actual = copy.deepcopy(cfg)
    actual.model.num_semantic_classes = 8
    actual.model.transformer_decoder.vocab_size = 4
    actual.model.transformer_decoder.seq_length = 6
    runner.actual_config_check(cfg, actual)
    actual.train.optim.lr = 0.001
    with pytest.raises(ValueError, match='train.optim'):
        runner.actual_config_check(cfg, actual)


def test_prediction_ids_reject_duplicates_and_incomplete(tmp_path):
    prediction, annotation = tmp_path/'pred.json', tmp_path/'refs.json'
    write(annotation, {'annotations': [{'image_id': 'a'}, {'image_id': 'b'}]})
    write(prediction, [{'image_id': 'a', 'caption': 'one'}])
    with pytest.raises(ValueError, match='Prediction IDs'):
        prediction_identity(prediction, annotation, ['a', 'b'])
    write(prediction, [{'image_id': 'a'}, {'image_id': 'a'}])
    with pytest.raises(ValueError, match='Prediction IDs'):
        prediction_identity(prediction, annotation, ['a'])


def test_freeze_test_skips_existing_scoring_and_rejects_tamper(tmp_path, monkeypatch):
    from utils.experiment_tracking import stable_hash
    run = tmp_path/'run'
    run.mkdir()
    checkpoint = run/'checkpoint'
    checkpoint.write_text('fixed')
    frozen = {'files': {str(checkpoint): sha256_file(checkpoint)}}
    write(run/'best_snapshot_p1.json', {'best_snapshot': str(checkpoint), 'best_snapshot_sha256': sha256_file(checkpoint)})
    annotation = run/'refs.json'
    write(annotation, {'annotations': [{'image_id': 'a'}]})
    write(run/'request_config.json', {'data': {'eval_anno_path': str(annotation)}})
    write(tmp_path/'inputs_levir_cc.json', {'samples': [{'split': 'test', 'sample_id': 'a'}]})
    prediction = run/'test_output/captions/controls_locked/sc_results.json'
    write(prediction, [{'image_id': 'a', 'caption': 'a building'}])
    identity = prediction_identity(prediction, annotation, ['a'])
    identity['checkpoint_sha256'] = sha256_file(checkpoint)
    write(run/'test_prediction_identity.json', identity)
    write(run/'test_metrics.json', {'metrics': {m: 1.0 for m in METRICS}})
    write(run/'test_score_identity.json', {'score_sha256': sha256_file(run/'test_metrics.json'), 'identity': identity})
    monkeypatch.setattr(runner, 'execute', lambda *args: pytest.fail('Must not repeat inference or scoring'))
    runner.test_one(tmp_path, frozen, 'levir_cc', 1111, 'card', run)
    prediction.write_text('[]')
    with pytest.raises(ValueError):
        runner.test_one(tmp_path, frozen, 'levir_cc', 1111, 'card', run)


def test_fixed_sample_diagnostic_real_checkpoint(tmp_path):
    cfg = small(dataset_fixture(tmp_path))
    detector, speaker = build_control_models(cfg, CARD, DynamicSpeaker)
    checkpoint = tmp_path/'diagnostic.pt'
    torch.save({'change_detector_state': detector.state_dict(), 'speaker_state': speaker.state_dict()}, checkpoint)
    config_path, output = tmp_path/'config.json', tmp_path/'diagnostic.json'
    write(config_path, cfg)
    subprocess.run([sys.executable, str(ROOT/'scripts/diagnose_semantic_controls.py'),
        '--cfg', str(config_path), '--checkpoint', str(checkpoint), '--output', str(output),
        '--count', '1', '--device', 'cpu'], check=True, stdout=subprocess.DEVNULL)
    record = read(output)['records'][0]
    assert record['sample_id'] == 'val.png'
    assert {'gamma', 'gate', 'coverage', 'relative_residual'} <= set(record)


def test_complete_36_run_audit_freeze_and_mutation(tmp_path, monkeypatch):
    from utils.semantic_controls import ENV_ROOTS
    from utils.semantic_control_audit import matrix
    from utils.experiment_tracking import stable_hash
    root = tmp_path/'protocol'
    lock = {'source': {}, 'input_hashes': {}, 'configurations': {}}
    for dataset in DATASETS:
        cfg = dataset_fixture(tmp_path/dataset, dataset)
        monkeypatch.setenv(ENV_ROOTS[dataset], cfg.data.data_root)
        manifest = input_manifest(cfg)
        write(root/('inputs_'+dataset+'.json'), manifest)
        lock['input_hashes'][dataset] = stable_hash(manifest)
    for dataset, seed, arm, run in matrix(root):
        request = runner.expected(root, dataset, seed, arm)
        lock['configurations'][str(run.relative_to(root))] = request
        write(run/'request_config.json', request)
        write(run/'resolved_config.json', request)
        actual = copy.deepcopy(request)
        actual.model.transformer_decoder.vocab_size = 4
        actual.model.transformer_decoder.seq_length = 6
        actual.model.num_semantic_classes = 7 if arm == 'card' else 8
        write(run/'actual_model_config.json', actual)
        write(run/'controls_provenance.json', {'source': {}, 'input_hash': lock['input_hashes'][dataset]})
        params = {'public': torch.ones(1)}
        if arm != 'card':
            params['semantic'] = torch.ones(1)
        summaries = {n: {'shape': [1], 'dtype': 'torch.float32', 'sha256': 'test-initial-tensor'} for n in params}
        write(run/'initial_parameter_summary.json', dict(protocol_id=PROTOCOL, seed=seed,
              change_detector=summaries, speaker={'decoder': summaries['public']}))
        write(run/'run_summary.json', {'status': 'completed', 'final_global_step': 10000})
        (run/'command.txt').write_text('synthetic protocol fixture')
        (run/'environment.txt').write_text('synthetic CPU evidence; no training')
        prediction = run/'val_prediction.json'
        write(prediction, [{'image_id': 'val.png', 'caption': 'a building'}])
        identity = prediction_identity(prediction, request.data.eval_anno_path, ['val.png'])
        identity.update(prediction=str(prediction), reference=request.data.eval_anno_path)
        write(run/'validation_prediction_identity.json', {step: identity for step in range(1000, 10001, 1000)})
        rows = []
        for step in range(1000, 10001, 1000):
            path = run/'snapshots'/('run_checkpoint_%d.pt' % step)
            atomic_save_checkpoint({'global_step': step, 'model_cfg': actual,
                'change_detector_state': params, 'speaker_state': {'decoder': torch.ones(1)}},
                str(path), immutable=True, write_checksum=True, expected_step=step)
            # D and G are intentionally worse; only C0 is required to improve.
            score = 1.1 if arm == 'rsaca' else (1.0 if arm == 'card' else 0.9)
            rows.append(dict(iter=step, snapshot_path=str(path), **{m: score for m in METRICS}))
        with (run/'val_metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['iter', 'snapshot_path', *METRICS])
            writer.writeheader()
            writer.writerows(rows)
        subprocess.run([sys.executable, str(ROOT/'scripts/select_best_snapshot_p1.py'),
            '--exp-dir', str(run), '--protocol-id', PROTOCOL], check=True, stdout=subprocess.DEVNULL)
    write(root/'protocol.json', lock)
    runner.freeze(root, lock)
    frozen = read(root/'frozen.json')
    assert frozen['validation_statistics']['positive_mean_metrics'] == 15
    assert len(frozen['validation_statistics']['datasets']['levir_cc']['arms']) == 4
    verify_frozen_files(frozen)
    changed = next(root.glob('*/card*/actual_model_config.json'))
    changed.write_text('{}')
    with pytest.raises(ValueError, match='Frozen evidence changed'):
        verify_frozen_files(frozen)
