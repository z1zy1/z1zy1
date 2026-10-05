"""Synthetic CPU checks; no research model is trained or scored here."""
import copy
import collections
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from models.CARD import SemanticCrossAttentionFusion, CARD
from models.transformer_decoder import DynamicSpeaker
from utils import semantic_followup as followup
from utils.semantic_controls import FOLLOWUP_PROTOCOL, PROTOCOL, ARMS, DATASETS, SEEDS, build_control_models
from utils.semantic_control_audit import enforce_test_admission
from utils.checkpoint_integrity import atomic_save_checkpoint, sha256_file
from utils.config_validation import validate_resolved_config
from utils.experiment_tracking import cfg_to_plain, stable_hash

ROOT = Path(__file__).resolve().parents[1]


def fusion(**kwargs):
    return SemanticCrossAttentionFusion(8, 8, num_heads=2, dropout=0,
        use_sparse_change_tokens=True, use_reliability_gate=True,
        use_global_semantic_token=True, gate_whole_adapter=True, **kwargs)


@pytest.mark.parametrize('gate', [0.0, 0.25, 1.0])
def test_constant_gate_formula_empty_and_gradients(gate):
    torch.manual_seed(5)
    one, actual = fusion(fixed_nonempty_gate=True), fusion(fixed_nonempty_gate=True, fixed_gate_value=gate)
    actual.load_state_dict(one.state_dict(), strict=True)
    x = torch.randn(2, 4, 8, requires_grad=True)
    sem = torch.zeros(2, 2, 2, dtype=torch.long)
    sem[1, 0, 0] = 1
    y_one, y = one(x, semantic_diff=sem), actual(x, semantic_diff=sem)
    assert torch.equal(y[0], x[0])
    assert torch.allclose(y, x + gate * (y_one - x))
    assert actual.last_reliability_gate.tolist() == [[0.0], [gate]]
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is None for p in actual.reliability_gate.parameters())
    if gate:
        assert actual.attention.in_proj_weight.grad.abs().sum() > 0
    assert set(actual.state_dict()) == set(one.state_dict())


def test_dense_access_keeps_coverage_empty_invalid_and_local_values():
    sparse, dense = fusion(), fusion(dense_local_access=True)
    dense.load_state_dict(sparse.state_dict(), strict=True)
    query = torch.randn(1, 4, 8)
    sem = torch.tensor([[[1, 0], [-1, 0]]])
    sparse(query, semantic_diff=sem)
    dense(query, semantic_diff=sem)
    assert torch.equal(sparse.last_change_coverage, dense.last_change_coverage)
    assert sparse.last_attention[..., 1].eq(0).all()
    assert dense.last_attention[..., 1].gt(0).all()
    assert dense.last_attention[..., 2].eq(0).all()  # ignore remains inaccessible
    assert sparse.last_attention.shape == dense.last_attention.shape  # same global/fallback
    features = torch.randn(1, 4, 8, requires_grad=True)
    mask = torch.tensor([[True, False, False, False]])
    valid = torch.tensor([[True, True, False, True]])
    context = dense._sparse_semantic_context(query, features, mask, local_valid_mask=valid)[0]
    context.sum().backward()
    assert features.grad[0, 1].abs().sum() > 0  # unchanged local V was not multiplied by zero
    for module in (sparse, dense):
        assert torch.equal(module(query, semantic_diff=torch.zeros_like(sem)), query)
        assert module.last_reliability_gate.eq(0).all()


@pytest.mark.parametrize('value', [float('nan'), -0.1, 1.1])
def test_invalid_gate_rejected(value):
    with pytest.raises(ValueError):
        fusion(fixed_nonempty_gate=True, fixed_gate_value=value)


def test_matched_candidate_config_and_test_block(tmp_path):
    candidate = {'name': 'dense', 'model': {'semantic_fusion_dense_local_access': True}}
    configs = {a: followup.candidate_configuration(ROOT, tmp_path, 'levir_cc', 1111, a, {}, candidate, .25) for a in ARMS}
    c, g = copy.deepcopy(configs['rsaca'].model), copy.deepcopy(configs['fixed_gate'].model)
    g.semantic_fusion_fixed_nonempty_gate = False
    g.semantic_fusion_fixed_gate_value = 1.0
    assert c == g
    assert not configs['plain_fusion'].model.semantic_fusion_dense_local_access
    for cfg in configs.values():
        validate_resolved_config(cfg, phase='train')
        with pytest.raises(ValueError, match='validation-only'):
            validate_resolved_config(cfg, phase='test')
        with pytest.raises(ValueError, match='validation-only'):
            enforce_test_admission(ROOT, cfg, 'missing.pt', str(tmp_path/'test.json'))
    assert not (tmp_path/'test.json').exists()


def test_followup_small_models_share_initialization(tmp_path):
    models = {}
    for arm in ARMS:
        cfg = followup.candidate_configuration(ROOT, tmp_path, 'levir_cc', 1111, arm, {},
            {'name': 'dense', 'model': {'semantic_fusion_dense_local_access': True}}, .25)
        enc, dec = cfg.model.transformer_encoder, cfg.model.transformer_decoder
        enc.feat_dim = enc.att_dim = enc.emb_dim = dec.att_dim = dec.word_dim = 16
        enc.att_head = dec.att_head = cfg.model.semantic_fusion_heads = 2
        enc.att_layer = dec.att_layer = 1
        dec.vocab_size, dec.seq_length = 13, 6
        models[arm] = build_control_models(cfg, CARD, DynamicSpeaker)
    for arm in ARMS:
        assert all(torch.equal(v, models[arm][0].state_dict()[k]) for k, v in models['card'][0].state_dict().items())
        assert all(torch.equal(v, models[arm][1].state_dict()[k]) for k, v in models['card'][1].state_dict().items())
    assert all(torch.equal(v, models['fixed_gate'][0].state_dict()[k]) for k, v in models['rsaca'][0].state_dict().items())


@pytest.fixture
def registered(tmp_path, monkeypatch):
    """Actual tiny files/checkpoint, mocked dataset resolver and runtime only."""
    monkeypatch.setattr(followup, 'runtime_identity', lambda: {'environment': 'synthetic_cpu'})
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    artifacts = {}
    def put(name, payload):
        path = inputs/(name + '.json')
        path.write_text(json.dumps(payload), encoding='utf-8')
        artifacts[name] = {'path': str(path), 'sha256': sha256_file(str(path))}
        return path
    reference = put('reference', {'annotations': [{'image_id': 'a', 'id': 1, 'caption': 'a building'}]})
    put('prediction', [{'image_id': 'a', 'caption': 'a building'}])
    put('ids', ['a'])
    put('metrics', {m: 1 for m in followup.METRICS})
    put('resource', {'synthetic': True})
    put('before', [1]); put('after', [2])
    put('images', {'a': {'before': 'before', 'after': 'after'}})
    cfg = {'model': {'semantic_input_mode': 'none'}, 'train': {'protocol_id': PROTOCOL},
           'data': {'dataset': 'levir_cc', 'use_semantic_maps': True, 'semantic_diff_only': True,
                    'semantic_diff_binary': True, 'eval_anno_path': str(reference)},
           'exp_dir': str(inputs), 'exp_name': 'historical'}
    put('config', cfg)
    manifest = {'samples': [{'sample_id': 'a', 'split': 'val', 'semantic': []}], 'missing': []}
    put('manifest', manifest)
    monkeypatch.setattr(followup, 'input_manifest', lambda cfg, source: copy.deepcopy(manifest))
    summary = {'seed': 1111, 'change_detector': {'x': {'sha256': 'abc'}}, 'speaker': {'x': {'sha256': 'def'}}}
    put('initial', summary)
    checkpoint = inputs/'model_checkpoint_1000.pt'
    atomic_save_checkpoint({'global_step': 1000, 'model_cfg': cfg}, str(checkpoint),
                          write_checksum=True, expected_step=1000)
    artifacts['checkpoint'] = {'path': str(checkpoint), 'sha256': sha256_file(str(checkpoint))}
    stages = {s: {'environment': 'cpu'} for s in followup.STAGES}
    stages['P0'].update(required_artifacts=['initial', 'checkpoint'], initialization_summaries=['initial'],
                        checkpoints=[{'artifact': 'checkpoint', 'step': 1000}])
    job = dict(prediction='prediction', reference='reference', expected_ids='ids',
               original_metrics='metrics', scorer_resources=['resource'], dataset='levir_cc')
    stages['P1a'].update(metric_tolerance=0.0, jobs=[job])
    stages['P1b'].update(max_changed_fraction=0.0, jobs=[dict(job, config='config', checkpoint='checkpoint')])
    stages['P1d'].update(sample_count=1, sample_seed=42, expected_ids='ids', reference='reference',
                         image_manifest='images', predictions={'CARD': 'prediction', 'RSACA': 'prediction'})
    for stage in ('P1c', 'P2', 'P3'):
        stages[stage].update(budget_steps=10000, selection_rule='five_metric_equal_weight_log',
            datasets=['levir_cc'], seeds=[1111], candidates=[{'name': 'original', 'model': {}}], constant_gates=[0.25, 1.0])
    stages['P1c']['initialization_references'] = {'levir_cc/1111/card': 'initial', 'levir_cc/1111/rsaca': 'initial'}
    spec = {'protocol_id': FOLLOWUP_PROTOCOL, 'validation_only': True, 'output_root': str(tmp_path/'runs'),
            'environments': {'cpu': {'environment': 'synthetic_cpu'}}, 'artifacts': artifacts, 'stages': stages,
            'datasets': {'levir_cc': {'manifest': 'manifest', 'config': 'config', 'source_kind': 'annotation'}},
            'data_environment': {}}
    spec_path = tmp_path/'spec.json'
    spec_path.write_text(json.dumps(spec), encoding='utf-8')
    registration = tmp_path/'runs/registration.json'
    followup.register(spec_path, registration, execute=True)
    return registration, spec, tmp_path


def synthetic_score(monkeypatch, difference=False):
    """Stand-in scorer to exercise receipt execution/outcome, not real COCO."""
    from utils import semantic_diagnostics
    def score(prediction, reference, **kw):
        assert kw['scorer_command'][0] == sys.executable
        assert '--stage' in kw['scorer_command'] and 'val' in kw['scorer_command']
        out = kw['output_dir']; out.mkdir()
        (out/'metrics.json').write_text(json.dumps({m: 1 for m in followup.METRICS}), encoding='utf-8')
        return {'status': 'different' if difference else 'match', 'executed_fixture': True}
    monkeypatch.setattr(semantic_diagnostics, 'reproduce_scores', score)


def synthetic_decode(monkeypatch, spec, different=False):
    def run(command, output):
        assert command[1].endswith('test_card_spot.py')
        assert command[command.index('--split')+1] == 'val'
        cfg = yaml.safe_load(Path(command[command.index('--cfg')+1]).read_text(encoding='utf-8'))
        assert Path(spec['output_root']) in Path(cfg['exp_dir']).parents
        assert cfg['exp_name'] == 'isolated_validation'
        path = Path(command[command.index('--result_json')+1])
        assert Path(cfg['exp_dir']) in path.parent.parent.parent.parents
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps([{'image_id': 'a', 'caption': 'different' if different else 'a building'}]), encoding='utf-8')
        output.mkdir(); (output/'fixture.log').write_text('synthetic decoder', encoding='utf-8')
    monkeypatch.setattr(followup, '_run', run)


def test_stage_lifecycle_receipts_conditional_training_and_no_test(registered, monkeypatch):
    registration, spec, _ = registered
    root = Path(spec['output_root'])
    synthetic_score(monkeypatch); synthetic_decode(monkeypatch, spec)
    for stage in ('P0', 'P1a', 'P1b'):
        result = followup.execute_stage(registration, stage, execute=True)
        assert result['status'] == 'PASS', result['result']
        followup.receipt(followup.load_registration(registration), stage)
    result = followup.execute_stage(registration, 'P1c', execute=True)
    assert result['status'] == 'NOT_REQUIRED'
    assert not list((root/'P1c').glob('train*'))
    deps = followup.dependencies(followup.load_registration(registration), 'P2')
    assert 'P1c' not in deps
    assert not list(root.rglob('*test*'))
    with pytest.raises(ValueError, match='already exists'):
        followup.execute_stage(registration, 'P1a', execute=True)


def test_content_can_run_before_p0_and_has_no_machine_labels(registered):
    registration, spec, _ = registered
    record = followup.execute_stage(registration, 'P1d', execute=True)
    assert record['status'] == 'READY_FOR_ANNOTATION', record['result']
    root = Path(spec['output_root'])
    assert not (root/'P0').exists()
    packet = json.loads((root/'P1d/blind_packet.json').read_text(encoding='utf-8'))
    assert packet[0]['labels'] is None
    assert set(packet[0]['captions']) == {'system_0', 'system_1'}


def test_difference_is_evidence_but_does_not_release_development(registered, monkeypatch):
    registration, spec, _ = registered
    synthetic_score(monkeypatch, difference=True); synthetic_decode(monkeypatch, spec)
    for stage in ('P0', 'P1a', 'P1b'):
        followup.execute_stage(registration, stage, execute=True)
    lock = followup.load_registration(registration)
    assert followup.receipt(lock, 'P1a', ('DIFFERENT',))['status'] == 'DIFFERENT'
    followup.dependencies(lock, 'P1c')
    with pytest.raises(ValueError, match='did not pass'):
        followup.dependencies(lock, 'P2')


def test_scoring_subset_cannot_be_registered_as_full_validation(registered, monkeypatch):
    _, spec, tmp_path = registered
    manifest_path = Path(spec['artifacts']['manifest']['path'])
    manifest = followup.read(manifest_path)
    manifest['samples'].append({'sample_id': 'b', 'split': 'val', 'semantic': []})
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    spec['artifacts']['manifest']['sha256'] = sha256_file(str(manifest_path))
    from utils import semantic_diagnostics
    monkeypatch.setattr(semantic_diagnostics, 'reproduce_scores', lambda *a, **k: pytest.fail('subset reached scorer'))
    with pytest.raises(ValueError, match='cover the registered validation split'):
        followup._p1a(spec, spec['stages']['P1a'], tmp_path/'never_created')
    assert not (tmp_path/'never_created').exists()


@pytest.mark.parametrize('status', ['scorer_error', 'parse_error', 'blocked', 'unverifiable',
    'new_environment_recomputed', 'not_run', 'dry_run', 'unknown', None])
def test_unsuccessful_scoring_is_failed_not_difference(status):
    assert followup._receipt_status('P1a', {'jobs': [{'status': status}]}) == 'FAILED'
    assert followup._receipt_status('P1a', {'jobs': [{'status': 'match'}, {'status': status}]}) == 'FAILED'
    assert followup._receipt_status('P1a', {'jobs': []}) == 'FAILED'


def test_scorer_failure_blocks_receipt_training_and_cli(registered, monkeypatch):
    registration, spec, _ = registered
    assert followup.execute_stage(registration, 'P0', execute=True)['status'] == 'PASS'
    from utils import semantic_diagnostics
    monkeypatch.setattr(semantic_diagnostics, 'reproduce_scores', lambda *a, **k: {'status': 'scorer_error'})
    record = followup.execute_stage(registration, 'P1a', execute=True)
    assert record['status'] == 'FAILED'
    lock = followup.load_registration(registration)
    with pytest.raises(ValueError, match='did not pass'):
        followup.receipt(lock, 'P1a', ('PASS', 'DIFFERENT'))
    with pytest.raises(ValueError, match='did not pass'):
        followup.dependencies(lock, 'P1c')
    from scripts import run_semantic_followup as cli
    monkeypatch.setattr(cli, 'execute_stage', lambda *a: record)
    monkeypatch.setattr(sys, 'argv', ['run_semantic_followup.py', 'stage', '--registration',
        str(registration), '--stage', 'P1a', '--execute'])
    assert cli.main() == 1


def test_receipt_tamper_and_input_changes_rejected(registered):
    registration, spec, _ = registered
    followup.execute_stage(registration, 'P0', execute=True)
    lock = followup.load_registration(registration)
    result_path = Path(spec['output_root'])/'P0/result.json'
    result_path.write_text('{"verified": true}', encoding='utf-8')
    with pytest.raises(ValueError, match='output changed'):
        followup.receipt(lock, 'P0')
    Path(spec['artifacts']['initial']['path']).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='changed registered artifact'):
        followup.artifact(spec, 'initial')


def test_dryrun_no_heavy_import_or_output(registered, monkeypatch):
    registration, spec, _ = registered
    monkeypatch.setattr(followup, 'runtime_identity', lambda: pytest.fail('dryrun imported runtime'))
    for stage in followup.STAGES:
        assert followup.execute_stage(registration, stage)['status'] == 'NOT_RUN'
        assert not (Path(spec['output_root'])/stage).exists()
    process = subprocess.run([sys.executable, str(ROOT/'scripts/run_semantic_followup.py'),
        'stage', '--registration', str(registration), '--stage', 'P1b'], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)['status'] == 'NOT_RUN'


def test_preview_reports_missing_parameters_and_renders_commands_without_imports(registered, monkeypatch):
    registration, spec, _ = registered
    assert followup.execute_stage(registration, 'P0', execute=True)['status'] == 'PASS'
    lock = followup.load_registration(registration)
    import builtins
    old_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'numpy', 'datasets'):
            pytest.fail('Preview imported heavy model/data dependency: ' + name)
        return old_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    ready = followup.preview_stage(lock, 'P1a')
    assert ready['readiness'] == 'READY_FOR_EXECUTION_CHECKS'
    assert ready['prerequisites'] == {'P0': 'PASS'}
    assert ready['commands'][0][1].endswith('score_predictions.py')
    missing = copy.deepcopy(lock)
    missing['spec']['stages']['P1a']['metric_tolerance'] = None
    blocked = followup.preview_stage(missing, 'P1a')
    assert blocked['readiness'] == 'BLOCKED'
    assert any('metric_tolerance' in item for item in blocked['issues'])
    infer = followup.preview_stage(lock, 'P1b')
    assert infer['commands'][0][infer['commands'][0].index('--split')+1] == 'val'
    train = followup.preview_stage(lock, 'P2')
    assert train['readiness'] == 'BLOCKED'  # missing score/decode receipts
    assert len(train['commands']) == 10  # 5 configs and their selection commands
    assert all(not (Path(spec['output_root'])/stage).exists() for stage in ('P1a', 'P1b', 'P2'))


def test_runtime_mismatch_records_failure_and_user_pass_not_enough(registered, monkeypatch):
    registration, spec, _ = registered
    monkeypatch.setattr(followup, 'runtime_identity', lambda: {'different': True})
    record = followup.execute_stage(registration, 'P0', execute=True)
    assert record['status'] == 'FAILED'
    record['status'] = 'PASS'
    followup.write(Path(spec['output_root'])/'P0/receipt.json', record)
    with pytest.raises(ValueError, match='Runtime differs'):
        followup.receipt(followup.load_registration(registration), 'P0')


def test_matrix_and_registration_initial_tensor_guard(registered):
    registration, spec, _ = registered
    lock = followup.load_registration(registration)
    plans = list(followup.training_configs(lock, registration, 'P2'))
    assert len(plans) == 5  # CARD, D, candidate, g=.25, g=1
    missing = copy.deepcopy(lock); missing['spec']['stages']['P2']['constant_gates'] = None
    with pytest.raises(ValueError, match='constant-g'):
        list(followup.training_configs(missing, registration, 'P2'))
    _, _, _, _, _, cfg = next(followup.training_configs(lock, registration, 'P1c'))
    good = followup.read(followup.artifact(spec, 'initial'))
    followup.validate_initialization_registration(cfg, good)
    bad = copy.deepcopy(good); bad['speaker'] = {'different': True}
    with pytest.raises(ValueError, match='NOT_COMPARABLE'):
        followup.validate_initialization_registration(cfg, bad)


def test_complete_matrix_uses_selected_candidate_with_matched_fixed(monkeypatch, tmp_path):
    spec = {'output_root': str(tmp_path), 'data_environment': {}, 'stages': {'P3': {
        'budget_steps': 10000, 'selection_rule': 'five_metric_equal_weight_log',
        'datasets': list(DATASETS), 'seeds': list(SEEDS)}}}
    candidate = {'name': 'dense', 'model': {'semantic_fusion_dense_local_access': True}}
    monkeypatch.setattr(followup, 'receipt', lambda *a, **k: {'result': {'selected_candidate': candidate}})
    plans = list(followup.training_configs({'spec': spec}, tmp_path/'registration.json', 'P3'))
    assert len(plans) == 36
    assert len({(d, s, a) for _, d, s, a, _, _ in plans}) == 36
    for _, _, _, arm, _, cfg in plans:
        assert cfg.train.protocol_id == FOLLOWUP_PROTOCOL
        assert cfg.model.semantic_fusion_dense_local_access == (arm in ('rsaca', 'fixed_gate'))


@pytest.mark.parametrize('stage', ['P1a', 'P1b'])
@pytest.mark.parametrize('spelling', ['absolute', 'relative', 'dot_segments'])
def test_reference_paths_from_nonproject_cwd_reach_actual_handlers(registered, monkeypatch, stage, spelling):
    _, spec, tmp_path = registered
    reference = Path(spec['artifacts']['reference']['path'])
    relative = os.path.relpath(reference, followup.PROJECT)
    value = str(reference) if spelling == 'absolute' else relative
    if spelling == 'dot_segments':
        (reference.parent/'existing_segment').mkdir()
        value = str(reference.parent/'existing_segment/../reference.json')
    config_path = Path(spec['artifacts']['config']['path'])
    config = followup.read(config_path)
    config['data']['eval_anno_path'] = value
    config_path.write_text(json.dumps(config), encoding='utf-8')
    spec['artifacts']['config']['sha256'] = sha256_file(str(config_path))
    config_bytes = config_path.read_bytes()
    manifest_path = Path(spec['artifacts']['manifest']['path'])
    manifest_bytes = manifest_path.read_bytes()
    manifest = followup.read(manifest_path)
    visited = []
    def resolver(cfg, source_kind):
        visited.append(Path.cwd())
        assert Path.cwd() == followup.PROJECT
        # Exercise the legacy resolver's ordinary cwd-relative file opening.
        assert followup.read(cfg.data.eval_anno_path)['annotations']
        return copy.deepcopy(manifest)
    monkeypatch.setattr(followup, 'input_manifest', resolver)
    synthetic_score(monkeypatch); synthetic_decode(monkeypatch, spec)
    caller = tmp_path/'caller'; caller.mkdir(); monkeypatch.chdir(caller)
    monkeypatch.setattr(followup, 'receipt', lambda *a, **k: {'status': 'PASS'})
    preview = followup.preview_stage({'spec': spec}, stage)
    assert preview['readiness'] == 'READY_FOR_EXECUTION_CHECKS', preview['issues']
    assert not (Path(spec['output_root'])/stage).exists()
    assert Path.cwd() == caller
    output = Path(spec['output_root'])/stage; output.mkdir()
    result = (followup._p1a if stage == 'P1a' else followup._p1b)(spec, spec['stages'][stage], output)
    assert followup._receipt_status(stage, result) == 'PASS'
    assert Path.cwd() == caller
    if stage == 'P1b':
        assert visited == [followup.PROJECT]
    assert config_path.read_bytes() == config_bytes
    assert manifest_path.read_bytes() == manifest_bytes


@pytest.mark.parametrize('stage', ['P1a', 'P1b'])
def test_reference_mismatch_blocked_in_preview_and_handler(registered, monkeypatch, stage):
    _, spec, tmp_path = registered
    config_path = Path(spec['artifacts']['config']['path'])
    config = followup.read(config_path)
    other = tmp_path/'other_reference.json'; other.write_text('{}', encoding='utf-8')
    config['data']['eval_anno_path'] = str(other)
    config_path.write_text(json.dumps(config), encoding='utf-8')
    spec['artifacts']['config']['sha256'] = sha256_file(str(config_path))
    monkeypatch.setattr(followup, 'receipt', lambda *a, **k: {'status': 'PASS'})
    preview = followup.preview_stage({'spec': spec}, stage)
    assert preview['readiness'] == 'BLOCKED'
    assert any('references differ' in issue for issue in preview['issues'])
    with pytest.raises(ValueError, match='references differ'):
        (followup._p1a if stage == 'P1a' else followup._p1b)(spec, spec['stages'][stage], tmp_path/'never_created')
    assert not (tmp_path/'never_created').exists()


def test_dataset_cwd_restored_on_exception(registered, monkeypatch):
    _, spec, tmp_path = registered
    caller = tmp_path/'caller'; caller.mkdir(); monkeypatch.chdir(caller)
    def fail(cfg, source_kind):
        assert Path.cwd() == followup.PROJECT
        raise RuntimeError('resolver fixture failure')
    monkeypatch.setattr(followup, 'input_manifest', fail)
    cfg = followup._cfg(followup.artifact(spec, 'config'))
    with pytest.raises(RuntimeError, match='resolver fixture failure'):
        followup.verify_dataset(spec, 'levir_cc', cfg)
    assert Path.cwd() == caller


def test_register_and_receipt_hash_once_but_next_validation_is_fresh(registered, monkeypatch):
    registration, spec, tmp_path = registered
    counts = collections.Counter()
    original = followup.sha256_file
    def counted(path):
        counts[str(Path(path).resolve())] += 1
        return original(path)
    monkeypatch.setattr(followup, 'sha256_file', counted)
    new_spec = copy.deepcopy(spec); new_spec['output_root'] = str(tmp_path/'uncreated_registration')
    spec_path = tmp_path/'new_spec.json'; spec_path.write_text(json.dumps(new_spec), encoding='utf-8')
    followup.register(spec_path, tmp_path/'uncreated_registration/registration.json', execute=False)
    assert all(counts[str(Path(item['path']).resolve())] == 1 for item in spec['artifacts'].values())
    assert not (tmp_path/'uncreated_registration').exists()
    result = followup.execute_stage(registration, 'P0', execute=True)
    assert result['status'] == 'PASS'
    lock = followup.load_registration(registration)
    counts.clear()
    followup.receipt(lock, 'P0')
    assert all(counts[str(Path(spec['artifacts'][name]['path']).resolve())] == 1 for name in result['inputs'])
    initial = Path(spec['artifacts']['initial']['path'])
    initial.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='changed registered artifact'):
        followup.receipt(lock, 'P0')


def test_stage_end_still_rehashes_used_input(registered, monkeypatch):
    registration, spec, _ = registered
    def mutate_after_first_validation(spec, settings):
        path = followup.artifact(spec, 'initial')
        path.write_text('{}', encoding='utf-8')
        return {'verified': True}
    monkeypatch.setattr(followup, '_p0', mutate_after_first_validation)
    result = followup.execute_stage(registration, 'P0', execute=True)
    assert result['status'] == 'FAILED'
    assert 'changed registered artifact' in result['result']['error']
