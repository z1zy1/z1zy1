import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from models.CARD import CARD, SemanticCrossAttentionFusion
from models.transformer_decoder import DynamicSpeaker
from scripts import run_rsaca_supplement as runner
from tests.test_semantic_controls import dataset_fixture, small
from utils import rsaca_supplement as supplement
from utils.checkpoint_integrity import atomic_save_checkpoint, sha256_file
from utils.experiment_tracking import cfg_to_plain
from utils.semantic_control_audit import enforce_test_admission, read, statistics_report, write
from utils.semantic_controls import ARMS, METRICS, SUPPLEMENT_PROTOCOL, build_control_models, supplement_configuration

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('size', [2, 14])
def test_ablation_values_masks_and_empty_identity(size, monkeypatch):
    base = dict(embed_dim=8, num_semantic_classes=8, num_heads=2, dropout=0,
                use_sparse_change_tokens=True, use_reliability_gate=True,
                gate_whole_adapter=True)
    modules = [
        SemanticCrossAttentionFusion(**base, use_global_semantic_token=True),
        SemanticCrossAttentionFusion(**base, use_global_semantic_token=True, dense_local_access=True),
        SemanticCrossAttentionFusion(**base, use_global_semantic_token=False),
        SemanticCrossAttentionFusion(**base, use_global_semantic_token=True, global_token_mode='changed_mean'),
        SemanticCrossAttentionFusion(**base, use_global_semantic_token=True, fixed_nonempty_gate=True),
    ]
    for model in modules[1:]:
        model.load_state_dict(modules[0].state_dict(), strict=True)
    x = torch.randn(3, size*size, 8, requires_grad=True)
    before = torch.zeros(3, size*2, size*2, dtype=torch.long)
    after = before.clone()
    before[1] = after[1] = 4
    after[1, 0, 0] = 2
    after[1, 0, 2] = -1
    after[2, 1, 1] = 1  # nearest-neighbor resampling removes this change
    observed = []
    for fusion in modules:
        original = fusion.attention.forward

        def track(q, k, v, _forward=original, **kwargs):
            assert torch.equal(k, v)
            observed.append((k.detach(), kwargs['key_padding_mask'].detach()))
            return _forward(q, k, v, **kwargs)

        monkeypatch.setattr(fusion.attention, 'forward', track)
        result = fusion(x, before, after, spatial_size=(size, size), detach_ratio=0.5)
        assert torch.isfinite(result).all()
        assert torch.equal(result[[0, 2]], x[[0, 2]])
        (result * torch.randn_like(result)).sum().backward()
        assert torch.isfinite(x.grad).all()
    kv, pad = observed[0]
    dense_kv, dense_pad = observed[1]
    ng_kv, ng_pad = observed[2]
    n = size*size
    assert kv.shape[0] == dense_kv.shape[0] == observed[3][0].shape[0] == n+2
    assert ng_kv.shape[0] == n+1
    assert torch.equal(kv[:n], ng_kv[:n])
    assert torch.equal(kv[-1], ng_kv[-1])
    assert torch.equal(pad[:, :n], ng_pad[:, :n])
    assert torch.equal(kv[-2:], dense_kv[-2:])
    assert torch.equal(modules[0].last_reliability_gate, modules[1].last_reliability_gate)
    assert torch.equal(modules[0].last_reliability_gate, modules[2].last_reliability_gate)
    assert dense_pad[1, 1] and pad[1, size] and not dense_pad[1, size]
    assert torch.count_nonzero(dense_kv[size, 1]) and not torch.count_nonzero(kv[size, 1])


@pytest.mark.parametrize('arm', ARMS)
def test_historical_model_caption_equivalence(arm):
    source = subprocess.check_output(
        ['git', 'show', supplement.HISTORICAL['O']+':models/CARD.py'], cwd=ROOT)
    namespace = {'__name__': 'historical_card'}
    exec(compile(source, 'historical_CARD.py', 'exec'), namespace)
    cfg = small(supplement_configuration(ROOT, '.', 'levir_mci', 1111, arm, {}))
    detector, speaker = build_control_models(cfg, CARD, DynamicSpeaker)
    old = namespace['CARD'](cfg)
    old.load_state_dict(detector.state_dict(), strict=True)
    assert set(old.state_dict()) == set(detector.state_dict())
    assert all(torch.equal(old.state_dict()[name], value) for name, value in detector.state_dict().items())
    first, second = torch.randn(1, 16, 14, 14), torch.randn(1, 16, 14, 14)
    semantics = dict(semantic_diff=None if arm == 'card' else torch.randint(0, 5, (1, 14, 14)))
    for module in (old, detector, speaker):
        module.eval()
    observed = []
    if arm != 'card':
        for module in (detector, old):
            module.semantic_cross_fusion.attention.register_forward_hook(
                lambda _, inputs, outputs: observed.append((inputs, outputs)))
    with torch.no_grad():
        actual, expected = detector(first, second, **semantics)[0], old(first, second, **semantics)[0]
        if observed:
            for current, historical in zip(observed[0], observed[1]):
                assert len(current) == len(historical)
                assert all(torch.equal(a, b) for a, b in zip(current, historical))
        assert torch.equal(actual, expected)
        assert torch.isfinite(actual).all()
        if arm in ('rsaca', 'fixed_gate'):
            assert torch.equal(old.semantic_cross_fusion.last_reliability_gate,
                               detector.semantic_cross_fusion.last_reliability_gate)
        labels = torch.tensor([[2, 5, 6, 4, 0, 0]])
        logits = lambda memory: speaker.logit(speaker.core(labels, labels.ne(0).float(), memory,
                                                          diagonal_mask=True)[0])
        assert torch.equal(logits(actual), logits(expected))
        generated, _ = speaker.sample(actual, sample_max=1)
        assert torch.equal(generated, speaker.sample(expected, sample_max=1)[0])
        assert torch.equal(generated, speaker.sample(actual, sample_max=1, fixed_steps=speaker.seq_length)[0])


@pytest.mark.parametrize('positive', [0, 11, 14])
def test_negative_validation_directions_admit_complete_matrix(tmp_path, positive):
    rows = []
    for dataset, seed, arm, _ in supplement.entries(tmp_path, 'O'):
        offset = list(supplement.COUNTS).index(dataset)*5
        values = {m: 1 if arm != 'rsaca' else (1.1 if offset+i < positive else 0.9)
                  for i, m in enumerate(METRICS)}
        rows.append(dict(dataset=dataset, arm=arm, seed=seed, metrics=values))
    validation = statistics_report(rows)
    assert validation['positive_mean_metrics'] == positive
    payload = dict(series='O', errors=[], validation=validation, runs={
        p.relative_to(tmp_path).as_posix(): {'strict_load': True}
        for _, _, _, p in supplement.entries(tmp_path, 'O')})
    assert supplement.freeze_payload(tmp_path, payload)['validation'] == validation
    payload['runs'].pop(next(iter(payload['runs'])))
    with pytest.raises(ValueError, match='matrix'):
        supplement.freeze_payload(tmp_path, payload)


def test_cli_fixture_freeze_admission_scoring_and_retry(tmp_path, monkeypatch):
    """Mock scorer and decoder only for the fixture's prediction/score chain."""
    root, series = tmp_path/'new-ablation', 'new-ablation'
    ref, checkpoint, manifest_file = (tmp_path/n for n in ('refs.json', 'model.pt', 'inputs.json'))
    write(ref, dict(annotations=[dict(id=1, image_id='x', caption='a')]))
    manifest = dict(samples=[dict(sample_id='x', split='test')])
    write(manifest_file, manifest)
    digest = atomic_save_checkpoint({'global_step': 1000}, str(checkpoint),
                                    write_checksum=True, expected_step=1000)
    source, scorer = {'source': {'commit': 'fixture'}}, {'mocked': True}
    monkeypatch.setattr(supplement, 'execution_identity', lambda _: source)
    monkeypatch.setattr(supplement, 'scorer_identity', lambda: scorer)
    monkeypatch.setattr(supplement, 'get_inputs', lambda *a, **k: manifest)
    runs = {}
    for d, seed, arm, run in supplement.entries(root, series):
        cfg = supplement.node(dict(exp_dir=str(root/'evaluations'/arm), exp_name=run.name,
                                   train=dict(protocol_id=SUPPLEMENT_PROTOCOL),
                                   data=dict(dataset=d, eval_anno_path=str(ref))))
        runs[run.relative_to(root).as_posix()] = dict(
            strict_load=True, config=cfg_to_plain(cfg), checkpoint=str(checkpoint), step=1000,
            checkpoint_sha256=digest, selection_sha256='selected', files={str(ref): sha256_file(ref)},
            manifest=str(manifest_file), manifest_sha256=sha256_file(manifest_file))
    payload = dict(series=series, errors=[], runs=runs, rows=[], assets=[], validation={},
                   execution=source, scorer=scorer, training_identity=source)
    monkeypatch.setattr(runner, 'audit', lambda *a: payload)
    monkeypatch.setattr(runner, 'score_prediction', lambda *a: dict.fromkeys(METRICS, 0.25))

    def invoke(stage, *flags):
        monkeypatch.setattr(sys, 'argv', ['run_rsaca_supplement', '--root', str(tmp_path),
                                         '--series', series, '--stage', stage, *flags])
        runner.main()

    def mock_inference(*args):
        assert args[0] == 'test_card_spot.py' and args[args.index('--split')+1] == 'test'
        config = supplement.node(read(args[args.index('--cfg')+1]))
        output = args[args.index('--result_json')+1]
        assert enforce_test_admission(ROOT, config, checkpoint, output) == 'new'
        supplement.publish_prediction(config, output, dict(x='a'))

    monkeypatch.setattr(runner, 'execute', mock_inference)
    invoke('freeze')
    item = runs['runs/rsaca/rsaca_levir_cc_seed1111']
    cfg = supplement.node(item['config'])
    run = supplement.training_run(cfg)
    prediction = run/'captions'/'test.json'
    write(prediction, [])  # interrupted pre-receipt publication
    with pytest.raises(ValueError, match='Incomplete output'):
        enforce_test_admission(ROOT, cfg, checkpoint, prediction)
    filters = ('--dataset', 'levir_cc', '--arm', 'rsaca', '--seed', '1111')
    def broken_score(*args):
        raise RuntimeError('fixture scorer interrupted')

    with monkeypatch.context() as scored:
        scored.setattr(runner, 'score_prediction', broken_score)
        with pytest.raises(RuntimeError, match='scorer interrupted'):
            invoke('test', *filters, '--retry-incomplete')
    failure = next((root/'evaluation_attempts').glob('*/execution.json'))
    assert read(failure)['status'] == 'failed'
    assert enforce_test_admission(ROOT, cfg, checkpoint, prediction) == 'prediction_only'
    invoke('test', *filters)
    assert list((root/'failed_evaluations').glob('*/evaluation/captions/test.json'))
    assert enforce_test_admission(ROOT, cfg, checkpoint, prediction) == 'complete'
    invoke('test', *filters)
    invoke('summary')
    summary = read(root/'reports/test/statistics.json')
    assert summary['complete'] == 1 and summary['planned'] == 27
    assert len(summary['anomalies']) == 26
    with pytest.raises(ValueError, match='output'):
        enforce_test_admission(ROOT, cfg, checkpoint, run/'renamed.json')
    write(prediction, [])
    with pytest.raises(ValueError, match='IDs'):
        enforce_test_admission(ROOT, cfg, checkpoint, prediction)
    checkpoint.write_bytes(b'tampered')
    with pytest.raises(ValueError):
        enforce_test_admission(ROOT, cfg, checkpoint, prediction)


def test_preflight_respects_blank_maps_and_card_without_maps(tmp_path, monkeypatch):
    cfg = dataset_fixture(tmp_path, 'second_cc')
    monkeypatch.setitem(supplement.COUNTS, 'second_cc', 1)
    manifest = supplement.get_inputs(cfg)
    path = Path(manifest['samples'][2]['semantic'][1]['path'])
    np.save(path, np.zeros((28, 28), dtype=np.uint8))
    assert not supplement.get_inputs(cfg)['missing']
    np.save(path, np.zeros((13, 28), dtype=np.uint8))
    with pytest.raises(ValueError, match='misaligned temporal'):
        supplement.get_inputs(cfg)
    path.unlink()
    with pytest.raises(ValueError, match='Missing input'):
        supplement.get_inputs(cfg)
    card = copy.deepcopy(cfg)
    card.data.use_semantic_maps = False
    card.model.semantic_input_mode = 'none'
    assert not supplement.get_inputs(card)['missing']


def test_dry_matrix_does_not_create_output(tmp_path):
    root = tmp_path/'unused'
    result = subprocess.check_output(
        [sys.executable, 'scripts/run_rsaca_supplement.py', '--root', str(root),
         '--stage', 'test', '--dry-run'], cwd=ROOT, text=True)
    plan = json.loads(result)
    assert {s: len(set(keys)) for s, keys in plan.items()} == {
        'O': 36, 'S': 36, 'new-ablation': 27, 'new-full': 54}
    assert sum('/dense_local/' in k for k in plan['new-ablation']) == 9
    assert sum('/no_global/' in k for k in plan['new-ablation']) == 9
    assert sum('/rsaca/' in k for k in plan['new-full']) == 9
    assert not root.exists()


@pytest.mark.parametrize('protocol,expected', [
    (SUPPLEMENT_PROTOCOL, 'frozen.json'),
    ('legacy', 'cannot be relabeled'),
])
def test_real_low_level_cli_blocks_missing_freeze_and_relabel(tmp_path, protocol, expected):
    ckpt = tmp_path/'checkpoint.pt'
    atomic_save_checkpoint({'model_cfg': {'train': {'protocol_id': SUPPLEMENT_PROTOCOL}}},
                           str(ckpt))
    cfg = supplement_configuration(ROOT, tmp_path/'evaluations', 'levir_cc', 1111, 'card', {})
    cfg.train.protocol_id = protocol
    path, out = tmp_path/'config.json', tmp_path/'output.json'
    write(path, cfg_to_plain(cfg))
    result = subprocess.run([sys.executable, 'test_card_spot.py', '--cfg', str(path), '--split', 'test',
                             '--snapshot_path', str(ckpt), '--result_json', str(out)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode and expected in result.stderr
    assert not out.exists()
