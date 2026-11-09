"""CPU FP32 acceptance on the actual V1 fusion path (plan section 5.2)."""
import json
import math
from pathlib import Path
import subprocess

import pytest
import torch

from models.CARD import CARD, SemanticCrossAttentionFusion
from models.transformer_decoder import DynamicSpeaker
from utils.semantic_controls import configuration, build_control_models, parameter_summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def deterministic_cpu():
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1111)
    yield
    torch.set_num_threads(old_threads)
    torch.use_deterministic_algorithms(old_deterministic)


def fusion(mode='none', cls=SemanticCrossAttentionFusion, explicit=True):
    kwargs = dict(num_heads=8, dropout=0.1, gamma_init=0.01, gamma_max=0.5,
                  use_sparse_change_tokens=True, use_reliability_gate=True,
                  use_global_semantic_token=True, gate_whole_adapter=True)
    if explicit:
        kwargs['local_position'] = mode
    model = cls(512, 8, **kwargs).eval()
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model


def probe(model, q, semantic=None, before=None, after=None):
    result = {}
    def attention_input(module, args):
        assert torch.equal(args[1], args[2])  # the original MHA receives identical K/V memory
        result['memory'] = args[1].transpose(0, 1).detach().clone()
    def attention_output(module, args, output):
        result['U'] = output[0].transpose(0, 1).detach().clone()
    handles = [model.attention.register_forward_pre_hook(attention_input),
               model.attention.register_forward_hook(attention_output),
               model.norm.register_forward_hook(
                   lambda m, args, output: result.update(adapter=output.detach().clone())),
               model.diff_projection.register_forward_hook(
                   lambda m, args, output: result.update(T=output.detach().clone())),
               model.reliability_gate.register_forward_pre_hook(
                   lambda m, args: result.update(gate_inputs=args[0].detach().clone()))]
    try:
        z = model(q, semantic_diff=semantic, sem_before=before, sem_after=after,
                  spatial_size=(14, 14), detach_ratio=0.5)
    finally:
        for handle in handles:
            handle.remove()
    result.update(Z=z, deltaZ=z-q, deltaA=result['adapter']-q,
                  gate=model.last_reliability_gate.clone(), coverage=model.last_change_coverage.clone(),
                  attention=model.last_attention.detach().clone())
    assert all(torch.isfinite(value).all() for value in result.values())
    return result


def test_strict_v1_replay_and_no_rng_or_parameter_delta():
    source = subprocess.check_output(['git', 'show', '995471b:models/CARD.py'], cwd=ROOT)
    old = {}
    exec(compile(source, 'V1@995471b/models/CARD.py', 'exec'), old)
    torch.manual_seed(1111)
    original = fusion(cls=old['SemanticCrossAttentionFusion'], explicit=False)
    expected_rng = torch.get_rng_state()
    for explicit, mode in ((False, 'none'), (True, 'none'), (True, 'sincos2d_kv')):
        torch.manual_seed(1111)
        current = fusion(mode, explicit=explicit)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        current.load_state_dict(original.state_dict(), strict=True)
        assert set(current.state_dict()) == set(original.state_dict())
        assert [(n, p.requires_grad) for n, p in current.named_parameters()] == [
            (n, p.requires_grad) for n, p in original.named_parameters()]
        q = torch.randn(2, 196, 512)
        sem = torch.randint(0, 3, (2, 14, 14))
        if mode == 'none':
            assert torch.equal(current(q, semantic_diff=sem), original(q, semantic_diff=sem))


def test_position_formula_rectangular_row_major_and_memory_contract():
    model = fusion()
    q = torch.randn(1, 196, 512)
    sem = torch.zeros(1, 14, 14, dtype=torch.long)
    sem[0, 1, 1] = 1
    e1 = probe(model, q, sem)
    model.local_position = 'sincos2d_kv'
    e2 = probe(model, q, sem)
    p = model._local_position_encoding((14, 14), q)
    weights = sem.flatten(1).ne(0).unsqueeze(-1)
    torch.testing.assert_close(e2['memory'][:, :196]-e1['memory'][:, :196], p*weights,
                               atol=1e-6, rtol=1e-5)
    for key in ('T', 'gate_inputs', 'coverage', 'gate'):
        assert torch.equal(e1[key], e2[key])
    assert torch.equal(e1['memory'][:, 196:], e2['memory'][:, 196:])
    torch.testing.assert_close(p.norm(dim=-1), torch.full((1, 196), 1.6))
    rectangular = model._local_position_encoding((3, 5), torch.zeros(1, 15, 512))
    for r, c in ((0, 0), (0, 4), (2, 0), (2, 4)):
        assert rectangular[0, r*5+c, 0].item() == pytest.approx(0.1*math.sin(r+0.5), abs=1e-7)
        assert rectangular[0, r*5+c, 256].item() == pytest.approx(0.1*math.sin(c+0.5), abs=1e-7)
    for bad in ('unknown',):
        with pytest.raises(ValueError, match='local position'):
            fusion(bad)
    with pytest.raises(ValueError, match='sparse local'):
        SemanticCrossAttentionFusion(8, 8, local_position='sincos2d_kv')


@pytest.mark.parametrize('layouts', [([(1, 1)], [(12, 12)]),
                                  ([(3, 2), (3, 10)], [(1, 6), (5, 6)])])
def test_output_position_response_locked_thresholds(layouts):
    model = fusion()
    q = torch.randn(1, 196, 512)
    maps = []
    for points in layouts:
        sem = torch.zeros(1, 14, 14, dtype=torch.long)
        for r, c in points:
            sem[0, r, c] = 1
        maps.append(sem)
    report = {}
    for mode in ('none', 'sincos2d_kv'):
        model.local_position = mode
        left, right, repeat = probe(model, q, maps[0]), probe(model, q, maps[1]), probe(model, q, maps[0])
        report[mode] = {}
        for key in ('U', 'deltaA', 'gate', 'deltaZ'):
            delta = (left[key]-right[key]).detach()
            noise = (left[key]-repeat[key]).abs().max().item()
            report[mode][key] = dict(maxabs=delta.abs().max().item(), l2=delta.norm().item(),
                                    relative_l2=delta.norm().item()/max(left[key].norm().item(), 1e-12), noise=noise)
        if mode == 'sincos2d_kv':
            for key in ('U', 'deltaA', 'deltaZ'):
                limit = max(1e-6, 10*report[mode][key]['noise'], 10*report['none'][key]['maxabs'])
                report[mode][key]['threshold'] = limit
                assert report[mode][key]['maxabs'] > limit
        assert report[mode]['gate']['maxabs'] <= 1e-6
    p = model._local_position_encoding((14, 14), q)
    v_weight = model.attention.in_proj_weight[1024:]
    report['norms'] = dict(P=p.norm().item(), T=left['T'].norm().item(),
                           projected_local_V=torch.nn.functional.linear(
                               left['memory'][:, :196], v_weight, model.attention.in_proj_bias[1024:]).norm().item())
    print(json.dumps(dict(layouts=layouts, diagnostics=report), sort_keys=True))


@pytest.mark.parametrize('mode', ['none', 'sincos2d_kv'])
def test_empty_ignore_downsample_single_cell_mixed_and_gradient(mode):
    model = fusion(mode)
    q = torch.randn(5, 196, 512, requires_grad=True)
    sem = torch.zeros(5, 28, 28, dtype=torch.long)
    sem[1].fill_(-1)
    sem[2, 1, 1] = 1  # nearest to 14x14 does not sample this pixel
    sem[3, 2, 2] = 1
    sem[4, 6:12, 6:12] = 2
    mixed = probe(model, q, sem)
    assert torch.equal(mixed['Z'][:3], q[:3])
    assert torch.equal(mixed['gate'][:3], torch.zeros(3, 1))
    for i in range(5):
        single = probe(model, q[i:i+1], sem[i:i+1])
        for key in ('deltaZ', 'coverage', 'gate'):
            torch.testing.assert_close(mixed[key][i:i+1], single[key], atol=1e-6, rtol=1e-5)
    reordered = probe(model, q.flip(0), sem.flip(0))
    torch.testing.assert_close(mixed['Z'], reordered['Z'].flip(0), atol=1e-6, rtol=1e-5)
    all_empty = probe(model, q[:3], sem[:3])
    assert torch.equal(all_empty['Z'], q[:3])
    valid = torch.randint(0, 7, (5, 14, 14))
    identical_pair = probe(model, q, before=valid, after=valid)
    assert torch.equal(identical_pair['Z'], q)
    vector = torch.randn_like(q[:3])
    gradient, = torch.autograd.grad((mixed['Z'][:3]*vector).sum(), q)
    assert torch.equal(gradient[:3], vector)
    assert torch.equal(gradient[3:], torch.zeros_like(gradient[3:]))


def small_config():
    cfg = configuration(ROOT, 'unused', 'levir_mci', 1111, 'rsaca', {})
    enc, dec = cfg.model.transformer_encoder, cfg.model.transformer_decoder
    enc.feat_dim = 16
    enc.att_dim = enc.emb_dim = dec.att_dim = dec.word_dim = 16
    enc.att_head = dec.att_head = cfg.model.semantic_fusion_heads = 2
    enc.att_layer = dec.att_layer = 1
    dec.vocab_size, dec.seq_length = 13, 6
    cfg.model.num_semantic_classes = 8
    return cfg


@pytest.mark.parametrize('mode', ['none', 'sincos2d_kv'])
def test_actual_card_decoder_empty_identity_and_caption_gradient(mode, tmp_path):
    cfg = small_config()
    cfg.model.semantic_fusion_local_position = mode
    detector, speaker = build_control_models(cfg, CARD, DynamicSpeaker)
    detector.eval()
    speaker.eval()
    assert detector.semantic_cross_fusion.local_position == mode
    before, after = torch.randn(2, 16, 14, 14), torch.randn(2, 16, 14, 14)
    captured = {}
    def capture_q(module, args):
        captured['Q'] = args[0]
    hook = detector.semantic_cross_fusion.register_forward_pre_hook(capture_q)
    z = detector(before, after, semantic_diff=torch.zeros(2, 14, 14, dtype=torch.long))[0]
    hook.remove()
    q = captured['Q']
    assert torch.equal(z, q)
    seq = torch.randint(1, 12, (2, 5))
    loss, logits, _ = speaker._forward(z, seq, torch.ones_like(seq), labels_with_ignore=seq)
    direct = speaker._forward(q, seq, torch.ones_like(seq))[1]
    assert torch.equal(logits, direct)
    assert torch.equal(speaker.sample(z, sample_max=1)[0], speaker.sample(q, sample_max=1)[0])
    # Train/validation/inference all construct this same class from the archived config.
    from utils.checkpoint_integrity import atomic_save_checkpoint
    from utils.checkpointing import load_checkpoint_file
    from utils.config_validation import validate_resolved_config
    from utils.experiment_tracking import key_switch_summary
    for phase in ('train', 'val', 'test'):
        validate_resolved_config(cfg, phase=phase)
        assert key_switch_summary(cfg)['semantic_fusion_local_position'] == mode
    checkpoint = tmp_path/'model_checkpoint_1000.pt'
    atomic_save_checkpoint(dict(global_step=1000, model_cfg=cfg,
                                change_detector_state=detector.state_dict()),
                           str(checkpoint), immutable=True, expected_step=1000)
    saved = load_checkpoint_file(str(checkpoint), map_location='cpu')
    restored = CARD(saved['model_cfg']).eval()
    restored.load_state_dict(saved['change_detector_state'], strict=True)
    assert restored.semantic_cross_fusion.local_position == mode
    assert torch.equal(restored(before, after, semantic_diff=torch.zeros(2, 14, 14, dtype=torch.long))[0], z)
    loss.backward()
    assert detector.img[0].weight.grad.abs().sum() > 0
    assert speaker.logit.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
               for p in detector.semantic_cross_fusion.parameters())


@pytest.mark.parametrize('seed', [1111, 2222, 3333])
def test_position_arms_shared_initialization_and_legacy_config(seed):
    cfg = small_config()
    cfg.train.seed = seed
    del cfg.model['semantic_fusion_local_position']
    old_detector, old_speaker = build_control_models(cfg, CARD, DynamicSpeaker)
    summaries = []
    for mode in ('none', 'sincos2d_kv'):
        cfg.model.semantic_fusion_local_position = mode
        detector, speaker = build_control_models(cfg, CARD, DynamicSpeaker)
        summaries.append((parameter_summary(detector), parameter_summary(speaker)))
    assert summaries[0] == summaries[1] == (parameter_summary(old_detector), parameter_summary(old_speaker))


@pytest.mark.parametrize('paired', [False, True])
def test_joint_layout_permutation_preserves_summaries(paired):
    model = fusion()
    q = torch.randn(1, 196, 512)
    after = torch.zeros(1, 14, 14, dtype=torch.long)
    before = torch.zeros_like(after)
    after[0, 1:3, 1:3] = torch.tensor([[1, 2], [2, 1]])
    after[0, 4, 5] = -1
    before[0, 1, 1] = 3
    before[0, 4, 5] = -1
    results = []
    for mode in ('none', 'sincos2d_kv'):
        model.local_position = mode
        def run(b, a):
            return probe(model, q, before=b, after=a) if paired else probe(model, q, a)
        left, right = run(before, after), run(before.flip((1, 2)), after.flip((1, 2)))
        for key in ('gate_inputs', 'gate', 'coverage'):
            torch.testing.assert_close(left[key], right[key], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(left['T'].view(1, 14, 14, 512).flip((1, 2)).flatten(1, 2),
                                   right['T'], atol=1e-6, rtol=1e-5)
        results.append((left['Z']-right['Z']).abs().max().item())
    assert results[0] < 1e-6
    assert results[1] > max(1e-6, 10*results[0])
