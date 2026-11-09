"""Small on-disk fixtures exercise the real loader, without training or scoring."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from imageio.v2 import imwrite

from datasets.rcc_dataset_transformer_levir import RCCDataset, RCCDataLoader, rcc_collate
from utils.semantic_controls import configuration, ENV_ROOTS, POSITION_PROTOCOL
from utils.semantic_input_preflight import preflight_semantic_input


ROOT = Path(__file__).parents[1]


def fixture(tmp_path, dataset='second_cc', arm='E1'):
    data = tmp_path/'data'
    cfg = configuration(ROOT, tmp_path/'runs', dataset, 1111,
                        'card' if arm == 'E0' else 'rsaca',
                        {ENV_ROOTS[dataset]: str(data)})
    cfg.train.protocol_id = POSITION_PROTOCOL
    cfg.train.semantic_control_arm = arm
    cfg.data.num_workers = 0
    cfg.data.return_dict = True
    names = ('complete.png', 'neighbour.png')
    splits = {'train': [0, 1], 'val': [], 'test': [],
              'idx_to_filename': dict(enumerate(names)),
              'idx_to_split': {0: 'train', 1: 'train'}}
    data.mkdir(parents=True)
    for path, payload in ((cfg.data.splits_json, splits),
                          (cfg.data.vocab_json, {'<NULL>': 0, '<START>': 1, 'building': 2, '<END>': 3}),
                          (cfg.data.caption_json, {'images': []}),
                          (cfg.data.eval_anno_path, {'annotations': [
                              {'image_id': name, 'id': i, 'caption': 'building'}
                              for i, name in enumerate(names)]})):
        Path(path).write_text(json.dumps(payload), encoding='utf-8')
    with h5py.File(cfg.data.h5_label_file, 'w') as handle:
        handle['labels'] = np.array([[1, 2, 3, 0], [1, 2, 3, 0]], dtype=np.int32)
        handle['label_start_idx'] = np.arange(2)
        handle['label_end_idx'] = np.arange(2)
    for number, name in enumerate(names):
        for phase in (cfg.data.default_phase, cfg.data.semantic_phase):
            feature = Path(cfg.data.default_feature_dir)/'train'/phase/(name+'.npy')
            feature.parent.mkdir(parents=True, exist_ok=True)
            np.save(feature, np.ones((4, 14, 14), dtype=np.float32))
            source = Path(cfg.data.default_img_dir)/'train'/phase/name
            source.parent.mkdir(parents=True, exist_ok=True)
            imwrite(source, np.zeros((28, 28, 3), dtype=np.uint8))
        phases = ('sem/A', 'sem/B') if dataset == 'second_cc' else ('label',)
        root = data if dataset == 'second_cc' else data/'images'
        for phase in phases:
            target = root/'train'/phase/name
            target.parent.mkdir(parents=True, exist_ok=True)
            values = np.zeros((28, 28), dtype=np.uint8)
            if number and phase != 'sem/A':
                values[:4, :4] = 1 if dataset == 'second_cc' else 255
            imwrite(target, values)
    return cfg


def semantic_path(cfg, phase, filename='neighbour.png'):
    root = cfg.data.semantic_map_root if cfg.data.dataset == 'second_cc' else cfg.data.semantic_diff_root
    return Path(root)/'train'/phase/filename


@pytest.mark.parametrize('dataset,phase,time', [
    ('second_cc', 'sem/A', 'before'), ('second_cc', 'sem/B', 'after'),
    ('levir_mci', 'label', 'diff')])
def test_missing_inputs_fail_before_first_collate(tmp_path, monkeypatch, dataset, phase, time):
    cfg = fixture(tmp_path, dataset)
    missing = semantic_path(cfg, phase)
    missing.unlink()
    calls = []
    import datasets.rcc_dataset_transformer_levir as module
    monkeypatch.setattr(module, 'rcc_collate', lambda batch: calls.append(batch))
    with pytest.raises(ValueError) as error:
        RCCDataset(cfg, 'train')
    text = str(error.value)
    for expected in ('dataset='+dataset, 'split=train', 'ID=neighbour.png', 'time='+time, 'phase='+phase, 'path='):
        assert expected in text
    assert not calls


@pytest.mark.parametrize('dataset,phase', [('second_cc', 'sem/B'), ('levir_mci', 'label')])
def test_corrupt_input_fails_with_identity(tmp_path, dataset, phase):
    cfg = fixture(tmp_path, dataset)
    semantic_path(cfg, phase).write_bytes(b'not a decodable image')
    with pytest.raises(ValueError, match='ID=neighbour.png.*decode failed'):
        RCCDataset(cfg, 'train')


@pytest.mark.parametrize('dataset,phase', [('second_cc', 'sem/B'), ('levir_mci', 'label')])
def test_sizes_are_checked_against_temporal_rgb(tmp_path, dataset, phase):
    cfg = fixture(tmp_path, dataset)
    imwrite(semantic_path(cfg, phase), np.zeros((30, 30), dtype=np.uint8))
    with pytest.raises(ValueError, match='ID=neighbour.png.*sizes differ'):
        RCCDataset(cfg, 'train')


def test_root_fallback_is_rejected_instead_of_becoming_both_times(tmp_path):
    cfg = fixture(tmp_path)
    for phase in ('sem/A', 'sem/B'):
        semantic_path(cfg, phase).unlink()
    # The legacy resolver would return this same file for BOTH requested times.
    imwrite(Path(cfg.data.semantic_map_root)/'neighbour.png', np.zeros((28, 28), dtype=np.uint8))
    with pytest.raises(ValueError, match='wrong split, phase or source root'):
        RCCDataset(cfg, 'train')


def test_wrong_time_configuration_is_rejected(tmp_path):
    cfg = fixture(tmp_path)
    cfg.data.semantic_after_phase = 'sem/A'
    with pytest.raises(ValueError, match='time=after.*expected temporal/source phase sem/B'):
        RCCDataset(cfg, 'train')


def test_unregistered_confidence_input_is_rejected(tmp_path):
    cfg = fixture(tmp_path)
    cfg.data.semantic_diff_confidence_root = str(tmp_path/'confidence')
    with pytest.raises(ValueError, match='position protocol disables semantic confidence'):
        RCCDataset(cfg, 'train')


def test_hardlinked_temporal_maps_are_rejected(tmp_path):
    import os
    cfg = fixture(tmp_path)
    target = semantic_path(cfg, 'sem/B')
    target.unlink()
    os.link(semantic_path(cfg, 'sem/A'), target)
    with pytest.raises(ValueError, match='before and after resolve to the same file'):
        RCCDataset(cfg, 'train')


@pytest.mark.parametrize('dataset', ['second_cc', 'levir_mci'])
def test_zero_and_complete_neighbour_decode_exactly_like_legacy(tmp_path, dataset):
    cfg = fixture(tmp_path, dataset)
    ds = RCCDataset(cfg, 'train')
    assert [r['group'] for r in ds.semantic_input_records] == ['original_empty', 'valid_nonempty']
    checked = [ds._load_semantic_pair('train', name) for name in ('complete.png', 'neighbour.png')]
    ds.require_semantic_inputs = False
    legacy = [ds._load_semantic_pair('train', name) for name in ('complete.png', 'neighbour.png')]
    for actual, expected in zip(checked, legacy):
        for a, b in zip(actual, expected):
            assert a is b if a is None or b is None else torch.equal(a, b)
    ds.require_semantic_inputs = True
    batch = rcc_collate([ds[0], ds[1]])
    assert batch['semantic_diff'].shape == (2, 28, 28)
    assert torch.count_nonzero(batch['semantic_diff'][0]) == 0
    assert torch.count_nonzero(batch['semantic_diff'][1]) > 0
    assert torch.equal(batch['semantic_diff'][1], checked[1][2])


def test_file_removed_after_preflight_fails_before_collate(tmp_path, monkeypatch):
    cfg = fixture(tmp_path)
    ds = RCCDataset(cfg, 'train')
    semantic_path(cfg, 'sem/B').unlink()
    calls = []
    import datasets.rcc_dataset_transformer_levir as module
    monkeypatch.setattr(module, 'rcc_collate', lambda batch: calls.append(batch))
    loader = RCCDataLoader(ds, batch_size=2, num_workers=0)
    with pytest.raises(ValueError, match='ID=neighbour.png.*time=after.*required file is missing'):
        next(iter(loader))
    assert not calls


@pytest.mark.parametrize('dataset', ['second_cc', 'levir_mci'])
def test_e0_uses_only_visual_caches_without_semantic_or_rgb_files(tmp_path, dataset):
    cfg = fixture(tmp_path, dataset, arm='E0')
    data = Path(cfg.data.data_root)
    for path in data.rglob('*.png'):
        path.unlink()
    ds = RCCDataset(cfg, 'train')
    assert not ds.semantic_input_records
    batch = rcc_collate([ds[0], ds[1]])
    assert batch['semantic_diff'] is None
    assert batch['feature_before'].shape[0] == 2


def test_raw_encoding_exceptions_are_recorded_without_relabeling(tmp_path):
    cfg = fixture(tmp_path, 'levir_mci')
    values = np.zeros((28, 28), dtype=np.uint8)
    values[0, 0] = 17  # Existing decoder maps unknown positive greys to class 1.
    imwrite(semantic_path(cfg, 'label'), values)
    record, pair = preflight_semantic_input(cfg, 'train', 'neighbour.png', return_tensors=True)
    assert record['inputs'][0]['raw_encoding']['unrecognized_pixels'] == 1
    assert pair[2][0, 0] == 1
    assert record['original_changed'] == 1
    assert record['geometry_status'].endswith('augmentation geometry unverified')


def test_group_registry_keeps_ignore_and_downsampled_empty_separate(tmp_path):
    cfg = fixture(tmp_path)
    for phase in ('sem/A', 'sem/B'):
        semantic_path(cfg, phase).unlink()
        np.save(semantic_path(cfg, phase).with_suffix('.npy'), np.full((28, 28), -1, dtype=np.int64))
    ignored = preflight_semantic_input(cfg, 'train', 'neighbour.png')
    assert ignored['group'] == 'all_invalid'
    assert ignored['original_valid'] == ignored['grid_valid'] == 0
    np.save(semantic_path(cfg, 'sem/A').with_suffix('.npy'), np.zeros((28, 28), dtype=np.int64))
    after = np.zeros((28, 28), dtype=np.int64)
    after[1, 1] = 1
    np.save(semantic_path(cfg, 'sem/B').with_suffix('.npy'), after)
    downsampled = preflight_semantic_input(cfg, 'train', 'neighbour.png')
    assert downsampled['group'] == 'downsampled_empty'
    assert downsampled['original_changed'] == 1 and downsampled['grid_changed'] == 0
