"""Read-only validation of the position protocol's required semantic inputs.

The ordinary dataset's optional semantic behaviour is deliberately unchanged.
Resolution and decoding use the same methods as training, before collation can
turn one missing field into a missing field for all its neighbours.
"""
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def requires_semantic_preflight(cfg):
    from utils.semantic_controls import POSITION_PROTOCOL
    return (getattr(cfg.train, 'protocol_id', '') == POSITION_PROTOCOL
            and getattr(cfg.train, 'semantic_control_arm', '') in ('E1', 'E2'))


def _reader(cfg):
    from datasets.rcc_dataset_transformer_levir import RCCDataset
    ds = RCCDataset.__new__(RCCDataset)
    ds.cfg = cfg
    ds.dataset_name = str(cfg.data.dataset).lower()
    ds.semantic_map_root = cfg.data.semantic_map_root or cfg.data.semantic_img_dir
    ds.s_img_dir = cfg.data.semantic_img_dir
    ds.semantic_diff_root = cfg.data.semantic_diff_root
    ds.semantic_diff_phase = cfg.data.semantic_diff_phase
    ds.semantic_diff_binary = bool(cfg.data.semantic_diff_binary)
    ds.semantic_unknown_change_class = int(cfg.data.semantic_unknown_change_class)
    ds.mask_ignore_index = int(getattr(cfg.train, 'semantic_ignore_index', -1))
    ds._missing_semantic_warnings = 5
    return ds


def preflight_semantic_input(cfg, split, filename, dataset=None, spatial_size=(14, 14),
                             return_tensors=False):
    """Return serializable decode/group evidence without creating label caches.

    Only split-qualified source paths are accepted for the new protocol. The
    legacy resolver still chooses the file; fallback to another phase/root is
    rejected here rather than quietly changing the experiment's input source.
    ``return_tensors`` lets the dataset reuse the checked decode at runtime.
    """
    from datasets.rcc_dataset_transformer_levir import (
        _candidate_paths, _first_existing, _load_image_or_array,
        LEVIR_LABEL_VALUE_TO_CLASS, LEVIR_LABEL_RGB_TO_CLASS,
        SECOND_SEMANTIC_RGB_TO_CLASS,
    )
    ds = dataset or _reader(cfg)
    name = str(cfg.data.dataset).lower()

    def fail(role, phase, path, reason):
        raise ValueError('Semantic input preflight failed: dataset=%s split=%s ID=%s '
                         'time=%s phase=%s path=%s: %s'
                         % (name, split, filename, role, phase, path, reason))

    if name not in ('second_cc', 'levir_mci'):
        fail('configuration', '', '', 'unsupported position-protocol dataset')
    if split not in ('train', 'val', 'test') or filename != os.path.basename(filename):
        fail('configuration', '', '', 'invalid split or sample ID')
    is_second = name == 'second_cc'
    if getattr(cfg.data, 'semantic_diff_confidence_root', ''):
        fail('configuration', '', cfg.data.semantic_diff_confidence_root,
             'the position protocol disables semantic confidence maps')
    if is_second and getattr(cfg.data, 'semantic_diff_root', ''):
        fail('configuration', '', cfg.data.semantic_diff_root,
             'SECOND derives diff from its required pair in the position protocol')
    phases = (('before', cfg.data.semantic_before_phase, 'sem/A'),
              ('after', cfg.data.semantic_after_phase, 'sem/B')) if is_second else (
                  ('diff', cfg.data.semantic_diff_phase, 'label'),)
    if bool(cfg.data.semantic_diff_only) == is_second:
        fail('configuration', '', '', 'SECOND requires a temporal pair; MCI requires diff-only')
    decoded, records, paths = [], [], []
    for role, phase, expected_phase in phases:
        if str(phase).replace('\\', '/') != expected_phase:
            fail(role, phase, '', 'expected temporal/source phase '+expected_phase)
        root = (cfg.data.semantic_map_root or cfg.data.semantic_img_dir) if is_second else cfg.data.semantic_diff_root
        path = (ds._resolve_semantic_path(split, phase, filename) if is_second
                else ds._resolve_semantic_diff_path(split, filename))
        expected_bases = (Path(root)/split/phase, Path(root)/phase/split)
        expected = [candidate for candidate in _candidate_paths(root, split, phase, filename)
                    if Path(candidate).parent in expected_bases]
        if path is None:
            fail(role, phase, expected[0] if expected else root, 'required file is missing')
        if Path(path).resolve() not in {Path(p).resolve() for p in expected}:
            fail(role, phase, path, 'resolved fallback has the wrong split, phase or source root')
        paths.append(Path(path).resolve())
        try:
            raw = np.asarray(_load_image_or_array(path))
            if raw.ndim not in (2, 3) or not raw.size or not np.isfinite(raw).all():
                raise ValueError('expected a finite, nonempty 2-D class map or RGB image')
            if raw.ndim == 3 and raw.shape[-1] not in (3, 4):
                raise ValueError('expected RGB/RGBA channels')
            value = (ds._load_semantic_map(path) if is_second else ds._load_semantic_diff_map(path))
            if value is None or value.ndim != 2 or tuple(value.shape) != tuple(raw.shape[:2]):
                raise ValueError('decoder did not return one class ID per input pixel')
            if raw.ndim == 2:
                codes, counts = np.unique(raw, return_counts=True)
                allowed = set(range(int(cfg.data.num_semantic_classes))) | {-1} if is_second else set(LEVIR_LABEL_VALUE_TO_CLASS)
                unknown = sum(int(c) for v, c in zip(codes, counts) if int(v) not in allowed)
                encoding = {'values': {str(int(v)): int(c) for v, c in zip(codes, counts)},
                            'unrecognized_pixels': unknown}
            else:
                colors, counts = np.unique(raw[..., :3].reshape(-1, 3), axis=0, return_counts=True)
                allowed = SECOND_SEMANTIC_RGB_TO_CLASS if is_second else LEVIR_LABEL_RGB_TO_CLASS
                encoding = {'colors': {','.join(map(str, color.tolist())): int(c)
                                       for color, c in zip(colors, counts)},
                            'unrecognized_pixels': sum(int(c) for color, c in zip(colors, counts)
                                                       if tuple(color) not in allowed)}
        except Exception as error:
            fail(role, phase, path, 'decode failed: '+str(error))
        decoded.append(value)
        records.append({'role': role, 'phase': phase, 'path': str(Path(path)),
                        'shape': list(value.shape), 'raw_encoding': encoding})
    if len(paths) == 2 and (paths[0] == paths[1] or os.path.samefile(paths[0], paths[1])):
        fail('after', phases[1][1], paths[1], 'before and after resolve to the same file')

    # Raw RGB sizes establish the same split/ID/view, while cache tensors have a
    # different spatial resolution. Pixel-level augmentation geometry cannot be
    # inferred from sizes alone and is explicitly left unverified in evidence.
    sources = []
    for role, root, phase, expected_phase in (
            ('before', cfg.data.default_img_dir, cfg.data.default_phase, 'rgb/A' if is_second else 'A'),
            ('after', cfg.data.semantic_img_dir, cfg.data.semantic_phase, 'rgb/B' if is_second else 'B')):
        if str(phase).replace('\\', '/') != expected_phase:
            fail(role, phase, '', 'expected visual temporal phase '+expected_phase)
        bases = (Path(root)/split/phase, Path(root)/phase/split)
        candidates = [p for p in _candidate_paths(root, split, phase, filename) if Path(p).parent in bases]
        path = _first_existing(candidates)
        try:
            raw = _load_image_or_array(path)
            if raw is None or raw.ndim != 3 or raw.shape[-1] not in (3, 4) or not raw.size:
                raise ValueError('required source RGB is missing or malformed')
            if any(tuple(v.shape) != tuple(raw.shape[:2]) for v in decoded):
                raise ValueError('semantic/source RGB sizes differ')
        except Exception as error:
            fail(role, phase, path or candidates[0], str(error))
        sources.append({'role': role, 'phase': phase, 'path': str(Path(path)), 'shape': list(raw.shape[:2])})

    ignore = int(getattr(cfg.train, 'semantic_ignore_index', -1))
    before, after = decoded if is_second else (torch.zeros_like(decoded[0]), decoded[0])

    def counts(b, a):
        valid = (b != ignore) & (a != ignore)
        changed = valid & (b != a)
        return int(valid.sum()), int(changed.sum())

    original_valid, original_changed = counts(before, after)
    small = [F.interpolate(v[None, None].float(), size=spatial_size, mode='nearest')[0, 0].long()
             for v in (before, after)]
    grid_valid, grid_changed = counts(*small)
    group = ('all_invalid' if original_valid == 0 else 'valid_nonempty' if grid_changed > 0
             else 'downsampled_empty' if original_changed > 0 else 'original_empty')
    result = {'inputs': records, 'source_images': sources,
              'original_valid': original_valid, 'original_changed': original_changed,
              'grid_valid': grid_valid, 'grid_changed': grid_changed, 'group': group,
              'geometry_status': 'split/ID/phase/size verified; augmentation geometry unverified'}
    if return_tensors:
        if is_second:
            valid = (before != ignore) & (after != ignore)
            diff = torch.zeros_like(after)
            changed = valid & (before != after)
            diff[changed] = after[changed] + 1
            diff[~valid] = ignore
            return result, (before, after, diff, None)
        return result, (None, None, after, None)
    return result
