import json
import os
import random

import h5py
import numpy as np
import torch
from imageio.v2 import imread
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from utils.dataset_config import feature_extraction_command
from utils.semantic_tags import build_semantic_label, read_semantic_tags
from utils.semantic_label import (
    ACTION_VOCAB,
    OBJECT_VOCAB,
    build_semantic_targets,
    load_semantic_target_cache,
    save_semantic_target_cache,
    semantic_targets_to_tensors,
)


IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.npy')

LEVIR_LABEL_VALUE_TO_CLASS = {
    0: 0,
    128: 1,  # road
    255: 2,  # building
}
LEVIR_LABEL_RGB_TO_CLASS = {
    (0, 0, 0): 0,
    (128, 128, 128): 1,
    (255, 255, 255): 2,
    (0, 0, 255): 1,
    (255, 255, 0): 1,
    (255, 0, 0): 2,
}
SECOND_SEMANTIC_RGB_TO_CLASS = {
    (0, 0, 255): 0,
    (80, 80, 255): 0,
    (0, 128, 0): 1,
    (0, 255, 0): 1,
    (80, 208, 80): 1,
    (80, 255, 80): 1,
    (128, 0, 0): 2,
    (255, 0, 0): 2,
    (208, 80, 80): 2,
    (128, 128, 128): 3,
    (208, 208, 208): 3,
    (255, 255, 255): 4,
    (0, 0, 0): 5,
}
SECOND_KNOWN_COLORS = np.array(list(SECOND_SEMANTIC_RGB_TO_CLASS.keys()), dtype=np.float32)
SECOND_KNOWN_CLASSES = np.array(list(SECOND_SEMANTIC_RGB_TO_CLASS.values()), dtype=np.int64)


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() in ('1', 'true', 'yes', 'y', 'on')
    return bool(value)


def _candidate_paths(root, split_name, phase_name, filename):
    if not root:
        return []
    stem = os.path.splitext(filename)[0]
    names = [filename, stem + '.png', stem + '.jpg', stem + '.jpeg', stem + '.npy']
    bases = []
    if split_name and phase_name:
        bases.extend([
            os.path.join(root, split_name, phase_name),
            os.path.join(root, phase_name, split_name),
        ])
    if split_name:
        bases.append(os.path.join(root, split_name))
    if phase_name:
        bases.append(os.path.join(root, phase_name))
    bases.append(root)
    paths = []
    for base in bases:
        for name in names:
            paths.append(os.path.join(base, name))
    return paths


def _first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _load_image_or_array(path):
    if path is None:
        return None
    if path.lower().endswith('.npy'):
        return np.load(path)
    return imread(path)


def _rgb_to_class_map(rgb, mapping, ignore_index=-1, known_colors=None, known_classes=None):
    if rgb.ndim == 2:
        return rgb.astype(np.int64)
    if rgb.ndim == 3 and rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Expected RGB semantic map, got shape %s.' % (rgb.shape,))
    flat = rgb.reshape(-1, 3).astype(np.int64)
    out = np.full((flat.shape[0],), int(ignore_index), dtype=np.int64)
    unknown = np.ones((flat.shape[0],), dtype=bool)
    for color, cls_id in mapping.items():
        match = np.all(flat == np.array(color, dtype=np.int64), axis=1)
        out[match] = int(cls_id)
        unknown[match] = False
    if unknown.any() and known_colors is not None and len(known_colors) > 0:
        values = flat[unknown].astype(np.float32)
        distances = ((values[:, None, :] - known_colors[None, :, :]) ** 2).sum(axis=2)
        out[unknown] = known_classes[distances.argmin(axis=1)]
    return out.reshape(rgb.shape[0], rgb.shape[1])


class RCCDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.dataset_name = str(getattr(cfg.data, 'dataset', '')).lower()

        print('Speaker Dataset loading vocab json file: ', cfg.data.vocab_json)
        with open(cfg.data.vocab_json, 'r', encoding='utf-8') as f:
            self.word_to_idx = json.load(f)
        self.idx_to_word = {int(idx): word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.idx_to_word)
        print('vocab size is ', self.vocab_size)

        with open(cfg.data.splits_json, 'r', encoding='utf-8') as f:
            self.splits = json.load(f)
        self.idx_to_filename = self.splits['idx_to_filename']
        self.idx_to_split = self.splits['idx_to_split']

        self.d_feat_dir = cfg.data.default_feature_dir
        self.s_feat_dir = cfg.data.semantic_feature_dir
        self.d_img_dir = cfg.data.default_img_dir
        self.s_img_dir = cfg.data.semantic_img_dir
        self.default_phase = getattr(cfg.data, 'default_phase', 'A')
        self.semantic_phase = getattr(cfg.data, 'semantic_phase', 'B')
        self.pseudo_mask_root = getattr(cfg.data, 'pseudo_mask_root', '')
        self.pseudo_mask_phase = getattr(cfg.data, 'pseudo_mask_phase', '')
        self.allow_missing_pseudo_mask = _as_bool(getattr(cfg.data, 'allow_missing_pseudo_mask', False))
        self.enable_aux_mask = _as_bool(cfg.model.enable_aux_mask)
        self.mask_ignore_index = getattr(cfg.train, 'mask_ignore_index', -1)
        self.mask_type = str(getattr(cfg.data, 'mask_type', 'binary')).lower()
        self.num_mask_classes = int(getattr(cfg.data, 'num_mask_classes', getattr(cfg.model, 'num_mask_classes', 1)) or 1)
        self.require_aux_mask = self.enable_aux_mask and float(cfg.train.lambda_mask) > 0
        self.use_change_mask = _as_bool(getattr(cfg.data, 'use_change_mask', False)) or self.require_aux_mask
        self.mask_label_dir = getattr(cfg.data, 'mask_label_dir', '')

        self.semantic_input_mode = str(getattr(cfg.model, 'semantic_input_mode', 'none')).lower()
        self.use_semantic_maps = _as_bool(getattr(cfg.data, 'use_semantic_maps', False)) or self.semantic_input_mode not in ('', 'none')
        self.semantic_map_root = getattr(cfg.data, 'semantic_map_root', '') or self.s_img_dir
        self.semantic_before_phase = getattr(cfg.data, 'semantic_before_phase', '') or 'sem/A'
        self.semantic_after_phase = getattr(cfg.data, 'semantic_after_phase', '') or 'sem/B'
        self.semantic_diff_root = getattr(cfg.data, 'semantic_diff_root', '')
        self.num_semantic_classes = int(getattr(cfg.data, 'num_semantic_classes', getattr(cfg.model, 'num_semantic_classes', 0)) or 0)
        if self.dataset_name == 'second_cc' and self.use_semantic_maps and self.num_semantic_classes <= 0:
            self.num_semantic_classes = 7
        self.semantic_diff_classes = self.num_semantic_classes + 1 if self.use_semantic_maps else self.num_semantic_classes
        if self.use_semantic_maps:
            cfg.model.num_semantic_classes = max(1, self.semantic_diff_classes)
            cfg.data.num_semantic_classes = max(1, self.num_semantic_classes)

        self.train_semantic_aux = _as_bool(cfg.train.use_semantic_aux) and split == 'train'
        self.use_dense_semantic_aux = self.train_semantic_aux and (
            self.use_semantic_maps
            or str(getattr(cfg.train, 'semantic_loss_type', '')).lower() in ('ce', 'cross_entropy', 'ce_dice', 'multiclass_ce', 'dense_ce')
        )
        self.use_tag_semantic_aux = self.train_semantic_aux and not self.use_dense_semantic_aux
        self.use_relation_aux = _as_bool(cfg.train.use_relation_aux)
        self.semantic_normalize_synonyms = _as_bool(cfg.train.semantic_normalize_synonyms)
        self.semantic_relation_cache = getattr(cfg.data, 'semantic_relation_cache', '')
        self.semantic_tags = []
        self.num_semantic_tags = 0
        self.semantic_labels_by_img_idx = {}
        self.semantic_label_stats = None
        self.semantic_targets_by_img_idx = {}
        self.changeflags_by_filename = self._load_changeflags()
        self._missing_pseudo_mask_warnings = 0
        self._missing_semantic_warnings = 0

        if self.use_tag_semantic_aux:
            self.semantic_tags = read_semantic_tags(cfg.train.semantic_tag_file)
            self.num_semantic_tags = len(self.semantic_tags)
        if self.require_aux_mask and not self.pseudo_mask_root:
            print(
                'Warning: aux mask is enabled but cfg.data.pseudo_mask_root is empty. '
                'Mask supervision will be ignored for missing masks.'
            )

        if split == 'train':
            self.batch_size = cfg.data.train.batch_size
            self.seq_per_img = cfg.data.train.seq_per_img
            self.split_idxs = self.splits['train']
            self.num_samples = len(self.split_idxs)
            if cfg.data.train.max_samples is not None:
                self.num_samples = min(cfg.data.train.max_samples, self.num_samples)
        elif split == 'val':
            self.batch_size = cfg.data.val.batch_size
            self.seq_per_img = cfg.data.val.seq_per_img
            self.split_idxs = self.splits['val']
            self.num_samples = len(self.split_idxs)
            if cfg.data.val.max_samples is not None:
                self.num_samples = min(cfg.data.val.max_samples, self.num_samples)
        elif split == 'test':
            self.batch_size = cfg.data.test.batch_size
            self.seq_per_img = cfg.data.test.seq_per_img
            self.split_idxs = self.splits['test']
            self.num_samples = len(self.split_idxs)
            if cfg.data.test.max_samples is not None:
                self.num_samples = min(cfg.data.test.max_samples, self.num_samples)
        else:
            raise Exception('Unknown data split %s' % split)

        print('Dataset size for %s: %d' % (split, self.num_samples))

        with h5py.File(cfg.data.h5_label_file, 'r') as h5_label_file:
            seq_size = h5_label_file['labels'].shape
            self.labels = h5_label_file['labels'][:]
            self.max_seq_length = seq_size[1]
            self.IGNORE = -1
            self.label_start_idx = h5_label_file['label_start_idx'][:]
            self.label_end_idx = h5_label_file['label_end_idx'][:]
        print('Max sequence length is %d' % self.max_seq_length)
        self._preflight_feature_paths()

        if self.use_tag_semantic_aux:
            self.semantic_labels_by_img_idx = self._build_semantic_labels_by_img_idx()
            self.semantic_label_stats = self._summarize_semantic_labels()
            print(
                'Semantic labels enabled for %s split: %d tags from %s'
                % (split, self.num_semantic_tags, cfg.train.semantic_tag_file)
            )
            self._print_semantic_label_stats()
        if self.use_dense_semantic_aux:
            print(
                'Dense semantic auxiliary targets enabled for %s split: semantic_maps=%s classes=%d'
                % (split, self.use_semantic_maps, int(getattr(cfg.model, 'num_semantic_classes', 0)))
            )
        if self.use_relation_aux:
            self.semantic_targets_by_img_idx = self._build_relation_targets_by_img_idx()
            print(
                'Relation auxiliary targets enabled for %s split: %d objects, %d actions'
                % (split, len(OBJECT_VOCAB), len(ACTION_VOCAB))
            )

    def __len__(self):
        return self.num_samples

    def _feature_source_path(self, img_dir, split_name, phase_name, filename):
        return os.path.join(img_dir, split_name, phase_name, filename)

    def _format_feature_preflight_error(self, missing):
        lines = [
            'Feature preflight failed for dataset "%s" split "%s".'
            % (getattr(self.cfg.data, 'dataset', 'unknown'), self.split),
            'The CARD baseline expects pre-extracted ResNet .npy features before training/testing.',
        ]
        for item in missing:
            lines.extend([
                'Missing %s feature for image "%s".' % (item['role'], item['filename']),
                '  feature_root: %s' % item['feature_root'],
                '  phase: %s' % item['phase'],
                '  source_image: %s' % item['source_image'],
            ])
        lines.extend([
            'Generate features before training/testing, for example:',
            '  %s' % feature_extraction_command(self.cfg),
        ])
        return '\n'.join(lines)

    def _preflight_feature_paths(self):
        if self.num_samples <= 0:
            return
        img_idx = int(self.split_idxs[0])
        idx_key = str(img_idx)
        filename = self.idx_to_filename[idx_key]
        split_name = self.idx_to_split[idx_key]
        checks = [
            {
                'role': 'before/default',
                'feature_root': self.d_feat_dir,
                'image_root': self.d_img_dir,
                'phase': self.default_phase,
            },
            {
                'role': 'after/semantic',
                'feature_root': self.s_feat_dir,
                'image_root': self.s_img_dir,
                'phase': self.semantic_phase,
            },
        ]
        missing = []
        for item in checks:
            try:
                self._resolve_feature_path(item['feature_root'], split_name, item['phase'], filename)
            except FileNotFoundError:
                missing.append({
                    'role': item['role'],
                    'filename': filename,
                    'feature_root': item['feature_root'],
                    'phase': item['phase'],
                    'source_image': self._feature_source_path(item['image_root'], split_name, item['phase'], filename),
                })
        if missing:
            raise FileNotFoundError(self._format_feature_preflight_error(missing))

    @staticmethod
    def _resolve_feature_path(base_dir, split_name, phase_name, filename):
        stem = os.path.splitext(filename)[0]
        candidates = [
            os.path.join(base_dir, split_name, phase_name, filename + '.npy'),
            os.path.join(base_dir, split_name, phase_name, stem + '.npy'),
            os.path.join(base_dir, phase_name, split_name, filename + '.npy'),
            os.path.join(base_dir, phase_name, split_name, stem + '.npy'),
            os.path.join(base_dir, split_name, filename + '.npy'),
            os.path.join(base_dir, split_name, stem + '.npy'),
            os.path.join(base_dir, filename + '.npy'),
            os.path.join(base_dir, stem + '.npy'),
        ]
        found = _first_existing(candidates)
        if found:
            return found
        raise FileNotFoundError(
            'Cannot find feature for "%s" under "%s" and split "%s".' % (filename, base_dir, split_name)
        )

    def _load_changeflags(self):
        paths = []
        for attr in ('changeflag_json', 'caption_json'):
            path = getattr(self.cfg.data, attr, '')
            if path and path not in paths:
                paths.append(path)
        result = {}
        for path in paths:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
            except Exception as exc:
                print('Warning: could not load changeflag JSON %s: %s' % (path, exc))
                continue
            if isinstance(payload, dict) and 'images' in payload:
                for item in payload['images']:
                    filename = item.get('filename')
                    if filename is not None and 'changeflag' in item:
                        result[filename] = int(item.get('changeflag'))
        return result

    def _resolve_pseudo_mask_path(self, split_name, filename):
        roots_and_phases = []
        if self.pseudo_mask_root:
            roots_and_phases.append((self.pseudo_mask_root, self.pseudo_mask_phase))
        if self.mask_label_dir:
            roots_and_phases.append((self.mask_label_dir, 'label'))
        if self.dataset_name == 'levir_mci':
            roots_and_phases.append((self.d_img_dir, 'label'))
        for root, phase in roots_and_phases:
            found = _first_existing(_candidate_paths(root, split_name, phase, filename))
            if found:
                return found
        if not self.allow_missing_pseudo_mask and self._missing_pseudo_mask_warnings < 5:
            print(
                'Warning: cannot find mask for "%s" under "%s" split "%s". '
                'This sample will be ignored for mask loss.' % (filename, self.pseudo_mask_root or self.mask_label_dir, split_name)
            )
            self._missing_pseudo_mask_warnings += 1
        return None

    def _load_mask_class(self, mask_path):
        arr = _load_image_or_array(mask_path)
        if arr is None:
            return None
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            if np.all(arr[..., 0] == arr[..., 1]) and np.all(arr[..., 1] == arr[..., 2]):
                arr = arr[..., 0]
            else:
                return _rgb_to_class_map(arr[..., :3], LEVIR_LABEL_RGB_TO_CLASS, ignore_index=self.mask_ignore_index)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = arr.astype(np.int64)
        out = np.full(arr.shape, int(self.mask_ignore_index), dtype=np.int64)
        for value, cls_id in LEVIR_LABEL_VALUE_TO_CLASS.items():
            out[arr == value] = cls_id
        unknown = out == int(self.mask_ignore_index)
        if unknown.any():
            out[unknown] = (arr[unknown] > 0).astype(np.int64)
        return out

    def _load_mask(self, mask_path):
        mask_class = self._load_mask_class(mask_path)
        if mask_class is None:
            return None
        if self.mask_type == 'multiclass' or self.num_mask_classes > 1:
            return torch.from_numpy(mask_class.astype(np.int64))
        binary = (mask_class > 0).astype(np.float32)
        binary[mask_class == int(self.mask_ignore_index)] = float(self.mask_ignore_index)
        return torch.from_numpy(binary).unsqueeze(0)

    def _resolve_semantic_path(self, split_name, phase_name, filename):
        roots = [self.semantic_map_root, self.s_img_dir]
        for root in roots:
            found = _first_existing(_candidate_paths(root, split_name, phase_name, filename))
            if found:
                return found
        if self._missing_semantic_warnings < 5:
            print(
                'Warning: cannot find semantic map for "%s" phase "%s" split "%s".'
                % (filename, phase_name, split_name)
            )
            self._missing_semantic_warnings += 1
        return None

    def _load_semantic_map(self, path):
        arr = _load_image_or_array(path)
        if arr is None:
            return None
        if self.dataset_name == 'second_cc':
            class_map = _rgb_to_class_map(
                arr,
                SECOND_SEMANTIC_RGB_TO_CLASS,
                ignore_index=getattr(self.cfg.train, 'semantic_ignore_index', -1),
                known_colors=SECOND_KNOWN_COLORS,
                known_classes=SECOND_KNOWN_CLASSES,
            )
        else:
            class_map = self._load_mask_class(path)
        return torch.from_numpy(class_map.astype(np.int64))

    def _load_semantic_pair(self, split_name, filename):
        if not self.use_semantic_maps:
            return None, None, None
        before_path = self._resolve_semantic_path(split_name, self.semantic_before_phase, filename)
        after_path = self._resolve_semantic_path(split_name, self.semantic_after_phase, filename)
        before = self._load_semantic_map(before_path) if before_path else None
        after = self._load_semantic_map(after_path) if after_path else None
        if before is None or after is None:
            return before, after, None
        diff_path = _first_existing(_candidate_paths(self.semantic_diff_root, split_name, '', filename)) if self.semantic_diff_root else None
        if diff_path:
            diff = self._load_semantic_map(diff_path)
        else:
            valid = before != int(getattr(self.cfg.train, 'semantic_ignore_index', -1))
            valid = valid & (after != int(getattr(self.cfg.train, 'semantic_ignore_index', -1)))
            changed = (before != after) & valid
            diff = torch.zeros_like(after, dtype=torch.long)
            diff[changed] = after[changed].long() + 1
            diff[~valid] = int(getattr(self.cfg.train, 'semantic_ignore_index', -1))
        return before, after, diff

    def _decode_reference_caption(self, token_ids):
        words = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == 0 or token_id == 3:
                break
            word = self.idx_to_word.get(token_id)
            if word is None:
                continue
            if word.startswith('<') and word.endswith('>'):
                continue
            words.append(word)
        return ' '.join(words)

    def _get_reference_captions(self, img_idx):
        ix1 = int(self.label_start_idx[img_idx])
        ix2 = int(self.label_end_idx[img_idx])
        return [
            self._decode_reference_caption(self.labels[ix, :self.max_seq_length])
            for ix in range(ix1, ix2 + 1)
        ]

    def _build_semantic_labels_by_img_idx(self):
        labels_by_img_idx = {}
        for img_idx in self.split_idxs[:self.num_samples]:
            img_idx = int(img_idx)
            captions = self._get_reference_captions(img_idx)
            semantic_label = build_semantic_label(
                captions,
                self.semantic_tags,
                normalize_synonyms=self.semantic_normalize_synonyms,
            )
            labels_by_img_idx[img_idx] = torch.from_numpy(semantic_label)
        return labels_by_img_idx

    def _summarize_semantic_labels(self):
        if not self.semantic_labels_by_img_idx:
            return {
                'total_samples': 0,
                'tag_counts': np.zeros(self.num_semantic_tags, dtype=np.float32),
                'all_zero_samples': 0,
                'avg_positive_tags': 0.0,
            }
        label_matrix = torch.stack([label.float() for label in self.semantic_labels_by_img_idx.values()], dim=0)
        positive_per_sample = label_matrix.sum(dim=1)
        return {
            'total_samples': int(label_matrix.size(0)),
            'tag_counts': label_matrix.sum(dim=0).cpu().numpy(),
            'all_zero_samples': int((positive_per_sample == 0).sum().item()),
            'avg_positive_tags': float(positive_per_sample.mean().item()),
        }

    def _print_semantic_label_stats(self):
        if self.semantic_label_stats is None:
            return
        total_samples = self.semantic_label_stats['total_samples']
        tag_counts = self.semantic_label_stats['tag_counts']
        all_zero_samples = self.semantic_label_stats['all_zero_samples']
        avg_positive_tags = self.semantic_label_stats['avg_positive_tags']
        print('Semantic label stats for %s split:' % self.split)
        print('  samples: %d' % total_samples)
        print('  all_zero_samples: %d' % all_zero_samples)
        print('  avg_positive_tags_per_sample: %.4f' % avg_positive_tags)
        for tag, count in zip(self.semantic_tags, tag_counts):
            print('  semantic_tag_count %s: %.0f' % (tag, float(count)))
        if total_samples > 0 and float(np.sum(tag_counts)) == 0.0:
            print('WARNING: all semantic labels are zero. Check semantic tag rules and captions.')
        elif total_samples > 0 and avg_positive_tags < 0.25:
            print('WARNING: semantic labels are very sparse. Inspect tag rules before long training.')

    def _build_relation_targets_by_img_idx(self):
        cached_targets = {}
        if self.semantic_relation_cache and os.path.exists(self.semantic_relation_cache):
            cached_targets = load_semantic_target_cache(self.semantic_relation_cache)
        targets_by_img_idx = {}
        cache_updated = False
        for img_idx in self.split_idxs[:self.num_samples]:
            img_idx = int(img_idx)
            cache_key = str(img_idx)
            if cache_key in cached_targets:
                semantic_targets = cached_targets[cache_key]
            else:
                captions = self._get_reference_captions(img_idx)
                semantic_targets = build_semantic_targets(captions)
                cached_targets[cache_key] = semantic_targets
                cache_updated = True
            targets_by_img_idx[img_idx] = semantic_targets_to_tensors(semantic_targets)
        if self.semantic_relation_cache and cache_updated:
            save_semantic_target_cache(self.semantic_relation_cache, cached_targets)
        return targets_by_img_idx

    def __getitem__(self, index):
        img_idx = int(self.split_idxs[index])
        idx_key = str(img_idx)
        filename = self.idx_to_filename[idx_key]
        split_name = self.idx_to_split[idx_key]

        d_feat_path = self._resolve_feature_path(self.d_feat_dir, split_name, self.default_phase, filename)
        q_feat_path = self._resolve_feature_path(self.s_feat_dir, split_name, self.semantic_phase, filename)

        d_img_path = os.path.join(self.d_img_dir, split_name, self.default_phase, filename)
        q_img_path = os.path.join(self.s_img_dir, split_name, self.semantic_phase, filename)

        d_feature = torch.FloatTensor(np.load(d_feat_path))
        q_feature = torch.FloatTensor(np.load(q_feat_path))

        pseudo_mask = None
        if self.use_change_mask:
            pseudo_mask_path = self._resolve_pseudo_mask_path(split_name, filename)
            if pseudo_mask_path is not None:
                pseudo_mask = self._load_mask(pseudo_mask_path)
            elif self.require_aux_mask:
                if self.mask_type == 'multiclass' or self.num_mask_classes > 1:
                    pseudo_mask = torch.full((d_feature.shape[-2], d_feature.shape[-1]), int(self.mask_ignore_index), dtype=torch.long)
                else:
                    pseudo_mask = torch.full((1, d_feature.shape[-2], d_feature.shape[-1]), float(self.mask_ignore_index), dtype=torch.float32)

        semantic_before, semantic_after, semantic_diff = self._load_semantic_pair(split_name, filename)
        semantic_dense = None
        if self.use_dense_semantic_aux:
            if self.use_semantic_maps:
                semantic_dense = semantic_diff
            elif pseudo_mask is not None and (self.mask_type == 'multiclass' or self.num_mask_classes > 1):
                semantic_dense = pseudo_mask.long()

        semantic_labels = None
        if self.use_tag_semantic_aux:
            semantic_labels = self.semantic_labels_by_img_idx.get(img_idx)
        semantic_targets = None
        if self.use_relation_aux:
            semantic_targets = self.semantic_targets_by_img_idx.get(img_idx)
            if semantic_targets is None:
                raise ValueError('Missing relation semantic targets for image index %d.' % img_idx)

        ix1 = int(self.label_start_idx[img_idx])
        ix2 = int(self.label_end_idx[img_idx])
        n_cap = ix2 - ix1 + 1

        seq = np.zeros([self.seq_per_img, self.max_seq_length], dtype=int)
        if n_cap < self.seq_per_img:
            for q in range(self.seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :self.max_seq_length] = self.labels[ixl, :self.max_seq_length]
        else:
            ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
            seq[:, :self.max_seq_length] = self.labels[ixl: ixl + self.seq_per_img, :self.max_seq_length]

        mask = np.zeros_like(seq)
        nonzeros = np.array(list(map(lambda x: (x != 0).sum(), seq)))
        for ix, row in enumerate(mask):
            row[:nonzeros[ix]] = 1
        if seq.size == self.max_seq_length:
            labels_with_ignore_tolish = [
                self.IGNORE if m == 0 else tid
                for tid, m in zip(seq.squeeze(0).tolist(), mask.squeeze(0).tolist())
            ][1:] + [self.IGNORE]
            labels_with_ignore = np.array(labels_with_ignore_tolish)
            labels_with_ignore = np.expand_dims(labels_with_ignore, 0)
        else:
            labels_with_ignore = np.zeros_like(seq)

        changeflag = self.changeflags_by_filename.get(filename)
        if changeflag is None:
            if pseudo_mask is not None:
                valid = pseudo_mask != float(self.mask_ignore_index)
                changeflag = int((pseudo_mask[valid] > 0).any().item()) if valid.any() else -1
            elif semantic_diff is not None:
                changeflag = int((semantic_diff > 0).any().item())
            else:
                changeflag = -1

        if self.return_dict:
            mask_class = None
            if pseudo_mask is not None and pseudo_mask.dim() == 2:
                mask_class = pseudo_mask
            return {
                'feature_before': d_feature,
                'feature_after': q_feature,
                'caption_tokens': seq,
                'labels_with_ignore': labels_with_ignore,
                'caption_mask': mask,
                'image_before': d_img_path,
                'image_after': q_img_path,
                'mask': pseudo_mask,
                'mask_class': mask_class,
                'semantic_labels': semantic_labels,
                'semantic_targets': semantic_targets,
                'semantic_dense': semantic_dense,
                'semantic_before': semantic_before,
                'semantic_after': semantic_after,
                'semantic_diff': semantic_diff,
                'changeflag': int(changeflag),
                'image_id': filename,
                'split': split_name,
                'has_mask': pseudo_mask is not None,
                'has_semantic': semantic_diff is not None,
            }

        return (
            d_feature,
            q_feature,
            seq,
            labels_with_ignore,
            mask,
            d_img_path,
            q_img_path,
            pseudo_mask,
            semantic_labels,
            semantic_targets,
            semantic_dense,
            semantic_before,
            semantic_after,
            semantic_diff,
            int(changeflag),
            filename,
        )

    def get_vocab_size(self):
        return self.vocab_size

    def get_idx_to_word(self):
        return self.idx_to_word

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_max_seq_length(self):
        return self.max_seq_length

    def get_num_semantic_tags(self):
        return self.num_semantic_tags

    def get_semantic_tags(self):
        return self.semantic_tags

    def get_semantic_label_stats(self):
        return self.semantic_label_stats

    def get_num_relation_objects(self):
        return len(OBJECT_VOCAB)

    def get_num_relation_actions(self):
        return len(ACTION_VOCAB)


def _collate_optional(values):
    values = list(values)
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        return None
    first = values[0]
    if isinstance(first, str):
        return tuple(values)
    if isinstance(first, dict):
        return default_collate(values)
    if isinstance(first, (int, np.integer)):
        return torch.tensor(values, dtype=torch.long)
    return default_collate(values)


def rcc_collate(batch):
    if batch and isinstance(batch[0], dict):
        keys = batch[0].keys()
        return {key: _collate_optional([item.get(key) for item in batch]) for key in keys}
    transposed = list(zip(*batch))
    collated = []
    for index, values in enumerate(transposed):
        if index in (5, 6, 15):
            collated.append(tuple(values))
        else:
            collated.append(_collate_optional(values))
    return tuple(collated)


class RCCDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        kwargs['collate_fn'] = rcc_collate
        super().__init__(dataset, **kwargs)
