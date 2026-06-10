import json
import os
import random

import h5py
import numpy as np
import torch
from imageio.v2 import imread
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from utils.semantic_tags import build_semantic_label, read_semantic_tags
from utils.semantic_label import (
    ACTION_VOCAB,
    OBJECT_VOCAB,
    build_semantic_targets,
    load_semantic_target_cache,
    save_semantic_target_cache,
    semantic_targets_to_tensors,
)


class RCCDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split

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
        self.pseudo_mask_root = cfg.data.pseudo_mask_root
        self.allow_missing_pseudo_mask = cfg.data.allow_missing_pseudo_mask
        self.enable_aux_mask = cfg.model.enable_aux_mask
        self.require_aux_mask = self.enable_aux_mask and cfg.train.lambda_mask > 0
        self.mask_ignore_index = getattr(cfg.train, 'mask_ignore_index', -1)
        self.use_semantic_aux = bool(cfg.train.use_semantic_aux) and split == 'train'
        self.use_relation_aux = bool(cfg.train.use_relation_aux)
        self.semantic_normalize_synonyms = bool(cfg.train.semantic_normalize_synonyms)
        self.semantic_relation_cache = getattr(cfg.data, 'semantic_relation_cache', '')
        self.semantic_tags = []
        self.num_semantic_tags = 0
        self.semantic_labels_by_img_idx = {}
        self.semantic_label_stats = None
        self.semantic_targets_by_img_idx = {}
        self._missing_pseudo_mask_warnings = 0

        if self.use_semantic_aux:
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

        print("Dataset size for %s: %d" % (split, self.num_samples))

        with h5py.File(cfg.data.h5_label_file, 'r') as h5_label_file:
            seq_size = h5_label_file['labels'].shape
            self.labels = h5_label_file['labels'][:]
            self.max_seq_length = seq_size[1]
            self.IGNORE = -1
            self.label_start_idx = h5_label_file['label_start_idx'][:]
            self.label_end_idx = h5_label_file['label_end_idx'][:]
        print('Max sequence length is %d' % self.max_seq_length)

        if self.use_semantic_aux:
            self.semantic_labels_by_img_idx = self._build_semantic_labels_by_img_idx()
            self.semantic_label_stats = self._summarize_semantic_labels()
            print(
                'Semantic labels enabled for %s split: %d tags from %s'
                % (split, self.num_semantic_tags, cfg.train.semantic_tag_file)
            )
            self._print_semantic_label_stats()
        if self.use_relation_aux:
            self.semantic_targets_by_img_idx = self._build_relation_targets_by_img_idx()
            print(
                'Relation auxiliary targets enabled for %s split: %d objects, %d actions'
                % (split, len(OBJECT_VOCAB), len(ACTION_VOCAB))
            )

    def __len__(self):
        return self.num_samples

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
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f'Cannot find feature for "{filename}" under "{base_dir}" and split "{split_name}".'
        )

    def _load_mask(self, mask_path):
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path)
        else:
            mask = imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            ignore_mask = mask == float(self.mask_ignore_index)
            mask = np.where(ignore_mask, mask, mask / 255.0)
        return torch.from_numpy(mask).unsqueeze(0)

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
        label_matrix = torch.stack(
            [label.float() for label in self.semantic_labels_by_img_idx.values()],
            dim=0,
        )
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

    def _resolve_pseudo_mask_path(self, split_name, filename):
        if not self.pseudo_mask_root:
            return None

        stem = os.path.splitext(filename)[0]
        candidates = [
            os.path.join(self.pseudo_mask_root, split_name, filename),
            os.path.join(self.pseudo_mask_root, split_name, stem + '.png'),
            os.path.join(self.pseudo_mask_root, split_name, stem + '.jpg'),
            os.path.join(self.pseudo_mask_root, split_name, stem + '.jpeg'),
            os.path.join(self.pseudo_mask_root, split_name, stem + '.npy'),
            os.path.join(self.pseudo_mask_root, filename),
            os.path.join(self.pseudo_mask_root, stem + '.png'),
            os.path.join(self.pseudo_mask_root, stem + '.jpg'),
            os.path.join(self.pseudo_mask_root, stem + '.jpeg'),
            os.path.join(self.pseudo_mask_root, stem + '.npy'),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

        if not self.allow_missing_pseudo_mask and self._missing_pseudo_mask_warnings < 5:
            print(
                f'Warning: cannot find pseudo mask for "{filename}" under "{self.pseudo_mask_root}" '
                f'and split "{split_name}". This sample will be ignored for mask loss.'
            )
            self._missing_pseudo_mask_warnings += 1
        return None

    def __getitem__(self, index):
        img_idx = int(self.split_idxs[index])
        idx_key = str(img_idx)
        filename = self.idx_to_filename[idx_key]
        split_name = self.idx_to_split[idx_key]

        d_feat_path = self._resolve_feature_path(
            self.d_feat_dir, split_name, self.default_phase, filename
        )
        q_feat_path = self._resolve_feature_path(
            self.s_feat_dir, split_name, self.semantic_phase, filename
        )

        d_img_path = os.path.join(self.d_img_dir, split_name, self.default_phase, filename)
        q_img_path = os.path.join(self.s_img_dir, split_name, self.semantic_phase, filename)

        d_feature = torch.FloatTensor(np.load(d_feat_path))
        q_feature = torch.FloatTensor(np.load(q_feat_path))
        pseudo_mask = None
        if self.require_aux_mask:
            pseudo_mask_path = self._resolve_pseudo_mask_path(split_name, filename)
            if pseudo_mask_path is not None:
                pseudo_mask = self._load_mask(pseudo_mask_path)
            else:
                pseudo_mask = torch.full(
                    (1, d_feature.shape[-2], d_feature.shape[-1]),
                    float(self.mask_ignore_index),
                    dtype=torch.float32,
                )
        semantic_labels = None
        if self.use_semantic_aux:
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

        sample = (
            d_feature,
            q_feature,
            seq,
            labels_with_ignore,
            mask,
            d_img_path,
            q_img_path,
            pseudo_mask,
        )
        if self.use_semantic_aux:
            sample = sample + (semantic_labels,)
        if self.use_relation_aux:
            sample = sample + (semantic_targets,)
        return sample

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


def rcc_collate(batch):
    transposed = list(zip(*batch))
    d_feat_batch = transposed[0]
    q_feat_batch = transposed[1]
    seq_batch = default_collate(transposed[2])
    label_with_ignore_batch = default_collate(transposed[3])
    mask_batch = default_collate(transposed[4])

    if any(f is not None for f in d_feat_batch):
        d_feat_batch = default_collate(d_feat_batch)
    if any(f is not None for f in q_feat_batch):
        q_feat_batch = default_collate(q_feat_batch)

    d_img_batch = transposed[5]
    q_img_batch = transposed[6]
    pseudo_mask_batch = None
    if len(transposed) > 7 and any(m is not None for m in transposed[7]):
        pseudo_mask_batch = default_collate(transposed[7])
    output = (
        d_feat_batch,
        q_feat_batch,
        seq_batch,
        label_with_ignore_batch,
        mask_batch,
        d_img_batch,
        q_img_batch,
        pseudo_mask_batch,
    )
    extra_batches = []
    if len(transposed) > 8:
        extra_batches.append(default_collate(transposed[8]) if all(x is not None for x in transposed[8]) else None)
    if len(transposed) > 9:
        extra_batches.append(default_collate(transposed[9]) if all(x is not None for x in transposed[9]) else None)
    if extra_batches:
        return output + tuple(extra_batches)
    return output


class RCCDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        kwargs['collate_fn'] = rcc_collate
        super().__init__(dataset, **kwargs)
