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
        self.use_semantic_aux = bool(cfg.train.use_semantic_aux) and split == 'train'
        self.semantic_normalize_synonyms = bool(cfg.train.semantic_normalize_synonyms)
        self.semantic_tags = []
        self.num_semantic_tags = 0
        self.semantic_labels_by_img_idx = {}

        if self.use_semantic_aux:
            self.semantic_tags = read_semantic_tags(cfg.train.semantic_tag_file)
            self.num_semantic_tags = len(self.semantic_tags)

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
            print(
                'Semantic labels enabled for %s split: %d tags from %s'
                % (split, self.num_semantic_tags, cfg.train.semantic_tag_file)
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

    @staticmethod
    def _load_mask(mask_path):
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path)
        else:
            mask = imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
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

    def _resolve_pseudo_mask_path(self, split_name, filename):
        if not self.pseudo_mask_root:
            if self.enable_aux_mask:
                raise ValueError('cfg.data.pseudo_mask_root must be set when aux mask is enabled.')
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

        if self.allow_missing_pseudo_mask:
            return None
        raise FileNotFoundError(
            f'Cannot find pseudo mask for "{filename}" under "{self.pseudo_mask_root}" and split "{split_name}".'
        )

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
        if self.enable_aux_mask:
            pseudo_mask_path = self._resolve_pseudo_mask_path(split_name, filename)
            if pseudo_mask_path is not None:
                pseudo_mask = self._load_mask(pseudo_mask_path)
        semantic_labels = None
        if self.use_semantic_aux:
            semantic_labels = self.semantic_labels_by_img_idx.get(img_idx)
            if semantic_labels is None:
                raise ValueError('Missing semantic labels for image index %d.' % img_idx)

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
            return sample + (semantic_labels,)
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
    if len(transposed) > 8:
        semantic_label_batch = default_collate(transposed[8])
        return output + (semantic_label_batch,)
    return output


class RCCDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        kwargs['collate_fn'] = rcc_collate
        super().__init__(dataset, **kwargs)
