import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config_transformer import cfg, merge_cfg_from_file, merge_cfg_from_list
from datasets.datasets import create_dataset
from models.CARD import CARD
from models.transformer_decoder import DynamicSpeaker
from utils.semantic_tags import build_semantic_label


def unpack_change_detector_output(outputs):
    if len(outputs) == 6:
        encoder_output, con_loss, ind_loss, att1, att2, mask_pred = outputs
        semantic_logits = None
    elif len(outputs) == 7:
        encoder_output, con_loss, ind_loss, att1, att2, mask_pred, semantic_logits = outputs
    elif len(outputs) == 8:
        encoder_output, con_loss, ind_loss, att1, att2, mask_pred, semantic_logits, _ = outputs
    else:
        raise RuntimeError('Unexpected CARD output size: %d' % len(outputs))
    return encoder_output, con_loss, ind_loss, att1, att2, mask_pred, semantic_logits


def apply_semantic_debug_overrides():
    cfg.train.use_semantic_aux = True
    cfg.train.use_relation_aux = False
    cfg.train.use_content_word_weight = False
    if hasattr(cfg.train, 'use_content_word_weighted_ce'):
        cfg.train.use_content_word_weighted_ce = False
    cfg.train.content_word_token_ids = []


def check_labels(max_samples):
    dataset, _ = create_dataset(cfg, 'train')
    if dataset.get_num_semantic_tags() <= 0:
        raise RuntimeError('No semantic tags were loaded.')
    if not getattr(dataset, 'semantic_labels_by_img_idx', None):
        raise RuntimeError('No semantic labels were built. Check cfg.train.use_semantic_aux.')

    stats = dataset.get_semantic_label_stats()
    print(
        'semantic_label_summary samples=%d all_zero=%d avg_positive=%.4f'
        % (stats['total_samples'], stats['all_zero_samples'], stats['avg_positive_tags'])
    )
    for tag, count in zip(dataset.get_semantic_tags(), stats['tag_counts']):
        print('semantic_tag_count %s %.0f' % (tag, float(count)))

    inspected = min(max_samples, len(dataset.split_idxs))
    for raw_img_idx in dataset.split_idxs[:inspected]:
        img_idx = int(raw_img_idx)
        label = dataset.semantic_labels_by_img_idx[img_idx]
        positives = [
            tag for tag, value in zip(dataset.get_semantic_tags(), label.tolist())
            if float(value) > 0.5
        ]
        filename = dataset.idx_to_filename[str(img_idx)]
        print(
            'semantic_label_sample img_idx=%d file=%s positive_count=%d positives=%s'
            % (img_idx, filename, len(positives), ','.join(positives) if positives else 'all_zero')
        )


def make_synthetic_caption_batch(word_to_idx, batch_size, seq_length, device):
    start_id = int(word_to_idx.get('<START>', 2))
    end_id = int(word_to_idx.get('<END>', 3))
    words = ['new', 'buildings', 'are', 'constructed', 'beside', 'the', 'road']
    token_ids = [start_id] + [int(word_to_idx[w]) for w in words if w in word_to_idx] + [end_id]
    token_ids = token_ids[:seq_length]

    seq = torch.zeros(batch_size, seq_length, dtype=torch.long, device=device)
    seq[:, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long, device=device)
    dec_mask = seq.ne(0).float()

    labels_with_ignore = torch.full_like(seq, -1)
    labels_with_ignore[:, :-1] = seq[:, 1:].clone()
    labels_with_ignore[labels_with_ignore == 0] = -1
    return seq, dec_mask, labels_with_ignore


def synthetic_forward(batch_size, use_cpu):
    train_dataset, _ = create_dataset(cfg, 'train')
    cfg.model.transformer_decoder.vocab_size = train_dataset.get_vocab_size()
    cfg.model.transformer_decoder.seq_length = train_dataset.get_max_seq_length()

    device = torch.device('cpu' if use_cpu or not torch.cuda.is_available() else 'cuda')
    change_detector = CARD(cfg).to(device)
    speaker = DynamicSpeaker(cfg).to(device)
    change_detector.train()
    speaker.train()

    feat_dim = cfg.model.transformer_encoder.feat_dim
    before_feats = torch.randn(batch_size, feat_dim, 14, 14, device=device)
    after_feats = torch.randn(batch_size, feat_dim, 14, 14, device=device)
    seq, dec_mask, labels_with_ignore = make_synthetic_caption_batch(
        train_dataset.get_word_to_idx(),
        batch_size,
        cfg.model.transformer_decoder.seq_length,
        device,
    )

    semantic_label = build_semantic_label(
        ['new buildings are constructed beside the road'],
        train_dataset.get_semantic_tags(),
        normalize_synonyms=cfg.train.semantic_normalize_synonyms,
    )
    semantic_labels = torch.as_tensor(
        np.stack([semantic_label] * batch_size),
        dtype=torch.float32,
        device=device,
    )

    outputs = change_detector(before_feats, after_feats)
    encoder_output, con_loss, ind_loss, _, _, _, semantic_logits = unpack_change_detector_output(outputs)
    if semantic_logits is None:
        raise RuntimeError('CARD did not return semantic_logits.')
    if semantic_logits.shape != semantic_labels.shape:
        raise RuntimeError(
            'semantic_logits shape %s does not match semantic_labels shape %s.'
            % (tuple(semantic_logits.shape), tuple(semantic_labels.shape))
        )

    caption_loss, prediction_scores, _ = speaker._forward(
        encoder_output,
        seq,
        dec_mask,
        labels_with_ignore=labels_with_ignore,
    )
    semantic_loss = nn.BCEWithLogitsLoss()(semantic_logits, semantic_labels)
    total_loss = caption_loss + 0.001 * con_loss + 0.001 * ind_loss + cfg.train.lambda_semantic * semantic_loss
    total_loss.backward()

    print('synthetic_forward device=%s' % device)
    print('encoder_output_shape=%s' % (tuple(encoder_output.shape),))
    print('prediction_scores_shape=%s' % (tuple(prediction_scores.shape),))
    print('semantic_logits_shape=%s semantic_labels_shape=%s' % (
        tuple(semantic_logits.shape),
        tuple(semantic_labels.shape),
    ))
    print(
        'losses caption=%.6f semantic=%.6f total=%.6f finite=%s'
        % (
            caption_loss.item(),
            semantic_loss.item(),
            total_loss.item(),
            bool(torch.isfinite(total_loss).item()),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--check_labels', action='store_true')
    parser.add_argument('--synthetic_forward', action='store_true')
    parser.add_argument('--max_samples', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    merge_cfg_from_file(args.cfg)
    if args.opts:
        merge_cfg_from_list(args.opts)
    apply_semantic_debug_overrides()

    if not args.check_labels and not args.synthetic_forward:
        args.check_labels = True

    if args.check_labels:
        check_labels(args.max_samples)
    if args.synthetic_forward:
        synthetic_forward(args.batch_size, args.cpu)


if __name__ == '__main__':
    main()
