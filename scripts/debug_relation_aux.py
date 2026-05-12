import argparse
import os
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config_transformer import cfg, merge_cfg_from_file, merge_cfg_from_list
from datasets.datasets import create_dataset
from models.CARD import CARD
from models.transformer_decoder import DynamicSpeaker
from utils.semantic_label import build_content_word_token_ids, build_semantic_targets, semantic_targets_to_tensors


def apply_cli_overrides(args):
    if args.check_labels or args.synthetic_forward or args.use_relation_aux:
        cfg.train.use_relation_aux = True
    if args.use_content_word_weight:
        cfg.train.use_content_word_weight = True
    if args.content_word_weight is not None:
        cfg.train.content_word_weight = args.content_word_weight
    if args.max_content_word_weight is not None:
        cfg.train.max_content_word_weight = args.max_content_word_weight
    if args.relation_aux_dropout is not None:
        cfg.train.relation_aux_dropout = args.relation_aux_dropout


def check_labels(max_samples):
    dataset, _ = create_dataset(cfg, 'train')
    if not getattr(dataset, 'semantic_targets_by_img_idx', None):
        raise RuntimeError('No semantic relation targets were built. Check cfg.train.use_relation_aux.')

    total_objects = 0.0
    total_actions = 0.0
    total_relations = 0.0
    inspected = min(max_samples, len(dataset.split_idxs))
    for raw_img_idx in dataset.split_idxs[:inspected]:
        img_idx = int(raw_img_idx)
        targets = dataset.semantic_targets_by_img_idx[img_idx]
        object_count = targets['objects'].sum().item()
        action_count = targets['actions'].sum().item()
        relation_count = targets['relations'].sum().item()
        total_objects += object_count
        total_actions += action_count
        total_relations += relation_count
        filename = dataset.idx_to_filename[str(img_idx)]
        print(
            'label_check img_idx=%d file=%s objects=%.0f actions=%.0f relations=%.0f shapes=%s/%s/%s'
            % (
                img_idx,
                filename,
                object_count,
                action_count,
                relation_count,
                tuple(targets['objects'].shape),
                tuple(targets['actions'].shape),
                tuple(targets['relations'].shape),
            )
        )

    print(
        'label_check_summary samples=%d objects=%.0f actions=%.0f relations=%.0f'
        % (inspected, total_objects, total_actions, total_relations)
    )


def make_synthetic_caption_batch(word_to_idx, batch_size, seq_length, device):
    start_id = int(word_to_idx.get('<START>', 2))
    end_id = int(word_to_idx.get('<END>', 3))
    words = ['a', 'building', 'was', 'added', 'and', 'trees', 'were', 'removed']
    token_ids = [start_id] + [int(word_to_idx[w]) for w in words if w in word_to_idx] + [end_id]
    token_ids = token_ids[:seq_length]

    seq = torch.zeros(batch_size, seq_length, dtype=torch.long, device=device)
    seq[:, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long, device=device)
    dec_mask = seq.ne(0).float()

    labels_with_ignore = torch.full_like(seq, -1)
    shifted = seq[:, 1:].clone()
    labels_with_ignore[:, :-1] = shifted
    labels_with_ignore[labels_with_ignore == 0] = -1
    return seq, dec_mask, labels_with_ignore


def synthetic_forward(batch_size, use_cpu):
    train_dataset, _ = create_dataset(cfg, 'train')
    cfg.model.transformer_decoder.vocab_size = train_dataset.get_vocab_size()
    cfg.model.transformer_decoder.seq_length = train_dataset.get_max_seq_length()
    if cfg.train.use_content_word_weight:
        cfg.train.content_word_token_ids = build_content_word_token_ids(train_dataset.get_word_to_idx())

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

    semantic_targets = semantic_targets_to_tensors(
        build_semantic_targets(['a building was added and trees were removed'])
    )
    semantic_targets = {
        key: value.unsqueeze(0).repeat(batch_size, *([1] * value.dim())).to(device)
        for key, value in semantic_targets.items()
    }

    outputs = change_detector(before_feats, after_feats)
    if len(outputs) != 8:
        raise RuntimeError('Expected CARD to return 8 outputs with relation aux enabled, got %d.' % len(outputs))
    encoder_output, con_loss, ind_loss, _, _, _, _, relation_aux_logits = outputs
    caption_loss, prediction_scores, _ = speaker._forward(
        encoder_output,
        seq,
        dec_mask,
        labels_with_ignore=labels_with_ignore,
    )

    bce = nn.BCEWithLogitsLoss()
    object_loss = bce(relation_aux_logits['objects'], semantic_targets['objects'])
    action_loss = bce(relation_aux_logits['actions'], semantic_targets['actions'])
    relation_loss = bce(relation_aux_logits['relations'], semantic_targets['relations'])
    total_loss = caption_loss + 0.001 * con_loss + 0.001 * ind_loss + object_loss + action_loss + relation_loss
    total_loss.backward()

    print('synthetic_forward device=%s' % device)
    print('encoder_output_shape=%s' % (tuple(encoder_output.shape),))
    print('prediction_scores_shape=%s' % (tuple(prediction_scores.shape),))
    print('object_logits_shape=%s target_shape=%s' % (tuple(relation_aux_logits['objects'].shape), tuple(semantic_targets['objects'].shape)))
    print('action_logits_shape=%s target_shape=%s' % (tuple(relation_aux_logits['actions'].shape), tuple(semantic_targets['actions'].shape)))
    print('relation_logits_shape=%s target_shape=%s' % (tuple(relation_aux_logits['relations'].shape), tuple(semantic_targets['relations'].shape)))
    print(
        'losses caption=%.6f object=%.6f action=%.6f relation=%.6f total=%.6f finite=%s'
        % (
            caption_loss.item(),
            object_loss.item(),
            action_loss.item(),
            relation_loss.item(),
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
    parser.add_argument('--use_relation_aux', action='store_true')
    parser.add_argument('--use_content_word_weight', action='store_true')
    parser.add_argument('--content_word_weight', type=float, default=None)
    parser.add_argument('--max_content_word_weight', type=float, default=None)
    parser.add_argument('--relation_aux_dropout', type=float, default=None)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    merge_cfg_from_file(args.cfg)
    if args.opts:
        merge_cfg_from_list(args.opts)
    apply_cli_overrides(args)

    if not args.check_labels and not args.synthetic_forward:
        args.check_labels = True

    if args.check_labels:
        check_labels(args.max_samples)
    if args.synthetic_forward:
        synthetic_forward(args.batch_size, args.cpu)


if __name__ == '__main__':
    main()
