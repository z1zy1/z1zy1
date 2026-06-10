import os
import argparse
import json
import time
import numpy as np
import torch
torch.backends.cudnn.enabled  = True
import torch.nn as nn
import torch.nn.functional as F

from configs.config_transformer import cfg, merge_cfg_from_file
from configs.config_transformer import merge_cfg_from_list
from datasets.datasets import create_dataset
from models.CARD import CARD
from models.transformer_decoder import DynamicSpeaker
from utils.semantic_label import build_content_word_token_ids

from utils.utils import AverageMeter, accuracy, set_mode, load_checkpoint, \
                        decode_sequence, decode_sequence_transformer, coco_gen_format_save
from utils.vis_utils import visualize_att
from tqdm import tqdm


def unpack_batch(batch):
    semantic_labels = None
    semantic_targets = None
    if len(batch) == 7:
        d_feats, sc_feats, labels, labels_with_ignore, masks, d_img_paths, sc_img_paths = batch
        pseudo_masks = None
    elif len(batch) == 8:
        d_feats, sc_feats, labels, labels_with_ignore, masks, d_img_paths, sc_img_paths, pseudo_masks = batch
    elif len(batch) == 9:
        d_feats, sc_feats, labels, labels_with_ignore, masks, d_img_paths, sc_img_paths, pseudo_masks, extra = batch
        if isinstance(extra, dict):
            semantic_targets = extra
        else:
            semantic_labels = extra
    elif len(batch) == 10:
        d_feats, sc_feats, labels, labels_with_ignore, masks, d_img_paths, sc_img_paths, pseudo_masks, semantic_labels, semantic_targets = batch
    else:
        raise ValueError('Unexpected batch size: %d' % len(batch))
    return d_feats, sc_feats, labels, labels_with_ignore, masks, d_img_paths, sc_img_paths, pseudo_masks, semantic_labels, semantic_targets


def unpack_change_detector_output(outputs):
    relation_aux_logits = None
    if len(outputs) == 6:
        encoder_output, con_loss, ind_loss, att1, att2, mask_pred = outputs
        semantic_logits = None
    elif len(outputs) == 7:
        encoder_output, con_loss, ind_loss, att1, att2, mask_pred, semantic_logits = outputs
    elif len(outputs) == 8:
        encoder_output, con_loss, ind_loss, att1, att2, mask_pred, semantic_logits, relation_aux_logits = outputs
    else:
        raise ValueError('Unexpected CARD output size: %d' % len(outputs))
    return encoder_output, con_loss, ind_loss, att1, att2, mask_pred, semantic_logits, relation_aux_logits


def is_content_word_weighted_ce_enabled(cfg):
    return bool(
        getattr(cfg.train, 'use_content_word_weight', False)
        or getattr(cfg.train, 'use_content_word_weighted_ce', False)
    )


def load_change_detector_state_compat(model, state_dict):
    current_keys = set(model.state_dict().keys())
    state_keys = set(state_dict.keys())
    missing_keys = sorted(current_keys - state_keys)
    unexpected_keys = sorted(state_keys - current_keys)
    semantic_only = all(
        key.startswith('semantic_head.')
        for key in missing_keys + unexpected_keys
    )
    if not missing_keys and not unexpected_keys:
        model.load_state_dict(state_dict)
        return
    if semantic_only:
        print('Loading change detector with semantic_head checkpoint compatibility.')
        incompatible = model.load_state_dict(state_dict, strict=False)
        non_semantic_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith('semantic_head.')
        ]
        non_semantic_unexpected = [
            key for key in incompatible.unexpected_keys
            if not key.startswith('semantic_head.')
        ]
        if non_semantic_missing or non_semantic_unexpected:
            raise RuntimeError(
                'Non-semantic checkpoint mismatch: missing=%s unexpected=%s'
                % (non_semantic_missing, non_semantic_unexpected)
            )
        return
    model.load_state_dict(state_dict)


def apply_cli_overrides(args, cfg):
    if args.use_relation_aux:
        cfg.train.use_relation_aux = True
    if args.use_content_word_weight:
        cfg.train.use_content_word_weight = True
        if hasattr(cfg.train, 'use_content_word_weighted_ce'):
            cfg.train.use_content_word_weighted_ce = True
    if args.content_word_weight is not None:
        cfg.train.content_word_weight = args.content_word_weight
    if args.max_content_word_weight is not None:
        cfg.train.max_content_word_weight = args.max_content_word_weight
    if args.use_weak_mask_prior:
        cfg.train.use_weak_mask_prior = True
    if args.mask_alpha is not None:
        cfg.train.mask_alpha = args.mask_alpha
    if getattr(cfg.train, 'use_content_word_weighted_ce', False):
        cfg.train.use_content_word_weight = True

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--snapshot', type=int, default=None)
parser.add_argument('--snapshot_path', type=str, default=None)
parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
parser.add_argument('--result_json', type=str, default=None)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--use_relation_aux', action='store_true')
parser.add_argument('--use_content_word_weight', action='store_true')
parser.add_argument('--content_word_weight', type=float, default=None)
parser.add_argument('--max_content_word_weight', type=float, default=None)
parser.add_argument('--use_weak_mask_prior', action='store_true')
parser.add_argument('--mask_alpha', type=float, default=None)
parser.add_argument('opts', nargs=argparse.REMAINDER)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
if args.opts:
    merge_cfg_from_list(args.opts)
apply_cli_overrides(args, cfg)

# Device configuration
use_cuda = torch.cuda.is_available()
if args.gpu == -1:
    gpu_ids = cfg.gpu_id
else:
    gpu_ids = [args.gpu]
torch.backends.cudnn.enabled  = True
if use_cuda:
    default_gpu_device = gpu_ids[0]
    torch.cuda.set_device(default_gpu_device)
    device = torch.device("cuda", default_gpu_device)
else:
    default_gpu_device = None
    device = torch.device("cpu")

# Experiment configuration
exp_dir = cfg.exp_dir
exp_name = cfg.exp_name

output_dir = os.path.join(exp_dir, exp_name)

if args.result_json is not None:
    result_save_path_pos = os.path.normpath(args.result_json)
    caption_output_path = os.path.dirname(result_save_path_pos)
    if caption_output_path and not os.path.exists(caption_output_path):
        os.makedirs(caption_output_path)
    eval_output_dir = os.path.dirname(os.path.dirname(caption_output_path))
else:
    eval_output_dir = os.path.join(output_dir, '%s_output' % args.split)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    caption_output_path = os.path.join(eval_output_dir, 'captions', args.split)
    if not os.path.exists(caption_output_path):
        os.makedirs(caption_output_path)
    result_save_path_pos = os.path.join(caption_output_path, 'sc_results.json')

att_output_path = os.path.join(eval_output_dir, 'attentions', args.split)
if not os.path.exists(att_output_path):
    os.makedirs(att_output_path)

if args.visualize:
    visualize_save_dir = os.path.join(eval_output_dir, 'visualizations')
    if not os.path.exists(visualize_save_dir):
        os.makedirs(visualize_save_dir)

if args.snapshot_path is not None:
    snapshot_full_path = os.path.normpath(args.snapshot_path)
elif args.snapshot is not None:
    snapshot_dir = os.path.join(output_dir, 'snapshots')
    snapshot_file = '%s_checkpoint_%d.pt' % (exp_name, args.snapshot)
    snapshot_full_path = os.path.join(snapshot_dir, snapshot_file)
else:
    raise ValueError('Either --snapshot or --snapshot_path must be provided.')
checkpoint = load_checkpoint(snapshot_full_path)
change_detector_state = checkpoint['change_detector_state']
speaker_state = checkpoint['speaker_state']

# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
idx_to_word = train_dataset.get_idx_to_word()
eval_dataset, eval_loader = create_dataset(cfg, args.split)

# Keep decoder vocabulary / max length aligned with dataset preprocessing outputs.
cfg.model.transformer_decoder.vocab_size = train_dataset.get_vocab_size()
cfg.model.transformer_decoder.seq_length = train_dataset.get_max_seq_length()
if is_content_word_weighted_ce_enabled(cfg):
    cfg.train.content_word_token_ids = build_content_word_token_ids(train_dataset.get_word_to_idx())
else:
    cfg.train.content_word_token_ids = []

# Load modules
change_detector = CARD(cfg)
load_change_detector_state_compat(change_detector, change_detector_state)
change_detector = change_detector.to(device)

speaker = DynamicSpeaker(cfg)
speaker.load_state_dict(speaker_state)
speaker.to(device)

print(change_detector)
print(speaker)

set_mode('eval', [change_detector, speaker])
with torch.no_grad():
    test_iter_start_time = time.time()

    result_sents_pos = {}
    for i, batch in tqdm(enumerate(eval_loader)):

        d_feats, sc_feats, labels, labels_with_ignore, masks, d_img_paths, sc_img_paths, _, _, _ = unpack_batch(batch)

        batch_size = d_feats.size(0)

        d_feats, sc_feats = d_feats.to(device),  sc_feats.to(device)

        labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(device)

        change_outputs = change_detector(d_feats, sc_feats)
        encoder_output, _, _, _, _, _, _, _ = unpack_change_detector_output(change_outputs)

        speaker_output_pos, pos_dynamic_atts = speaker.sample(encoder_output, sample_max=1)

        gen_sents_pos = decode_sequence_transformer(idx_to_word, speaker_output_pos[:, 1:])

        for j in range(batch_size):
            gts = decode_sequence_transformer(idx_to_word, labels[j][:, 1:])
            sent_pos = gen_sents_pos[j]
            image_id = os.path.basename(d_img_paths[j])
            result_sents_pos[image_id] = sent_pos
            image_num = image_id.split('.')[0]

    test_iter_end_time = time.time() - test_iter_start_time
    print('%s inference took %.4f seconds' % (args.split, test_iter_end_time))
    coco_gen_format_save(result_sents_pos, result_save_path_pos)
    print('Saved captions to %s' % result_save_path_pos)


