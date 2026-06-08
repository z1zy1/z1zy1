import os
import sys
import json
import argparse
import time
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import random

from configs.config_transformer import cfg, merge_cfg_from_file
from configs.config_transformer import merge_cfg_from_list
from datasets.datasets import create_dataset
from models.CARD import CARD
from models.transformer_decoder import DynamicSpeaker
from utils.logger import Logger
from utils.semantic_label import build_content_word_token_ids
from utils.semantic_warmup import get_effective_lambda_semantic
from utils.utils import AverageMeter, accuracy, set_mode, save_checkpoint, \
                        LanguageModelCriterion, decode_sequence, decode_sequence_transformer, decode_beams, \
                        build_optimizer, coco_gen_format_save, one_hot_encode, \
                        EntropyLoss, LabelSmoothingLoss

from utils.vis_utils import visualize_att


def align_mask_tensor(tensor, spatial_size, mode):
    if tensor is None:
        return None
    tensor = tensor.float()
    if tensor.shape[-2:] != spatial_size:
        if mode == 'nearest':
            tensor = F.interpolate(tensor, size=spatial_size, mode=mode)
        else:
            tensor = F.interpolate(tensor, size=spatial_size, mode=mode, align_corners=False)
    return tensor


def compute_dice_loss(pred, target, valid_mask=None, eps=1e-6):
    if valid_mask is None:
        valid_mask = torch.ones_like(pred)
    pred = pred * valid_mask
    target = target * valid_mask
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice_score.mean()


def compute_mask_loss(pred, target, ignore_mask=None, loss_type='bce', bce_dice_alpha=0.5):
    target = align_mask_tensor(target, pred.shape[-2:], mode='bilinear')
    valid_mask = None
    if ignore_mask is not None:
        ignore_mask = align_mask_tensor(ignore_mask, pred.shape[-2:], mode='nearest')
        valid_mask = (~ignore_mask.bool()).float()

    bce = F.binary_cross_entropy(pred, target, reduction='none')
    if valid_mask is not None:
        bce = bce * valid_mask
        valid_count = valid_mask.sum()
        if valid_count.item() == 0:
            bce_loss = pred.sum() * 0.0
        else:
            bce_loss = bce.sum() / valid_count
    else:
        bce_loss = bce.mean()

    if loss_type == 'bce':
        return bce_loss
    if loss_type == 'bce_dice':
        dice_loss = compute_dice_loss(pred, target, valid_mask=valid_mask)
        return bce_dice_alpha * bce_loss + (1.0 - bce_dice_alpha) * dice_loss
    raise ValueError('Unknown mask_loss_type: %s' % loss_type)


def get_effective_lambda_mask(cfg, global_step):
    if not cfg.model.enable_aux_mask:
        return 0.0
    if cfg.train.lambda_mask <= 0:
        return 0.0
    if not cfg.train.use_mask_warmup:
        return cfg.train.lambda_mask
    if cfg.train.mask_warmup_steps <= 0:
        return cfg.train.lambda_mask
    return cfg.train.lambda_mask * min(1.0, float(global_step) / float(cfg.train.mask_warmup_steps))


def get_warmup_lambda(target_lambda, global_step, warmup_steps):
    if target_lambda <= 0:
        return 0.0
    if warmup_steps <= 0:
        return target_lambda
    return target_lambda * min(1.0, float(global_step) / float(warmup_steps))


def move_semantic_targets_to_device(semantic_targets, device):
    if semantic_targets is None:
        return None
    return {
        key: value.to(device).float()
        for key, value in semantic_targets.items()
    }


def update_micro_f1_counts(counts, logits, targets, threshold):
    preds = torch.sigmoid(logits) >= threshold
    gold = targets >= 0.5
    counts['tp'] += (preds & gold).sum().item()
    counts['fp'] += (preds & (~gold)).sum().item()
    counts['fn'] += ((~preds) & gold).sum().item()


def micro_f1_from_counts(counts):
    denom = 2 * counts['tp'] + counts['fp'] + counts['fn']
    if denom == 0:
        return 0.0
    return 2.0 * counts['tp'] / float(denom)


def is_content_word_weighted_ce_enabled(cfg):
    return bool(
        getattr(cfg.train, 'use_content_word_weight', False)
        or getattr(cfg.train, 'use_content_word_weighted_ce', False)
    )


def apply_cli_overrides(args, cfg):
    if args.output_dir is not None:
        output_dir = os.path.normpath(args.output_dir)
        exp_dir, exp_name = os.path.split(output_dir)
        if not exp_name:
            raise ValueError('--output_dir must point to an experiment directory, not a root directory.')
        if args.exp_name is not None and args.exp_name != exp_name:
            raise ValueError(
                '--exp_name (%s) must match the basename of --output_dir (%s).'
                % (args.exp_name, exp_name)
            )
        cfg.exp_dir = exp_dir or '.'
        cfg.exp_name = exp_name
    elif args.exp_name is not None:
        cfg.exp_name = args.exp_name
    if args.use_mask_aux:
        cfg.model.enable_aux_mask = True
    if args.use_relation_aux:
        cfg.train.use_relation_aux = True
    if args.use_semantic_aux:
        cfg.train.use_semantic_aux = True
    if args.use_semantic_detach:
        cfg.train.use_semantic_detach = True
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.debug_semantic_detach:
        cfg.train.use_semantic_aux = True
        cfg.train.use_semantic_detach = True
        cfg.train.log_interval = 1
        cfg.train.max_iter = min(cfg.train.max_iter, 2)
        cfg.train.snapshot_interval = max(cfg.train.snapshot_interval, cfg.train.max_iter + 1)
    if args.lambda_obj is not None:
        cfg.train.lambda_obj = args.lambda_obj
    if args.lambda_act is not None:
        cfg.train.lambda_act = args.lambda_act
    if args.lambda_rel is not None:
        cfg.train.lambda_rel = args.lambda_rel
    if args.relation_aux_dropout is not None:
        cfg.train.relation_aux_dropout = args.relation_aux_dropout
    if args.semantic_warmup_steps is not None:
        cfg.train.semantic_warmup_steps = args.semantic_warmup_steps
    if args.lambda_semantic is not None:
        cfg.train.lambda_semantic = args.lambda_semantic
    if args.use_semantic_warmup:
        cfg.train.use_semantic_warmup = True
    if args.semantic_warmup_start is not None:
        cfg.train.semantic_warmup_start = args.semantic_warmup_start
    if args.semantic_warmup_end is not None:
        cfg.train.semantic_warmup_end = args.semantic_warmup_end
    if args.semantic_warmup_type is not None:
        cfg.train.semantic_warmup_type = args.semantic_warmup_type
    if args.semantic_late_start:
        cfg.train.semantic_late_start = True
    if args.semantic_start_iter is not None:
        cfg.train.semantic_start_iter = args.semantic_start_iter
    if args.semantic_threshold is not None:
        cfg.train.semantic_threshold = args.semantic_threshold
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
    if args.lambda_mask is not None:
        cfg.train.lambda_mask = args.lambda_mask
    if args.mask_warmup_steps is not None:
        cfg.train.mask_warmup_steps = args.mask_warmup_steps
        if args.mask_warmup_steps > 0:
            cfg.train.use_mask_warmup = True
    if getattr(cfg.train, 'use_content_word_weighted_ce', False):
        cfg.train.use_content_word_weight = True


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

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--visualize_every', type=int, default=10)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--use_mask_aux', action='store_true')
parser.add_argument('--use_relation_aux', action='store_true')
parser.add_argument('--use_semantic_aux', action='store_true')
parser.add_argument('--use_semantic_detach', action='store_true')
parser.add_argument('--debug_semantic_detach', action='store_true')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--lambda_obj', type=float, default=None)
parser.add_argument('--lambda_act', type=float, default=None)
parser.add_argument('--lambda_rel', type=float, default=None)
parser.add_argument('--relation_aux_dropout', type=float, default=None)
parser.add_argument('--semantic_warmup_steps', type=int, default=None)
parser.add_argument('--lambda_semantic', type=float, default=None)
parser.add_argument('--use_semantic_warmup', action='store_true')
parser.add_argument('--semantic_warmup_start', type=int, default=None)
parser.add_argument('--semantic_warmup_end', type=int, default=None)
parser.add_argument('--semantic_warmup_type', type=str, default=None)
parser.add_argument('--semantic_late_start', action='store_true')
parser.add_argument('--semantic_start_iter', type=int, default=None)
parser.add_argument('--semantic_threshold', type=float, default=None)
parser.add_argument('--use_content_word_weight', action='store_true')
parser.add_argument('--content_word_weight', type=float, default=None)
parser.add_argument('--max_content_word_weight', type=float, default=None)
parser.add_argument('--use_weak_mask_prior', action='store_true')
parser.add_argument('--mask_alpha', type=float, default=None)
parser.add_argument('--lambda_mask', type=float, default=None)
parser.add_argument('--mask_warmup_steps', type=int, default=None)
parser.add_argument('opts', nargs=argparse.REMAINDER)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
if args.opts:
    merge_cfg_from_list(args.opts)
apply_cli_overrides(args, cfg)

# Device configuration
use_cuda = torch.cuda.is_available()
gpu_ids = cfg.gpu_id
torch.backends.cudnn.enabled = False
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
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

output_dir = os.path.join(exp_dir, exp_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cfg_file_save = os.path.join(output_dir, 'cfg.json')
json.dump(cfg, open(cfg_file_save, 'w'))

sample_dir = os.path.join(output_dir, 'eval_gen_samples')
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
sample_subdir_format = '%s_samples_%d'

sent_dir = os.path.join(output_dir, 'eval_sents')
if not os.path.exists(sent_dir):
    os.makedirs(sent_dir)
sent_subdir_format = '%s_sents_%d'

snapshot_dir = os.path.join(output_dir, 'snapshots')
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)
snapshot_file_format = '%s_checkpoint_%d.pt'

train_logger = Logger(cfg, output_dir, is_train=True)
val_logger = Logger(cfg, output_dir, is_train=False)

random.seed(cfg.train.seed)
np.random.seed(cfg.train.seed)
torch.manual_seed(cfg.train.seed)
if use_cuda:
    torch.cuda.manual_seed_all(cfg.train.seed)

if not cfg.model.enable_aux_mask:
    experiment_mode = 'baseline'
elif cfg.train.use_mask_conf_filter:
    experiment_mode = 'aux_mask + conf_filter'
else:
    experiment_mode = 'aux_mask'
if cfg.train.use_semantic_aux:
    experiment_mode += ' + semantic_aux'
if cfg.train.use_relation_aux:
    experiment_mode += ' + relation_aux'
if is_content_word_weighted_ce_enabled(cfg):
    experiment_mode += ' + content_word_weight'
if cfg.train.use_weak_mask_prior:
    experiment_mode += ' + weak_mask_prior'

# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
val_dataset, val_loader = create_dataset(cfg, 'val')
train_size = len(train_dataset)
val_size = len(val_dataset)
num_semantic_tags = train_dataset.get_num_semantic_tags() if hasattr(train_dataset, 'get_num_semantic_tags') else 0
num_relation_objects = train_dataset.get_num_relation_objects() if hasattr(train_dataset, 'get_num_relation_objects') else 0
num_relation_actions = train_dataset.get_num_relation_actions() if hasattr(train_dataset, 'get_num_relation_actions') else 0

if cfg.train.use_semantic_aux:
    if cfg.train.semantic_loss_type != 'multilabel_bce':
        raise ValueError('Unknown semantic_loss_type: %s' % cfg.train.semantic_loss_type)
    if num_semantic_tags <= 0:
        raise ValueError('Semantic auxiliary branch is enabled but no semantic tags were loaded.')
    print('Semantic auxiliary branch enabled.')
    print('Semantic supervision type: object/action multi-label BCE.')
    print('Number of semantic tags: %d' % num_semantic_tags)
    print('Semantic tag file: %s' % cfg.train.semantic_tag_file)
    print('lambda_semantic: %s' % cfg.train.lambda_semantic)
    print('use_semantic_detach: %s' % cfg.train.use_semantic_detach)
    print(
        'Semantic warmup: enabled=%s start=%s end=%s type=%s.'
        % (
            cfg.train.use_semantic_warmup,
            cfg.train.semantic_warmup_start,
            cfg.train.semantic_warmup_end,
            cfg.train.semantic_warmup_type,
        )
    )
    print(
        'Semantic late start: enabled=%s start_iter=%s.'
        % (cfg.train.semantic_late_start, cfg.train.semantic_start_iter)
    )
    print('Relation auxiliary loss: %s.' % ('enabled' if cfg.train.use_relation_aux else 'disabled'))
    print(
        'Content word weighted CE: %s.'
        % ('enabled' if is_content_word_weighted_ce_enabled(cfg) else 'disabled')
    )
    if hasattr(train_dataset, 'get_semantic_label_stats'):
        semantic_label_stats = train_dataset.get_semantic_label_stats()
        if semantic_label_stats is not None:
            print(
                'Semantic label check: samples=%d all_zero_samples=%d avg_positive_tags=%.4f'
                % (
                    semantic_label_stats['total_samples'],
                    semantic_label_stats['all_zero_samples'],
                    semantic_label_stats['avg_positive_tags'],
                )
            )

if cfg.train.use_relation_aux:
    if num_relation_objects <= 0 or num_relation_actions <= 0:
        raise ValueError('Relation auxiliary branch is enabled but the dataset does not expose object/action vocab sizes.')
    print('Relation auxiliary branch enabled.')
    print('Number of relation objects: %d' % num_relation_objects)
    print('Number of relation actions: %d' % num_relation_actions)

# Keep decoder vocabulary / max length aligned with dataset preprocessing outputs.
cfg.model.transformer_decoder.vocab_size = train_dataset.get_vocab_size()
cfg.model.transformer_decoder.seq_length = train_dataset.get_max_seq_length()
if is_content_word_weighted_ce_enabled(cfg):
    cfg.train.content_word_token_ids = build_content_word_token_ids(train_dataset.get_word_to_idx())
    print('Content word weighted CE enabled. Token ids: %s' % cfg.train.content_word_token_ids)
else:
    cfg.train.content_word_token_ids = []

# Create model
change_detector = CARD(cfg)
change_detector.to(device)

speaker = DynamicSpeaker(cfg)
speaker.to(device)

print(change_detector)
print(speaker)

experiment_summary = [
    'Experiment Summary:',
    f'  exp_name: {exp_name}',
    f'  mode: {experiment_mode}',
    f'  seed: {cfg.train.seed}',
    f'  enable_aux_mask: {cfg.model.enable_aux_mask}',
    f'  lambda_mask: {cfg.train.lambda_mask}',
    f'  use_mask_conf_filter: {cfg.train.use_mask_conf_filter}',
    f'  mask_conf_threshold: {cfg.train.mask_conf_threshold}',
    f'  use_mask_warmup: {cfg.train.use_mask_warmup}',
    f'  mask_warmup_steps: {cfg.train.mask_warmup_steps}',
    f'  mask_loss_type: {cfg.train.mask_loss_type}',
    f'  mask_bce_dice_alpha: {cfg.train.mask_bce_dice_alpha}',
    f'  use_weak_mask_prior: {cfg.train.use_weak_mask_prior}',
    f'  mask_alpha: {cfg.train.mask_alpha}',
    f'  pseudo_mask_root: {cfg.data.pseudo_mask_root}',
    f'  use_semantic_aux: {cfg.train.use_semantic_aux}',
    f'  lambda_semantic: {cfg.train.lambda_semantic}',
    f'  use_semantic_detach: {cfg.train.use_semantic_detach}',
    f'  use_semantic_warmup: {cfg.train.use_semantic_warmup}',
    f'  semantic_warmup_start: {cfg.train.semantic_warmup_start}',
    f'  semantic_warmup_end: {cfg.train.semantic_warmup_end}',
    f'  semantic_warmup_type: {cfg.train.semantic_warmup_type}',
    f'  semantic_late_start: {cfg.train.semantic_late_start}',
    f'  semantic_start_iter: {cfg.train.semantic_start_iter}',
    f'  semantic_loss_type: {cfg.train.semantic_loss_type}',
    f'  semantic_tag_file: {cfg.train.semantic_tag_file}',
    f'  semantic_aux_dropout: {cfg.train.semantic_aux_dropout}',
    f'  semantic_normalize_synonyms: {cfg.train.semantic_normalize_synonyms}',
    f'  num_semantic_tags: {num_semantic_tags}',
    f'  use_relation_aux: {cfg.train.use_relation_aux}',
    f'  lambda_obj: {cfg.train.lambda_obj}',
    f'  lambda_act: {cfg.train.lambda_act}',
    f'  lambda_rel: {cfg.train.lambda_rel}',
    f'  relation_aux_dropout: {cfg.train.relation_aux_dropout}',
    f'  semantic_warmup_steps: {cfg.train.semantic_warmup_steps}',
    f'  semantic_threshold: {cfg.train.semantic_threshold}',
    f'  num_relation_objects: {num_relation_objects}',
    f'  num_relation_actions: {num_relation_actions}',
    f'  use_content_word_weight: {cfg.train.use_content_word_weight}',
    f'  use_content_word_weighted_ce: {cfg.train.use_content_word_weighted_ce}',
    f'  content_word_weight: {cfg.train.content_word_weight}',
    f'  max_content_word_weight: {cfg.train.max_content_word_weight}',
    f'  content_word_token_count: {len(cfg.train.content_word_token_ids)}',
]
for line in experiment_summary:
    print(line)

with open(os.path.join(output_dir, 'model_print'), 'w') as f:
    print(change_detector, file=f)
    print(speaker, file=f)
    for line in experiment_summary:
        print(line, file=f)

all_params = list(change_detector.parameters()) + list(speaker.parameters())
optimizer = build_optimizer(all_params, cfg)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cfg.train.optim.step_size,
    gamma=cfg.train.optim.gamma)
semantic_loss_func = nn.BCEWithLogitsLoss() if cfg.train.use_semantic_aux else None
relation_loss_func = nn.BCEWithLogitsLoss() if cfg.train.use_relation_aux else None

# Train loop
t = 0
epoch = 0
printed_binary_mask_warning = False

set_mode('train', [change_detector, speaker])

while t < cfg.train.max_iter:
    epoch += 1
    print('Starting epoch %d' % epoch)

    speaker_loss_avg = AverageMeter()
    constraint_loss_avg = AverageMeter()
    total_loss_avg = AverageMeter()
    if epoch > cfg.train.scheduled_sampling_start and cfg.train.scheduled_sampling_start >= 0:
        frac = (epoch - cfg.train.scheduled_sampling_start) // cfg.train.scheduled_sampling_increase_every
        ss_prob_prev = ss_prob
        ss_prob = min(cfg.train.scheduled_sampling_increase_prob * frac,
                      cfg.train.scheduled_sampling_max_prob)
        speaker.ss_prob = ss_prob
        if ss_prob_prev != ss_prob:
            print('Updating scheduled sampling rate: %.4f -> %.4f' % (ss_prob_prev, ss_prob))
    for i, batch in enumerate(train_loader):
        iter_start_time = time.time()

        d_feats, sc_feats, labels, labels_with_ignore, masks, d_img_paths, sc_img_paths, pseudo_masks, semantic_labels, semantic_targets = unpack_batch(batch)

        batch_size = d_feats.size(0)
        labels = labels.squeeze(1)
        labels_with_ignore = labels_with_ignore.squeeze(1)

        masks = masks.squeeze(1).float()

        d_feats,  sc_feats = d_feats.to(device), sc_feats.to(device)

        labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(device)
        if pseudo_masks is not None:
            pseudo_masks = pseudo_masks.to(device).float()
        if semantic_labels is not None:
            semantic_labels = semantic_labels.to(device).float()
        semantic_targets = move_semantic_targets_to_device(semantic_targets, device)

        optimizer.zero_grad()

        change_outputs = change_detector(d_feats, sc_feats)
        encoder_output, con_loss, ind_loss, _, _, mask_pred, semantic_logits, relation_aux_logits = unpack_change_detector_output(change_outputs)

        loss_pos, _, att_pos = speaker._forward(encoder_output,
                                                labels, masks, labels_with_ignore=labels_with_ignore)

        cap_loss = loss_pos
        cap_loss_val = cap_loss.item()
        con_loss_val = con_loss.item()
        ind_loss_val = ind_loss.item()
        mask_loss = None
        mask_loss_val = 0.0
        weighted_mask_loss_val = 0.0
        semantic_loss = None
        semantic_loss_val = 0.0
        effective_lambda_semantic = 0.0
        weighted_semantic_loss_val = 0.0
        object_loss = None
        action_loss = None
        relation_loss = None
        object_loss_val = 0.0
        action_loss_val = 0.0
        relation_loss_val = 0.0
        object_pos_count = 0.0
        action_pos_count = 0.0
        relation_pos_count = 0.0
        current_iter = t + 1
        effective_lambda_mask = get_effective_lambda_mask(cfg, t)
        if cfg.model.enable_aux_mask and cfg.train.lambda_mask > 0:
            if pseudo_masks is None:
                raise ValueError('Pseudo masks are required when aux mask is enabled.')
            if cfg.train.use_mask_conf_filter and not printed_binary_mask_warning:
                is_binary_mask = bool(torch.all((pseudo_masks == 0) | (pseudo_masks == 1)).item())
                if is_binary_mask:
                    print('Warning: pseudo masks appear binary. mask_conf_threshold will have little or no effect unless masks store soft confidence values.')
                printed_binary_mask_warning = True
            ignore_mask = None
            if cfg.train.use_mask_conf_filter:
                # Low-confidence pseudo-label regions are ignored in mask supervision.
                confidence = torch.maximum(pseudo_masks, 1.0 - pseudo_masks)
                ignore_mask = confidence < cfg.train.mask_conf_threshold
            mask_loss = compute_mask_loss(
                mask_pred,
                pseudo_masks,
                ignore_mask=ignore_mask,
                loss_type=cfg.train.mask_loss_type,
                bce_dice_alpha=cfg.train.mask_bce_dice_alpha,
            )
            mask_loss_val = mask_loss.item()
            weighted_mask_loss_val = effective_lambda_mask * mask_loss_val
        if cfg.train.use_semantic_aux:
            if semantic_labels is None:
                raise ValueError('Semantic labels are required when train.use_semantic_aux is enabled.')
            if semantic_logits is None:
                raise ValueError('CARD did not return semantic_logits while train.use_semantic_aux is enabled.')
            if semantic_logits.shape != semantic_labels.shape:
                raise ValueError(
                    'semantic_logits shape %s does not match semantic_labels shape %s.'
                    % (tuple(semantic_logits.shape), tuple(semantic_labels.shape))
            )
            semantic_loss = semantic_loss_func(semantic_logits, semantic_labels.float())
            semantic_loss_val = semantic_loss.item()
            effective_lambda_semantic = get_effective_lambda_semantic(
                current_iter,
                cfg.train.lambda_semantic,
                cfg.train.use_semantic_warmup,
                cfg.train.semantic_warmup_start,
                cfg.train.semantic_warmup_end,
                cfg.train.semantic_warmup_type,
                cfg.train.semantic_late_start,
                cfg.train.semantic_start_iter,
            )
            weighted_semantic_loss_val = effective_lambda_semantic * semantic_loss_val

        lambda_obj_current = get_warmup_lambda(cfg.train.lambda_obj, t, cfg.train.semantic_warmup_steps)
        lambda_act_current = get_warmup_lambda(cfg.train.lambda_act, t, cfg.train.semantic_warmup_steps)
        lambda_rel_current = get_warmup_lambda(cfg.train.lambda_rel, t, cfg.train.semantic_warmup_steps)
        if cfg.train.use_relation_aux:
            if semantic_targets is None:
                raise ValueError('Relation semantic targets are required when train.use_relation_aux is enabled.')
            if relation_aux_logits is None:
                raise ValueError('CARD did not return relation_aux_logits while train.use_relation_aux is enabled.')
            for key in ('objects', 'actions', 'relations'):
                if key not in semantic_targets or key not in relation_aux_logits:
                    raise ValueError('Missing relation auxiliary key: %s' % key)
                if relation_aux_logits[key].shape != semantic_targets[key].shape:
                    raise ValueError(
                        'relation_aux_logits[%s] shape %s does not match semantic_targets[%s] shape %s.'
                        % (key, tuple(relation_aux_logits[key].shape), key, tuple(semantic_targets[key].shape))
                    )
            object_loss = relation_loss_func(relation_aux_logits['objects'], semantic_targets['objects'])
            action_loss = relation_loss_func(relation_aux_logits['actions'], semantic_targets['actions'])
            relation_loss = relation_loss_func(relation_aux_logits['relations'], semantic_targets['relations'])
            object_loss_val = object_loss.item()
            action_loss_val = action_loss.item()
            relation_loss_val = relation_loss.item()
            object_pos_count = semantic_targets['objects'].sum().item()
            action_pos_count = semantic_targets['actions'].sum().item()
            relation_pos_count = semantic_targets['relations'].sum().item()

        total_loss = cap_loss + 0.001 * con_loss + 0.001 * ind_loss
        if mask_loss is not None:
            total_loss = total_loss + effective_lambda_mask * mask_loss
        if semantic_loss is not None:
            total_loss = total_loss + effective_lambda_semantic * semantic_loss
        if object_loss is not None:
            total_loss = total_loss + lambda_obj_current * object_loss
        if action_loss is not None:
            total_loss = total_loss + lambda_act_current * action_loss
        if relation_loss is not None:
            total_loss = total_loss + lambda_rel_current * relation_loss

        total_loss_val = total_loss.item()

        speaker_loss_avg.update(cap_loss_val, 2 * batch_size)
        constraint_loss_avg.update(con_loss_val + ind_loss_val, 2 * batch_size)
        total_loss_avg.update(total_loss_val, 2 * batch_size)

        stats = {}

        stats['cap_loss'] = cap_loss_val
        stats['loss_cap'] = cap_loss_val
        stats['loss_caption'] = cap_loss_val
        stats['avg_cap_loss'] = speaker_loss_avg.avg
        stats['con_loss'] = con_loss_val
        stats['ind_loss'] = ind_loss_val
        stats['avg_constraint_loss'] = constraint_loss_avg.avg
        stats['lambda_mask'] = cfg.train.lambda_mask
        stats['use_mask_aux'] = float(cfg.model.enable_aux_mask)
        stats['effective_lambda_mask'] = effective_lambda_mask
        stats['lambda_semantic'] = cfg.train.lambda_semantic
        stats['effective_lambda_semantic'] = effective_lambda_semantic
        stats['use_semantic_warmup'] = float(cfg.train.use_semantic_warmup)
        stats['semantic_warmup_start'] = cfg.train.semantic_warmup_start
        stats['semantic_warmup_end'] = cfg.train.semantic_warmup_end
        stats['semantic_late_start'] = float(cfg.train.semantic_late_start)
        stats['semantic_start_iter'] = cfg.train.semantic_start_iter
        stats['num_semantic_tags'] = num_semantic_tags
        stats['use_semantic_aux'] = float(cfg.train.use_semantic_aux)
        stats['use_semantic_detach'] = float(cfg.train.use_semantic_detach)
        stats['loss_mask'] = mask_loss_val
        stats['weighted_mask_loss'] = weighted_mask_loss_val
        stats['loss_semantic'] = semantic_loss_val
        stats['weighted_semantic_loss'] = weighted_semantic_loss_val
        semantic_debug_info = getattr(change_detector, 'semantic_detach_debug', {})
        for debug_key in (
            'semantic_input_requires_grad',
            'diff_features_requires_grad',
            'caption_input_requires_grad',
            'semantic_logits_requires_grad',
        ):
            stats[debug_key] = semantic_debug_info.get(debug_key, 0.0)
        stats['total_loss'] = total_loss_val
        stats['loss_total'] = total_loss_val
        stats['avg_total_loss'] = total_loss_avg.avg
        if cfg.model.enable_aux_mask and cfg.train.lambda_mask > 0:
            stats['mask_loss'] = mask_loss_val
        if cfg.train.use_relation_aux:
            stats['loss_obj'] = object_loss_val
            stats['loss_act'] = action_loss_val
            stats['loss_rel'] = relation_loss_val
            stats['lambda_obj_current'] = lambda_obj_current
            stats['lambda_act_current'] = lambda_act_current
            stats['lambda_rel_current'] = lambda_rel_current
            stats['object_pos_count'] = object_pos_count
            stats['action_pos_count'] = action_pos_count
            stats['relation_pos_count'] = relation_pos_count

        #results, sample_logprobs = model(d_feats, q_feats, labels, cfg=cfg, mode='sample')
        total_loss.backward()
        if cfg.train.grad_clip != -1.0:  # enable, -1 == disable
            nn.utils.clip_grad_norm_(change_detector.parameters(), cfg.train.grad_clip)
            nn.utils.clip_grad_norm_(speaker.parameters(), cfg.train.grad_clip)

        optimizer.step()

        iter_end_time = time.time() - iter_start_time

        t += 1

        if t % cfg.train.log_interval == 0:
            train_logger.print_current_stats(epoch, i, t, stats, iter_end_time)
            train_logger.plot_current_stats(
                epoch,
                float(i * batch_size) / train_size, stats, 'loss')

        if t % cfg.train.snapshot_interval == 0:
            speaker_state = speaker.state_dict()
            chg_det_state = change_detector.state_dict()
            checkpoint = {
                'change_detector_state': chg_det_state,
                'speaker_state': speaker_state,
                'model_cfg': cfg
            }
            save_path = os.path.join(snapshot_dir,
                                     snapshot_file_format % (exp_name, t))
            save_checkpoint(checkpoint, save_path)

            print('Running eval at iter %d' % t)
            set_mode('eval', [change_detector, speaker])
            with torch.no_grad():
                test_iter_start_time = time.time()

                idx_to_word = train_dataset.get_idx_to_word()

                if args.visualize:
                    sample_subdir_path = sample_subdir_format % (exp_name, t)
                    sample_save_dir = os.path.join(sample_dir, sample_subdir_path)
                    if not os.path.exists(sample_save_dir):
                        os.makedirs(sample_save_dir)
                sent_subdir_path = sent_subdir_format % (exp_name, t)
                sent_save_dir = os.path.join(sent_dir, sent_subdir_path)
                if not os.path.exists(sent_save_dir):
                    os.makedirs(sent_save_dir)

                result_sents_pos = {}
                val_f1_counts = {
                    'objects': {'tp': 0, 'fp': 0, 'fn': 0},
                    'actions': {'tp': 0, 'fp': 0, 'fn': 0},
                    'relations': {'tp': 0, 'fp': 0, 'fn': 0},
                }
                for val_i, val_batch in enumerate(val_loader):
                    d_feats, sc_feats, labels, labels_with_ignore, masks, d_img_paths, sc_img_paths, _, _, semantic_targets = unpack_batch(val_batch)

                    val_batch_size = d_feats.size(0)

                    d_feats, sc_feats = d_feats.to(device), sc_feats.to(device)

                    labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(device)
                    semantic_targets = move_semantic_targets_to_device(semantic_targets, device)


                    change_outputs = change_detector(d_feats, sc_feats)
                    encoder_output, _, _, att1, att2, _, _, relation_aux_logits = unpack_change_detector_output(change_outputs)
                    if cfg.train.use_relation_aux:
                        if relation_aux_logits is None or semantic_targets is None:
                            raise ValueError('Relation auxiliary validation requires logits and semantic targets.')
                        threshold = cfg.train.semantic_threshold
                        update_micro_f1_counts(val_f1_counts['objects'], relation_aux_logits['objects'], semantic_targets['objects'], threshold)
                        update_micro_f1_counts(val_f1_counts['actions'], relation_aux_logits['actions'], semantic_targets['actions'], threshold)
                        update_micro_f1_counts(val_f1_counts['relations'], relation_aux_logits['relations'], semantic_targets['relations'], threshold)


                    speaker_output_pos, _ = speaker.sample(encoder_output)

                    gen_sents_pos = decode_sequence_transformer(idx_to_word, speaker_output_pos[:, 1:]) # no start

                    for val_j in range(speaker_output_pos.size(0)):
                        gts = decode_sequence_transformer(idx_to_word, labels[val_j][:, 1:])

                        sent_pos = gen_sents_pos[val_j]

                        image_id = os.path.basename(d_img_paths[val_j])
                        result_sents_pos[image_id] = sent_pos

                        message = '%s results:\n' % d_img_paths[val_j]
                        message += '\t' + sent_pos + '\n'
                        message += '----------<GROUND TRUTHS>----------\n'
                        for gt in gts:
                            message += gt + '\n'
                        message += '===================================\n'
                        print(message)

                test_iter_end_time = time.time() - test_iter_start_time
                result_save_path_pos = os.path.join(sent_save_dir, 'sc_results.json')
                coco_gen_format_save(result_sents_pos, result_save_path_pos)
                if cfg.train.use_relation_aux:
                    val_stats = {
                        'object_f1': micro_f1_from_counts(val_f1_counts['objects']),
                        'action_f1': micro_f1_from_counts(val_f1_counts['actions']),
                        'relation_f1': micro_f1_from_counts(val_f1_counts['relations']),
                    }
                    val_logger.print_current_stats(epoch, 0, t, val_stats, test_iter_end_time)

            set_mode('train', [change_detector, speaker])
    lr_scheduler.step()
