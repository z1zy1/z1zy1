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
    if not cfg.train.use_mask_warmup:
        return cfg.train.lambda_mask
    if cfg.train.mask_warmup_steps <= 0:
        return cfg.train.lambda_mask
    if global_step < cfg.train.mask_warmup_steps:
        return 0.0
    return cfg.train.lambda_mask


def unpack_batch(batch):
    if len(batch) == 7:
        d_feats, sc_feats, labels, labels_with_ignore, masks, d_img_paths, sc_img_paths = batch
        pseudo_masks = None
    elif len(batch) == 8:
        d_feats, sc_feats, labels, labels_with_ignore, masks, d_img_paths, sc_img_paths, pseudo_masks = batch
    else:
        raise ValueError('Unexpected batch size: %d' % len(batch))
    return d_feats, sc_feats, labels, labels_with_ignore, masks, d_img_paths, sc_img_paths, pseudo_masks

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--visualize_every', type=int, default=10)
parser.add_argument('opts', nargs=argparse.REMAINDER)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
if args.opts:
    merge_cfg_from_list(args.opts)

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

random.seed(1111)
np.random.seed(1111)
torch.manual_seed(1111)

if not cfg.model.enable_aux_mask:
    experiment_mode = 'baseline'
elif cfg.train.use_mask_conf_filter:
    experiment_mode = 'aux_mask + conf_filter'
else:
    experiment_mode = 'aux_mask'

# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
val_dataset, val_loader = create_dataset(cfg, 'val')
train_size = len(train_dataset)
val_size = len(val_dataset)

# Keep decoder vocabulary / max length aligned with dataset preprocessing outputs.
cfg.model.transformer_decoder.vocab_size = train_dataset.get_vocab_size()
cfg.model.transformer_decoder.seq_length = train_dataset.get_max_seq_length()

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
    f'  enable_aux_mask: {cfg.model.enable_aux_mask}',
    f'  lambda_mask: {cfg.train.lambda_mask}',
    f'  use_mask_conf_filter: {cfg.train.use_mask_conf_filter}',
    f'  mask_conf_threshold: {cfg.train.mask_conf_threshold}',
    f'  use_mask_warmup: {cfg.train.use_mask_warmup}',
    f'  mask_warmup_steps: {cfg.train.mask_warmup_steps}',
    f'  mask_loss_type: {cfg.train.mask_loss_type}',
    f'  mask_bce_dice_alpha: {cfg.train.mask_bce_dice_alpha}',
    f'  pseudo_mask_root: {cfg.data.pseudo_mask_root}',
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

        d_feats, sc_feats, labels, labels_with_ignore, masks, d_img_paths, sc_img_paths, pseudo_masks = unpack_batch(batch)

        batch_size = d_feats.size(0)
        labels = labels.squeeze(1)
        labels_with_ignore = labels_with_ignore.squeeze(1)

        masks = masks.squeeze(1).float()

        d_feats,  sc_feats = d_feats.to(device), sc_feats.to(device)

        labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(device)
        if pseudo_masks is not None:
            pseudo_masks = pseudo_masks.to(device).float()

        optimizer.zero_grad()

        encoder_output, con_loss, ind_loss, _, _, mask_pred = change_detector(d_feats, sc_feats)

        loss_pos, _, att_pos = speaker._forward(encoder_output,
                                                labels, masks, labels_with_ignore=labels_with_ignore)

        cap_loss = loss_pos
        cap_loss_val = cap_loss.item()
        con_loss_val = con_loss.item()
        ind_loss_val = ind_loss.item()
        mask_loss = None
        mask_loss_val = 0.0
        effective_lambda_mask = get_effective_lambda_mask(cfg, t)
        if cfg.model.enable_aux_mask:
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

        total_loss = cap_loss + 0.001 * con_loss + 0.001 * ind_loss
        if mask_loss is not None:
            total_loss = total_loss + effective_lambda_mask * mask_loss

        total_loss_val = total_loss.item()

        speaker_loss_avg.update(cap_loss_val, 2 * batch_size)
        constraint_loss_avg.update(con_loss_val + ind_loss_val, 2 * batch_size)
        total_loss_avg.update(total_loss_val, 2 * batch_size)

        stats = {}

        stats['cap_loss'] = cap_loss_val
        stats['avg_cap_loss'] = speaker_loss_avg.avg
        stats['con_loss'] = con_loss_val
        stats['ind_loss'] = ind_loss_val
        stats['avg_constraint_loss'] = constraint_loss_avg.avg
        stats['lambda_mask'] = cfg.train.lambda_mask
        stats['effective_lambda_mask'] = effective_lambda_mask
        stats['total_loss'] = total_loss_val
        stats['avg_total_loss'] = total_loss_avg.avg
        if cfg.model.enable_aux_mask:
            stats['mask_loss'] = mask_loss_val

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
                for val_i, val_batch in enumerate(val_loader):
                    d_feats, sc_feats, labels, labels_with_ignore, masks, d_img_paths, sc_img_paths, _ = unpack_batch(val_batch)

                    val_batch_size = d_feats.size(0)

                    d_feats, sc_feats = d_feats.to(device), sc_feats.to(device)

                    labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(device)


                    encoder_output, _, _, att1, att2, _ = change_detector(d_feats, sc_feats)


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

            set_mode('train', [change_detector, speaker])
    lr_scheduler.step()
