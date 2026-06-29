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
from utils.dataset_config import apply_dataset_cli_overrides
from utils.experiment_tracking import save_resolved_config, sync_wcsg_config_aliases, write_single_row_csv
from utils.semantic_label import build_content_word_token_ids

from utils.utils import AverageMeter, accuracy, set_mode, load_checkpoint, \
                        decode_sequence, decode_sequence_transformer, coco_gen_format_save
from utils.vis_utils import visualize_att
from tqdm import tqdm


def unpack_batch(batch):
    if isinstance(batch, dict):
        return batch
    if len(batch) < 8:
        raise ValueError('Unexpected batch size: %d' % len(batch))
    return {
        'feature_before': batch[0],
        'feature_after': batch[1],
        'caption_tokens': batch[2],
        'labels_with_ignore': batch[3],
        'caption_mask': batch[4],
        'image_before': batch[5],
        'image_after': batch[6],
        'mask': batch[7],
        'semantic_labels': batch[8] if len(batch) > 8 else None,
        'semantic_targets': batch[9] if len(batch) > 9 else None,
        'semantic_dense': batch[10] if len(batch) > 10 else None,
        'semantic_before': batch[11] if len(batch) > 11 else None,
        'semantic_after': batch[12] if len(batch) > 12 else None,
        'semantic_diff': batch[13] if len(batch) > 13 else None,
        'changeflag': batch[14] if len(batch) > 14 else None,
        'image_id': batch[15] if len(batch) > 15 else None,
    }


def move_optional_tensor(value, device, dtype=None):
    if value is None:
        return None
    value = value.to(device)
    return value.to(dtype=dtype) if dtype is not None else value


def align_mask_tensor(tensor, spatial_size):
    if tensor is None:
        return None
    tensor = tensor.float()
    if tensor.shape[-2:] != spatial_size:
        tensor = F.interpolate(tensor, size=spatial_size, mode='nearest')
    return tensor


def align_class_tensor(tensor, spatial_size):
    if tensor is None:
        return None
    if tensor.dim() == 4 and tensor.size(1) == 1:
        tensor = tensor[:, 0]
    if tensor.shape[-2:] != spatial_size:
        tensor = F.interpolate(tensor.unsqueeze(1).float(), size=spatial_size, mode='nearest').squeeze(1)
    return tensor.long()


def segmentation_metrics(logits, target, ignore_index=-1):
    if logits is None or target is None:
        return None
    if logits.size(1) == 1:
        pred = (torch.sigmoid(logits) >= 0.5).long()[:, 0]
        target = align_mask_tensor(target, logits.shape[-2:])[:, 0]
        valid = target != float(ignore_index)
        gold = (target > 0.5).long()
        classes = [1]
    else:
        pred = torch.argmax(logits, dim=1)
        target = align_class_tensor(target, logits.shape[-2:])
        valid = target != int(ignore_index)
        gold = target.long()
        classes = list(range(1, logits.size(1))) or [0]
    if valid.sum().item() == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0, 'miou': 0.0}
    pred_pos = torch.zeros_like(pred, dtype=torch.bool)
    gold_pos = torch.zeros_like(gold, dtype=torch.bool)
    for cls_id in classes:
        pred_pos |= pred == int(cls_id)
        gold_pos |= gold == int(cls_id)
    pred_pos &= valid
    gold_pos &= valid
    tp = (pred_pos & gold_pos).sum().item()
    fp = (pred_pos & (~gold_pos) & valid).sum().item()
    fn = ((~pred_pos) & gold_pos & valid).sum().item()
    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) else 0.0
    iou = tp / float(tp + fp + fn) if (tp + fp + fn) else 0.0
    result = {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}
    ious = []
    for cls_id in classes:
        cls_pred = (pred == int(cls_id)) & valid
        cls_gold = (gold == int(cls_id)) & valid
        union = (cls_pred | cls_gold).sum().item()
        if union:
            cls_iou = ((cls_pred & cls_gold).sum().item()) / float(union)
            result['class_%d_iou' % cls_id] = cls_iou
            ious.append(cls_iou)
    result['miou'] = float(np.mean(ious)) if ious else 0.0
    return result


def average_metric_list(items):
    if not items:
        return {}
    keys = sorted({key for item in items for key in item.keys()})
    return {key: float(np.mean([item[key] for item in items if key in item])) for key in keys}

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
    optional_prefixes = (
        'semantic_head.', 'semantic_dense_head.', 'semantic_embedding.', 'semantic_cross.', 'semantic_cross_fusion.', 'aux_mask_head.'
    )
    semantic_only = all(key.startswith(optional_prefixes) for key in missing_keys + unexpected_keys)
    if not missing_keys and not unexpected_keys:
        model.load_state_dict(state_dict)
        return
    if semantic_only:
        print('Loading change detector with semantic_head checkpoint compatibility.')
        incompatible = model.load_state_dict(state_dict, strict=False)
        non_semantic_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(optional_prefixes)
        ]
        non_semantic_unexpected = [
            key for key in incompatible.unexpected_keys
            if not key.startswith(optional_prefixes)
        ]
        if non_semantic_missing or non_semantic_unexpected:
            raise RuntimeError(
                'Non-semantic checkpoint mismatch: missing=%s unexpected=%s'
                % (non_semantic_missing, non_semantic_unexpected)
            )
        return
    model.load_state_dict(state_dict)


def apply_cli_overrides(args, cfg):
    apply_dataset_cli_overrides(args, cfg)
    if args.model is not None:
        cfg.model.type = args.model
        if args.model == 'card':
            cfg.model.enable_aux_mask = False
            cfg.model.use_aux_mask = False
            cfg.train.use_aux_mask = False
            cfg.train.use_semantic_aux = False
            cfg.train.use_aux_semantic = False
            cfg.train.use_semantic_cross_attention = False
            cfg.train.use_semantic_hard_gate = False
            cfg.train.use_feature_reweight = False
            cfg.model.semantic_input_mode = 'none'
    if args.use_aux_mask:
        cfg.model.enable_aux_mask = True
    if args.use_aux_semantic:
        cfg.train.use_semantic_aux = True
    if args.use_semantic_partial_detach:
        cfg.train.use_semantic_partial_detach = True
    if args.use_feature_reweight:
        cfg.train.use_feature_reweight = True
    if getattr(args, 'no_feature_reweight', False):
        cfg.train.use_feature_reweight = False
    if getattr(args, 'no_semantic_hard_gate', False):
        cfg.train.use_semantic_hard_gate = False
        if str(getattr(cfg.model, 'semantic_input_mode', 'none')).lower() == 'hard_gate':
            cfg.model.semantic_input_mode = 'none'
    if getattr(args, 'use_semantic_cross_attention', False):
        cfg.train.use_semantic_cross_attention = True
        cfg.model.semantic_input_mode = 'cross_attention'
    if getattr(args, 'use_semantic_hard_gate', False):
        cfg.train.use_semantic_hard_gate = True
        cfg.model.semantic_input_mode = 'hard_gate'
    if args.reweight_alpha is not None:
        cfg.train.reweight_alpha = args.reweight_alpha
    if args.lmask is not None:
        cfg.train.lambda_mask = args.lmask
    if args.lsem is not None:
        cfg.train.lambda_semantic = args.lsem
    if args.mask_loss_type is not None:
        cfg.train.mask_loss_type = args.mask_loss_type
    if args.semantic_loss_type is not None:
        cfg.train.semantic_loss_type = args.semantic_loss_type
    if args.semantic_detach_ratio is not None:
        cfg.train.semantic_detach_ratio = args.semantic_detach_ratio
    if args.paper_selection_mode:
        cfg.train.paper_selection_mode = True
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
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
parser.add_argument('--result_json', type=str, default=None)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--model', type=str, default=None, choices=['card', 'sgc_card'])
parser.add_argument('--data_root', type=str, default=None)
parser.add_argument('--levir_mci_root', type=str, default=None)
parser.add_argument('--second_cc_root', type=str, default=None)
parser.add_argument('--feature_root', type=str, default=None)
parser.add_argument('--use_change_mask', action='store_true')
parser.add_argument('--mask_type', type=str, default=None, choices=['binary', 'multiclass'])
parser.add_argument('--num_mask_classes', type=int, default=None)
parser.add_argument('--use_semantic_maps', action='store_true')
parser.add_argument('--semantic_input_mode', type=str, default=None, choices=['none', 'aux', 'early_fusion', 'cross_attention', 'hard_gate', 'weak_coupled'])
parser.add_argument('--num_semantic_classes', type=int, default=None)
parser.add_argument('--eval_change_nochange_split', action='store_true')
parser.add_argument('--paper_selection_mode', action='store_true')
parser.add_argument('--use_aux_mask', action='store_true')
parser.add_argument('--use_aux_semantic', action='store_true')
parser.add_argument('--use_semantic_partial_detach', action='store_true')
parser.add_argument('--semantic_detach_ratio', type=float, default=None)
parser.add_argument('--lmask', type=float, default=None)
parser.add_argument('--lsem', type=float, default=None)
parser.add_argument('--use_feature_reweight', action='store_true')
parser.add_argument('--no_feature_reweight', action='store_true')
parser.add_argument('--use_semantic_cross_attention', action='store_true')
parser.add_argument('--use_semantic_hard_gate', action='store_true')
parser.add_argument('--no_semantic_hard_gate', action='store_true')
parser.add_argument('--reweight_alpha', type=float, default=None)
parser.add_argument('--mask_loss_type', type=str, default=None)
parser.add_argument('--semantic_loss_type', type=str, default=None)
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
sync_wcsg_config_aliases(cfg)

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
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

snapshot_path_arg = args.snapshot_path or args.checkpoint
if args.snapshot_path is not None and args.checkpoint is not None:
    if os.path.normpath(args.snapshot_path) != os.path.normpath(args.checkpoint):
        raise ValueError('--snapshot_path and --checkpoint point to different files.')

if snapshot_path_arg is not None:
    snapshot_full_path = os.path.normpath(snapshot_path_arg)
elif args.snapshot is not None:
    snapshot_dir = os.path.join(output_dir, 'snapshots')
    snapshot_file = '%s_checkpoint_%d.pt' % (exp_name, args.snapshot)
    snapshot_full_path = os.path.join(snapshot_dir, snapshot_file)
else:
    raise ValueError('Either --snapshot, --snapshot_path, or --checkpoint must be provided.')
save_resolved_config(output_dir, cfg, args=args, checkpoint_path=snapshot_full_path, phase=args.split, log_path=os.path.join(output_dir, 'eval_log.txt'))
test_metrics_csv_path = os.path.join(output_dir, 'test_metrics.csv')
if args.split == 'test' and not os.path.exists(test_metrics_csv_path):
    write_single_row_csv(
        test_metrics_csv_path,
        {'checkpoint_path': '', 'Bleu_1': '', 'Bleu_2': '', 'Bleu_3': '', 'Bleu_4': '', 'METEOR': '', 'ROUGE_L': '', 'CIDEr': '', 'SPICE': '', 'Mask_Precision': '', 'Mask_Recall': '', 'Mask_F1': '', 'Mask_IoU': '', 'Mask_mIoU': ''},
        fieldnames=['checkpoint_path', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE', 'Mask_Precision', 'Mask_Recall', 'Mask_F1', 'Mask_IoU', 'Mask_mIoU'],
    )
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
    mask_metric_items = []
    semantic_metric_items = []
    for i, batch in tqdm(enumerate(eval_loader)):

        batch_data = unpack_batch(batch)
        d_feats = batch_data['feature_before']
        sc_feats = batch_data['feature_after']
        labels = batch_data['caption_tokens']
        masks = batch_data.get('mask')
        semantic_dense = batch_data.get('semantic_dense')
        semantic_before = batch_data.get('semantic_before')
        semantic_after = batch_data.get('semantic_after')
        semantic_diff = batch_data.get('semantic_diff')
        d_img_paths = batch_data.get('image_before')
        image_ids = batch_data.get('image_id')

        batch_size = d_feats.size(0)

        d_feats, sc_feats = d_feats.to(device), sc_feats.to(device)
        labels = labels.to(device)
        masks = move_optional_tensor(masks, device)
        semantic_dense = move_optional_tensor(semantic_dense, device)
        semantic_before = move_optional_tensor(semantic_before, device)
        semantic_after = move_optional_tensor(semantic_after, device)
        semantic_diff = move_optional_tensor(semantic_diff, device)

        change_outputs = change_detector(
            d_feats,
            sc_feats,
            semantic_before=semantic_before,
            semantic_after=semantic_after,
            semantic_diff=semantic_diff,
        )
        encoder_output, _, _, _, _, mask_pred, semantic_logits, _ = unpack_change_detector_output(change_outputs)

        mask_metrics = segmentation_metrics(mask_pred, masks, getattr(cfg.train, 'mask_ignore_index', -1))
        if mask_metrics is not None:
            mask_metric_items.append(mask_metrics)
        dense_sem_target = semantic_dense if semantic_dense is not None else semantic_diff
        if semantic_logits is not None and semantic_logits.dim() == 4:
            semantic_metrics = segmentation_metrics(
                semantic_logits,
                dense_sem_target,
                getattr(cfg.train, 'semantic_ignore_index', -1),
            )
            if semantic_metrics is not None:
                semantic_metric_items.append(semantic_metrics)

        speaker_output_pos, pos_dynamic_atts = speaker.sample(encoder_output, sample_max=1)

        gen_sents_pos = decode_sequence_transformer(idx_to_word, speaker_output_pos[:, 1:])

        for j in range(batch_size):
            gts = decode_sequence_transformer(idx_to_word, labels[j][:, 1:])
            sent_pos = gen_sents_pos[j]
            if image_ids is not None:
                image_id = os.path.basename(str(image_ids[j]))
            elif d_img_paths is not None:
                image_id = os.path.basename(str(d_img_paths[j]))
            else:
                image_id = str(i * cfg.data.batch_size + j)
            result_sents_pos[image_id] = sent_pos
            image_num = image_id.split('.')[0]

    test_iter_end_time = time.time() - test_iter_start_time
    print('%s inference took %.4f seconds' % (args.split, test_iter_end_time))
    coco_gen_format_save(result_sents_pos, result_save_path_pos)
    print('Saved captions to %s' % result_save_path_pos)
    aux_metrics = {}
    averaged_mask_metrics = average_metric_list(mask_metric_items)
    if averaged_mask_metrics:
        aux_metrics.update({
            'Mask_Precision': averaged_mask_metrics.get('precision', 0.0),
            'Mask_Recall': averaged_mask_metrics.get('recall', 0.0),
            'Mask_F1': averaged_mask_metrics.get('f1', 0.0),
            'Mask_IoU': averaged_mask_metrics.get('iou', 0.0),
            'Mask_mIoU': averaged_mask_metrics.get('miou', 0.0),
        })
        if 'class_1_iou' in averaged_mask_metrics:
            aux_metrics['IoU_road'] = averaged_mask_metrics['class_1_iou']
        if 'class_2_iou' in averaged_mask_metrics:
            aux_metrics['IoU_building'] = averaged_mask_metrics['class_2_iou']
    averaged_semantic_metrics = average_metric_list(semantic_metric_items)
    if averaged_semantic_metrics:
        aux_metrics.update({
            'Semantic_mIoU': averaged_semantic_metrics.get('miou', 0.0),
            'Semantic_IoU': averaged_semantic_metrics.get('iou', 0.0),
            'Semantic_F1': averaged_semantic_metrics.get('f1', 0.0),
        })
    if aux_metrics:
        aux_metric_path = os.path.join(os.path.dirname(result_save_path_pos), 'aux_metrics.json')
        with open(aux_metric_path, 'w', encoding='utf-8') as f:
            json.dump(aux_metrics, f, indent=2)
        print('Saved auxiliary metrics to %s' % aux_metric_path)



