import argparse
import json
import os
from collections import Counter

import numpy as np
from imageio.v2 import imread

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


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


PALETTE = np.array([
    [0, 0, 255],
    [0, 180, 0],
    [220, 0, 0],
    [160, 160, 160],
    [255, 255, 255],
    [0, 0, 0],
    [255, 180, 0],
    [180, 0, 255],
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description='SECOND-CC semantic head sanity check.')
    parser.add_argument('--cfg', default=os.path.join('configs', 'dynamic', 'transformer_second_cc_aug_sgc_card.yaml'))
    parser.add_argument('--data_root', default='SECOND-CC-AUG')
    parser.add_argument('--feature_root', default=None)
    parser.add_argument('--output_dir', default=os.path.join('experiments', 'sanity_check', 'second_cc_semantic_head'))
    parser.add_argument('--split', default='train')
    parser.add_argument('--max_samples', type=int, default=50)
    parser.add_argument('--forward_batch', action='store_true', help='Run a random-initialized semantic head forward pass when features are available.')
    return parser.parse_args()


class DatasetArgs:
    def __init__(self, data_root, feature_root):
        self.dataset = 'second_cc'
        self.data_root = data_root
        self.levir_mci_root = None
        self.second_cc_root = data_root
        self.feature_root = feature_root
        self.use_change_mask = False
        self.mask_type = None
        self.num_mask_classes = None
        self.use_semantic_maps = True
        self.semantic_input_mode = 'cross_attention'
        self.num_semantic_classes = 7
        self.eval_change_nochange_split = False


def candidate_paths(root, split_name, phase_name, filename):
    stem = os.path.splitext(filename)[0]
    names = [filename, stem + '.png', stem + '.jpg', stem + '.jpeg', stem + '.npy']
    bases = []
    if split_name and phase_name:
        bases.extend([os.path.join(root, split_name, phase_name), os.path.join(root, phase_name, split_name)])
    if split_name:
        bases.append(os.path.join(root, split_name))
    if phase_name:
        bases.append(os.path.join(root, phase_name))
    bases.append(root)
    for base in bases:
        for name in names:
            yield os.path.join(base, name)


def first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def load_splits(root, split):
    path = os.path.join(root, 'splits.json')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    indices = data.get(split, [])
    return data, indices


def load_semantic_map(path, ignore_index=-1):
    if not path:
        return None
    arr = np.load(path) if path.lower().endswith('.npy') else imread(path)
    return _rgb_to_class_map(
        arr,
        SECOND_SEMANTIC_RGB_TO_CLASS,
        ignore_index=ignore_index,
        known_colors=SECOND_KNOWN_COLORS,
        known_classes=SECOND_KNOWN_CLASSES,
    )


def class_counter_update(counter, arr):
    if arr is None:
        return
    values, counts = np.unique(arr, return_counts=True)
    for value, count in zip(values, counts):
        counter[int(value)] += int(count)


def semantic_diff(before, after, ignore_index=-1):
    valid = (before != ignore_index) & (after != ignore_index)
    diff = np.zeros_like(after, dtype=np.int64)
    changed = (before != after) & valid
    diff[changed] = after[changed] + 1
    diff[~valid] = ignore_index
    return diff


def colorize_class_map(arr):
    if arr is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    safe = np.where(arr < 0, 0, arr).astype(np.int64)
    safe = np.clip(safe, 0, len(PALETTE) - 1)
    return PALETTE[safe]


def make_visualization(paths, sem_before, sem_after, diff, output_path):
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        return 'PIL unavailable: %s' % exc
    panels = []
    labels = []
    for label, path in [('rgb_before', paths.get('rgb_before')), ('rgb_after', paths.get('rgb_after'))]:
        if path and os.path.exists(path):
            arr = imread(path)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
        else:
            arr = np.zeros((256, 256, 3), dtype=np.uint8)
        panels.append(Image.fromarray(arr.astype(np.uint8)).resize((160, 160)))
        labels.append(label)
    for label, arr in [('sem_before', sem_before), ('sem_after', sem_after), ('sem_diff', diff)]:
        panels.append(Image.fromarray(colorize_class_map(arr)).resize((160, 160), resample=Image.Resampling.NEAREST))
        labels.append(label)
    canvas = Image.new('RGB', (160 * len(panels), 184), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, panel in enumerate(panels):
        canvas.paste(panel, (160 * i, 24))
        draw.text((160 * i + 4, 4), labels[i], fill=(0, 0, 0))
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    canvas.save(output_path)
    return None


def inspect_semantic_maps(root, split, max_samples, output_dir):
    splits, indices = load_splits(root, split)
    idx_to_filename = splits.get('idx_to_filename', {})
    idx_to_split = splits.get('idx_to_split', {})
    before_counter = Counter()
    after_counter = Counter()
    diff_counter = Counter()
    samples = []
    visual_dir = os.path.join(output_dir, 'visualizations')
    for n, img_idx in enumerate(indices[:max_samples]):
        key = str(int(img_idx))
        filename = idx_to_filename.get(key)
        split_name = idx_to_split.get(key, split)
        if not filename:
            continue
        sem_before_path = first_existing(candidate_paths(root, split_name, os.path.join('sem', 'A'), filename))
        sem_after_path = first_existing(candidate_paths(root, split_name, os.path.join('sem', 'B'), filename))
        rgb_before_path = first_existing(candidate_paths(root, split_name, os.path.join('rgb', 'A'), filename))
        rgb_after_path = first_existing(candidate_paths(root, split_name, os.path.join('rgb', 'B'), filename))
        sem_before = load_semantic_map(sem_before_path) if sem_before_path else None
        sem_after = load_semantic_map(sem_after_path) if sem_after_path else None
        diff = semantic_diff(sem_before, sem_after) if sem_before is not None and sem_after is not None else None
        class_counter_update(before_counter, sem_before)
        class_counter_update(after_counter, sem_after)
        class_counter_update(diff_counter, diff)
        vis_path = os.path.join(visual_dir, '%03d_%s.png' % (n, os.path.splitext(filename)[0]))
        vis_error = make_visualization(
            {'rgb_before': rgb_before_path, 'rgb_after': rgb_after_path},
            sem_before,
            sem_after,
            diff,
            vis_path,
        )
        samples.append({
            'filename': filename,
            'split': split_name,
            'semantic_before_path': sem_before_path or '',
            'semantic_after_path': sem_after_path or '',
            'rgb_before_path': rgb_before_path or '',
            'rgb_after_path': rgb_after_path or '',
            'visualization': '' if vis_error else vis_path,
            'visualization_error': vis_error or '',
            'before_shape': list(sem_before.shape) if sem_before is not None else None,
            'after_shape': list(sem_after.shape) if sem_after is not None else None,
            'diff_has_change': bool(diff is not None and (diff > 0).any()),
        })
    return {
        'sample_count': len(samples),
        'class_distribution_before': dict(sorted(before_counter.items())),
        'class_distribution_after': dict(sorted(after_counter.items())),
        'class_distribution_diff': dict(sorted(diff_counter.items())),
        'samples': samples,
    }


def align_class_tensor(tensor, spatial_size, ignore_index=-1):
    if tensor.dim() == 4 and tensor.size(1) == 1:
        tensor = tensor[:, 0]
    if tensor.shape[-2:] != spatial_size:
        tensor = F.interpolate(tensor.unsqueeze(1).float(), size=spatial_size, mode='nearest').squeeze(1)
    return tensor.long()


def segmentation_miou(logits, target, ignore_index=-1):
    pred = torch.argmax(logits, dim=1)
    target = align_class_tensor(target, logits.shape[-2:], ignore_index=ignore_index)
    valid = target != int(ignore_index)
    ious = []
    for cls_id in range(1, logits.size(1)):
        cls_pred = (pred == cls_id) & valid
        cls_gold = (target == cls_id) & valid
        union = (cls_pred | cls_gold).sum().item()
        if union:
            ious.append(((cls_pred & cls_gold).sum().item()) / float(union))
    return float(np.mean(ious)) if ious else 0.0


def run_forward_check(args):
    if torch is None:
        return {'semantic_head_forward_ok': False, 'error': 'PyTorch is not installed in this environment.'}
    try:
        from configs.config_transformer import cfg, merge_cfg_from_file
        from datasets.datasets import create_dataset
        from models.CARD import CARD
        from utils.dataset_config import apply_dataset_cli_overrides
        from utils.experiment_tracking import sync_wcsg_config_aliases

        merge_cfg_from_file(args.cfg)
        apply_dataset_cli_overrides(DatasetArgs(args.data_root, args.feature_root), cfg)
        cfg.model.type = 'sgc_card'
        cfg.model.enable_aux_mask = False
        cfg.train.use_semantic_aux = True
        cfg.train.use_aux_semantic = True
        cfg.train.lambda_semantic = 0.0005
        cfg.train.lambda_mask = 0.0
        cfg.train.use_semantic_partial_detach = True
        cfg.train.semantic_detach_ratio = 0.5
        cfg.train.semantic_loss_type = 'ce_dice'
        cfg.model.semantic_input_mode = 'cross_attention'
        cfg.data.use_semantic_maps = True
        cfg.data.return_dict = True
        cfg.data.train.batch_size = 2
        sync_wcsg_config_aliases(cfg)
        dataset, loader = create_dataset(cfg, args.split)
        batch = next(iter(loader))
        model = CARD(cfg)
        model.eval()
        with torch.no_grad():
            outputs = model(
                batch['feature_before'].float(),
                batch['feature_after'].float(),
                semantic_before=batch.get('semantic_before'),
                semantic_after=batch.get('semantic_after'),
                semantic_diff=batch.get('semantic_diff'),
            )
            semantic_logits = outputs[6] if len(outputs) > 6 else None
            target = batch.get('semantic_dense')
            if target is None:
                target = batch.get('semantic_diff')
            result = {
                'semantic_head_forward_ok': semantic_logits is not None,
                'semantic_logits_shape': list(semantic_logits.shape) if semantic_logits is not None else None,
                'semantic_loss_nonzero': False,
                'semantic_loss': None,
                'semantic_prediction_class_distribution': {},
                'semantic_mIoU': None,
                'semantic_mIoU_close_to_zero': None,
            }
            if semantic_logits is not None and target is not None:
                target = align_class_tensor(target.long(), semantic_logits.shape[-2:])
                loss = F.cross_entropy(semantic_logits, target, ignore_index=int(getattr(cfg.train, 'semantic_ignore_index', -1)))
                pred = torch.argmax(semantic_logits, dim=1)
                values, counts = torch.unique(pred.cpu(), return_counts=True)
                miou = segmentation_miou(semantic_logits, target, int(getattr(cfg.train, 'semantic_ignore_index', -1)))
                result.update({
                    'semantic_loss_nonzero': bool(loss.item() != 0.0),
                    'semantic_loss': float(loss.item()),
                    'semantic_prediction_class_distribution': {str(int(v)): int(c) for v, c in zip(values, counts)},
                    'semantic_mIoU': miou,
                    'semantic_mIoU_close_to_zero': bool(miou < 0.01),
                })
            return result
    except Exception as exc:
        return {'semantic_head_forward_ok': False, 'error': str(exc)}


def write_report(output_dir, payload):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'sanity_report.json')
    txt_path = os.path.join(output_dir, 'sanity_report.txt')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    lines = [
        '# SECOND-CC semantic head sanity check',
        'semantic_class_number: %s' % payload['semantic_class_number'],
        'ignore_index: %s' % payload['ignore_index'],
        'semantic_map_resize_interpolation: nearest for class-id targets; semantic embeddings may be bilinear-resized only after class IDs are embedded.',
        'uses_nearest_for_class_id_resize: %s' % payload['uses_nearest_for_class_id_resize'],
        'class_distribution_before: %s' % payload['map_inspection']['class_distribution_before'],
        'class_distribution_after: %s' % payload['map_inspection']['class_distribution_after'],
        'class_distribution_diff: %s' % payload['map_inspection']['class_distribution_diff'],
        'visualization_dir: %s' % os.path.join(output_dir, 'visualizations'),
        'forward_check: %s' % payload['forward_check'],
    ]
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(json_path)
    print(txt_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    map_info = inspect_semantic_maps(args.data_root, args.split, args.max_samples, args.output_dir)
    forward = run_forward_check(args) if args.forward_batch else {'semantic_head_forward_ok': False, 'skipped': 'pass --forward_batch to instantiate CARD and run one batch'}
    payload = {
        'semantic_class_number': 7,
        'ignore_index': -1,
        'known_rgb_to_class': {str(key): value for key, value in SECOND_SEMANTIC_RGB_TO_CLASS.items()},
        'uses_nearest_for_class_id_resize': True,
        'semantic_map_resize_interpolation': 'nearest for class-id maps/targets',
        'map_inspection': map_info,
        'forward_check': forward,
    }
    write_report(args.output_dir, payload)


if __name__ == '__main__':
    main()



