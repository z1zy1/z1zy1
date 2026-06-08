import argparse
import os
import sys

import torch
import torch.nn as nn


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config_transformer import cfg, merge_cfg_from_file, merge_cfg_from_list
from models.CARD import CARD


def unpack_change_detector_output(outputs):
    if len(outputs) == 7:
        encoder_output, con_loss, ind_loss, att1, att2, mask_pred, semantic_logits = outputs
        relation_aux_logits = None
    elif len(outputs) == 8:
        encoder_output, con_loss, ind_loss, att1, att2, mask_pred, semantic_logits, relation_aux_logits = outputs
    else:
        raise RuntimeError('Expected CARD output with semantic logits, got %d tensors.' % len(outputs))
    return encoder_output, con_loss, ind_loss, att1, att2, mask_pred, semantic_logits, relation_aux_logits


def parse_args():
    parser = argparse.ArgumentParser(description='Quickly verify semantic detach gradient flags.')
    parser.add_argument(
        '--cfg',
        default='configs/dynamic/transformer_levir_cc_aux_mask_conf_semantic.yaml',
        help='Config file used to build CARD.',
    )
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--height', type=int, default=14)
    parser.add_argument('--width', type=int, default=14)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    merge_cfg_from_file(args.cfg)
    if args.opts:
        merge_cfg_from_list(args.opts)

    cfg.train.use_semantic_aux = True
    cfg.train.use_semantic_detach = True
    cfg.train.use_relation_aux = False
    cfg.train.use_weak_mask_prior = False

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device('cuda', cfg.gpu_id[0]) if use_cuda else torch.device('cpu')
    if use_cuda:
        torch.cuda.set_device(cfg.gpu_id[0])

    torch.manual_seed(cfg.train.seed)
    feat_dim = cfg.model.transformer_encoder.feat_dim

    model = CARD(cfg).to(device)
    model.train()

    input_1 = torch.randn(
        args.batch_size,
        feat_dim,
        args.height,
        args.width,
        device=device,
        requires_grad=True,
    )
    input_2 = torch.randn(
        args.batch_size,
        feat_dim,
        args.height,
        args.width,
        device=device,
        requires_grad=True,
    )

    outputs = model(input_1, input_2)
    encoder_output, _, _, _, _, _, semantic_logits, _ = unpack_change_detector_output(outputs)
    semantic_labels = torch.zeros_like(semantic_logits)
    loss_semantic = nn.BCEWithLogitsLoss()(semantic_logits, semantic_labels)
    loss_semantic.backward()

    debug_info = getattr(model, 'semantic_detach_debug', {})
    expected = {
        'semantic_input_requires_grad': 0.0,
        'diff_features_requires_grad': 1.0,
        'caption_input_requires_grad': 1.0,
        'semantic_logits_requires_grad': 1.0,
    }

    print('use_semantic_detach=True')
    print('encoder_output_shape=%s' % (tuple(encoder_output.shape),))
    for key in (
        'semantic_input_requires_grad',
        'diff_features_requires_grad',
        'caption_input_requires_grad',
        'semantic_logits_requires_grad',
    ):
        value = float(debug_info.get(key, -1.0))
        print('%s: %s' % (key, bool(value)))
        if value != expected[key]:
            raise RuntimeError(
                '%s expected %s but got %s.'
                % (key, bool(expected[key]), bool(value))
            )

    semantic_head_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.semantic_head.parameters()
    )
    print('semantic_head_has_grad: %s' % semantic_head_grad)
    if not semantic_head_grad:
        raise RuntimeError('semantic loss did not train semantic_head parameters.')

    print('semantic branch detach check passed.')


if __name__ == '__main__':
    main()
