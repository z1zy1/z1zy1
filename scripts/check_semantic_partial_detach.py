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
from models.transformer_decoder import DynamicSpeaker


def unpack_change_detector_output(outputs):
    if len(outputs) == 7:
        encoder_output, con_loss, ind_loss, att1, att2, mask_pred, semantic_logits = outputs
        relation_aux_logits = None
    elif len(outputs) == 8:
        encoder_output, con_loss, ind_loss, att1, att2, mask_pred, semantic_logits, relation_aux_logits = outputs
    else:
        raise RuntimeError('Expected CARD output with semantic logits, got %d tensors.' % len(outputs))
    return encoder_output, con_loss, ind_loss, att1, att2, mask_pred, semantic_logits, relation_aux_logits


def grad_norm(parameters):
    total_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        value = p.grad.detach().float().norm(2).item()
        total_sq += value * value
    return total_sq ** 0.5


def set_requires_grad(parameters, requires_grad):
    previous = []
    for p in parameters:
        previous.append(p.requires_grad)
        p.requires_grad_(requires_grad)
    return previous


def restore_requires_grad(parameters, previous):
    for p, requires_grad in zip(parameters, previous):
        p.requires_grad_(requires_grad)


def parse_args():
    parser = argparse.ArgumentParser(description='Dry-run semantic partial detach gradient blocking.')
    parser.add_argument(
        '--cfg',
        default='configs/dynamic/transformer_levir_cc_aux_mask_conf_semantic.yaml',
        help='Config file used to build CARD and the decoder.',
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
    cfg.train.use_semantic_detach = False
    cfg.train.use_semantic_partial_detach = True
    cfg.train.semantic_update_visual = True
    cfg.train.use_relation_aux = False
    cfg.train.use_weak_mask_prior = False

    if cfg.train.use_semantic_detach and cfg.train.use_semantic_partial_detach:
        raise RuntimeError('use_semantic_detach and use_semantic_partial_detach cannot both be True')

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device('cuda', cfg.gpu_id[0]) if use_cuda else torch.device('cpu')
    if use_cuda:
        torch.cuda.set_device(cfg.gpu_id[0])

    torch.manual_seed(cfg.train.seed)
    feat_dim = cfg.model.transformer_encoder.feat_dim

    change_detector = CARD(cfg).to(device)
    decoder = DynamicSpeaker(cfg).to(device)
    optimizer = torch.optim.SGD(
        list(change_detector.parameters()) + list(decoder.parameters()),
        lr=1e-5,
    )
    change_detector.train()
    decoder.train()

    input_1 = torch.randn(args.batch_size, feat_dim, args.height, args.width, device=device)
    input_2 = torch.randn(args.batch_size, feat_dim, args.height, args.width, device=device)

    optimizer.zero_grad()
    outputs = change_detector(input_1, input_2)
    encoder_output, con_loss, ind_loss, _, _, _, semantic_logits, _ = unpack_change_detector_output(outputs)
    semantic_labels = torch.zeros_like(semantic_logits)

    main_loss = encoder_output.sum() * 0.0 + con_loss * 0.001 + ind_loss * 0.001
    semantic_loss = nn.BCEWithLogitsLoss()(semantic_logits, semantic_labels)

    main_loss.backward(retain_graph=True)
    decoder_grad_after_main = grad_norm(decoder.parameters())

    decoder_requires_grad = set_requires_grad(decoder.parameters(), False)
    decoder_frozen_before_semantic = all(not p.requires_grad for p in decoder.parameters())
    try:
        semantic_loss.backward()
    finally:
        restore_requires_grad(decoder.parameters(), decoder_requires_grad)
    decoder_restored_after_semantic = all(p.requires_grad == prev for p, prev in zip(decoder.parameters(), decoder_requires_grad))

    decoder_grad_after_semantic = grad_norm(decoder.parameters())
    semantic_head_grad = grad_norm(change_detector.semantic_head.parameters())
    debug_info = getattr(change_detector, 'semantic_detach_debug', {})
    semantic_input_requires_grad = bool(debug_info.get('semantic_input_requires_grad', 0.0))

    optimizer.step()

    print('use_semantic_partial_detach=True')
    print('semantic_update_visual=True')
    print('semantic_branch_depends_on_decoder=False')
    print('semantic_input_requires_grad: %s' % semantic_input_requires_grad)
    print('semantic_head_grad_norm: %.8f' % semantic_head_grad)
    print('decoder_requires_grad_false_during_semantic_backward: %s' % decoder_frozen_before_semantic)
    print('decoder_requires_grad_restored_after_semantic_backward: %s' % decoder_restored_after_semantic)
    print('decoder_grad_norm_after_main_loss: %.8f' % decoder_grad_after_main)
    print('decoder_grad_norm_after_semantic_loss: %.8f' % decoder_grad_after_semantic)

    if not semantic_input_requires_grad:
        raise RuntimeError('semantic_input.requires_grad should be True for partial detach with semantic_update_visual=True.')
    if semantic_head_grad <= 0:
        raise RuntimeError('semantic_head did not receive gradients.')
    if abs(decoder_grad_after_semantic - decoder_grad_after_main) > 1e-12:
        raise RuntimeError('decoder gradients changed during semantic backward.')
    if not decoder_frozen_before_semantic or not decoder_restored_after_semantic:
        raise RuntimeError('decoder parameters were not correctly frozen/restored.')

    print('semantic partial detach check passed.')


if __name__ == '__main__':
    main()
