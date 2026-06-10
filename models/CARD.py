import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_uniform_

from utils.semantic_label import ACTION_VOCAB, OBJECT_VOCAB
from utils.semantic_tags import read_semantic_tags


class CrossTransformer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, input1, input2):
        attn_output, attn_weight = self.attention(input1, input2, input2)
        output = input1 + self.dropout1(attn_output)
        output = self.norm1(output)
        return output, attn_weight


class AuxMaskHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = max(input_dim // 2, 1)
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)


class SemanticAuxHead(nn.Module):
    def __init__(self, input_dim, num_semantic_tags, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_semantic_tags),
        )

    def _pool(self, x):
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() == 4:
            return F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        raise ValueError('SemanticAuxHead expects [B, N, D] or [B, D, H, W], got %s' % (tuple(x.shape),))

    def forward(self, x):
        return self.net(self._pool(x))


class RelationAuxiliaryHead(nn.Module):
    def __init__(self, input_dim, num_objects, num_actions, dropout=0.1):
        super().__init__()
        self.num_objects = num_objects
        self.num_actions = num_actions
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.object_classifier = nn.Linear(input_dim, num_objects)
        self.action_classifier = nn.Linear(input_dim, num_actions)
        self.relation_classifier = nn.Linear(input_dim, num_actions * num_objects)

    def _pool(self, x):
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() == 4:
            return x.mean(dim=(2, 3))
        raise ValueError('RelationAuxiliaryHead expects [B, N, D] or [B, D, H, W], got %s' % (tuple(x.shape),))

    def forward(self, x):
        pooled = self._pool(x)
        hidden = self.proj(pooled)
        relation_logits = self.relation_classifier(hidden)
        relation_logits = relation_logits.view(-1, self.num_actions, self.num_objects)
        return {
            'objects': self.object_classifier(hidden),
            'actions': self.action_classifier(hidden),
            'relations': relation_logits,
        }


def partial_detach_feature(feat, detach_ratio: float):
    if detach_ratio <= 0:
        return feat
    if detach_ratio >= 1:
        return feat.detach()
    return feat.detach() * detach_ratio + feat * (1.0 - detach_ratio)


def reweight_feature_with_mask(feat, mask_logits, alpha: float, spatial_size=None):
    if mask_logits is None or alpha <= 0:
        return feat
    if mask_logits.dim() != 4 or mask_logits.size(1) != 1:
        raise ValueError(
            'Feature reweight expects mask logits with shape [B, 1, H, W], got %s.'
            % (tuple(mask_logits.shape),)
        )

    mask_prob = torch.sigmoid(mask_logits)
    if feat.dim() == 3:
        batch_size, num_tokens, _ = feat.shape
        if mask_prob.size(0) != batch_size:
            raise ValueError(
                'Feature reweight batch mismatch: feat=%s mask=%s.'
                % (tuple(feat.shape), tuple(mask_logits.shape))
            )
        if spatial_size is not None:
            expected_tokens = int(spatial_size[0]) * int(spatial_size[1])
            if num_tokens != expected_tokens:
                raise ValueError(
                    'Feature reweight token count %d does not match spatial size %s.'
                    % (num_tokens, spatial_size)
                )
            if mask_prob.shape[-2:] != spatial_size:
                mask_prob = F.interpolate(mask_prob, size=spatial_size, mode='bilinear', align_corners=False)
        mask_tokens = mask_prob.flatten(2).transpose(1, 2)
        if mask_tokens.size(1) != num_tokens:
            raise ValueError(
                'Feature reweight mask token count %d does not match feat token count %d.'
                % (mask_tokens.size(1), num_tokens)
            )
        return feat * (1.0 + float(alpha) * mask_tokens)

    if feat.dim() == 4:
        batch_size, _, height, width = feat.shape
        if mask_prob.size(0) != batch_size:
            raise ValueError(
                'Feature reweight batch mismatch: feat=%s mask=%s.'
                % (tuple(feat.shape), tuple(mask_logits.shape))
            )
        if mask_prob.shape[-2:] != (height, width):
            mask_prob = F.interpolate(mask_prob, size=(height, width), mode='bilinear', align_corners=False)
        return feat * (1.0 + float(alpha) * mask_prob)

    raise ValueError('Feature reweight expects [B, N, C] or [B, C, H, W], got %s.' % (tuple(feat.shape),))


class CARD(nn.Module):

    def __init__(self, cfg, temp=0.07):
        super().__init__()
        self.enable_aux_mask = cfg.model.enable_aux_mask
        self.use_semantic_aux = bool(cfg.train.use_semantic_aux)
        self.use_semantic_detach = bool(getattr(cfg.train, 'use_semantic_detach', False))
        self.use_semantic_partial_detach = bool(getattr(cfg.train, 'use_semantic_partial_detach', False))
        self.semantic_update_visual = bool(getattr(cfg.train, 'semantic_update_visual', True))
        if self.use_semantic_detach and self.use_semantic_partial_detach:
            raise ValueError('use_semantic_detach and use_semantic_partial_detach cannot both be True')
        self.use_relation_aux = bool(cfg.train.use_relation_aux)
        self.use_weak_mask_prior = bool(cfg.train.use_weak_mask_prior)
        self.mask_alpha = cfg.train.mask_alpha
        self.use_feature_reweight = bool(getattr(cfg.train, 'use_feature_reweight', False))
        self.reweight_alpha = float(getattr(cfg.train, 'reweight_alpha', 0.2))
        self.semantic_detach_ratio = float(getattr(cfg.train, 'semantic_detach_ratio', 0.5))
        self.semantic_detach_debug = {}
        self.temp = nn.Parameter(torch.ones([]) * temp)
        self.feat_dim = cfg.model.transformer_encoder.feat_dim
        self.att_dim = cfg.model.transformer_encoder.att_dim
        self.att_head = cfg.model.transformer_encoder.att_head

        self.embed_dim = cfg.model.transformer_encoder.emb_dim

        self.img = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.att_dim, kernel_size=1, padding=0),
        )

        self.positions = nn.Parameter(torch.randn(14 ** 2+1, self.att_dim))

        self.encoder = CrossTransformer(self.att_dim, self.att_head)
        self.bef_cls_token = nn.Parameter(torch.randn(1, 1, self.att_dim))
        self.aft_cls_token = nn.Parameter(torch.randn(1, 1, self.att_dim))
        self.common_proj = nn.Linear(self.att_dim, self.att_dim)

        self.bef_diff = nn.Linear(self.att_dim, self.att_dim)

        self.aft_diff = nn.Linear(self.att_dim, self.att_dim)

        self.common_trans = nn.Sequential(
            nn.Linear(self.att_dim * 2, self.embed_dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        self.diff_trans = nn.Sequential(
            nn.Linear(self.att_dim * 2, self.embed_dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.diff_trans2 = nn.Sequential(
            nn.Linear(self.att_dim * 2, self.embed_dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.num_hidden_layers = cfg.model.transformer_encoder.att_layer

        self.transformer = CrossTransformer(self.att_dim, self.att_head)

        self.fc = nn.Sequential(
            nn.Linear(self.att_dim * 2, self.embed_dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        self.aux_mask_head = AuxMaskHead(self.embed_dim) if (
            self.enable_aux_mask or self.use_weak_mask_prior or self.use_feature_reweight
        ) else None
        self.semantic_tags = []
        self.num_semantic_tags = 0
        self.semantic_head = None
        if self.use_semantic_aux:
            self.semantic_tags = read_semantic_tags(cfg.train.semantic_tag_file)
            self.num_semantic_tags = len(self.semantic_tags)
            self.semantic_head = SemanticAuxHead(
                self.embed_dim,
                self.num_semantic_tags,
                dropout=cfg.train.semantic_aux_dropout,
            )
        self.relation_aux_head = None
        if self.use_relation_aux:
            self.relation_aux_head = RelationAuxiliaryHead(
                self.embed_dim,
                len(OBJECT_VOCAB),
                len(ACTION_VOCAB),
                dropout=cfg.train.relation_aux_dropout,
            )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def pairwise_distances(self, x):
        # x should be two dimensional
        instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
        return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()

    def GaussianKernelMatrix(self, x, sigma=1):
        pairwise_distances_ = self.pairwise_distances(x)
        return torch.exp(-pairwise_distances_ / sigma)

    def forward(self, input_1, input_2):
        self.semantic_detach_debug = {}
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        batch_size, C, H, W = input_1.size()

        input_1 = self.img(input_1)  # (128,196, 512)
        input_2 = self.img(input_2)

        input_1 = input_1.view(batch_size, self.att_dim, -1).permute(2, 0, 1) # (L, B, D)
        input_2 = input_2.view(batch_size, self.att_dim, -1).permute(2, 0, 1)

        bef_cls_token = self.bef_cls_token.repeat(1, batch_size, 1)
        aft_cls_token = self.aft_cls_token.repeat(1, batch_size, 1)

        input_1 = torch.cat([bef_cls_token, input_1], 0)
        input_2 = torch.cat([aft_cls_token, input_2], 0)

        positions = self.positions.unsqueeze(1).repeat(1, batch_size, 1)

        input_1 = input_1 + positions  # (batch, att_dim, h, w)
        input_2 = input_2 + positions

        input_1, _ = self.encoder(input_1, input_1)
        input_2, _ = self.encoder(input_2, input_2)

        ########################################
        ### common context feature decoupling and consistency constraint
        bef_common = self.common_proj(input_1[0, :, :])
        aft_common = self.common_proj(input_2[0, :, :])
        bef_cls_feat = F.normalize(bef_common, dim=-1)
        aft_cls_feat = F.normalize(aft_common, dim=-1)

        sim_b2a = bef_cls_feat @ aft_cls_feat.t() / self.temp
        sim_a2b = aft_cls_feat @ bef_cls_feat.t() / self.temp

        sim_targets = torch.zeros_like(sim_b2a)
        sim_targets[:, :] = torch.eye(batch_size, device=sim_targets.device)

        loss_b2a = -torch.sum(F.log_softmax(sim_b2a, dim=1) * sim_targets, dim=1).mean()
        loss_a2b = -torch.sum(F.log_softmax(sim_a2b, dim=1) * sim_targets, dim=1).mean()
        loss_con = (loss_b2a + loss_a2b) / 2
        ##################################################

        ##################################################
        ### difference context feature decoupling and independence constraint
        bef_diff = self.bef_diff(input_1[0, :, :])
        aft_diff = self.aft_diff(input_2[0, :, :])

        bef_diff_norm = F.normalize(bef_diff, dim=-1)
        aft_diff_norm = F.normalize(aft_diff, dim=-1)

        # HISC
        s_x = 1
        s_y = 1
        m, _, = bef_diff_norm.shape  # batch size
        K = self.GaussianKernelMatrix(bef_diff_norm, s_x)
        L = self.GaussianKernelMatrix(aft_diff_norm, s_y)
        hsic_center = torch.eye(m, device=bef_diff_norm.device) - 1.0 / m * torch.ones((m, m), device=bef_diff_norm.device)
        loss_ind = torch.trace(torch.mm(L, torch.mm(hsic_center, torch.mm(K, hsic_center)))) / ((m - 1) ** 2)
        ################

        input_1_common = torch.cat([bef_common.unsqueeze(0).expand_as(input_1[1:, :, :]), input_1[1:, :, :]], dim=-1)
        input_1_common = self.common_trans(input_1_common)
        input_2_common = torch.cat([aft_common.unsqueeze(0).expand_as(input_2[1:, :, :]), input_2[1:, :, :]], dim=-1)
        input_2_common = self.common_trans(input_2_common)
        input_pre1 = input_1[1:, :, :]
        input_pre2 = input_2[1:, :, :]

        input_1_common, att1 = self.transformer(input_1_common, input_2_common)
        input_2_common, att2 = self.transformer(input_2_common, input_1_common)

        input_1_diff = input_pre1 - input_1_common

        input_1_diff = torch.cat([bef_diff.unsqueeze(0).expand_as(input_1[1:, :, :]), input_1_diff[:, :, :]], dim=-1)
        input_1_diff = self.diff_trans(input_1_diff)
        input_2_diff = input_pre2 - input_2_common
        input_2_diff = torch.cat([aft_diff.unsqueeze(0).expand_as(input_2[1:, :, :]), input_2_diff[:, :, :]], dim=-1)
        input_2_diff = self.diff_trans2(input_2_diff)
        input_1_diff = input_1_diff.permute(1, 0, 2)
        input_2_diff = input_2_diff.permute(1, 0, 2)

        output = torch.cat([input_1_diff, input_2_diff], -1)
        output = self.fc(output)
        mask_pred = None
        semantic_logits = None
        relation_aux_logits = None
        if self.aux_mask_head is not None:
            aux_feat = output.permute(0, 2, 1).contiguous().view(batch_size, self.embed_dim, H, W)
            mask_pred = self.aux_mask_head(aux_feat)
        if self.use_weak_mask_prior:
            if mask_pred is None:
                raise ValueError('Weak mask prior requires an auxiliary mask prediction.')
            mask_tokens = torch.sigmoid(mask_pred).flatten(2).transpose(1, 2)
            output = output * (1.0 + self.mask_alpha * mask_tokens)
        diff_features = output
        caption_input = diff_features
        if self.use_feature_reweight:
            caption_input = reweight_feature_with_mask(
                diff_features,
                mask_pred,
                self.reweight_alpha,
                spatial_size=(H, W),
            )
        if self.use_semantic_aux:
            if self.use_semantic_detach:
                semantic_input = diff_features.detach()
            elif self.use_semantic_partial_detach:
                semantic_input = partial_detach_feature(diff_features, self.semantic_detach_ratio)
            elif not self.semantic_update_visual:
                semantic_input = diff_features.detach()
            else:
                semantic_input = diff_features
            semantic_logits = self.semantic_head(semantic_input)
            self.semantic_detach_debug = {
                'semantic_input_requires_grad': float(semantic_input.requires_grad),
                'diff_features_requires_grad': float(diff_features.requires_grad),
                'caption_input_requires_grad': float(caption_input.requires_grad),
                'semantic_logits_requires_grad': float(semantic_logits.requires_grad),
                'semantic_branch_depends_on_decoder': 0.0,
                'semantic_detach_ratio': self.semantic_detach_ratio if self.use_semantic_partial_detach else 0.0,
                'feature_reweight_enabled': float(self.use_feature_reweight),
            }
        if self.use_relation_aux:
            relation_aux_logits = self.relation_aux_head(caption_input)

        if self.use_relation_aux:
            return caption_input, loss_con, loss_ind, att1, att2, mask_pred, semantic_logits, relation_aux_logits
        if self.use_semantic_aux:
            return caption_input, loss_con, loss_ind, att1, att2, mask_pred, semantic_logits
        return caption_input, loss_con, loss_ind, att1, att2, mask_pred


class AddSpatialInfo(nn.Module):

    def _create_coord(self, img_feat):
        batch_size, _, h, w = img_feat.size()
        coord_map = img_feat.new_zeros(2, h, w)
        for i in range(h):
            for j in range(w):
                coord_map[0][i][j] = (j * 2.0 / w) - 1
                coord_map[1][i][j] = (i * 2.0 / h) - 1
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, img_feat):
        coord_map = self._create_coord(img_feat)
        img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
        return img_feat_aug

