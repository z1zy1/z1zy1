import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_uniform_

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
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
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

    def forward(self, x):
        return self.net(x)


class CARD(nn.Module):

    def __init__(self, cfg, temp=0.07):
        super().__init__()
        self.enable_aux_mask = cfg.model.enable_aux_mask
        self.use_semantic_aux = bool(cfg.train.use_semantic_aux)
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
        self.aux_mask_head = AuxMaskHead(self.embed_dim) if self.enable_aux_mask else None
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
        sim_targets[:, :] = torch.eye(batch_size)

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
        if self.enable_aux_mask:
            aux_feat = output.permute(0, 2, 1).contiguous().view(batch_size, self.embed_dim, H, W)
            mask_pred = self.aux_mask_head(aux_feat)
        if self.use_semantic_aux:
            semantic_logits = self.semantic_head(output.mean(dim=1))

        if self.use_semantic_aux:
            return output, loss_con, loss_ind, att1, att2, mask_pred, semantic_logits
        return output, loss_con, loss_ind, att1, att2, mask_pred


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

