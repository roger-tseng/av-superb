# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import numpy as np
import timm.models.vision_transformer
import torch
import torch.nn as nn
import torchaudio
from timm.models.vision_transformer import Block, PatchEmbed

from .util.patch_embed import PatchEmbed3D_new, PatchEmbed_new
from .util.pos_embed import get_3d_sincos_pos_embed


class VisionTransformerMM(timm.models.vision_transformer.VisionTransformer):
    """Mulitmodal Vision Transformer with support for global average pooling (assume loading both audio and video)"""

    def __init__(
        self,
        global_pool=False,
        mask_2d=True,
        av_fusion=False,
        n_frm=8,
        depth_av=0,
        pos_train=False,
        ft=False,
        roll_mag_aug=False,
        dataset="audioset",
        **kwargs
    ):
        super(VisionTransformerMM, self).__init__(**kwargs)
        self.av_fusion = av_fusion
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]

            if not self.av_fusion:
                self.fc_norm_a = norm_layer(embed_dim)
                self.fc_norm_v = norm_layer(embed_dim)
                self.head_a = nn.Linear(embed_dim, self.num_classes)
                self.head_v = nn.Linear(embed_dim, self.num_classes)

            del self.norm
            # must deleete otherwise distributed
        else:
            assert 0  # not implemented yet
        self.mask_2d = mask_2d

        # video branch
        num_heads = kwargs["num_heads"]
        depth = kwargs["depth"]
        mlp_ratio = kwargs["mlp_ratio"]

        patch_size = 16
        self.temporal_kernel = 2
        self.temporal_stride = 2
        self.n_frm = n_frm

        self.patch_embed_v = PatchEmbed3D_new(
            video_size=(n_frm, 224, 224),
            patch_size=(self.temporal_kernel, patch_size, patch_size),
            stride=(self.temporal_stride, patch_size, patch_size),
            in_chans=3,
            embed_dim=embed_dim,
        )

        self.num_patches_v = self.patch_embed_v.num_patches
        self.cls_token_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_v = nn.Parameter(
            torch.zeros(1, self.num_patches_v + 1, embed_dim), requires_grad=pos_train
        )
        self.blocks_v = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.av_fusion = av_fusion
        self.depth_av = depth_av
        if self.av_fusion:
            self.blocks_av = nn.ModuleList(
                [
                    Block(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        qk_scale=None,
                        norm_layer=norm_layer,
                    )
                    for i in range(self.depth_av)
                ]
            )
            if ft:
                self.fc_norm_av = norm_layer(2 * embed_dim)
                self.head_av = nn.Linear(2 * embed_dim, self.num_classes)

        del self.head
        # must deleete otherwise distributed complains
        self.initialize_weights()

    def initialize_weights(self):
        pass

    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "pos_embed_v", "cls_token_v"}

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, length, dim
        # ## for AS
        T = 64
        F = 8

        # mask T
        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        x = torch.gather(x, dim=1, index=index)  # N, len_keep_T(T'), F, D

        # mask F
        x = x.permute(0, 2, 1, 3)  # N T' F D => N F T' D
        len_keep_F = int(F * (1 - mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0, 2, 1, 3)  # N F' T' D => N T' F' D
        x_masked = x_masked.reshape(N, len_keep_F * len_keep_T, D)

        return x_masked, None, None

    def forward_features_a(self, x):
        B = x.shape[0]

        # audio
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(
            B, -1, -1
        )  # cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm_a(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward_features_av(self, x, v):
        B = x.shape[0]

        # audio
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(
            B, -1, -1
        )  # cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        # audio encoding
        for blk in self.blocks:
            x = blk(x)

        # video
        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v[:, 1:, :]
        cls_token_v = self.cls_token_v + self.pos_embed_v[:, :1, :]
        cls_tokens_v = cls_token_v.expand(B, -1, -1)
        v = torch.cat((cls_tokens_v, v), dim=1)
        v = self.pos_drop(v)

        # video encoding
        for blk in self.blocks_v:
            v = blk(v)

        if self.av_fusion:
            xv = torch.cat((x, v), dim=1)
            for blk in self.blocks_av:
                xv = blk(xv)
            x_len = x.shape[1]
            x = xv[:, 1:x_len, :].mean(dim=1)  # global pool without cls token
            v = xv[:, x_len + 1 :, :].mean(dim=1)  # global pool without cls token
            outcome_av = self.fc_norm_av(torch.cat((x, v), dim=1))
            return outcome_av, outcome_av
        else:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            v = v[:, 1:, :].mean(dim=1)  # global pool without cls token

            outcome_a = self.fc_norm_a(x)
            outcome_v = self.fc_norm_v(v)
        return outcome_a, outcome_v

    def forward_features_av_mask(self, x, v, mask_t_prob, mask_f_prob, mask_v_prob):
        B = x.shape[0]  # 4,1,1024,128
        # audio
        x = self.patch_embed(x)  # 4, 512, 768
        x = x + self.pos_embed[:, 1:, :]
        if self.random_masking_2d:
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob, mask_f_prob)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_t_prob)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # video
        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v[:, 1:, :]
        v, mask, ids_restore = self.random_masking(v, mask_v_prob)
        cls_token_v = self.cls_token_v + self.pos_embed_v[:, :1, :]
        cls_tokens_v = cls_token_v.expand(B, -1, -1)
        v = torch.cat((cls_tokens_v, v), dim=1)
        v = self.pos_drop(v)

        for blk in self.blocks_v:
            v = blk(v)

        # av fusion
        if self.av_fusion:
            xv = torch.cat((x, v), dim=1)
            for blk in self.blocks_av:
                xv = blk(xv)
            x_len = x.shape[1]
            x = xv[:, 1:x_len, :].mean(dim=1)  # global pool without cls token
            v = xv[:, x_len + 1 :, :].mean(dim=1)
            outcome_av = self.fc_norm_av(torch.cat((x, v), dim=1))
            return outcome_av, outcome_av
        else:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            v = v[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome_a = self.fc_norm_a(x)
            outcome_v = self.fc_norm_v(v)
            return outcome_a, outcome_v

    # overwrite original timm
    # def forward(self, x, v, mask_t_prob=0.0, mask_f_prob=0.0, mask_v_prob=0.0):
    def forward(self, source, mask_t_prob=0.0, mask_f_prob=0.0, mask_v_prob=0.0):
        x, v = [], []
        for audio, video in source:
            x.append(audio.unsqueeze(0))
            v.append(video.unsqueeze(0))
        x = torch.cat(x, dim=0)
        v = torch.cat(v, dim=0)
        # x = x.to(device, non_blocking=True)
        # v = v.to(device, non_blocking=True)

        v = v[:, :, : self.n_frm, :, :]
        # add 10/23 for test
        if mask_t_prob > 0.0 or mask_f_prob > 0.0 or mask_v_prob > 0.0:
            x, v = self.forward_features_av_mask(
                x,
                v,
                mask_t_prob=mask_t_prob,
                mask_f_prob=mask_f_prob,
                mask_v_prob=mask_v_prob,
            )
        else:
            x, v = self.forward_features_av(x, v)
        if self.av_fusion:
            x = self.head_av(x)
            return x, x.detach()
        else:
            x = self.head_a(x)
            v = self.head_v(v)
            return x, v  # x.detach()


def vitmm_small_patch16(**kwargs):
    model = VisionTransformerMM(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vitmm_base_patch16(**kwargs):
    model = VisionTransformerMM(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vitmm_large_patch16(**kwargs):
    model = VisionTransformerMM(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vitmm_huge_patch14(**kwargs):
    model = VisionTransformerMM(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
