# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = [
    "avid_spec_cnn_9",
]


class Basic2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=(1, 1)):
        self.__dict__.update(locals())
        super(Basic2DBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Conv2D(nn.Module):
    def __init__(self, channels=1, depth=10):
        super(Conv2D, self).__init__()
        assert depth == 10

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.block1 = Basic2DBlock(64, 64, stride=(2, 2))
        self.block2 = Basic2DBlock(64, 128, stride=(2, 2))
        self.block3 = Basic2DBlock(128, 256, stride=(2, 2))
        self.block4 = Basic2DBlock(256, 512)
        # original
        # self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.output_dim = 512

    def forward(self, x, return_embs=False):
        seq_len = x.shape[-2]
        x_c1 = self.conv1(x)
        x_b1 = self.block1(x_c1)
        x_b2 = self.block2(x_b1)
        x_b3 = self.block3(x_b2)
        x_b4 = self.block4(x_b3)

        # original
        # x_pool = self.pool(x_b4)
        # revised: ( generalize temporal pooling to variabble sequence length)
        x_pool = F.adaptive_max_pool2d(x_b4, (seq_len // 128, 1), False)

        if return_embs:
            return {
                "conv2x": x_b1,
                "conv3x": x_b2,
                "conv4x": x_b3,
                "conv5x": x_b4,
                "pool": x_pool,
            }
        else:
            return x_b4


def avid_spec_cnn_9(channels=1, pretrained=False):
    model = Conv2D(channels=channels)

    if pretrained:
        import os

        base_dir = os.environ["PYTHONPATH"].split(":")[0]
        try:
            ckpt_fn = "/glusterfs/hmittal/ssrl/pretrained_av_weights/AVID_Audioset_Cross-N1024_checkpoint.pth.tar"
            ckp = torch.load(ckpt_fn, map_location="cpu")
        except FileNotFoundError:
            ckpt_fn = "/Users/morgado/Projects/ssrl/models/ckpt/AVID_Audioset_Cross-N1024_checkpoint.pth.tar"
            ckp = torch.load(ckpt_fn, map_location="cpu")
        prefix = "module.audio_model."
        model.load_state_dict(
            {
                k.replace(prefix, ""): ckp["model"][k]
                for k in ckp["model"]
                if k.startswith(prefix)
            }
        )
        print(f"Audio model initialized from {ckpt_fn}")

    return model
