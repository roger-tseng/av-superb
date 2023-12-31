# One needs to agree with the AV-HuBERT license (see COPYRIGHT under root 
# directory of source tree) to use pretrained models listed in this file.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/avhubert/hubconf.py ]
#   Synopsis     [ the avhubert torch hubconf ]
#   Author       [ S3PRL / Puyuan Peng, UT Austin]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import os

from utils.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert
from .expert import FinetunedUpstreamExpert as _FinetunedUpstreamExpert

# -------------#


def avhubert_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    if kwargs.get("finetune"):
        return _FinetunedUpstreamExpert(ckpt, *args, **kwargs)
    else:
        return _UpstreamExpert(ckpt, *args, **kwargs)


def avhubert_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return avhubert_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def avhubert(refresh=False, *args, **kwargs):
    """
    The default model
        refresh (bool): whether to download ckpt/config again if existed
    """
    return avhubert_base_lrs3(refresh=refresh, *args, **kwargs)

def avhubert_audio(refresh=False, *args, **kwargs):
    """
    The default model
        refresh (bool): whether to download ckpt/config again if existed
    """
    model = avhubert(refresh=refresh, *args, **kwargs)
    model.model.modality_dropout = 1
    model.model.audio_dropout = 0
    return model

def avhubert_video(refresh=False, *args, **kwargs):
    """
    The default model
        refresh (bool): whether to download ckpt/config again if existed
    """
    model = avhubert(refresh=refresh, *args, **kwargs)
    model.model.modality_dropout = 1
    model.model.audio_dropout = 1
    return model

def avhubert_fusion(refresh=False, *args, **kwargs):
    """
    The default model
        refresh (bool): whether to download ckpt/config again if existed
    """
    model = avhubert(refresh=refresh, *args, **kwargs)
    model.model.modality_dropout = 0
    return model

############### AV-HuBERT trained on LRS3 ###############
def avhubert_base_lrs3(refresh=False, *args, **kwargs):
    """
    The avhubert base model trained on LRS3
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3/clean-pretrain/base_lrs3_iter5.pt"
    return avhubert_url(refresh=refresh, *args, **kwargs)

def avhubert_ft_lrs3_433(refresh=False, *args, **kwargs):
    """
    The avhubert base model trained on LRS3 then fine-tuned on LRS3-433h
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3/vsr/base_lrs3_433h.pt"
    kwargs["finetune"] = True
    return avhubert_url(refresh=refresh, *args, **kwargs)

def avhubert_ft_lrs3_433_audio(refresh=False, *args, **kwargs):
    """
    The avhubert base model trained on LRS3 then fine-tuned on LRS3-433h
        refresh (bool): whether to download ckpt/config again if existed
    """
    model = avhubert_ft_lrs3_433(refresh=refresh, *args, **kwargs)
    model.model.modality_dropout = 1
    model.model.audio_dropout = 0
    return model

def avhubert_ft_lrs3_433_video(refresh=False, *args, **kwargs):
    """
    The avhubert base model trained on LRS3 then fine-tuned on LRS3-433h
        refresh (bool): whether to download ckpt/config again if existed
    """
    model = avhubert_ft_lrs3_433(refresh=refresh, *args, **kwargs)
    model.model.modality_dropout = 1
    model.model.audio_dropout = 1
    return model

def avhubert_ft_lrs3_433_fusion(refresh=False, *args, **kwargs):
    """
    The avhubert base model trained on LRS3 then fine-tuned on LRS3-433h
        refresh (bool): whether to download ckpt/config again if existed
    """
    model = avhubert_ft_lrs3_433(refresh=refresh, *args, **kwargs)
    model.model.modality_dropout = 0
    return model

def avhubert_large_lrs3(refresh=False, *args, **kwargs):
    """
    The avhubert large model trained on LRS3
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3/clean-pretrain/large_lrs3_iter5.pt"
    return avhubert_url(refresh=refresh, *args, **kwargs)


############### AV-HuBERT trained on both LRS3 and VoxCeleb2 ###############
def avhubert_base_lrs3vox(refresh=False, *args, **kwargs):
    """
    The avhubert base model trained on LRS3 and VoxCeleb2
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter5.pt"
    return avhubert_url(refresh=refresh, *args, **kwargs)


def avhubert_large_lrs3vox(refresh=False, *args, **kwargs):
    """
    The avhubert large model trained on LRS3 and VoxCeleb2
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt"
    return avhubert_url(refresh=refresh, *args, **kwargs)


############### noise-augmented AV-HuBERT ###############
def avhubert_base_lrs3vox_na(refresh=False, *args, **kwargs):
    """
    The avhubert base model trained on LRS3 and VoxCeleb2, trained with noise augmentation
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/base_vox_iter5.pt"
    return avhubert_url(refresh=refresh, *args, **kwargs)


def avhubert_large_lrs3vox_na(refresh=False, *args, **kwargs):
    """
    The avhubert large model trained on LRS3 and VoxCeleb2, trained with noise augmentation
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/large_vox_iter5.pt"
    return avhubert_url(refresh=refresh, *args, **kwargs)
