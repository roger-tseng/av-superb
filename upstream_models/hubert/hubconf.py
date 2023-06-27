# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/hubert/hubconf.py ]
#   Synopsis     [ the HuBERT torch hubconf ]
#   Author       [ S3PRL / Kushal Lakhotia ]
"""*********************************************************************************************"""


import logging
import os
import time
from pathlib import Path

from filelock import FileLock

from utils.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)

NEW_ENOUGH_SECS = 2.0


def hubert_custom(
    ckpt: str,
    legacy: bool = False,
    refresh: bool = False,
    **kwargs,
):

    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, **kwargs)


def hubert_local(*args, **kwargs):
    return hubert_custom(*args, **kwargs)


def hubert_url(*args, **kwargs):
    return hubert_custom(*args, **kwargs)


def hubert(refresh=False, *args, **kwargs):
    """
    The default model - Base
        refresh (bool): whether to download ckpt/config again if existed
    """
    return hubert_base(refresh=refresh, *args, **kwargs)


def hubert_base(refresh=False, legacy=True, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
    return hubert_custom(refresh=refresh, legacy=legacy, **kwargs)


def hubert_large_ll60k(refresh=False, legacy=False, **kwargs):
    """
    The Large model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt"
    return hubert_custom(refresh=refresh, legacy=legacy, **kwargs)


def hubert_base_robust_mgr(refresh=False, legacy=False, **kwargs):
    """
    The Base model, continually trained with Libri 960 hr with Musan noise, Gaussian noise and Reverberation.
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/kphuang68/HuBERT_base_robust_mgr/resolve/main/HuBERT_base_robust_mgr_best_loss_2.7821.pt"
    return hubert_custom(refresh=refresh, legacy=legacy, **kwargs)
