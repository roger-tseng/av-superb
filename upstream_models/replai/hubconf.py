# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/avhubert/hubconf.py ]
#   Synopsis     [ the RepLai torch hubconf ]
#   Author       [ S3PRL / Yi-Jen Shih, NTU]
"""*********************************************************************************************"""

# for more information, go to https://github.com/HimangiM/RepLAI

import os

from s3prl.utility.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

# -------------#


def google_large_file_link(file_id):
    return f"https://drive.google.com/u/0/uc?id={file_id}&export=download&confirm=t&uuid=b73dc818-c9b0-49e8-a488-13b829fdbb7e&at=ANzk5s4TfW9_erYs4M9PH7KYuKzy:1681055195563"


def replai_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def replai_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return replai_local(
        _urls_to_filepaths(ckpt, refresh=refresh, agent="gdown"), *args, **kwargs
    )


def replai(refresh=False, *args, **kwargs):
    """
    The default model
        refresh (bool): whether to download ckpt/config again if existed
    """
    return replai_ek100(refresh=refresh, *args, **kwargs)


############### RepLAI trained on EPIC-KITCHENS-100 ###############
def replai_ek100(refresh=False, *args, **kwargs):
    """
    The replai model trained on EPIC-KITCHENS-100
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = google_large_file_link("1FlRJxKo0gYGZTzxPUB4hloQWA4cNG4ZU")
    return replai_url(refresh=refresh, *args, **kwargs)


def replai_ek100_scratch(refresh=False, *args, **kwargs):
    """
    The replai model trained on EPIC-KITCHENS-100 (scratch)
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = google_large_file_link("1aVKCLD6DWZYafvrA-rODLyYMw0X_hj9d")
    return replai_url(refresh=refresh, *args, **kwargs)


def replai_ek100_woMoI(refresh=False, *args, **kwargs):
    """
    The replai model (w/o MoI) trained on EPIC-KITCHENS-100
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = google_large_file_link("1KAtIvgK4RxcgYPK8aHTbnJxawEpY0Q4w")
    return replai_url(refresh=refresh, *args, **kwargs)


def replai_ek100_woAStC(refresh=False, *args, **kwargs):
    """
    The replai model (w/o AStC) trained on EPIC-KITCHENS-100
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = google_large_file_link("1ls8MFxxaUr_D8KKzqANUfslU90DQYUxU")
    return replai_url(refresh=refresh, *args, **kwargs)


############### RepLAI trained on Ego4D ###############
def replai_ego4d(refresh=False, *args, **kwargs):
    """
    The replai model trained on Ego4D
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = google_large_file_link("1uw_fAB9N3y9--vGegKYdK_H2SSMbk4d5")
    return replai_url(refresh=refresh, *args, **kwargs)


def replai_ego4d_scratch(refresh=False, *args, **kwargs):
    """
    The replai model trained on Ego4D (scratch)
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = google_large_file_link("1g5DhV7z5W5kNBJSMQ2cHK0iWU0KAyecX")
    return replai_url(refresh=refresh, *args, **kwargs)


def replai_ego4d_woMoI(refresh=False, *args, **kwargs):
    """
    The replai model (w/o MoI) trained on Ego4D
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = google_large_file_link("1F6ZOuyS9C92OLrunA7zQkWp7vb6eQkOT")
    return replai_url(refresh=refresh, *args, **kwargs)


def replai_ego4d_woAStC(refresh=False, *args, **kwargs):
    """
    The replai model (w/o AStC) trained on Ego4D
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = google_large_file_link("1hQH1UWgC_-EztKpmma3TNST7OYGkQyI_")
    return replai_url(refresh=refresh, *args, **kwargs)
