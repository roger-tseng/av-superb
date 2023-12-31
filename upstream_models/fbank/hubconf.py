# Author: Yuan Tseng
# Modified from S3PRL
# (Authors: Leo Yang, Andy T. Liu and S3PRL team, https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/baseline/hubconf.py)

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/baseline/hubconf.py ]
#   Synopsis     [ the baseline torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import os

# -------------#
from .expert import UpstreamExpert as _UpstreamExpert


def baseline_local(model_config, *args, **kwargs):
    """
    Baseline feature
        model_config: PATH
    """
    assert os.path.isfile(model_config)
    return _UpstreamExpert(model_config, *args, **kwargs)

def fbank(*args, **kwargs):
    """
    Baseline feature - Fbank, or Mel-scale spectrogram
    """
    kwargs["model_config"] = os.path.join(os.path.dirname(__file__), "fbank.yaml")
    return baseline_local(*args, **kwargs)
