# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/baseline/hubconf.py ]
#   Synopsis     [ the baseline torch hubconf ]
#   Author       [ S3PRL (Leo Yang) ]
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


def hog(*args, **kwargs):
    """
    Baseline feature - Histogram of gradients
    """
    kwargs["model_config"] = os.path.join(os.path.dirname(__file__), "hog.yaml")
    return baseline_local(*args, **kwargs)
