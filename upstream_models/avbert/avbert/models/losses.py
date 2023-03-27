"""Loss functions."""

import torch.nn as nn

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "multi_margin": nn.MultiMarginLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
