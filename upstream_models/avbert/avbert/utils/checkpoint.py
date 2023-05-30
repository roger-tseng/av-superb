import os
import shutil
from collections import OrderedDict

import torch

import utils.logging as logging

logger = logging.get_logger(__name__)


def load_checkpoint(model, state_dict, data_parallel=False):
    """
    Load the trained weights from the checkpoint.
    Args:
        model (model): model to load the weights from the checkpoint.
        state_dict (OrderedDict): checkpoint.
        data_parallel (bool): if true, model is wrapped by
            torch.nn.parallel.DistributedDataParallel.
    """
    ms = model.module if data_parallel else model
    ms.load_state_dict(state_dict)


def load_finetune_checkpoint(model, state_dict, data_parallel=False, use_trans=False):
    """
    Load the pretrained weights from the checkpoint.
    Args:
        model (model): model to load the weights from the checkpoint.
        state_dict (OrderedDict): checkpoint.
        data_parallel (bool): if true, model is wrapped by
            torch.nn.parallel.DistributedDataParallel.
        use_trans (bool): if true, load the Transformer weights.
    """
    ms = model.module if data_parallel else model
    model_dict = ms.state_dict()
    partial_dict = OrderedDict()
    if use_trans:
        for key in state_dict.keys():
            if "_B" not in key and "head" not in key:
                partial_dict[key] = state_dict[key]
    else:
        for key in state_dict.keys():
            if 'visual_conv' in key and 'head' not in key:
                partial_dict[key[7:]] = state_dict[key]
            if 'audio_conv' in key and 'head' not in key:
                partial_dict[key[7:]] = state_dict[key]

    update_dict = {k: v for k, v in partial_dict.items() if k in model_dict}
    ms.load_state_dict(update_dict, strict=False)


def save_checkpoint(state, is_best=False, filename='checkpoint.pyth'):
    """
    Save the model weights to the checkpoint.
    Args:
        state (Dict): model states
        is_best (bool): whether the model has achieved the best performance so far.
        filename (str): path to the checkpoint to save.
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pyth')
