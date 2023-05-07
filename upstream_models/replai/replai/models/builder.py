from torch import nn
from torchvision import models

from ..utils import distributed_ops, network_ops
from . import audio as snd_models
from . import video as vid_models


def build_video_model(cfg, distributed=False, device=None, remove_head=False):
    if cfg.arch in vars(vid_models):
        model_builder = vid_models.__dict__[cfg.arch]
    elif cfg.arch in vars(models):
        model_builder = models.__dict__[cfg.arch]
    else:
        raise ValueError(f"Model {cfg.arch} not found.")

    model = model_builder(**cfg.args)
    if cfg.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if remove_head:
        model = network_ops.remove_classification_head(model)

    # Send model to gpus
    return distributed_ops.send_to_device(model, distributed=distributed, device=device)


def build_audio_model(cfg, distributed=False, device=None, remove_head=False):
    if cfg.arch in vars(snd_models):
        model_builder = snd_models.__dict__[cfg.arch]
    elif cfg.arch in vars(models):
        model_builder = models.__dict__[cfg.arch]
    else:
        raise ValueError(f"Model {cfg.arch} not found.")

    model = model_builder(**cfg.args)
    if cfg.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if remove_head:
        model = network_ops.remove_classification_head(model)

    # Send model to gpus
    return distributed_ops.send_to_device(model, distributed=distributed, device=device)


def build_audio_video_model(cfg, distributed=False, device=None, remove_head=False):
    video_model = build_video_model(
        cfg.video, distributed=distributed, device=device, remove_head=remove_head
    )
    audio_model = build_audio_model(
        cfg.audio, distributed=distributed, device=device, remove_head=remove_head
    )
    return nn.ModuleDict({"video": video_model, "audio": audio_model})
