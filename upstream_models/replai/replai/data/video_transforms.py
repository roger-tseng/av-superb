import numpy as np
import torch
from pytorchvideo import transforms as vT
from torchvision import transforms as T

from .transforms import video as vT2

__all__ = ["ResizeCropFlip", "MultiResizeCropFlip", "MultiScaleCropFlipColorJitter"]


class MultiScaleCropFlipColorJitter:
    def __init__(
        self,
        num_frames=8,
        crop=(224, 224),
        color=(0.4, 0.4, 0.4, 0.1),
        min_area=0.08,
        augment=True,
    ):
        from collections.abc import Iterable

        if isinstance(crop, Iterable):
            crop = tuple(crop)
        self.crop = crop
        self.augment = augment
        self.num_frames = num_frames

        if augment:
            transforms = [
                vT2.RandomResizedCrop(crop, scale=(min_area, 1.0)),
                vT2.RandomHorizontalFlip(),
                vT2.ColorJitter(*color),
            ]
        else:
            transforms = [
                vT2.Resize(int(crop[0] / 0.875)),
                vT2.CenterCrop(crop),
            ]

        transforms += [
            vT2.ClipToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vT.UniformTemporalSubsample(num_frames),
        ]
        self.t = T.Compose(transforms)

    def __call__(self, x, fps):
        return self.t(x)


class ResizeCropFlip:
    def __init__(
        self, num_frames=8, min_size=256, max_size=360, crop_size=224, augment=True
    ):
        if augment:
            transforms = [
                vT2.RandomShortSideScale(min_size=min_size, max_size=max_size),
                vT2.RandomCrop(crop_size),
                vT2.RandomHorizontalFlip(),
            ]
        else:
            transforms = [
                vT2.Resize(min_size),
                vT2.CenterCrop(crop_size),
            ]
        transforms += [
            vT2.ClipToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vT.UniformTemporalSubsample(num_frames),
        ]
        self.t = T.Compose(transforms)

    def __call__(self, x, fps):
        return self.t(x)


class UniformClips:
    def __init__(
        self, num_clips, clip_duration, transform, transform_args, augment=True
    ):
        self.num_clips = num_clips
        self.clip_duration = clip_duration
        self.transform = eval(transform)(**transform_args, augment=augment)

    def __call__(self, x, fps):
        nt = len(x)
        clip_len = int(
            self.clip_duration * fps
        )  # [0, 1, ..., 96] -> [0, 1, ..., 88] -> [0, 11, 22, 33, .. 88]
        start_times = [int(ss) for ss in np.linspace(0, nt - clip_len, self.num_clips)]
        clips = [self.transform(x[ss : ss + clip_len], fps) for ss in start_times]
        return clips


class MultiResizeCropFlip:
    def __init__(
        self,
        num_augm=2,
        num_frames=8,
        min_size=256,
        max_size=360,
        crop_size=224,
        augment=True,
    ):
        self.t = ResizeCropFlip(num_frames, min_size, max_size, crop_size, augment)
        self.num_augm = num_augm

    def __call__(self, x):
        return [self.t(x) for _ in range(self.num_augm)]
