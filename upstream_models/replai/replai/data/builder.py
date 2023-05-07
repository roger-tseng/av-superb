import torch

from . import audio_transforms, video_transforms


class Transform:
    def __init__(self, transforms, data_shapes):
        self.transforms = transforms
        self.data_shapes = data_shapes

    def __call__(self, x):
        for k in self.transforms:
            if k not in x or x[k] is None:
                x[k] = torch.zeros(tuple(self.data_shapes[k])).float()
            else:
                x[k] = self.transforms[k](x[k][0], x[k][1])
        return x


def build_transforms(cfg, augment):
    transforms, data_shapes = {}, {}
    for k in cfg:
        if cfg[k].name in vars(video_transforms):
            transforms[k] = video_transforms.__dict__[cfg[k].name](
                **cfg[k].args, augment=augment
            )
        elif cfg[k].name in vars(audio_transforms):
            transforms[k] = audio_transforms.__dict__[cfg[k].name](
                **cfg[k].args, augment=augment
            )
        else:
            raise NotImplementedError(f"Transform {cfg.name} not found.")
        data_shapes[k] = cfg[k].data_shape
    return Transform(transforms, data_shapes)
