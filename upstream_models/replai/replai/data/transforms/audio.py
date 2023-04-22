import torch
import torch.nn as nn


class RandomVolume(object):
    def __init__(self, volume):
        assert len(volume) == 2
        assert volume[1] >= volume[0] > 0
        self.volume = volume

    def __call__(self, x):
        if x.ndim == 1:
            x = x[None]
        assert x.ndim == 2
        n_waves, n_frames = x.shape
        max_mult = torch.minimum(1 / x.max(1)[0], torch.tensor(self.volume[1]))
        min_mult = torch.ones_like(max_mult) * self.volume[0]
        multiplier = torch.rand(n_waves) * (max_mult - min_mult) + min_mult
        return x * multiplier[:, None]


class UniformTemporalSubsample:
    def __init__(self, num_frames, time_dim=-2):
        self.num_frames = num_frames
        self.time_dim = time_dim

    def __call__(self, x):
        t = x.shape[self.time_dim]
        assert self.num_frames > 0 and t > 0

        # Sample by nearest neighbor interpolation if num_samples > t.
        indices = torch.linspace(0, t - 1, self.num_frames)
        indices = torch.clamp(indices, 0, t - 1).long()
        return torch.index_select(x, self.time_dim, indices)


class Log:
    def __init__(self, eps=1e-10):
        self.eps = eps

    def __call__(self, x):
        return torch.log(x.clamp(min=self.eps))


class Permute:
    def __init__(self, *dims):
        self.dims = dims

    def __call__(self, x):
        return x.permute(*self.dims)


class ToMono:
    def __call__(self, x):
        if x.ndim == 1:
            return x[None]
        elif x.ndim == 2:
            return x.mean(0, keepdims=True)
        else:
            raise ValueError("Audio tensor should have at most 2 dimensions (c,t)")


class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes.
        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]  # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width)

            return input

    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = 0


class SpecAugmentation(nn.Module):
    def __init__(
        self, time_drop_width, time_stripes_num, freq_drop_width, freq_stripes_num
    ):
        """Spec augmetation.
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(
            dim=2, drop_width=time_drop_width, stripes_num=time_stripes_num
        )

        self.freq_dropper = DropStripes(
            dim=3, drop_width=freq_drop_width, stripes_num=freq_stripes_num
        )

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x
