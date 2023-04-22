import numpy as np
import torch
from pytorchvideo import transforms as vT
from torchaudio import transforms as aT
from torchvision import transforms as T

from .transforms import audio as aT2


class ResampleLogMelSpectrogram:
    def __init__(
        self,
        raw_sample_rate=44100,
        audio_rate=16000,
        mel_window_size=32,
        mel_step_size=16,
        num_mels=80,
        num_temporal_frames=128,
        augment=False,
    ):
        self.outp_size = (1, num_temporal_frames, num_mels)
        self.raw_sample_rate = raw_sample_rate

        mel_mean = -7.03
        mel_std = 4.66
        n_fft = int(float(audio_rate) / 1000 * mel_window_size)
        hop_length = int(float(audio_rate) / 1000 * mel_step_size)
        eps = 1e-10

        transforms = [
            aT2.ToMono(),
            aT.Resample(
                orig_freq=raw_sample_rate,
                new_freq=audio_rate,
            ),
            aT.MelSpectrogram(
                sample_rate=audio_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=num_mels,
                center=False,
            ),
            aT2.Log(eps=eps),
            aT2.Permute(0, 2, 1),  # (1, F, T) -> (1, T, F)
            aT2.UniformTemporalSubsample(
                num_temporal_frames, time_dim=-2
            ),  # temporal and frequency axis, by here different
            # mean and different std dev, mean and
            T.Normalize((mel_mean,), (mel_std,)),  # remove
        ]
        self.t = T.Compose(transforms)

    def __call__(self, x, fps):
        assert fps == self.raw_sample_rate
        if isinstance(x, torch.Tensor):
            return self.t(x)
        elif isinstance(x, np.ndarray):
            return self.t(torch.from_numpy(x))
        else:
            raise TypeError(f"x is not a tensor or ndarray, but {type(x)}")

    def output_shape(self):
        return self.outp_size


class SpectrogramPrep:
    def __init__(self, args, augment=True):
        self.args = args
        self.augment = augment

        n_fft = int(
            float(args.audio_resampled_rate) / 1000 * args.audio_mel_window_size
        )
        hop_length = int(
            float(args.audio_resampled_rate) / 1000 * args.audio_mel_step_size
        )
        transforms = [
            aT2.RandomVolume(volume=args.volume_coeff),
            aT2.ToMono(),
            aT.Resample(
                orig_freq=args.audio_raw_sample_rate,
                new_freq=args.audio_resampled_rate,
            ),
            aT.MelSpectrogram(
                sample_rate=args.audio_resampled_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=args.audio_num_mels,
                center=False,
            ),
            aT.AmplitudeToDB(),
            T.Lambda(lambda x: x / 10.0),
            T.Lambda(
                lambda x: x.transpose(2, 1).unsqueeze(2)
            ),  # (C, F, T) -> (C, T, 1, F)
            vT.UniformTemporalSubsample(args.audio_mel_num_subsample),
            T.Lambda(lambda x: x.squeeze(2)),  # (C, T, 1, F) -> (C, T, F)
        ]
        if not augment:
            transforms.pop(0)
        self.t = T.Compose(transforms)

    def __call__(self, x, fps):
        assert fps == self.args.audio_raw_sample_rate
        return self.t(x)

    def output_shape(self):
        return 1, self.args.audio_mel_num_subsample, self.args.audio_num_mels
