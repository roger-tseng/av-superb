"""
Custom class for loading audio-visual model and extract features 
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/example/expert.py
"""
import math
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as aT
import torchvision
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from .avbert.dataset import transform


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                checkpoint path for loading pretrained weights.

            model_config:
                config path for your model.
        """
        super().__init__()

        self.audio_sample_rate = 44100
        self.audio_frequency = 80
        self.audio_time = 128

        self.video_frame_size = (128, 128)
        self.video_frame_rate = 16
        self.video_frame_mean = (0.45, 0.45, 0.45)
        self.video_frame_std = (0.225, 0.225, 0.225)

        # NOTE: Encoders should return (batch_size, seq_len, hidden_dims)
    
    def get_log_mel_spectrogram(
        waveform,
        audio_fps,
        frequency,
        time,
    ):
        """
        Convert the input waveform to log-mel-scaled spectrogram.
        args:
            waveform (tensor): input waveform. The dimension is
                `channel` x `time.`
            `audio_fps` (int): sampling rate of `waveform`.
            `frequency` (int): target frequecy dimension (number of mel bins).
            `time` (int): target time dimension.
        returns:
            (tensor): log-mel-scaled spectrogram with dimension of
                `channel` x `frequency` x `time`.
        """
        w = waveform.size(-1)
        n_fft = 2 * (math.floor(w / time) + 1)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            audio_fps, n_fft=n_fft, n_mels=frequency,
        )(waveform)
        log_mel_spectrogram = torch.log(1e-6 + mel_spectrogram)
        _nchannels, _frequency, _time = log_mel_spectrogram.size()
        assert _frequency == frequency, \
            f"frequency {_frequency} must be {frequency}"
        if _time != time:
            t = torch.zeros(
                _nchannels,
                frequency,
                time,
                dtype=log_mel_spectrogram.dtype,
            )
            min_time = min(time, _time)
            t[:, :, :min_time] = log_mel_spectrogram[:, :, :min_time]
            log_mel_spectrogram = t

        return log_mel_spectrogram

    def preprocess_video(self, video, video_frame_rate):
        """
        Replace this function to preprocess videos into your input format
        video: (video_length, video_channels, height, width), where video_channels is usually 3 for RGB or 1 for greyscale
        """
        # Resize video frames
        video_frames = []
        for frame in video:
            resized_frame = torchvision.transforms.functional.resize(
                frame, self.video_frame_size, antialias=False
            )
            normalized_frame = torchvision.transforms.functional.normalize(
                resized_frame, self.video_frame_mean, self.video_frame_std
            )
            video_frames.append(normalized_frame)
        video = torch.stack(video_frames)

        # Resample video
        # (from https://github.com/pytorch/vision/blob/5b07d6c9c6c14cf88fc545415d63021456874744/torchvision/datasets/video_utils.py#L278)
        step = float(video_frame_rate) / self.video_frame_rate
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            idxs = slice(None, None, step)
        else:
            num_frames = int(len(video) / step)
            idxs = torch.arange(num_frames, dtype=torch.float32) * step
            idxs = idxs.floor().to(torch.int64)
        video = video[idxs]

        # Other preprocessing steps (i.e. cropping, flipping, etc.)
        return video

    def preprocess_audio(self, audio, audio_sample_rate):
        """
        Replace this function to preprocess audio waveforms into your input format
        audio: (audio_channels, audio_length), where audio_channels is usually 1 or 2
        """
        # Resample audio
        if audio_sample_rate != self.audio_sample_rate:
            audio = torchaudio.functional.resample(
                audio, audio_sample_rate, self.audio_sample_rate
            )

        ratio = math.floor(audio.shape[-1] / (self.audio_sample_rate * 2)) + 1

        # Other preprocessing steps (e.g. trimming, transform to melspectrogram etc.)
        log_mel_spectrogram = self.get_log_mel_spectrogram(
            audio,
            self.audio_sample_rate,
            self.audio_frequency,
            self.audio_time * ratio,
        )

        return log_mel_spectrogram

    def forward(
        self, source: List[Tuple[Tensor, Tensor]]
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        Replace this function run a forward pass with your model
        source: list of audio-video Tensor tuples
                [(wav1,vid1), (wav2,vid2), ...]
                in your input format
        """
        audio, video = zip(*source)

        print(audio.shape, video.shape)

        # Collate audio and video into batch
        audios = pad_sequence(audio, batch_first=True)
        videos = torch.stack(video)

        # Run through audio and video encoders
        audio_feats = self.audio_encoder(audios)
        video_feats = self.video_encoder(videos)

        layer1 = self.model1(torch.cat((audio_feats, video_feats), dim=-1))

        layer2 = self.model2(layer1)

        # Return intermediate layer representations for potential layer-wise experiments
        # Dict should contain three items, with keys as listed below:
        # video_feats: features that only use visual modality as input
        # audio_feats: features that only use auditory modality as input
        # fusion_feats: features that consider both modalities
        # Each item should be a list of features that are of the same shape
        return {
            "video_feats": [video_feats],
            "audio_feats": [audio_feats],
            "fusion_feats": [layer1, layer2],
        }
