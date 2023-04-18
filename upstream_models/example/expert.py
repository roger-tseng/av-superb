"""
Custom class for loading audio-visual model and extract features
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/example/expert.py
"""
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as aT
import torchvision
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

HIDDEN_DIM = 8


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

        self.model1 = nn.Linear(2, HIDDEN_DIM)
        self.model2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        self.audio_sample_rate = 16000
        self.melspec_transform = [aT.MelSpectrogram(self.audio_sample_rate, n_mels=80)]

        self.video_frame_size = (224, 224)
        self.video_frame_rate = 25

        # NOTE: Encoders should return (batch_size, seq_len, hidden_dims)
        self.audio_encoder = lambda x: x.reshape(x.size(0), 1, -1).mean(dim=-1,keepdim=True)
        self.video_encoder = lambda x: x.reshape(x.size(0), 1, -1).mean(dim=-1,keepdim=True)

    def preprocess_video(self, video, video_frame_rate):
        """
        Replace this function to preprocess videos into your input format
        video: (video_length, video_channels, height, width), where video_channels is usually 3 for RGB or 1 for greyscale
        """
        # Resize video frames
        video_frames = []
        for frame in video:
            video_frames.append(
                torchvision.transforms.functional.resize(frame, self.video_frame_size, antialias=False)
            )
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
            num_frames = len(video)
            idxs = torch.arange(num_frames, dtype=torch.float32) * step
            idxs = idxs.floor().to(torch.int64)
        video = video[idxs]

        # Other preprocessing steps (i.e. cropping, flipping, etc.)
        # e.g. take first three frames to ensure all videos have same size
        video = video[:3]
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

        # Other preprocessing steps (e.g. trimming, transform to melspectrogram etc.)
        mel_specgram = self.melspec_transform[0](audio).transpose(0,1)

        return mel_specgram

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
