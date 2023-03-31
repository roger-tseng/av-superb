"""
Custom class for loading audio-visual model and extract features
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/example/expert.py
"""
from typing import Dict, List, Tuple, Union

import torch.nn as nn
import torchaudio
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

        self.model1 = nn.Linear(1, HIDDEN_DIM)
        self.model2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.audio_sample_rate = 16000
        self.video_frame_size = (224, 224)
        self.video_frame_rate = 25

    def preprocess_video(self, video, video_frame_rate):
        """
        Replace this function to preprocess videos into your input format
        video: (video_length, video_channels, height, width), where video_channels is usually 3 for RGB or 1 for greyscale
        """
        # Resize video frames
        video_frames = []
        for frame in video:
            video_frames.append(
                torchvision.transforms.functional.resize(frame, self.video_frame_size)
            )
        video = torch.stack(video_frames)

        # Resample video
        assert video_frame_rate == self.video_frame_rate

        # Other preprocessing steps (e.g. cropping, flipping, etc.)

        return video

    def preprocess_audio(self, audio, audio_sample_rate):
        """
        Replace this function to preprocessa audio waveforms into your input format
        audio: (audio_channels, audio_length), where audio_channels is usually 1 or 2
        """
        # Resample audio
        if audio_sample_rate != self.audio_sample_rate:
            audio = torchaudio.functional.resample(
                audio, audio_sample_rate, self.audio_sample_rate
            )

        # Other preprocessing steps (e.g. trimming, pitch shift, etc.)

        return audio

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
        wavs = pad_sequence(audio, batch_first=True).unsqueeze(-1)
        videos = torch.stack(video)

        # Run through audio and video encoders
        audio_feats = audio_encoder(wavs)
        video_feats = video_encoder(videos)

        layer1 = self.model1(torch.cat(audio_feats, video_feats))

        layer2 = self.model2(layer1)

        # Return intermediate layer representations for potential layer-wise experiments
        return {"hidden_states": [audio_feats, video_feats, layer1, layer2]}
