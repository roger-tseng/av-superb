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
from .preprocess_function import get_audio, get_visual_clip
from .avbert.utils import checkpoint as cu
from .avbert.config import get_audio_cfg, get_multi_cfg, get_video_cfg
from .avbert.models.video_model_builder import ResNet
from .avbert.models.audio_model_builder import AudioResNet
from .avbert.models.avbert import AVBert


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
        self.audio_duration = 2
        self.audio_temporal_samples = 10
        self.audio_frequency = 80
        self.audio_time = 128

        self.video_frame_size = 128
        self.video_frame_rate = 30
        self.video_num_frames = 32
        self.video_temporal_samples = 10
        self.video_spatial_samples = 10

        # NOTE: Encoders should return (batch_size, seq_len, hidden_dims)
        
        checkpoint = torch.load(
            ckpt, map_location='cpu'
        )

        cfg = get_video_cfg()
        self.video_encoder = ResNet(cfg)
        cu.load_finetune_checkpoint(
            self.video_encoder,
            checkpoint['state_dict'],
            cfg.NUM_GPUS > 1,
            cfg.MODEL.USE_TRANSFORMER,
        )

        cfg = get_audio_cfg()
        self.audio_encoder = AudioResNet(cfg)
        cu.load_finetune_checkpoint(
            self.audio_encoder,
            checkpoint['state_dict'],
            cfg.NUM_GPUS > 1,
            cfg.MODEL.USE_TRANSFORMER,
        )

        cfg = get_multi_cfg()
        self.multi_encoder = AVBert(cfg)
        cu.load_finetune_checkpoint(
            self.multi_encoder,
            checkpoint['state_dict'],
            cfg.NUM_GPUS > 1,
            cfg.MODEL.USE_TRANSFORMER,
        )

    def preprocess_video(self, video, video_frame_rate):
        """
        Replace this function to preprocess videos into your input format
        video: (video_length, video_channels, height, width), where video_channels is usually 3 for RGB or 1 for greyscale
        """
        visual_clips = []

        _num_frames = (
            self.video_num_frames *
            2 *
            video_frame_rate /
            self.video_frame_rate
        )

        delta = max(video.size(0) - _num_frames, 0)

        for spatial_sample_index in range(self.video_spatial_samples):
            for temporal_sample_index in range(self.video_temporal_samples):
                start_idx = delta * temporal_sample_index / (self.video_temporal_samples - 1)
                end_idx = start_idx + _num_frames - 1
                visual_clip = get_visual_clip(
                    video,
                    start_idx,
                    end_idx,
                    self.video_num_frames,
                    self.video_frame_size,
                    spatial_sample_index,
                )
                visual_clips.append(visual_clip)

        visual_clip = torch.stack(visual_clip)

        return visual_clips

    def preprocess_audio(self, audio, audio_sample_rate):
        """
        Replace this function to preprocess audio waveforms into your input format
        audio: (audio_channels, audio_length), where audio_channels is usually 1 or 2
        """
        audio_clips = []

        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        if audio_sample_rate != self.audio_sample_rate:
            audio = torchaudio.functional.resample(
                audio, audio_sample_rate, self.audio_sample_rate
            )

        total_length = audio.size(1)
        clip_length = (
            self.audio_sample_rate * self.audio_duration
        )
        delta = max(total_length - clip_length, 0)
        for temporal_sample_index in range(self.audio_temporal_samples):
            start_idx = int(
                delta * temporal_sample_index / (self.audio_temporal_samples - 1)
            )
            audio_clip = get_audio(
                audio,
                start_idx,
                clip_length,
                self.audio_sample_rate,
                self.audio_frequency,
                self.audio_time
            )
            audio_clips.append(audio_clip)

        audio_clips = torch.stack(audio_clips)

        return audio_clips

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

        print(audios[0].shape, videos[0].shape)

        # Run through audio and video encoders
        audio_feats = self.audio_encoder.get_feature_map(audios)
        print(audio_feats.shape)
        video_feats = self.video_encoder.get_feature_map([videos])[0]
        print(video_feats.shape)

        # conv_outputs, single_outputs, multi_output = self.multi_encoder(
        #     visual_seq=videos, audio_seq=audios
        # )
        # fusion_feats = (conv_outputs[0], conv_outputs[1], multi_output)
        # fusion_feats = (
        #     self.multi_encoder.visual_conv.head(fusion_feats[0]),
        #     self.multi_encoder.audio_conv.head(fusion_feats[1]),
        #     fusion_feats[2],
        # )

        # Return intermediate layer representations for potential layer-wise experiments
        # Dict should contain three items, with keys as listed below:
        # video_feats: features that only use visual modality as input
        # audio_feats: features that only use auditory modality as input
        # fusion_feats: features that consider both modalities
        # Each item should be a list of features that are of the same shape
        return {
            "video_feats": [video_feats],
            "audio_feats": [audio_feats],
            "fusion_feats": [],
        }
