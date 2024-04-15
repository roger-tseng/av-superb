"""
Custom class for loading audio-visual model and extract features
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/example/expert.py
"""

import sys
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from . import replai
from .replai import models
from .replai.data.builder import build_transforms

sys.modules["replai"] = replai  # create alias for unpickling


from munch import DefaultMunch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


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

        # load model weights
        _weights = torch.load(ckpt)
        _weights = _weights["model"]
        _weights = OrderedDict(
            {
                k[16:]: _weights[k]
                for k in _weights.keys()
                if k.startswith("module.backbone")
            }
        )

        # hardcode model config
        model_conf = DefaultMunch.fromDict(
            {
                "audio": {
                    "arch": "avid_spec_cnn_9",
                    "args": {"channels": 1, "pretrained": False},
                    "sync_bn": False,
                    "outp_dim": 512,
                },
                "video": {
                    "arch": "avid_r2plus1d_18",
                    "args": {"pretrained": False},
                    "sync_bn": False,
                    "outp_dim": 512,
                },
            }
        )

        # create model and load weights
        self.backbone = models.build_audio_video_model(model_conf, remove_head=True)
        self.backbone.load_state_dict(
            _weights, strict=True
        )

        self.audio_sample_rate = 16000
        self.video_frame_size = (112, 112)
        self.video_frame_rate = 16

    def preprocess_video(self, video, video_frame_rate):
        """
        Replace this function to preprocess videos into your input format
        video: (video_length, video_channels, height, width), where video_channels is usually 3 for RGB or 1 for greyscale
        in RepLAI, the default length is 0.5 secs for video, resulting in 8 frames (16FPS)
        """
        # Resample video
        # (from https://github.com/pytorch/vision/blob/5b07d6c9c6c14cf88fc545415d63021456874744/torchvision/datasets/video_utils.py#L278)
        step = float(video_frame_rate) / self.video_frame_rate
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            idxs = slice(None, None, step)
        else:
            num_frames = max(int(len(video) / step),1)
            idxs = torch.arange(num_frames, dtype=torch.float32) * step
            idxs = idxs.floor().to(torch.int64)
        video = video[idxs]

        _video_transform = build_transforms(
            cfg=DefaultMunch.fromDict(
                {
                    "video": {
                        "name": "ResizeCropFlip",
                        "args": {
                            "min_size": 128,
                            "max_size": 180,
                            "crop_size": self.video_frame_size[0],
                        },
                        "data_shape": [
                            3,
                            len(video),
                            self.video_frame_size[0],
                            self.video_frame_size[1],
                        ],
                    },
                }
            ),
            augment=False,
        )
        # Original uses OpenCV for resizing numpy tensors
        clips = {
            "video": (video.numpy().transpose(0, 2, 3, 1), self.video_frame_rate),
        }

        clips = _video_transform(clips)

        # output video shape (channel, length, w, h)
        return clips["video"]

    def preprocess_audio(self, audio, audio_sample_rate):
        """
        Replace this function to preprocess audio waveforms into your input format
        audio: (audio_channels, audio_length) or (audio_length,), where audio_channels is usually 1 or 2
        RepLAI uses 2.0 secs of audio at 16kHz, and take 128 temporal frames on the mel-spectrogram domain.
        In this implementation, we take variable lengths of audio as input, hence we rescale the number of mel frames accordingly.
        """
        if len(audio.shape) == 2:
            audio = audio.mean(dim=0)

        _audio_length_sec = len(audio) / audio_sample_rate
        num_temporal_frames = int(_audio_length_sec / 2.0 * 128)
        _audio_transform = build_transforms(
            cfg=DefaultMunch.fromDict(
                {
                    "audio": {
                        "name": "ResampleLogMelSpectrogram",
                        "args": {
                            "raw_sample_rate": audio_sample_rate,
                            "audio_rate": self.audio_sample_rate,
                            "mel_window_size": 32,
                            "mel_step_size": 16,
                            "num_mels": 80,
                            "num_temporal_frames": num_temporal_frames,
                        },
                        "data_shape": [1, num_temporal_frames, 80],
                    }
                }
            ),
            augment=False,
        )

        clips = {
            "audio": (audio, audio_sample_rate),
        }

        clips = _audio_transform(clips)

        return clips["audio"]

    def forward(
        self, source: List[Tuple[Tensor, Tensor]]
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        Replace this function run a forward pass with your model
        source: list of audio-video Tensor tuples
                [(wav1,vid1), (wav2,vid2), ...]
                in your input format
        """
        bsz = len(source)
        audio, video = zip(*source)

        # Collate audio and video into batch
        audio = [a.squeeze() for a in audio]
        wavs = pad_sequence(audio, batch_first=True).unsqueeze(dim=1)
        # Pad video along time axis, video starts with channel x time x height x width
        video = [v.permute(1, 0, 2, 3) for v in video]
        videos = pad_sequence(video, batch_first=True).permute(0, 2, 1, 3, 4)
        # videos = torch.stack(video)

        assert wavs.shape[0] == bsz
        assert videos.shape[0] == bsz
        assert videos.shape[-2] == 112
        assert videos.shape[-1] == 112

        # Run through audio and video encoders
        video_feats = self.backbone["video"](videos, return_embs=True)
        audio_feats = self.backbone["audio"](wavs, return_embs=True)

        # use the output of last CNN layer before pooling
        video_feats = video_feats["conv5x"]
        audio_feats = audio_feats["conv5x"]

        # convert video_feats to shape (bsz, T', hid_dim)
        video_feats = video_feats.flatten(start_dim=2, end_dim=-1)
        video_feats = video_feats.permute(0, 2, 1)

        # convert video_feats to shape (bsz, T', hid_dim)
        audio_feats = audio_feats.flatten(start_dim=2, end_dim=-1)
        audio_feats = audio_feats.permute(0, 2, 1)

        # Return intermediate layer representations for potential layer-wise experiments
        return {
            "video_feats": [video_feats],
            "audio_feats": [audio_feats],
            "fusion_feats": [],
        }