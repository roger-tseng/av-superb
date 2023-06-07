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
from .preprocess_function import get_audio_seq, get_visual_seq, resample
from .avbert.utils import checkpoint as cu
from .avbert.config import get_cfg
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

        self.cfg = get_cfg()

        # NOTE: Encoders should return (batch_size, seq_len, hidden_dims)
        
        checkpoint = torch.load(
            ckpt, map_location='cpu'
        )

        self.multi_encoder = AVBert(self.cfg)
        cu.load_finetune_checkpoint(
            self.multi_encoder,
            checkpoint['state_dict'],
            self.cfg.NUM_GPUS > 1,
            self.cfg.MODEL.USE_TRANSFORMER,
        )

    def preprocess_video(self, video, video_frame_rate):
        """
        Replace this function to preprocess videos into your input format
        video: (video_length, video_channels, height, width), where video_channels is usually 3 for RGB or 1 for greyscale
        """

        return video

    def preprocess_audio(self, audio, audio_sample_rate):
        """
        Replace this function to preprocess audio waveforms into your input format
        audio: (audio_channels, audio_length), where audio_channels is usually 1 or 2
        """

        return audio
    
    def preprocess(self, video, audio, video_frame_rate, audio_sample_rate):
        
        video = video.permute(0, 2, 3 ,1)

        visual_seqs = []

        num_frames = (
            self.cfg.DATA.NUM_FRAMES *
            self.cfg.DATA.SAMPLING_RATE *
            video_frame_rate /
            self.cfg.DATA.TARGET_FPS
        )

        waveform_size = int(
            self.cfg.DATA.TARGET_AUDIO_RATE *
            self.cfg.DATA.NUM_FRAMES *
            self.cfg.DATA.SAMPLING_RATE /
            self.cfg.DATA.TARGET_FPS
        )

        visual_delta = max(video.size(0) - num_frames, 0)
        for spatial_sample_index in range(self.cfg.TEST.NUM_SPATIAL_CROPS):
            visual_start_idx = [
                visual_delta * temporal_sample_index / (self.cfg.TEST.NUM_ENSEMBLE_VIEWS - 1)
                for temporal_sample_index in range(self.cfg.TEST.NUM_ENSEMBLE_VIEWS)
            ]
            visual_end_idx = [s + num_frames - 1 for s in visual_start_idx]
            visual_seq = get_visual_seq(
                video,
                visual_start_idx,
                visual_end_idx,
                self.cfg.DATA.NUM_FRAMES,
                self.cfg.DATA.TEST_CROP_SIZE,
                spatial_sample_index,
            )
            visual_seqs.append(visual_seq)

        visual_seqs = torch.stack(visual_seqs)

        if len(audio.shape) == 1:
            waveform = audio.unsqueeze(0)
        
        waveform = resample(
            waveform,
            audio_sample_rate,
            self.cfg.DATA.TARGET_AUDIO_RATE,
            use_mono=True,
        )

        audio_delta = max(waveform.size(-1) - waveform_size, 0)
        audio_start_idx = [
            int(audio_delta * (idx / visual_delta)) if visual_delta != 0 else 0
            for idx in visual_start_idx
        ]
        audio_end_idx = [s + waveform_size for s in audio_start_idx]

        audio_seq = get_audio_seq(
            waveform,
            audio_start_idx,
            audio_end_idx,
            self.cfg.DATA.TARGET_AUDIO_RATE,
            self.cfg.DATA.AUDIO_FREQUENCY,
            self.cfg.DATA.AUDIO_TIME,
        )

        return visual_seqs, audio_seq


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

        videos = videos.transpose(0, 1).contiguous()

        _ , single_outputs_0, multi_output_0 = self.multi_encoder(
            visual_seq=[videos[0]], audio_seq=audios
        )

        _ , single_outputs_1, multi_output_1 = self.multi_encoder(
            visual_seq=[videos[1]], audio_seq=audios
        )

        _ , single_outputs_2, multi_output_2 = self.multi_encoder(
            visual_seq=[videos[2]], audio_seq=audios
        )

        video_feats = torch.cat((single_outputs_0[0], single_outputs_1[0], single_outputs_2[0]), dim = 1)
        audio_feats = single_outputs_0[1]
        fusion_feats = torch.cat((multi_output_0[0], multi_output_1[0], multi_output_2[0]), dim = 1)

        # Return intermediate layer representations for potential layer-wise experiments
        # Dict should contain three items, with keys as listed below:
        # video_feats: features that only use visual modality as input
        # audio_feats: features that only use auditory modality as input
        # fusion_feats: features that consider both modalities
        # Each item should be a list of features that are of the same shape
        return {
            "video_feats": [video_feats],
            "audio_feats": [audio_feats],
            "fusion_feats": [fusion_feats],
        }
