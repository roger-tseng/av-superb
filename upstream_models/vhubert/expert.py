# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/avhubert/expert.py ]
#   Synopsis     [ the avhubert wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Puyuan Peng, UT Austin ]
"""*********************************************************************************************"""

import argparse
from concurrent.futures import process

import fairseq
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from packaging import version
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.functional import rgb_to_grayscale

from interfaces import UpstreamBase

from . import utils as custom_utils
from .hubert import AVHubertConfig, AVHubertModel


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        assert version.parse(fairseq.__version__) > version.parse(
            "0.10.2"
        ), "Please install the fairseq master branch."
        bundle = torch.load(ckpt)
        cfg = vars(AVHubertConfig())
        for key in bundle["cfg"]["model"]:
            cfg[key] = bundle["cfg"]["model"][key]
        cfg = argparse.Namespace(**cfg)

        # transfer some stuff from cfg.task
        cfg.sample_rate = bundle["cfg"]["task"]["sample_rate"]
        cfg.stack_order_audio = bundle["cfg"]["task"]["stack_order_audio"]
        cfg.normalize = bundle["cfg"]["task"]["normalize"]

        # hard code some stuff
        dictionaries = [[1] * bundle["model"]["label_embs_concat"].shape[0]]
        cfg.required_seq_len_multiple = 1  # newer fairseq wav2vec2 TranformerEncoder requires this parameter, which optimize the processing efficiency for FP16
        cfg.layer_type = "transformer"  # newer fairseq wav2vec2 TranformerEncoder requires this parameter, hard coded to fit avhubert
        cfg.checkpoint_activations = False  # newer fairseq wav2vec2 TranformerEncoder requires this parameter, hard coded to fit avhubert
        self.audio_sample_rate = 16000
        self.video_frame_rate = 25

        model = AVHubertModel(cfg, dictionaries)
        model.load_state_dict(bundle["model"])
        self.model = model.eval()
        self.cfg = cfg
        assert (
            self.video_frame_rate == cfg.sample_rate
        ), f"video sample rate should equal to the task sample rate, but it's not, video_sample_rate: {self.video_frame_rate}, task_cfg.sample_rate: {self.cfg.sample_rate}"
        self.transform = custom_utils.Compose(
            [
                custom_utils.Normalize(0.0, 255.0),
                custom_utils.CenterCrop((88, 88)),
                custom_utils.Normalize(0.421, 0.165),
            ]
        )

    def stacker(self, feats, stack_order):
        """
        TODO: need to make this a batch processing method
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(
            -1, stack_order * feat_dim
        )
        return feats

    def preprocess_audio(self, audio, audio_sample_rate):
        # audio: (audio_channels, audio_length), where audio_channels is usually 1 or 2
        # since using av-hubert native implementation, needs to work with numpy objects
        orig_device = audio.device
        audio = audio.cpu()
        if len(audio.shape) >= 3:
            raise NotImplementedError(
                f"input should be single sample, not a batch! shape of audio input to preprocess_audio: {audio.shape}"
            )
        elif len(audio.shape) == 2:
            assert (
                audio.shape[0] == 1 or audio.shape[0] == 2
            ), f"wrong audio shape to the preprocess_audio method: {audio.shape}"
            audio = audio.mean(0)
        # it can indeed do batch processing
        if audio_sample_rate != self.audio_sample_rate:
            audio = torchaudio.functional.resample(
                audio, audio_sample_rate, self.audio_sample_rate
            )
        audio_feats = custom_utils.logfbank(
            audio, samplerate=self.audio_sample_rate
        ).astype(
            np.float32
        )  # [T, F]
        in_data = self.stacker(audio_feats, self.cfg.stack_order_audio)
        # [T/stack_order_audio, stack_order_audio*F]
        in_data = torch.from_numpy(in_data.astype(np.float32))
        if self.cfg.normalize:
            with torch.no_grad():
                in_data = F.layer_norm(in_data, in_data.shape[1:])

        return in_data.to(orig_device)  # TxF

    def preprocess_video(self, video, video_frame_rate):
        # video: (video_length, video_channels, height, width), where video_channels is usually 3 for RGB or 1 for greyscale
        # avhubert will make image and audio have the same framerate so we can add them or concat them in feature dimension. image sample rate if 25Hz, audio sample rate is 100Hz (originally 16kHz, but after fbank it's 100Hz), four neighboring audio sample is stacked to get
        # since using av-hubert native implementation, needs to work with numpy objects
        orig_device = video.device
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

        # Transform to greyscale
        if video.shape[1] == 3:
            video = 0.2989 * video[:, 0] + 0.587 * video[:, 1] + 0.114 * video[:, 2]
        else:
            video = video.mean(dim=1)
        feats = self.transform(video)

        # T, H, W
        if isinstance(feats, np.ndarray):
            return torch.from_numpy(feats).to(orig_device)
        elif isinstance(feats, torch.Tensor):
            return feats.to(orig_device)

    def forward(self, processed_data):
        device = processed_data[0][0].device

        audio, video = [], []
        for audio_feats, video_feats in processed_data:
            diff = len(audio_feats) - len(video_feats)
            if diff > 0:
                audio_feats = audio_feats[:-diff] 
            elif diff < 0:
                audio_feats = F.pad(audio_feats, (0, 0, 0, -diff), "constant", 0)
            audio.append(audio_feats)
            video.append(video_feats)

        audio_length = torch.LongTensor([len(item) for item in audio])
        video_length = torch.LongTensor([len(item) for item in video])
        assert sum([a == v for a, v in zip(audio_length, video_length)]) == len(
            audio_length
        ), f"each audio should have the same temporal length with the corresponding video, but they are not: audio_length: {audio_length}, video_length: {video_length}"
        padding_mask = ~torch.lt(
            torch.arange(max(audio_length)).unsqueeze(0),
            (audio_length).unsqueeze(1),
        ).to(device)
        padded_audio = pad_sequence(audio, batch_first=True)
        padded_video = pad_sequence(video, batch_first=True)
        source = {
            "audio": padded_audio.transpose(1, 2),
            "video": padded_video.unsqueeze(dim=1),
        }
        result = self.model(
            source, padding_mask=padding_mask, mask=False, features_only=True
        )
        return {
            # "last_hidden_state": result["x"],
            "video_feats": result["features_video"],
            "audio_feats": result["features_audio"],
            "fusion_feats": [result["features"], result["x"]],
        }
