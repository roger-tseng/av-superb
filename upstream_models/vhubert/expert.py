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

from ..interfaces import UpstreamBase
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

    def processing_audio(self, audio, audio_sample_rate):
        # audio: (audio_channels, audio_length), where audio_channels is usually 1 or 2
        # since using av-hubert native implementation, needs to work with numpy objects
        audio = audio.cpu().numpy()
        if len(audio.shape) >= 3:
            raise NotImplementedError(
                f"input should be single sample, not a batch! shape of audio input to processing_audio: {audio.shape}"
            )
        elif len(audio.shape) == 2:
            assert (
                audio.shape[0] == 1 or audio.shape[0] == 2
            ), f"wrong audio shape to the processing_audio method: {audio.shape}"
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
        in_data = torch.unsqueeze(in_data, 0)
        return in_data  # BxTxF

    def processing_video(self, video, video_frame_rate):
        # video: (video_length, video_channels, height, width), where video_channels is usually 3 for RGB or 1 for greyscale
        # avhubert will make image and audio have the same framerate so we can add them or concat them in feature dimension. image sample rate if 25Hz, audio sample rate is 100Hz (originally 16kHz, but after fbank it's 100Hz), four neighboring audio sample is stacked to get
        # since using av-hubert native implementation, needs to work with numpy objects
        video = video.cpu().numpy()
        if video_frame_rate != self.video_frame_rate:
            if video_frame_rate > self.video_frame_rate:
                assert (
                    video_frame_rate % self.video_frame_rate == 0
                ), f"the input video frame rate is bigger than avhubert' required video framerate, but it's not a multiple of it, which makes it difficult to evenly downsample. input video frame rate: {video_frame_rate}, avhubert's required video frame rate: {self.video_frame_rate}"
                video = video[:: (video_frame_rate // self.video_frame_rate)]
            else:
                assert (
                    self.video_frame_rate % video_frame_rate == 0
                ), f"the avhubert' required video framerate is bigger than the input video frame rate, but it's not a multiple of it, which makes it difficult to evenly upsample. input video frame rate: {video_frame_rate}, avhubert's required video frame rate: {self.video_frame_rate}"
                video = torch.repeat_interleave(
                    video, self.video_frame_rate // video_frame_rate, dim=0
                )
        feats = self.transform(video)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def forward(self, processed_data):
        audio = [item[0] for item in processed_data]
        video = [item[1] for item in processed_data]
        audio_length = torch.LongTensor([len(item) for item in audio])
        video_length = torch.LongTensor([len(item) for item in video])
        assert sum([a == v for a, v in zip(audio_length, video_length)]) == len(
            audio_length
        ), f"each audio should have the same temporal length with the corresponding video, but they are not: audio_length: {audio_length}, video_length: {video_length}"
        padding_mask = ~torch.lt(
            torch.arange(max(audio_length)).unsqueeze(0),
            (audio_length).unsqueeze(1),
        )
        padded_audio = pad_sequence(audio.squeeze(0), batch_first=True)
        padded_video = pad_sequence(video.squeeze(0), batch_first=True)
        source = {
            "audio": padded_audio.transpose(1, 2).cuda(),
            "video": padded_video.cuda(),
        }
        result = self.model(
            source, padding_mask=padding_mask.cuda(), mask=False, features_only=True
        )
        return {"last_hidden_state": result["x"], "hidden_states": result["features"]}
        # ####################below will work for audio only s3prl#######################
        # new_audio = []
        # for audio in processed_data:
        #     processed = self.processing_audio(audio.cpu(), 16000)
        #     new_audio.append(processed)

        # audio_length = torch.LongTensor([len(item) for item in new_audio])
        # padding_mask = ~torch.lt(
        #     torch.arange(max(audio_length)).unsqueeze(0),
        #     (audio_length).unsqueeze(1),
        # )
        # padded_audio = pad_sequence([item.squeeze(0) for item in new_audio], batch_first=True)
        # source = {"audio": padded_audio.transpose(1,2).cuda(), "video": None}
        # result = self.model(source, padding_mask=padding_mask.cuda(), mask=False, features_only=True)
        # return {"last_hidden_state": result["x"], "hidden_states": result["features"]}
        # ####################above will work for audio only s3prl#######################
