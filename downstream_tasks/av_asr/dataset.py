"""
Custom class for loading audio-visual data 
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/dataset.py
"""
import json
import os
import random

import torch
import torch.nn as nn
import torchvision
from torch.utils.data.dataset import Dataset

from .fairseq_dictionary import Dictionary

class RandomDataset(Dataset):
    def __init__(
        self,
        preprocess=None,
        preprocess_audio=None,
        preprocess_video=None,
        split=None,
        **kwargs
    ):
        """
        Your dataset should take two preprocessing transform functions,
        preprocess_audio and preprocess_video as input.

        These two functions will be defined by the upstream models, and
        will transform raw waveform & video frames into the desired
        format of the upstream model.

        They take two arguments, the input audio/video Tensor, and the
        audio sample rate/video frame rate, respectively.

        Optionally, if you wish to obtain raw data for testing purposes,
        you may also specify these functions to be None, and return the
        raw data when the functions are not defined.
        """

        # Create Dictionary object
        self.dictionary = Dictionary.load("downstream_tasks/av_asr/char.dict")
        self.class_num = len(self.dictionary)

        self.preprocess = preprocess
        self.preprocess_audio = preprocess_audio
        self.preprocess_video = preprocess_video

        self.upstream_name = kwargs['upstream']
        self.upstream_feature_selection = kwargs['upstream_feature_selection']
        self.pooled_features_path = kwargs['pooled_features_path']

        self.full_path_root = kwargs['path_root'] + "/"

        if split == "train":
            self.dataset = json.load(
                open(self.full_path_root + "train_set_metadata_clean.json")
            )
        elif split == "val":
            self.dataset = json.load(
                open(self.full_path_root + "val_set_metadata_clean.json")
            )
        else:
            self.dataset = json.load(
                open(self.full_path_root + "test_set_metadata_clean.json")
            )

        self.skip_steps = 0

    def __getitem__(self, idx):
        if self.skip_steps > 0:
            # Skip this datapoint to resume training
            self.skip_steps -= 1
            return False, False, False, False

        path = self.full_path_root + self.dataset[idx]["path"]
        labels = self.dictionary.encode_line(
            " ".join(list(self.dataset[idx]["text"])),
            line_tokenizer=lambda x: x.split(),
        ).long()

        basename = path.replace('/', '_').rsplit('.')[0]
        if self.pooled_features_path:
            pooled_feature_path = f"{self.pooled_features_path}/{self.upstream_name}_{self.upstream_feature_selection}/{basename}_pooled.pt"
            if os.path.exists(pooled_feature_path):
                pooled_feature = torch.load(pooled_feature_path)
                return pooled_feature, pooled_feature, labels, True

        frames, wav, meta = torchvision.io.read_video(
            path,
            pts_unit="sec",
            output_format="TCHW",
        )
        audio_sr, video_fps = meta["audio_fps"], meta["video_fps"]

        wav = wav.squeeze(0)
        if self.preprocess is not None:
            processed_frames, processed_wav = self.preprocess(frames, wav, video_fps, audio_sr)
        else:    
            if self.preprocess_audio is not None:
                processed_wav = self.preprocess_audio(wav, audio_sr)
            else:
                processed_wav = wav
            if self.preprocess_video is not None:
                processed_frames = self.preprocess_video(frames, video_fps)
            else:
                processed_frames = frames

        return processed_wav, processed_frames, labels, basename

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, samples):
        wavs, videos, *others = zip(*samples)
        return wavs, videos, *others
