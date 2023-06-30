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

PATH_ROOT = (
    "/saltpool0/data/layneberry/lrs"  # Inside Dataset init, appends either '2/' or '3/'
)

# Other parameters
AUDIO_SAMPLE_RATE = 16000
VIDEO_FRAME_RATE = 25
MIN_SEC = 0.48
MAX_SEC = 6.2
HEIGHT = 224
WIDTH = 224


class RandomDataset(Dataset):
    def __init__(
        self,
        preprocess_audio=None,
        preprocess_video=None,
        split=None,
        lrs_version=None,
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

        self.AUDIO_SAMPLE_RATE = AUDIO_SAMPLE_RATE
        self.VIDEO_FRAME_RATE = VIDEO_FRAME_RATE

        self.preprocess_audio = preprocess_audio
        self.preprocess_video = preprocess_video

        self.lrs_version = lrs_version
        self.full_path_root = PATH_ROOT + str(lrs_version) + "/"

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

        self.all_lengths = json.load(open("all_audio_lengths.json"))

    def get_rates(self, idx):
        """
        Return audio sample rates and video frame rates
        """
        # This is constant for this dataset so don't need to use idx
        return {
            [self.AUDIO_SAMPLE_RATE] * len(self),
            [self.VIDEO_FRAME_RATE] * len(self),
        }

    def __getitem__(self, idx):
        file_path = self.full_path_root + self.dataset[idx]["path"]
        feature_path = (
            "/saltpool0/scratch/layneberry/avhubert_output/"
            + self.dataset[idx]["path"][:-4].replace("/", "_")
            + ".pth"
        )
        if os.path.exists(feature_path) and os.path.exists(feature_path + "_fusion"):
            audio_features, video_features = torch.load(feature_path)
            fusion_features = torch.load(feature_path + "_fusion")
            wav, frames = audio_features, video_features  # for returning easily
            length = self.all_lengths[self.dataset[idx]["path"]]
            feature_path = (feature_path, fusion_features)
        else:
            frames, wav, meta = torchvision.io.read_video(
                self.full_path_root + self.dataset[idx]["path"],
                pts_unit="sec",
                output_format="TCHW",
            )
            assert meta["audio_fps"] == self.AUDIO_SAMPLE_RATE
            assert meta["video_fps"] == self.VIDEO_FRAME_RATE

            wav = wav.squeeze(0)
            if self.preprocess_audio != None:
                wav = self.preprocess_audio(wav, self.AUDIO_SAMPLE_RATE)
            if self.preprocess_video != None:
                frames = self.preprocess_video(frames, self.VIDEO_FRAME_RATE)
            frames = frames.float()
            length = len(frames)

        labels = self.dictionary.encode_line(
            " ".join(list(self.dataset[idx]["text"])),
            line_tokenizer=lambda x: x.split(),
        ).long()
        return wav, frames, labels, feature_path, length

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, samples):
        wavs, videos, labels, paths, lens = [], [], [], [], []
        for wav, frames, label, pth, length in samples:
            wavs.append(wav)
            videos.append(frames)
            labels.append(label)
            paths.append(pth)
            lens.append(length)
        return wavs, videos, labels, paths, lens
