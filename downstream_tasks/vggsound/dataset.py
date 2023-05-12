import csv
import os
import random

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import torchvision.io
import torchvision.transforms as transforms
from scipy import signal
from torch.utils.data.dataset import Dataset
from torchaudio.transforms import Resample

"""
SAMPLE_RATE = 16000
EXAMPLE_WAV_MIN_SEC = 5
EXAMPLE_WAV_MAX_SEC = 20
EXAMPLE_DATASET_SIZE = 200
"""

# {'video_fps': 30.0, 'audio_fps': 44100}
SAMPLE_RATE = 16000
VIDEO_FRAME_RATE = 25
SEC = 10
HEIGHT = 224
WIDTH = 224


class VggsoundDataset(Dataset):
    def __init__(
        self,
        mode,
        vggsound_root,
        class_num,
        preprocess_audio=None,
        preprocess_video=None,
        **kwargs
    ):
        self.vggsound_root = vggsound_root
        self.mode = mode
        self.class_num = class_num

        if mode == "train":
            self.path = kwargs["train_location"]
        elif mode == "validation":
            self.path = kwargs["val_location"]
        elif mode == "test":
            self.path = kwargs["test_location"]
        print("dataset meta path", self.path)

        with open(self.path) as csvfile:
            self.data = list(csv.reader(csvfile, delimiter=","))

        self.preprocess_audio = preprocess_audio
        self.preprocess_video = preprocess_video

        print("dataset length:", len(self.data))
        if len(self.data) > 0:
            print("data example:", self.data[0])

    def get_rates(self, idx):
        return SAMPLE_RATE, VIDEO_FRAME_RATE
        # return self.audio_sample_rates[idx], self.video_frame_rates[idx]

    def __getitem__(self, idx):

        # only audio part
        """
        filename = "_".join(
                [
                    self.data[idx][0],
                    str(int(self.data[idx][1])*1000),
                    str(int(self.data[idx][1])*1000+10000)+".flac"
                ]
        )
        filepath = "/".join([self.vggsound_root,"data","vggsound","audio",filename])

        flac, sr = torchaudio.load(filepath)    # sr = 48000
        resampler = Resample(sr, SAMPLE_RATE)

        flac = resampler(flac)
        flac = flac.mean(dim=0).squeeze(0)
        
	label = int(self.data[idx][2])

        return flac, label
        """

        # video part
        # filename = "_".join(
        #        [
        #            self.data[idx][0],
        #            str(int(self.data[idx][1])*1000),
        #            str(int(self.data[idx][1])*1000+10000)+".mp4"
        #        ]
        # )
        # filepath = "/".join([self.vggsound_root,"data","vggsound","video",filename])

        start_time = str(int(self.data[idx][1]))
        filename = "_".join(
            [self.data[idx][0], (6 - len(start_time)) * "0" + start_time + ".mp4"]
        )
        filepath = "/".join([self.vggsound_root, filename])

        frames, wav, meta = torchvision.io.read_video(
            filepath, pts_unit="sec", output_format="TCHW"
        )

        frames = frames.float()
        wav = wav.mean(dim=0).squeeze(0)

        if self.preprocess_audio is not None:
            #print("audio length before preprocess", wav.shape)
            processed_wav = self.preprocess_audio(wav, meta["audio_fps"])
            #print("audio length after preprocess", processed_wav.shape)
        else:
            processed_wav = wav

        if self.preprocess_video is not None:
            #print("video length before preprocess", frames.shape)
            processed_frames = self.preprocess_video(frames, meta["video_fps"])
            #print("video length after preprocess", processed_frames.shape)
        else:
            processed_frames = frames

        # label
        label = int(self.data[idx][2])

        return processed_wav, processed_frames, label

    # len of the dataset
    def __len__(self):
        return len(self.data)

    """
    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels
    """

    def collate_fn(self, samples):
        wavs, videos, labels = [], [], []
        for wav, frames, label in samples:
            wavs.append(wav)
            videos.append(frames)
            labels.append(label)
        return wavs, videos, labels
