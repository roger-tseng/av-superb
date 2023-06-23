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


SAMPLE_RATE = 44100
VIDEO_FRAME_RATE = 30
SEC = 10
HEIGHT = 224
WIDTH = 224


class AudiosetDataset(Dataset):
    def __init__(
        self,
        csvname,
        audioset_root,
        preprocess=None,
        preprocess_audio=None,
        preprocess_video=None,
        **kwargs,
    ):
        self.audioset_root = audioset_root
        self.class_num = 527
        self.csv_root = kwargs["csv_root"]
        csvpath = "/".join([self.csv_root, csvname])
        with open(csvpath) as csvfile:
            self.data = list(csv.reader(csvfile))
        self.preprocess = preprocess
        self.preprocess_audio = preprocess_audio
        self.preprocess_video = preprocess_video
        self.upstream_name = kwargs["upstream"]
        # print("dataset length:", len(self.data))
        # print("data example:", self.data[0])

    def get_rates(self, idx):
        return SAMPLE_RATE, VIDEO_FRAME_RATE

    def __getitem__(self, idx):
        # Audio only part
        """
        filename = "_".join(
            [
                self.data[idx][0],
                str(int(float(self.data[idx][1]) * 1000)),
                str(int(float(self.data[idx][2]) * 1000)) + ".flac",
            ]
        )
        filepath = "/".join(
            [self.audioset_root, "data", "eval_segments", "audio", filename]
        )

        flac, sr = torchaudio.load(filepath)
        # flac=flac[0]
        def resampler(original_sample_rate, sample_rate):
            return torchaudio.transforms.Resample(original_sample_rate, sample_rate)

        flac = resampler(sr, SAMPLE_RATE)(flac)
        flac = flac.mean(dim=0).squeeze(0)

        # test
        # print(flac)
        #####
        origin_labels = [int(i) for i in self.data[idx][3:]]
        # print(origin_labels)
        labels = []
        for i in range(self.class_num):
            if i not in origin_labels:
                labels.append(0)
            else:
                labels.append(1)
        # label = int(self.data[idx][3])
        # print(labels)
        return flac, labels
        """
        # video part
        # print(idx)
        # print(self.data[idx][0])
        audio_sr, video_fps = self.get_rates(idx)
        filename = "_".join(
            [
                self.data[idx][0] + ".mp4",
            ]
        )
        filepath = "/".join([self.audioset_root, filename])
        feature_path = f"/work/u7196393/features/{self.upstream_name}/{filepath.rsplit('/')[-1].rsplit('.')[0]}.pt"
        if not os.path.exists(feature_path):
            filename = "_".join(
                [
                    self.data[idx][0] + ".mp4",
                ]
            )
            filepath = "/".join([self.audioset_root, filename])

            frames, wav, meta = torchvision.io.read_video(
                filepath, pts_unit="sec", output_format="TCHW"
            )
            frames = frames.float()
            # {'video_fps': 30.0, 'audio_fps': 44100}
            wav = wav.mean(dim=0).squeeze(0)

            # print(type(frames)) ; print(frames.size()) ; print(frames)
        # feature_path = f"/work/u7196393/features/{self.upstream_name}/{filepath.rsplit('/')[-1].rsplit('.')[0]}.pt"
        if os.path.exists(feature_path):
            processed_wav, processed_frames = torch.load(feature_path)
        else:
            if self.preprocess is not None:
                processed_frames, processed_wav = self.preprocess(frames, wav, video_fps, audio_sr)
            else:
                if self.preprocess_audio is not None:
                    processed_wav = self.preprocess_audio(wav, SAMPLE_RATE)
                else:
                    processed_wav = wav
                if self.preprocess_video is not None:
                    processed_frames = self.preprocess_video(frames, VIDEO_FRAME_RATE)
                else:
                    processed_frames = frames
            # uncomment next line to save feature
            # torch.save([processed_wav, processed_frames], feature_path)

        # label
        origin_labels = [int(i) for i in self.data[idx][3:]]
        # print(origin_labels)
        labels = []
        for i in range(self.class_num):
            if i not in origin_labels:
                labels.append(0)
            else:
                labels.append(1)
        # label = int(self.data[idx][3])
        # print(labels)
        return processed_wav, processed_frames, labels

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        wavs, videos, labels = [], [], []
        for wav, frames, label in samples:
            wavs.append(wav)
            videos.append(frames)
            labels.append(label)
        return wavs, videos, labels
