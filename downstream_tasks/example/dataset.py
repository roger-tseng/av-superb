"""
Custom class for loading audio-visual data 
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/dataset.py
"""
import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

# Example parameters
AUDIO_SAMPLE_RATE = 16000
VIDEO_FRAME_RATE = 25
MIN_SEC = 5
MAX_SEC = 20
DATASET_SIZE = 200
HEIGHT = 224
WIDTH = 224


class RandomDataset(Dataset):
    def __init__(self, **kwargs):
        self.class_num = 48
        self.AUDIO_SAMPLE_RATE = AUDIO_SAMPLE_RATE
        self.VIDEO_FRAME_RATE = VIDEO_FRAME_RATE

    def get_rates(self):
        """
        Return audio sample rates and video frame rates
        """
        return {
            [self.AUDIO_SAMPLE_RATE]*len(self),
            [self.VIDEO_FRAME_RATE]*len(self),
        }

    def __getitem__(self, idx):
        audio_samples = random.randint(MIN_SEC * AUDIO_SAMPLE_RATE, MAX_SEC * AUDIO_SAMPLE_RATE)
        video_samples = random.randint(MIN_SEC * VIDEO_FRAME_RATE, MAX_SEC * VIDEO_FRAME_RATE)
        
        # frames, wav = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
        wav = torch.randn(1,audio_samples)
        frames = torch.randn(video_samples, 3, HEIGHT, WIDTH)
        label = random.randint(0, self.class_num - 1)
        return wav, frames, label

    def __len__(self):
        return DATASET_SIZE

    def collate_fn(self, samples):
        wavs, videos, labels = [], [], []
        for wav, frames, label in samples:
            wavs.append(wav)
            videos.append(frames)
            labels.append(label)
        return wavs, videos, labels
