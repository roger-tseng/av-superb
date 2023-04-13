"""
Custom class for loading audio-visual data 
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/dataset.py
"""
import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

PATH_ROOT = '/saltpool0/data/layneberry/lrs3/lrs3_v0.4/'
DATASET_SIZE = 0
for split in ['pretrain', 'trainval', 'test']:
    video_list = open(PATH_ROOT+split+'.txt').readlines()
    DATASET_SIZE += len(video_list)

# Example parameters
AUDIO_SAMPLE_RATE = 16000 # DONE
VIDEO_FRAME_RATE = 25 # DONE
MIN_SEC = .48 # DONE
MAX_SEC = 6.2 # DONE
# DATASET_SIZE = 200 # DONE: written above
HEIGHT = 224 # DONE
WIDTH = 224 # DONE


class RandomDataset(Dataset):
    def __init__(self, **kwargs):
        self.class_num = 48 # TODO
        self.AUDIO_SAMPLE_RATE = AUDIO_SAMPLE_RATE
        self.VIDEO_FRAME_RATE = VIDEO_FRAME_RATE
        # TODO: Load metadata



    def get_rates(self): # DONE
        """
        Return audio sample rates and video frame rates
        """
        return {
            [self.AUDIO_SAMPLE_RATE] * len(self),
            [self.VIDEO_FRAME_RATE] * len(self),
        }

    def __getitem__(self, idx): # TODO
        audio_samples = random.randint(
            MIN_SEC * AUDIO_SAMPLE_RATE, MAX_SEC * AUDIO_SAMPLE_RATE
        )
        video_samples = random.randint(
            MIN_SEC * VIDEO_FRAME_RATE, MAX_SEC * VIDEO_FRAME_RATE
        )

        # frames, wav = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
        wav = torch.randn(1, audio_samples)
        frames = torch.randn(video_samples, 3, HEIGHT, WIDTH)
        label = random.randint(0, self.class_num - 1) # TODO
        return wav, frames, label

    def __len__(self): # DONE
        return DATASET_SIZE

    def collate_fn(self, samples): # DONE
        wavs, videos, labels = [], [], []
        for wav, frames, label in samples:
            wavs.append(wav)
            videos.append(frames)
            labels.append(label)
        return wavs, videos, labels
