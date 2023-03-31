import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchaudio

import csv
import os
from torchaudio.transforms import Resample

"""
SAMPLE_RATE = 16000
EXAMPLE_WAV_MIN_SEC = 5
EXAMPLE_WAV_MAX_SEC = 20
EXAMPLE_DATASET_SIZE = 200
"""


SAMPLE_RATE = 16000


class VggsoundDataset(Dataset):
    def __init__(self, mode, vggsound_root, **kwargs):
        self.vggsound_root = vggsound_root
        self.mode = mode
        self.class_num = 309

        if mode == "train":
            self.path = kwargs["train_location"]
        elif mode == "validation":
            self.path = kwargs["val_location"]
        elif mode == "test":
            self.path = kwargs["test_location"]
        print("dataset meta path", self.path)

        with open(self.path) as csvfile:
            self.data = list(csv.reader(csvfile,delimiter=","))
        print("data example:", self.data[0])




    def __getitem__(self, idx):
        
        filename = "_".join([self.data[idx][0],str(int(self.data[idx][1])*1000),str(int(self.data[idx][1])*1000+10000)+".flac"])
        filepath = "/".join([self.vggsound_root,"data","vggsound","audio",filename])

        flac, sr = torchaudio.load(filepath)

        resampler = Resample(sr, SAMPLE_RATE)
        flac = flac[0]
        flac = resampler(flac).squeeze(0)

        label = int(self.data[idx][2])
        
        return flac, label

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels
