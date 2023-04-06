import csv
import os
import random

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data.dataset import Dataset
from torchaudio.transforms import Resample

SAMPLE_RATE = 16000


class KineticsDataset(Dataset):
    def __init__(self, mode, kinetics_root, **kwargs):
        self.kinetics_root = kinetics_root
        self.mode = mode

        if mode == "train":
            self.path = kwargs["train_meta_location"]
        elif mode == "validation":
            self.path = kwargs["val_meta_location"]
        elif mode == "test":
            self.path = kwargs["test_meta_location"]
        print("dataset meta path", self.path)

        file = open(self.path, "r")
        data = list(csv.reader(file, delimiter=","))
        file.close()
        print("data example", data[0])

        self.dataset = data
        self.class_num = 700

    def __getitem__(self, idx):
        audio_path = os.path.join(self.kinetics_root, self.dataset[idx][0])

        wav, sr = torchaudio.load(audio_path)
        # print(wav.shape)
        resampler = Resample(sr, SAMPLE_RATE)
        wav = resampler(wav).mean(dim=0).squeeze(0)

        # print(self.dataset[idx][0], "wav length", len(wav))

        label = int(self.dataset[idx][1])

        # print(audio_path, label)

        return wav, label

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels
