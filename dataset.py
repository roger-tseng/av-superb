import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchaudio

import os
import csv

"""
SAMPLE_RATE = 16000
EXAMPLE_WAV_MIN_SEC = 5
EXAMPLE_WAV_MAX_SEC = 20
EXAMPLE_DATASET_SIZE = 200
"""


SAMPLE_RATE = 16000



class AudiosetDataset(Dataset):
    def __init__(self, csvname, audioset_root, **kwargs):
        self.audioset_root = audioset_root
        self.class_num = 527

        with open(audioset_root + "/csv/" + csvname) as csvfile:
            self.data = list(csv.reader(csvfile))




    def __getitem__(self, idx):              
        filename = "_".join([self.data[idx][0],str(int(float(self.data[idx][1])*1000)),str(int(float(self.data[idx][2])*1000))+".flac"])
        filepath = "/".join([self.audioset_root,"data","eval_segments","audio",filename])

        flac, sr = torchaudio.load(filepath)
        #flac=flac[0]
        def resampler(original_sample_rate, sample_rate):
            return torchaudio.transforms.Resample(original_sample_rate, sample_rate)

        flac = resampler(sr, SAMPLE_RATE)(flac)
        flac=flac.mean(dim=0).squeeze(0)
        
        #test
        #print(flac)
        #####
        origin_labels = [int(i) for i in self.data[idx][3:]]
        # print(origin_labels)
        labels=[]
        for i in range(self.class_num):
            if i not in origin_labels:
                labels.append(0)
            else:
                labels.append(1)
        # label = int(self.data[idx][3])
        # print(labels)
        return flac, labels

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels

