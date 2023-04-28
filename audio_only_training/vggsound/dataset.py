import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchaudio

import csv
import os
from torchaudio.transforms import Resample
import torchvision.transforms as transforms

import soundfile as sf
from scipy import signal

import numpy as np
import librosa
import torchvision.io
from .myutils import *

"""
SAMPLE_RATE = 16000
EXAMPLE_WAV_MIN_SEC = 5
EXAMPLE_WAV_MAX_SEC = 20
EXAMPLE_DATASET_SIZE = 200
"""


SAMPLE_RATE = 16000
VIDEO_FRAME_RATE = 25
SEC = 10
HEIGHT = 224
WIDTH = 224


class VggsoundDataset(Dataset):
    def __init__(self, mode, vggsound_root, class_num, **kwargs):
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
            self.data = list(csv.reader(csvfile,delimiter=","))
        print("dataset length:",len(self.data))
        print("data example:", self.data[0])


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


        filename = "_".join(
                [
                    self.data[idx][0],
                    str(int(self.data[idx][1])*1000),
                    str(int(self.data[idx][1])*1000+10000)+".mp4"
                ]
        )
        filepath = "/".join([self.vggsound_root,"data","vggsound","video",filename])

	

        # video part
        filename = "_".join(
                [
                    self.data[idx][0],
                    str(int(self.data[idx][1])*1000),
                    str(int(self.data[idx][1])*1000+10000)+".mp4"
                ]
        )

        filename = "_".join([self.data[idx][0],self.data[idx][1]+".mkv"])

        filepath = "/".join([self.vggsound_root,filename])

        video, audio, meta = torchvision.io.read_video(filepath, pts_unit='sec')
        resampler = Resample(meta['audio_fps'], SAMPLE_RATE)
        audio = resampler(audio)
        audio = audio.mean(dim=0).squeeze(0)


        # frames, wav = torchvision.io.read_video(filepath, pts_unit="sec", output_format="TCHW")
        

        # label
        label = int(self.data[idx][2])    

        return audio, label



    # len of the dataset
    def __len__(self):
        return len(self.data)

    
    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels
    

    """ 
    def collate_fn(self, samples):
        wavs, mp4s, labels = [], [], []
        for wav, mp4, label in samples:
            wavs.append(wav)
            mp4s.append(mp4)
            labels.append(label)
        return wavs, mp4s, labels
    """
