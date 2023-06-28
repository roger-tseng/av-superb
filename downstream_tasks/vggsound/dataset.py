import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchvision.io
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torchaudio.transforms import Resample

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
    def __init__(
        self,
        mode,
        preprocess=None,
        preprocess_audio=None,
        preprocess_video=None,
        **kwargs
    ):
        self.vggsound_root = kwargs['vggsound_root']
        self.class_num = kwargs['class_num']
        self.upstream_name = kwargs['upstream']
        self.mode = mode

        self.feature_root = kwargs['feature_root']
        if self.feature_root[-1] != '/':
            self.feature_root += '/'
        self.save_features = kwargs['save_features']

        if mode == "train":
            self.path = kwargs["train_location"]
        elif mode == "validation":
            self.path = kwargs["val_location"]
        elif mode == "test":
            self.path = kwargs["test_location"]

        with open(self.path) as csvfile: self.data = list(csv.reader(csvfile, delimiter=","))


        self.preprocess, self.preprocess_audio, self.preprocess_video = preprocess, preprocess_audio, preprocess_video

        self.upstream_name = kwargs['upstream']
        self.upstream_feature_selection = kwargs['upstream_feature_selection']
        self.pooled_features_path = kwargs['pooled_features_path']


        print("dataset meta path", self.path)
        print("dataset length:", len(self.data))
        if len(self.data) > 0: print("data example:", self.data[0])

    def get_rates(self, idx):
        return SAMPLE_RATE, VIDEO_FRAME_RATE
        # return self.audio_sample_rates[idx], self.video_frame_rates[idx]

    def __getitem__(self, idx):

        start_time = str(int(self.data[idx][1]))
        filename = "_".join([self.data[idx][0], (6 - len(start_time)) * "0" + start_time + ".mp4"])
        filepath = "/".join([self.vggsound_root, filename])

        basename = filepath.rsplit('/')[-1].rsplit('.')[0]

	# label
        label = int(self.data[idx][2])

        if self.pooled_features_path:
            pooled_feature_path = f"{self.pooled_features_path}/{self.upstream_name}_{self.upstream_feature_selection}/{basename}_pooled.pt"
            if os.path.exists(pooled_feature_path):
                # print("feature", pooled_feature_path)
                pooled_feature = torch.load(pooled_feature_path)
                return pooled_feature, pooled_feature, label, True        

        feature_path = f"/work/u2707828/testdata/{self.upstream_name}/{basename}.pt"

        if os.path.exists(feature_path):
            processed_wav, processed_frames = torch.load(feature_path)
        else:
            frames, wav, meta = torchvision.io.read_video(filepath, pts_unit="sec", output_format="TCHW")

            if "mavil" not in self.upstream_name:
                frames = frames.float()
            
            wav = wav.mean(dim=0).squeeze(0)

            audio_fps, video_fps = meta["audio_fps"], meta["video_fps"]

            if self.preprocess is not None:
                processed_frames, processed_wav = self.preprocess(frames, wav, video_fps, audio_fps)
            else:
                ###################preprocess audio############################
                if self.preprocess_audio is not None:
                    processed_wav = self.preprocess_audio(wav, audio_fps)
                else:
                    processed_wav = wav
                ###############################################################
                ###################preprocess video############################
                if self.preprocess_video is not None:
                    processed_frames = self.preprocess_video(frames, video_fps)
                else:
                    processed_frames = frames
                ################################################################

            # if self.mode == "test" and self.save_features:
                # print("save test data",basename)
                # torch.save([processed_wav, processed_frames], feature_path)

        return processed_wav, processed_frames, label, basename


    # len of the dataset
    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        wavs, videos, *others = zip(*samples)
        return wavs, videos, *others
        """
        wavs, videos, labels = [], [], []
        for wav, frames, label in samples:
            wavs.append(wav)
            videos.append(frames)
            labels.append(label)
        return wavs, videos, labels
        """
