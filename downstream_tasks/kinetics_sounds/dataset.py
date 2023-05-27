"""
Custom class for loading audio-visual data 
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/dataset.py
"""
import random

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

import torchaudio
import torchvision
from torchaudio.transforms import Resample

# Example parameters
AUDIO_SAMPLE_RATE = 44100
VIDEO_FRAME_RATE = 30
MIN_SEC = 5
MAX_SEC = 10
HEIGHT = 224
WIDTH = 224


class KineticsSoundsDataset(Dataset):
    def __init__(self, preprocess_audio, preprocess_video, mode, kinetics_root, class_num, **kwargs):
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
        self.class_num = class_num
        
        self.audio_sample_rates = [AUDIO_SAMPLE_RATE] * len(self)
        self.video_frame_rates = [VIDEO_FRAME_RATE] * len(self)
        self.preprocess_audio = preprocess_audio
        self.preprocess_video = preprocess_video
        self.upstream_name = kwargs['upstream']

        self.logs_file = open(kwargs["logs_file"], "w")


    # def get_rates(self, idx):
    #     """
    #     Return the audio sample rate and video frame rate of the idx-th video.
    #     (Datasets may contain data with different sample rates)
    #     """
    #     return self.audio_sample_rates[idx], self.video_frame_rates[idx]

    def __getitem__(self, idx):
        path = os.path.join(self.kinetics_root, self.dataset[idx][0])

        # You may use the following function to read video data:
        frames, wav, meta = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
        audio_sr, video_fps = meta.get('audio_fps'), meta.get('video_fps')

        label = int(self.dataset[idx][1])

        wav = wav.mean(dim=0).squeeze(0)

        if wav.shape[0] == 0 and frames.shape[0] == 0:
            self.logs_file.write("{0}, '{1}', {2}\n".format(path, frames.shape, video_fps))
            self.logs_file.flush()
            print("no data", path)

        if wav.shape[0] == 0:
            print("no audio", audio_sr)
            audio_samples = random.randint(
                MIN_SEC * AUDIO_SAMPLE_RATE, MAX_SEC * AUDIO_SAMPLE_RATE
            )
            wav = torch.zeros(audio_samples)
            audio_sr = AUDIO_SAMPLE_RATE
            # print(path)
            # rand_idx = random.randint(0, len(self.dataset)-1)
            # path = os.path.join(self.kinetics_root, self.dataset[rand_idx][0])
            # frames, wav, meta = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
            # label = int(self.dataset[rand_idx][1])
            # wav = wav.mean(dim=0).squeeze(0)

        if frames.shape[0] == 0:
            print("no video", video_fps)
            video_samples = random.randint(
                MIN_SEC * VIDEO_FRAME_RATE, MAX_SEC * VIDEO_FRAME_RATE
            )
            frames = torch.zeros([video_samples, 3, random.randint(50, HEIGHT), random.randint(50, WIDTH)])
            video_fps = VIDEO_FRAME_RATE
        
        # Run preprocessing only if features are not precomputed
        fname = path
        feature_path = f"/work/u8090533/features/{self.upstream_name}/{fname.rsplit('/')[-1].rsplit('.')[0]}.pt"
        if os.path.exists(feature_path):
            processed_wav, processed_frames = torch.load(feature_path)
        else:
            if self.preprocess_audio is not None:
                processed_wav = self.preprocess_audio(wav, audio_sr)
            else:
                processed_wav = wav
            if self.preprocess_video is not None:
                processed_frames = self.preprocess_video(frames, video_fps)
            else:
                processed_frames = frames
            # Uncomment the next line
            torch.save([processed_wav, processed_frames], feature_path)

        return processed_wav, processed_frames, label

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, samples):
        wavs, videos, labels = [], [], []
        for wav, frames, label in samples:
            wavs.append(wav)
            videos.append(frames)
            labels.append(label)
        return wavs, videos, labels
