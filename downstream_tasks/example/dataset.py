"""
Custom class for loading audio-visual data
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/dataset.py
"""
import random
import os

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

# Example parameters
AUDIO_SAMPLE_RATE = 44100
VIDEO_FRAME_RATE = 30
MIN_SEC = 5
MAX_SEC = 20
DATASET_SIZE = 200
HEIGHT = 224
WIDTH = 224


class RandomDataset(Dataset):
    def __init__(self, preprocess_audio=None, preprocess_video=None, **kwargs):
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
        self.class_num = 48
        self.audio_sample_rates = [AUDIO_SAMPLE_RATE] * len(self)
        self.video_frame_rates = [VIDEO_FRAME_RATE] * len(self)
        self.preprocess_audio = preprocess_audio
        self.preprocess_video = preprocess_video
        self.upstream_name = kwargs['upstream']

    def get_rates(self, idx):
        """
        Return the audio sample rate and video frame rate of the idx-th video.
        (Datasets may contain data with different sample rates)
        """
        return self.audio_sample_rates[idx], self.video_frame_rates[idx]

    def __getitem__(self, idx):
        length = random.randint(MIN_SEC, MAX_SEC)
        audio_samples = length * AUDIO_SAMPLE_RATE
        video_samples = length * VIDEO_FRAME_RATE
        audio_sr, video_fps = self.get_rates(idx)
        # You may use the following function to read video data:
        # frames, wav = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
        wav = torch.randn(audio_samples)
        frames = torch.randn(
            video_samples, 3, random.randint(50, HEIGHT), random.randint(50, WIDTH)
        )

        # Run preprocessing only if features are not precomputed
        fname = "path to your video"
        feature_path = f"/work/b07901163/features/{self.upstream_name}/{fname.rsplit('/')[-1].rsplit('.')[0]}.pt"
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
            # torch.save([processed_wav, processed_frames], feature_path)

        label = random.randint(0, self.class_num - 1)
        return processed_wav, processed_frames, label

    def __len__(self):
        return DATASET_SIZE

    def collate_fn(self, samples):
        wavs, videos, labels = [], [], []
        for wav, frames, label in samples:
            wavs.append(wav)
            videos.append(frames)
            labels.append(label)
        return wavs, videos, labels
