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
    def __init__(self, mode, preprocess=None, preprocess_audio=None, preprocess_video=None, kinetics_root=None, class_num=32, **kwargs):
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

        self.preprocess = preprocess
        self.preprocess_audio = preprocess_audio
        self.preprocess_video = preprocess_video
        
        self.upstream_name = kwargs['upstream']
        self.upstream_feature_selection = kwargs['upstream_feature_selection']
        self.pooled_features_path = kwargs['pooled_features_path']

        self.logs_file = open(kwargs["logs_file"], "w")

    def __getitem__(self, idx):
        path = os.path.join(self.kinetics_root, self.dataset[idx][0])
        
        label = int(self.dataset[idx][1])

        basename = path.rsplit('/')[-1].rsplit('.')[0]

        # Directly load pooled features if exist, 
        # skipping video loading and preprocessing
        if self.pooled_features_path:
            pooled_feature_path = f"{self.pooled_features_path}/{self.upstream_name}_{self.upstream_feature_selection}/{basename}_pooled.pt"
            if os.path.exists(pooled_feature_path):
                pooled_feature = torch.load(pooled_feature_path)
                return pooled_feature, pooled_feature, label, True

        feature_path = f"{self.kinetics_root}/features/{self.upstream_name}/{basename}.pt"
        if os.path.exists(feature_path):
            processed_wav, processed_frames = torch.load(feature_path)
        else:
            # You may use the following function to read video data:
            frames, wav, meta = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
            audio_sr, video_fps = meta.get('audio_fps'), meta.get('video_fps')

            wav = wav.mean(dim=0).squeeze(0)

            # no audio && no video
            if wav.shape[0] == 0 and frames.shape[0] == 0:
                self.logs_file.write("{0}, '{1}', {2}, no data\n".format(path, frames.shape, video_fps))
                self.logs_file.flush()
                print("no data", path)

            # no audio
            if wav.shape[0] == 0:
                self.logs_file.write("{0}, '{1}', {2}, no audio\n".format(path, frames.shape, video_fps))
                self.logs_file.flush()
                print("no audio", audio_sr)
                audio_samples = random.randint(
                    MIN_SEC * AUDIO_SAMPLE_RATE, MAX_SEC * AUDIO_SAMPLE_RATE
                )
                wav = torch.zeros(audio_samples)
                audio_sr = AUDIO_SAMPLE_RATE

            # no video
            if frames.shape[0] == 0:
                self.logs_file.write("{0}, '{1}', {2}, no video\n".format(path, frames.shape, video_fps))
                self.logs_file.flush()
                print("no video", video_fps)
                video_samples = random.randint(
                    MIN_SEC * VIDEO_FRAME_RATE, MAX_SEC * VIDEO_FRAME_RATE
                )
                frames = torch.ones(video_samples, 3, random.randint(50, HEIGHT), random.randint(50, WIDTH), dtype=torch.uint8)
                video_fps = VIDEO_FRAME_RATE

            # preprocess
            if self.preprocess is not None:
                processed_frames, processed_wav = self.preprocess(frames, wav, video_fps, audio_sr)
            else:
                if self.preprocess_audio is not None:
                    processed_wav = self.preprocess_audio(wav, audio_sr)
                else:
                    processed_wav = wav
                if self.preprocess_video is not None:
                    processed_frames = self.preprocess_video(frames, video_fps)
                else:
                    processed_frames = frames
            
            # save
            # torch.save([processed_wav, processed_frames], feature_path)

        return processed_wav, processed_frames, label, basename

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, samples):
        wavs, videos, *others = zip(*samples)
        return wavs, videos, *others
