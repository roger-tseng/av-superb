"""
Custom class for loading audio-visual data 
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/dataset.py
"""
import os
import csv

import torch
import torch.nn as nn
import torchvision
from torch.utils.data.dataset import Dataset

class UCF101Dataset(Dataset):
    def __init__(
        self,
        split,
        preprocess=None,
        preprocess_audio=None,
        preprocess_video=None,
        base_path=None,
        class_num=101,
        **kwargs,
    ):
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
        self.class_num = class_num
        self.split = split
        self.base_path = base_path

        videos = os.listdir(f"{base_path}")
        if split == "train":
            self.path = kwargs.get("train_meta_location")
        elif split == "dev":
            self.path = kwargs.get("val_meta_location")
        elif split == "test":
            self.path = kwargs.get("test_meta_location")

        try:
            file = open(self.path, "r")
            data = list(csv.reader(file, delimiter=","))
            file.close()
        except FileNotFoundError:
            data = []

        self.video_list = data
        self.preprocess = preprocess
        self.preprocess_audio = preprocess_audio
        self.preprocess_video = preprocess_video
        self.upstream_name = kwargs["upstream"]
        self.upstream_feature_selection = kwargs['upstream_feature_selection']
        self.pooled_features_path = kwargs['pooled_features_path']

    def __getitem__(self, idx):
        # You may use the following function to read video data:
        basename = self.video_list[idx][0]+".avi"
        video_path = os.path.join(self.base_path, basename)
        label = int(self.video_list[idx][1])

        # Directly load pooled features if exist, 
        # skipping video loading and preprocessing
        if self.pooled_features_path:
            pooled_feature_path = f"{self.pooled_features_path}/{self.upstream_name}_{self.upstream_feature_selection}/{basename}_pooled.pt"
            if os.path.exists(pooled_feature_path):
                pooled_feature = torch.load(pooled_feature_path)
                return pooled_feature, pooled_feature, label, True

        # Run preprocessing only if features are not precomputed
        feature_path = f"{self.base_path}/features/{self.upstream_name}/{basename}.pt"
        if os.path.exists(feature_path):
            processed_wav, processed_frames = torch.load(feature_path)
        else:
            frames, wav, meta = torchvision.io.read_video(
                video_path, pts_unit="sec", output_format="TCHW"
            )
            audio_sr, video_fps = meta.get('audio_fps'), meta.get('video_fps')
            assert audio_sr == 44100 and video_fps == 25.0, f"audio_sr: {audio_sr}, video_fps: {video_fps}, path: {video_path}"
            wav = wav.mean(dim=0).squeeze(0)

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
            # Uncomment the next line
            # torch.save([processed_wav, processed_frames], feature_path)

        return processed_wav, processed_frames, label, basename

    def __len__(self):
        return len(self.video_list)

    def collate_fn(self, samples):
        wavs, videos, *others = zip(*samples)
        # Concise way of doing:
        # wavs, videos, labels = [], [], []
        # for wav, frames, label in samples:
        #     wavs.append(wav)
        #     videos.append(frames)
        #     labels.append(label)
        return wavs, videos, *others
