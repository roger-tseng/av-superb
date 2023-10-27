import csv
import os
import torch
import torchvision.io
from torch.utils.data.dataset import Dataset


class AudiosetDataset(Dataset):
    def __init__(
        self,
        csvname,
        audioset_root,
        preprocess=None,
        preprocess_audio=None,
        preprocess_video=None,
        **kwargs,
    ):
        self.audioset_root = audioset_root
        self.class_num = 527
        self.csv_root = kwargs["csv_root"]
        csvpath = "/".join([self.csv_root, csvname])
        with open(csvpath) as csvfile:
            self.data = list(csv.reader(csvfile))
        self.preprocess = preprocess
        self.preprocess_audio = preprocess_audio
        self.preprocess_video = preprocess_video
        self.upstream_name = kwargs["upstream"]
        self.upstream_feature_selection = kwargs["upstream_feature_selection"]
        self.pooled_features_path = kwargs["pooled_features_path"]

    def __getitem__(self, idx):
        filename = "_".join(
            [
                self.data[idx][0] + ".mp4",
            ]
        )
        filepath = "/".join([self.audioset_root, filename])
        basename = filepath.rsplit("/")[-1].rsplit(".")[0]
        origin_labels = [int(i) for i in self.data[idx][3:]]
        labels = []
        for i in range(self.class_num):
            if i not in origin_labels:
                labels.append(0)
            else:
                labels.append(1)

        if self.pooled_features_path:
            pooled_feature_path = f"{self.pooled_features_path}/{self.upstream_name}_{self.upstream_feature_selection}/{basename}_pooled.pt"
            if os.path.exists(pooled_feature_path):
                pooled_feature = torch.load(pooled_feature_path)
                return pooled_feature, pooled_feature, labels, True

        feature_path = f"/work/u7196393/features/{self.upstream_name}/{basename}.pt"
        if not os.path.exists(feature_path):
            filename = "_".join(
                [
                    self.data[idx][0] + ".mp4",
                ]
            )
            filepath = "/".join([self.audioset_root, filename])

            frames, wav, meta = torchvision.io.read_video(
                filepath, pts_unit="sec", output_format="TCHW"
            )
            wav = wav.mean(dim=0).squeeze(0)
            audio_sr, video_fps = meta["audio_fps"], meta["video_fps"]
        if os.path.exists(feature_path):
            processed_wav, processed_frames = torch.load(feature_path)
        else:
            if self.preprocess is not None:
                processed_frames, processed_wav = self.preprocess(
                    frames, wav, video_fps, audio_sr
                )
            else:
                if self.preprocess_audio is not None:
                    # if "mavil" in self.upstream_name:
                    #     processed_wav = self.preprocess_audio(
                    #         wav, audio_sr, fbank_mean=-4.2677393, fbank_std=4.5689974
                    #     )
                    # else:
                    processed_wav = self.preprocess_audio(wav, audio_sr)
                else:
                    processed_wav = wav
                if self.preprocess_video is not None:
                    processed_frames = self.preprocess_video(frames, video_fps)
                else:
                    processed_frames = frames
            # uncomment next line to save feature
            # torch.save([processed_wav, processed_frames], feature_path)
        return processed_wav, processed_frames, labels, basename

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        wavs, videos, *others = zip(*samples)
        return wavs, videos, *others
