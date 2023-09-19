import random
import numpy as np
import os
import tqdm
from pathlib import Path
import itertools

import torch
import torch.nn as nn
import torchvision
from torch.utils.data.dataset import Dataset


# Example parameters
AUDIO_SAMPLE_RATE = 16000
VIDEO_FRAME_RATE = 25
AUDIO_VIDEO_RATE = 640
HEIGHT = 224
WIDTH = 224


def _findAllSeqs(dirName, extension=".mp4", speaker_level=1):
    r"""
    Lists all the sequences with the given extension in the dirName directory.
    Output:
        outSequences, speakers
        outSequence
        A list of tuples seq_path, speaker where:
            - seq_path is the relative path of each sequence relative to the
            parent directory
            - speaker is the corresponding speaker index
        outSpeakers
        The speaker labels (in order)
    The speaker labels are organized the following way
    \dirName
        \speaker_label
            \..
                ...
                seqName.extension
    Adjust the value of speaker_level if you want to choose which level of
    directory defines the speaker label. Ex if speaker_level == 2 then the
    dataset should be organized in the following fashion
    \dirName
        \crappy_label
            \speaker_label
                \..
                    ...
                    seqName.extension
    Set speaker_label == 0 if no speaker label will be retrieved no matter the
    organization of the dataset.
    """
    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)
    speakersTarget = {}
    outSequences = {}
    print("finding {}, Waiting...".format(extension))
    for root, dirs, filenames in tqdm.tqdm(os.walk(dirName, followlinks=True)):
        filtered_files = [f for f in filenames if f.endswith(extension)]
        if len(filtered_files) > 0:
            speakerStr = (os.sep).join(root[prefixSize:].split(os.sep)[:speaker_level])
            if speakerStr not in speakersTarget:
                speakersTarget[speakerStr] = len(speakersTarget)
            speaker = speakersTarget[speakerStr]
            for filename in filtered_files:
                full_path = os.path.join(root, filename)
                if outSequences.get(speakerStr) is None:
                    outSequences[speakerStr] = {}
                video_name = full_path.split("/")[-2]
                if outSequences[speakerStr].get(video_name) is None:
                    outSequences[speakerStr][video_name] = [
                        (speakerStr, speakersTarget[speakerStr], full_path)
                    ]
                else:
                    outSequences[speakerStr][video_name].append(
                        (speakerStr, speakersTarget[speakerStr], full_path)
                    )

    outSpeakers = [None for x in speakersTarget]

    for key, index in speakersTarget.items():
        outSpeakers[index] = key

    return outSequences, outSpeakers


def _get_test_list(
    dirName,
):
    """
    Get training data, validation data and testing trials
    Args:
        dirName: the dirctory to VoxCeleb2 dataroot,
        It should be
        \dirName
            \Test
                \mp4
                    \...
            \Train
                \mp4
                    \...
    """
    TestPath = Path(dirName, "test/mp4")
    TestSavePath = Path(dirName, "test.lst")

    TestSequences, _ = _findAllSeqs(dirName=str(TestPath))
    test_trials = []
    for spk in TestSequences.keys():
        videos = list(TestSequences[spk].keys())
        combinations_list = list(itertools.combinations(videos, 2))
        for v1, v2 in combinations_list:
            d1 = random.choice(TestSequences[spk][v1])[2]
            d2 = random.choice(TestSequences[spk][v2])[2]
            test_trials.append((1, d1, d2))

    spkIDs = list(TestSequences.keys())
    for spk in TestSequences.keys():
        otherspkIDs = [x for x in spkIDs if x != spk]
        videos = list(TestSequences[spk].keys())
        for video in videos:
            d1 = random.choice(TestSequences[spk][video])[2]
            otherspkID = random.choice(otherspkIDs)
            othervideo = random.choice(list(TestSequences[otherspkID].keys()))
            d2 = random.choice(TestSequences[otherspkID][othervideo])[2]
            test_trials.append((0, d1, d2))

    with open(TestSavePath, "w") as f:
        for item in test_trials:
            f.write(",".join(map(str, item)) + "\n")

    return test_trials


def _get_train_list(
    dirName,
):
    """
    Get training data, validation data and testing trials
    Args:
        dirName: the dirctory to VoxCeleb2 dataroot,
        It should be
        \dirName
            \Test
                \mp4
                    \...
            \Train
                \mp4
                    \...
    """
    TrainPath = Path(dirName, "dev/mp4")
    TrainSavePath = Path(dirName, "train.lst")
    ValidSavePath = Path(dirName, "valid.lst")

    TrainSequences, outSpeakers = _findAllSeqs(dirName=str(TrainPath))
    train_list = []
    dev_list = []
    for spk in TrainSequences.keys():
        videos = list(TrainSequences[spk].keys())
        dev_video = random.choice(videos)
        videos.remove(dev_video)
        dev_list.extend(TrainSequences[spk][dev_video])
        for train_video in videos:
            train_list.extend(TrainSequences[spk][train_video])

    with open(TrainSavePath, "w") as f:
        for item in train_list:
            f.write(",".join(map(str, item)) + "\n")
    with open(ValidSavePath, "w") as f:
        for item in dev_list:
            f.write(",".join(map(str, item)) + "\n")

    return train_list, dev_list, outSpeakers


class Classification_Dataset(Dataset):
    def __init__(
        self,
        preprocess=None,
        preprocess_audio=None,
        preprocess_video=None,
        dataroot=None,
        split=None,
        max_timestep=None,
        **kwargs
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

        Args:
            split (str): indicate the dataset is train or valid
            dataroot(str): root directory to VoxCeleb2
            \dataroot
                \Test
                    \mp4
                        \...
                \Train
                    \mp4
                        \...
        """
        self.dataroot = dataroot
        self.split = split
        self._preprocess_data(self.dataroot)

        self.preprocess = preprocess
        self.preprocess_audio = preprocess_audio
        self.preprocess_video = preprocess_video
        self.max_timestep = max_timestep

        self.upstream_name = kwargs['upstream']
        self.upstream_feature_selection = kwargs['upstream_feature_selection']
        self.pooled_features_path = kwargs['pooled_features_path']

        self.skip_steps = 0

    def _preprocess_data(self, roots):
        TrainSavePath = Path(roots, "train_5vid.lst")
        ValidSavePath = Path(roots, "valid_5vid.lst")

        if TrainSavePath.exists() and ValidSavePath.exists():
            with open(TrainSavePath, "r") as file:
                lines = file.readlines()
            train_list = [
                (
                    line.strip().split(",")[0],
                    line.strip().split(",")[1],
                    line.strip().split(",")[2],
                )
                for line in lines
            ]
            with open(ValidSavePath, "r") as file:
                lines = file.readlines()
            valid_list = [
                (
                    line.strip().split(",")[0],
                    line.strip().split(",")[1],
                    line.strip().split(",")[2],
                )
                for line in lines
            ]
            AllSpeakers = list(set([line.strip().split(",")[0] for line in lines]))
        else:
            train_list, valid_list, AllSpeakers = _get_train_list(roots)

        if self.split == "train":
            self.dataset = train_list
        elif self.split == "valid":
            self.dataset = valid_list
        else:
            raise

        self.all_speakers = AllSpeakers
        self.speaker_num = len(self.all_speakers)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.skip_steps > 0:
            # Skip this datapoint to resume training
            self.skip_steps -= 1
            return False, False, False, False
        path = self.dataset[idx][2]
        label = int(self.dataset[idx][1])
        basename = path.replace('/', '_').rsplit('.')[0]
        if self.pooled_features_path:
            pooled_feature_path = f"{self.pooled_features_path}/{self.upstream_name}_{self.upstream_feature_selection}/{basename}_pooled.pt"
            if os.path.exists(pooled_feature_path):
                pooled_feature = torch.load(pooled_feature_path)
                return pooled_feature, pooled_feature, label, True

        try:
            frames, wav, info = torchvision.io.read_video(
                path, pts_unit="sec", output_format="TCHW"
            )
            video_fps = info["video_fps"]
            audio_sr = info["audio_fps"]
        except:
            print(f"damaged file: {path}")

            path = self.dataset[0][2]
            label = int(self.dataset[0][1])
            frames, wav, info = torchvision.io.read_video(
                path, pts_unit="sec", output_format="TCHW"
            )
            video_fps = info["video_fps"]
            audio_sr = info["audio_fps"]
            
        length = wav.shape[1]
        if self.max_timestep != None:
            if length > self.max_timestep:
                audio_start = random.randint(0, int(length - self.max_timestep))
                video_start = audio_start // 640
                wav = wav[:, audio_start: audio_start + self.max_timestep]
                frames = frames[video_start: video_start + self.max_timestep // 640, :, :, :]

        if self.preprocess is not None:
            processed_frames, processed_wav = self.preprocess(frames, wav, video_fps, audio_sr)
        else:    
            if self.preprocess_audio is not None:
                processed_wav = self.preprocess_audio(wav, audio_sr)
            else:
                processed_wav = wav
            if self.preprocess_video is not None:
                try:
                    processed_frames = self.preprocess_video(frames, video_fps)
                except:
                    raise NotImplementedError(f"{idx}, {basename}, {path}, {frames.shape}, {video_fps}")
            else:
                processed_frames = frames

        return processed_wav, processed_frames, label, basename

    def collate_fn(self, samples):
        wavs, videos, *others = zip(*samples)
        return wavs, videos, *others


class Verification_Dataset(Dataset):
    def __init__(
        self, preprocess=None, preprocess_audio=None, preprocess_video=None, dataroot=None, max_timestep=None, **kwargs
    ):
        """
        Args:
            dataroot(str): data root for the VoxCeleb2
        """
        self.root = dataroot
        self.necessary_dict = self._processing()
        self.dataset = self.necessary_dict["spk_paths"]
        self.pair_table = self.necessary_dict["pair_table"]

        self.preprocess = preprocess
        self.preprocess_audio = preprocess_audio
        self.preprocess_video = preprocess_video
        self.max_timestep = max_timestep

        self.upstream_name = kwargs['upstream']
        self.upstream_feature_selection = kwargs['upstream_feature_selection']
        self.pooled_features_path = kwargs['pooled_features_path']

    def _processing(self):
        TestSavePath = Path(self.root, "test.lst")

        if TestSavePath.exists():
            with open(TestSavePath, "r") as file:
                lines = file.readlines()
            test_trials = [
                (
                    line.strip().split(",")[0],
                    line.strip().split(",")[1],
                    line.strip().split(",")[2],
                )
                for line in lines
            ]

        else:
            test_trials = _get_test_list(self.root)

        pair_table = []
        spk_paths = set()

        for pair in test_trials:
            list_pair = pair
            pair_1 = os.path.join(self.root, list_pair[1])
            pair_2 = os.path.join(self.root, list_pair[2])
            spk_paths.add(pair_1)
            spk_paths.add(pair_2)
            one_pair = [list_pair[0], pair_1, pair_2]
            pair_table.append(one_pair)
        return {
            "spk_paths": list(spk_paths),
            "total_spk_num": None,
            "pair_table": pair_table,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path = self.dataset[idx]
        x_name = path

        basename = path.replace('/', '_').rsplit('.')[0]
        if self.pooled_features_path:
            pooled_feature_path = f"{self.pooled_features_path}/{self.upstream_name}_{self.upstream_feature_selection}/{basename}_pooled.pt"
            if os.path.exists(pooled_feature_path):
                pooled_feature = torch.load(pooled_feature_path)
                return pooled_feature, pooled_feature, x_name, True

        # You may use the following function to read video data:
        frames, wav, info = torchvision.io.read_video(
            path, pts_unit="sec", output_format="TCHW"
        )
        video_fps = info["video_fps"]
        audio_sr = info["audio_fps"]

        length = wav.shape[1]
        if self.max_timestep != None:
            if length > self.max_timestep:
                audio_start = random.randint(0, int(length - self.max_timestep))
                video_start = audio_start // AUDIO_VIDEO_RATE
                wav = wav[:, audio_start: audio_start + self.max_timestep]
                frames = frames[video_start: video_start + self.max_timestep // AUDIO_VIDEO_RATE, :, :, :]

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

        return processed_wav, processed_frames, x_name, basename

    def collate_fn(self, samples):
        wavs, videos, *others = zip(*samples)
        return wavs, videos, *others

