"""
Custom class for loading audio-visual data 
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/dataset.py
"""
import random, json, torchvision

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

PATH_ROOT = '/saltpool0/data/layneberry/lrs3/lrs3_v0.4/'
DATASET_SIZE = 0
for split in ['pretrain', 'trainval', 'test']:
    video_list = open(PATH_ROOT+split+'.txt').readlines()
    DATASET_SIZE += len(video_list)
# This doesn't end up getting used, bc train and test sets are different sizes

# Example parameters
AUDIO_SAMPLE_RATE = 16000 # DONE
VIDEO_FRAME_RATE = 25 # DONE
MIN_SEC = .48 # DONE
MAX_SEC = 6.2 # DONE
# DATASET_SIZE = 200 # DONE: written above
HEIGHT = 224 # DONE
WIDTH = 224 # DONE


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

        self.class_num = 28 # DONE: Num chars plus one
        self.AUDIO_SAMPLE_RATE = AUDIO_SAMPLE_RATE
        self.VIDEO_FRAME_RATE = VIDEO_FRAME_RATE
        self.preprocess_audio = preprocess_audio
        self.preprocess_video = preprocess_video
        
        # print('Loading json')
        if split == 'train':
            self.dataset = json.load(open('/saltpool0/data/layneberry/lrs3/train_set_metadata_clean.json'))
        else:
            self.dataset = json.load(open('/saltpool0/data/layneberry/lrs3/test_set_metadata_clean.json'))
        # print('Json loading done!')

    def get_rates(self, idx): # DONE
        """
        Return audio sample rates and video frame rates
        """
        # This is constant for this dataset so don't need to use idx
        return {
            [self.AUDIO_SAMPLE_RATE] * len(self),
            [self.VIDEO_FRAME_RATE] * len(self),
        }

    def char2label(self, c):
        if c == '\'':
            return 26
        elif c == '|':
            return 27
        else:
            return ord(c)-65 # maps A-Z to 0-25

    def __getitem__(self, idx): # DONE
        # print('Reading video at idx', idx)
        frames, wav, meta = torchvision.io.read_video('/saltpool0/data/layneberry/lrs3/'+self.dataset[idx]['path'], pts_unit="sec", output_format="TCHW")
        assert(meta['audio_fps'] == self.AUDIO_SAMPLE_RATE)
        assert(meta['video_fps'] == self.VIDEO_FRAME_RATE)
        
        wav = wav.squeeze(0)
        if self.preprocess_audio != None:
            wav = self.preprocess_audio(wav, self.AUDIO_SAMPLE_RATE)
        if self.preprocess_video != None:
            frames = self.preprocess_video(frames, self.VIDEO_FRAME_RATE)

        labels = torch.Tensor([self.char2label(c) for c in self.dataset[idx]['text']])
        return wav, frames, labels

    def __len__(self): # DONE
        return len(self.dataset)

    def collate_fn(self, samples): # DONE
        # print('Collating')
        wavs, videos, labels = [], [], []
        for wav, frames, label in samples:
            wavs.append(wav)
            videos.append(frames)
            labels.append(label)
        return wavs, videos, labels
