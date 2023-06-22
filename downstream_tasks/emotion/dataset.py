import json
from pathlib import Path
from os.path import join as path_join
import numpy as np
import os
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
import imageio
import cv2
import torch


AUDIO_SAMPLE_RATE = 16000 #44100
VIDEO_FRAME_RATE = 25 #30
MIN_SEC = 5
MAX_SEC = 20
DATASET_SIZE = 200
HEIGHT = 224
WIDTH = 224

miss = '/data/member1/user_tahsieh/IEMOCAP/miss.txt' 
miss_list = []
with open(miss, 'r') as f:
    line = f.readline()
    while line:
        line = f.readline().replace('\n','') #去掉換行
        miss_list.append(line)
f.close()


class IEMOCAPDataset(Dataset):
    def __init__(self, iemocap_root, meta_path, preprocess=None, preprocess_audio=None, preprocess_video=None, **kwargs):
        
        self.iemocap_root = iemocap_root        
        with open(meta_path, 'r') as f:
            self.dataset = json.load(f)
            
        index = []
        for i in range(len(self.dataset['meta_data'])):
            path = self.dataset['meta_data'][i]['path']
            if path in miss_list:
                if i not in index:
                    index.append(i)             
        for i in index[::-1]:
            del self.dataset['meta_data'][i]
 
        self.class_dict = self.dataset['labels']
        self.idx2emotion = {value: key for key, value in self.class_dict.items()}
        self.class_num = len(self.class_dict)
        self.meta_data = self.dataset['meta_data']
        
        self.audio_sample_rates = [AUDIO_SAMPLE_RATE] * len(self)
        self.video_frame_rates = [VIDEO_FRAME_RATE] * len(self)
        self.preprocess = preprocess
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
        audio_sr, video_fps = self.get_rates(idx)
        wav, _ = torchaudio.load(path_join(self.iemocap_root, self.meta_data[idx]['path']))
        wav = wav.mean(dim=0).squeeze(0)

        avi_path = path_join(self.iemocap_root, os.path.splitext(self.meta_data[idx]['path'])[0].replace('wav', 'avi_sentence')+'.mp4')
        vid = imageio.get_reader(avi_path, 'ffmpeg')
        capture = cv2.VideoCapture(avi_path)
        frame_count=capture.get(cv2.CAP_PROP_FRAME_COUNT)
        images = []

        for frame_num in range(int(frame_count)):
            image = vid.get_data(frame_num)
            image = np.array(image)
            images.append(image)

        images = np.stack(images)
        images = torch.from_numpy(images)
        frames = torch.permute(images, (0, 3, 1, 2))
        frames = frames.float()

        label = self.meta_data[idx]['label']
        label = self.class_dict[label]
        
        fname = "pre_features"
        feature_path = f"/data/member1/user_tahsieh/IEMOCAP/pre_feature/{self.upstream_name}/{fname.rsplit('/')[-1].rsplit('.')[0]}.pt"
        
        if os.path.exists(feature_path):
            processed_wav, processed_frames = torch.load(feature_path)
        else:
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
                
                # torch.save([processed_wav, processed_frames], feature_path)

        return processed_wav, processed_frames, label, Path(self.meta_data[idx]['path']).stem
        
    def __len__(self):
        return len(self.meta_data)

def collate_fn(samples):
    wavs, videos, labels, files_name = [], [], [], []
    for wav, frames, label, file_name in samples:
        wavs.append(wav)
        videos.append(frames)
        labels.append(label)
        files_name.append(file_name)
    return wavs, videos, labels, files_name