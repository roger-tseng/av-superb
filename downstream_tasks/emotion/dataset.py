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



SAMPLE_RATE = 16000

miss = '/data/member1/user_tahsieh/IEMOCAP/miss.txt' 
miss_list = []
with open(miss, 'r') as f:
    line = f.readline()
    while line:
        line = f.readline().replace('\n','') #去掉換行
        miss_list.append(line)
f.close()


class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir, meta_path, pre_load=True):
        self.data_dir = data_dir
        self.pre_load = pre_load
        with open(meta_path, 'r') as f:
            self.data = json.load(f)
            
        index = []
        for i in range(len(self.data['meta_data'])):
            path = self.data['meta_data'][i]['path']
            if path in miss_list:
                if i not in index:
                    index.append(i)
             
        for i in index[::-1]:
            del self.data['meta_data'][i]
 
        self.class_dict = self.data['labels']
        self.idx2emotion = {value: key for key, value in self.class_dict.items()}
        self.class_num = len(self.class_dict)
        self.meta_data = self.data['meta_data']
        
        _, origin_sr = torchaudio.load(
            path_join(self.data_dir, self.meta_data[0]['path']))
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        if self.pre_load:
            self.wavs = self._load_all()
            

    def _load_wav(self, path):
        wav, _ = torchaudio.load(path_join(self.data_dir, path))
        wav = self.resampler(wav).squeeze(0)
        return wav
    
    def _load_avi_npy(self, path):
        avi_path = path_join(self.data_dir, os.path.splitext(path)[0].replace('wav', 'avi_sentence')+'.mp4')
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
        avi = torch.permute(images, (0, 3, 1, 2))
        
        return avi

    def _load_all(self):
        wavforms = []
        for info in self.meta_data:
            wav = self._load_wav(info['path'])
            wavforms.append(wav)
        return wavforms

    def __getitem__(self, idx):
        label = self.meta_data[idx]['label']
        label = self.class_dict[label]
        avi = self._load_avi_npy(self.meta_data[idx]['path'])

        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(self.meta_data[idx]['path'])
        return wav, avi, label, Path(self.meta_data[idx]['path']).stem

    def __len__(self):
        return len(self.meta_data)

def collate_fn(samples):
    return zip(*samples)