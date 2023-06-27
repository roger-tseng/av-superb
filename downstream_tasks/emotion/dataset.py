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


AUDIO_SAMPLE_RATE = 44100

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
        
        self.preprocess = preprocess
        self.preprocess_audio = preprocess_audio
        self.preprocess_video = preprocess_video
        self.upstream_name = kwargs['upstream']
        self.upstream_feature_selection = kwargs['upstream_feature_selection']
        self.pooled_features_path = kwargs['pooled_features_path']

        _, origin_sr = torchaudio.load(path_join(self.iemocap_root, self.meta_data[0]['path']))
        self.resampler = Resample(origin_sr, AUDIO_SAMPLE_RATE)

    def __getitem__(self, idx):
        wav, audio_sr = torchaudio.load(path_join(self.iemocap_root, self.meta_data[idx]['path']))
        wav = self.resampler(wav).squeeze(0)

        avi_path = path_join(self.iemocap_root, os.path.splitext(self.meta_data[idx]['path'])[0].replace('wav', 'avi_sentence')+'.mp4')
        vid = imageio.get_reader(avi_path, 'ffmpeg')
        capture = cv2.VideoCapture(avi_path)
        video_fps = capture.get(cv2.CAP_PROP_FPS)
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
        
        fname = self.meta_data[idx]['path']
        basename = Path(self.meta_data[idx]['path']).stem
        
        if self.pooled_features_path:
            pooled_feature_path = f"{self.pooled_features_path}/{self.upstream_name}_{self.upstream_feature_selection}/{basename}_pooled.pt"
            if os.path.exists(pooled_feature_path):
                pooled_feature = torch.load(pooled_feature_path)
                return pooled_feature, pooled_feature, label, True

        feature_path = f"/data/member1/user_tahsieh/IEMOCAP/preprocess_features/{self.upstream_name}/{fname.split('/')[0]}/{fname.split('/')[-2]}/{basename}.pt"
        
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
            
            # if not os.path.isdir(os.path.dirname(feature_path)):
            #     os.mkdir(os.path.dirname(feature_path))
            # torch.save([processed_wav, processed_frames], feature_path)

        return processed_wav, processed_frames, label, basename
        
    def __len__(self):
        return len(self.meta_data)

def collate_fn(samples):
    wavs, videos, *others = zip(*samples)
    return wavs, videos, *others
    # wavs, videos, labels, files_name = [], [], [], []
    # for wav, frames, label, file_name in samples:
    #     wavs.append(wav)
    #     videos.append(frames)
    #     labels.append(label)
    #     files_name.append(file_name)
    # return wavs, videos, labels, files_name